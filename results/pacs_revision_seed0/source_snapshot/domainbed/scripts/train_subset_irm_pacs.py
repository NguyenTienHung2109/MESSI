"""GPU-only, source-selected image-dataset runner for Subset-IRM."""

import argparse
import collections
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from domainbed import algorithms
from domainbed.lib import misc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_jsonl(path, value):
    def json_value(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu()
            return item.item() if item.ndim == 0 else item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): json_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_value(child) for child in item]
        return item
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(value), sort_keys=True) + "\n")


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def split_keys(length, fraction, seed, env):
    keys = list(range(length))
    np.random.RandomState(misc.seed_hash(seed, env)).shuffle(keys)
    count = int(length * fraction)
    return keys[count:], keys[:count]


def make_schedule(in_keys, steps, batch_size, seed, env):
    rng = np.random.RandomState(misc.seed_hash(seed, env, "subset_irm_schedule"))
    keys = np.asarray(in_keys, dtype=np.int64)
    return keys[rng.randint(0, len(keys), size=(steps, batch_size))]


class FixedBatchSampler:
    def __init__(self, schedule):
        self.schedule = schedule

    def __iter__(self):
        for row in self.schedule:
            yield row.tolist()

    def __len__(self):
        return len(self.schedule)


def accuracy(model, data, device, batch_size, num_workers):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            correct += int(model.predict(x).argmax(1).eq(y).sum().item())
            total += y.numel()
    model.train()
    return correct / total if total else 0.0


@torch.no_grad()
def component_accuracies(model, data, device, batch_size, num_workers):
    """Evaluate final, global-only, and subset-only predictions."""
    if not getattr(model, "global_expert_enabled", False):
        return {}
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct = collections.Counter()
    total = 0
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        components = model._subset_forward_components(x)
        for name, logits in (
                ("final", components[0]),
                ("subset_only", components[6]),
                ("global_only", components[8])):
            correct[name] += int(logits.argmax(1).eq(y).sum().item())
        total += y.numel()
    model.train()
    return {name: count / total if total else 0.0
            for name, count in correct.items()}


@torch.no_grad()
def source_expert_diagnostics(model, source_sets, device, batch_size,
                              num_workers):
    if not getattr(model, "subset_enabled", False):
        return {}
    risks = []
    accuracies = []
    component_scores = {}
    similarity_sums = collections.defaultdict(lambda: None)
    selected_cosine_weight = None
    selected_count_total = None
    similarity_count = 0
    for env, data in source_sets.items():
        ce_sum = torch.zeros(model.num_experts, device=device)
        correct = torch.zeros(model.num_experts, device=device)
        count = 0
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        model.eval()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            components = model._subset_forward_components(x)
            expert_logits = components[4]
            for expert in range(model.num_experts):
                ce_sum[expert] += torch.nn.functional.cross_entropy(
                    expert_logits[:, expert], y, reduction="sum"
                )
                correct[expert] += expert_logits[:, expert].argmax(1).eq(y).sum()
            count += y.numel()
            if getattr(model, "global_expert_enabled", False):
                from domainbed.subset_irm import expert_representation_diagnostics
                info = expert_representation_diagnostics(
                    components[3], components[4], components[7], components[8],
                    components[2],
                )
                for key in (
                    "feature_cosine_matrix", "feature_linear_cka_matrix",
                    "logit_cosine_matrix", "prediction_agreement_matrix",
                    "feature_variance",
                ):
                    weighted = info[key].float() * x.shape[0]
                    similarity_sums[key] = (
                        weighted if similarity_sums[key] is None
                        else similarity_sums[key] + weighted
                    )
                selected_count = info["subset_selected_count"].float()
                selected_count_total = (
                    selected_count if selected_count_total is None
                    else selected_count_total + selected_count
                )
                cosine_weight = (
                    info["global_to_selected_subset_cosine"].float()
                    * selected_count
                )
                selected_cosine_weight = (
                    cosine_weight if selected_cosine_weight is None
                    else selected_cosine_weight + cosine_weight
                )
                similarity_count += x.shape[0]
        risks.append((ce_sum / count).cpu().tolist())
        accuracies.append((correct / count).cpu().tolist())
        if getattr(model, "global_expert_enabled", False):
            component_scores[str(env)] = component_accuracies(
                model, data, device, batch_size, num_workers
            )
    model.train()
    result = {
        "source_val_all_expert_risk": risks,
        "source_val_per_expert_accuracy": accuracies,
        "risk_ema_matrix": model.risk_ema.risk.detach().cpu().tolist(),
        "risk_ema_valid_matrix": model.risk_ema.valid.detach().cpu().tolist(),
        "q_matrix": model._assignment(
            torch.full((model.num_domains, model.num_experts),
                       1 / model.num_experts, device=device),
            torch.arange(model.num_domains, device=device),
        ).cpu().tolist(),
    }
    if getattr(model, "global_expert_enabled", False):
        result["component_accuracy_by_source_env"] = component_scores
        result["expert_similarity_labels"] = ["global"] + [
            f"subset_{index}" for index in range(model.num_experts)
        ]
        for key, value in similarity_sums.items():
            result[f"expert_similarity_{key}"] = (
                value / max(similarity_count, 1)
            ).cpu().tolist()
        result["expert_similarity_subset_selected_count"] = (
            selected_count_total.cpu().tolist()
        )
        result["expert_similarity_global_to_selected_subset_cosine"] = (
            selected_cosine_weight / selected_count_total.clamp_min(1)
        ).cpu().tolist()
    return result


def heatmap(matrix, row_names, title, path, column_names=None):
    array = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    image = ax.imshow(array, aspect="auto", cmap="viridis")
    column_names = column_names or [f"E{i}" for i in range(array.shape[1])]
    ax.set_xticks(range(array.shape[1]), column_names)
    ax.set_yticks(range(array.shape[0]), row_names)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def resolve_run(config, run_name):
    if run_name not in config["runs"]:
        raise ValueError(f"unknown run {run_name}")
    spec = config["runs"][run_name]
    hparams = dict(config["base_hparams"])
    hparams.update(spec.get("hparams", {}))
    return spec, hparams


def save_checkpoint(path, model, metadata):
    payload = dict(metadata)
    payload["model_dict"] = collections.OrderedDict(
        (key, value.detach().cpu()) for key, value in model.state_dict().items()
    )
    torch.save(payload, path)


def target_checkpoint_diagnostics(checkpoint_records, target_env):
    """Summarize target metrics after source-only checkpoint selection."""
    if not checkpoint_records:
        raise ValueError("checkpoint_records must not be empty")

    target_in_key = f"env{target_env}_in_acc"
    target_out_key = f"env{target_env}_out_acc"
    last_record = checkpoint_records[-1]
    target_peak_record = max(
        checkpoint_records,
        key=lambda record: record[target_out_key],
    )
    return {
        "domainbed_oracle": {
            "selection": "last checkpoint (DomainBed OracleSelectionMethod)",
            "step": int(last_record["step"]),
            "val_accuracy": float(last_record[target_out_key]),
            "test_accuracy": float(last_record[target_in_key]),
        },
        "target_out_peak_diagnostic": {
            "selection": "argmax held-out env out_acc; diagnostic only",
            "step": int(target_peak_record["step"]),
            "val_accuracy": float(target_peak_record[target_out_key]),
            "test_accuracy": float(target_peak_record[target_in_key]),
        },
    }


def save_sirm_res_checkpoint(path, model, metadata):
    """Checkpoint sufficient to resume either S-IRM-res phase exactly."""
    payload = dict(metadata)
    payload.update({
        "algorithm": "S-IRM-res",
        "phase": model.phase,
        "global_step": int(model.global_step.item()),
        "residual_step": int(model.residual_step.item()),
        "model_state_dict": collections.OrderedDict(
            (key, value.detach().cpu()) for key, value in model.state_dict().items()
        ),
        "optimizer_state_dict": (
            model.global_optimizer.state_dict() if model.phase == "global"
            else model.residual_optimizer.state_dict()
        ),
        "scheduler_state_dict": None,
        "global_anchor_hash": model.global_anchor_hash,
        "config": metadata.get("manifest", {}).get("hparams", {}),
    })
    torch.save(payload, path)


@torch.no_grad()
def sirm_res_component_accuracies(model, data, device, batch_size, num_workers):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct = collections.Counter()
    total = 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        values = model.forward_components(x)
        for name, logits in (("full", values["final_logits"]),
                             ("global_only", values["global_logits"])):
            correct[name] += int(logits.argmax(1).eq(y).sum().item())
        if values["residual_logits"] is not None:
            for residual in range(model.num_residual_experts):
                logits = model.forward_components(
                    x, remove_residual=residual
                )["final_logits"]
                correct[f"remove_residual_{residual}"] += int(
                    logits.argmax(1).eq(y).sum().item()
                )
        total += y.numel()
    model.train()
    return {name: value / total if total else 0.0 for name, value in correct.items()}


@torch.no_grad()
def sirm_res_diagnostics(model, data_by_env, device, batch_size, num_workers):
    """Source/target diagnostic only; it never participates in selection."""
    result = {"phase": model.phase, "anchor_hash": model.anchor_hash(), "by_env": {}}
    for env, data in data_by_env.items():
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        total = 0
        global_correct = full_correct = 0
        routing_sum = routing_count = None
        residual_norm = global_norm = None
        feature_cosine_sum = None
        residual_competence_ce = None
        model.eval()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            values = model.forward_components(x)
            global_logits, final_logits = values["global_logits"], values["final_logits"]
            global_correct += int(global_logits.argmax(1).eq(y).sum().item())
            full_correct += int(final_logits.argmax(1).eq(y).sum().item())
            total += y.numel()
            if values["residual_logits"] is None:
                continue
            routing = values["routing"]
            residual_logits = values["residual_logits"]
            residual_features = values["residual_features"].float()
            global_features = values["global_features"].float()
            rs = routing.sum(0)
            rc = routing.gt(0).sum(0)
            rn = residual_logits.float().norm(dim=-1).sum(0)
            gn = global_logits.float().norm(dim=-1).sum()
            cosine = torch.nn.functional.cosine_similarity(
                residual_features, global_features.unsqueeze(1), dim=-1
            ).sum(0)
            competence = torch.stack([
                torch.nn.functional.cross_entropy(
                    global_logits + model.residual_scale * residual_logits[:, m],
                    y, reduction="sum"
                ) for m in range(model.num_residual_experts)
            ])
            routing_sum = rs if routing_sum is None else routing_sum + rs
            routing_count = rc if routing_count is None else routing_count + rc
            residual_norm = rn if residual_norm is None else residual_norm + rn
            global_norm = gn if global_norm is None else global_norm + gn
            feature_cosine_sum = cosine if feature_cosine_sum is None else feature_cosine_sum + cosine
            residual_competence_ce = competence if residual_competence_ce is None else residual_competence_ce + competence
        info = {
            "global_accuracy": global_correct / total if total else 0.0,
            "full_accuracy": full_correct / total if total else 0.0,
            "residual_gain": (full_correct - global_correct) / total if total else 0.0,
        }
        if routing_sum is not None:
            info.update({
                "routing_mean_mass": (routing_sum / total).cpu().tolist(),
                "routing_selected_fraction": (routing_count / total).cpu().tolist(),
                "residual_logit_norm": (residual_norm / total).cpu().tolist(),
                "global_logit_norm": float((global_norm / total).item()),
                "residual_to_global_logit_norm_ratio": (
                    residual_norm / global_norm.clamp_min(1e-8)
                ).cpu().tolist(),
                "global_residual_feature_cosine": (
                    feature_cosine_sum / total
                ).cpu().tolist(),
                "global_plus_residual_competence_ce": (
                    residual_competence_ce / total
                ).cpu().tolist(),
            })
        result["by_env"][str(env)] = info
    model.train()
    return result


def run_sirm_res(args, config, spec, hparams, output_dir):
    """Source-selected two-stage runner. Target data is touched only post-selection."""
    from domainbed import datasets
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    seed = int(config["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    dataset_name = config.get("dataset", "PACS")
    dataset_class = getattr(datasets, dataset_name)
    env_names = list(dataset_class.ENVIRONMENTS)
    target_env = int(config["target_env"] if args.target_env is None else args.target_env)
    source_envs = [env for env in range(len(env_names)) if env != target_env]
    total_steps = int(hparams["sirm_res_global_steps"] + hparams["sirm_res_residual_steps"])
    if args.steps is not None and int(args.steps) != total_steps:
        raise ValueError("S-IRM-res --steps must equal global_steps + residual_steps")
    checkpoint_freq = int(args.checkpoint_freq or config["checkpoint_freq"])
    train_hparams, eval_hparams = dict(hparams, data_augmentation=True), dict(hparams, data_augmentation=False)
    train_dataset = dataset_class(args.data_dir, [target_env], train_hparams)
    eval_dataset = dataset_class(args.data_dir, [target_env], eval_hparams)
    split = {env: dict(zip(("in", "out"), split_keys(
        len(train_dataset[env]), config["holdout_fraction"], seed, env
    ))) for env in range(len(env_names))}
    schedules = {env: make_schedule(split[env]["in"], total_steps, hparams["batch_size"], seed, env)
                 for env in source_envs}
    digest = hashlib.sha256()
    for step in range(total_steps):
        for env in source_envs:
            digest.update(schedules[env][step].tobytes())
    source_val_sets = {env: Subset(eval_dataset[env], split[env]["out"]) for env in source_envs}
    model = algorithms.get_algorithm_class("S-IRM-res")(
        train_dataset.input_shape, train_dataset.num_classes, len(source_envs), hparams
    ).to("cuda")
    parameter_count = lambda params: int(sum(parameter.numel() for parameter in params))
    total_parameter_count = parameter_count(model.parameters())
    global_trainable_count = parameter_count(model._global_parameters())
    residual_trainable_count = parameter_count(model._residual_parameters())
    activated_residual_count = parameter_count(
        list(model.residual_experts[:model.router_topk].parameters())
        + list(model.residual_heads.heads[:model.router_topk].parameters())
    )
    manifest = {
        "run": args.run, "description": spec["description"], "algorithm": "S-IRM-res",
        "dataset": dataset_name, "environment_order": env_names, "target_env": target_env,
        "source_envs": source_envs, "seed": seed, "steps": total_steps,
        "global_steps": model.global_steps, "residual_steps": model.residual_steps,
        "hparams": hparams, "sample_schedule_sha256": digest.hexdigest(),
        "selection": "stage 1: source-out global; stage 2: source-out full",
        "target_used_during_training": False, "target_used_for_checkpoint_selection": False,
        "compute_accounting": {
            "total_parameters": total_parameter_count,
            "stage1_trainable_parameters": global_trainable_count,
            "stage2_trainable_parameters": residual_trainable_count,
            "active_expert_parameters_per_inference": (
                parameter_count(model.global_expert.parameters()) + activated_residual_count
            ),
            "active_branches": f"1 global + top-{model.router_topk} residual",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    best_global = best_full = -1.0
    best_global_step = best_full_step = None
    start_step = 0
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        if restored.get("algorithm") != "S-IRM-res":
            raise ValueError("--resume is not an S-IRM-res checkpoint")
        model.load_state_dict(restored["model_state_dict"], strict=True)
        model.global_anchor_hash = restored.get("global_anchor_hash", "")
        if restored["phase"] == "residual":
            model.enter_residual_stage()
            model.residual_optimizer.load_state_dict(restored["optimizer_state_dict"])
        elif int(restored["global_step"]) >= model.global_steps:
            # A selected global-anchor checkpoint is phase="global" by
            # construction.  Its training budget is complete, so resume at
            # the residual transition rather than take an extra global step.
            model.enter_residual_stage()
        else:
            model.global_optimizer.load_state_dict(restored["optimizer_state_dict"])
        start_step = int(restored["global_step"] + restored["residual_step"])
        best_global, best_full = float(restored.get("best_global_source", -1)), float(restored.get("best_full_source", -1))
    # Slice fixed schedules directly on resume.  Replaying old batches is both
    # wasteful and can take many minutes on disk-backed image datasets.
    loaders = [DataLoader(
        train_dataset[env],
        batch_sampler=FixedBatchSampler(schedules[env][start_step:]),
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    ) for env in source_envs]
    iterator = [iter(loader) for loader in loaders]
    window, records = collections.defaultdict(list), []
    start = time.time()
    for step in range(start_step, total_steps):
        minibatches = []
        for item in iterator:
            x, y = next(item)
            minibatches.append((x.cuda(non_blocking=True), y.cuda(non_blocking=True)))
        values = model.update(minibatches)
        for key, value in values.items():
            if isinstance(value, (int, float)):
                window[key].append(float(value))
        append_jsonl(output_dir / "diagnostics.jsonl", {"step": step, **values})
        boundary = model.phase == "global" and int(model.global_step.item()) == model.global_steps
        due = step % checkpoint_freq == 0 or step == total_steps - 1 or boundary
        if due:
            source_scores = {str(env): accuracy(model, data, "cuda", config["eval_batch_size"],
                                                 max(0, config["num_workers"] // 2))
                             for env, data in source_val_sets.items()}
            source_mean = float(np.mean(list(source_scores.values())))
            record = {"step": step, "phase": model.phase, "source_val_by_env": source_scores,
                      "source_val_accuracy": source_mean,
                      "global_step": int(model.global_step.item()),
                      "residual_step": int(model.residual_step.item()),
                      **{key: float(np.mean(value)) for key, value in window.items()}}
            records.append(record); append_jsonl(output_dir / "train_log.jsonl", record)
            if model.phase == "global" and source_mean > best_global:
                best_global, best_global_step = source_mean, step
                save_sirm_res_checkpoint(checkpoints / "best_global.pkl", model, {
                    "step": step, "source_val_accuracy": source_mean, "manifest": manifest,
                    "best_global_source": best_global, "best_full_source": best_full,
                    "best_global_source_val": best_global, "best_full_source_val": best_full,
                })
            if model.phase == "residual" and source_mean > best_full:
                best_full, best_full_step = source_mean, step
                save_sirm_res_checkpoint(checkpoints / "best_full.pkl", model, {
                    "step": step, "source_val_accuracy": source_mean, "manifest": manifest,
                    "best_global_source": best_global, "best_full_source": best_full,
                    "best_global_source_val": best_global, "best_full_source_val": best_full,
                })
            print(json.dumps({"run": args.run, **record}, sort_keys=True), flush=True)
            window = collections.defaultdict(list)
        if boundary:
            anchor = torch.load(checkpoints / "best_global.pkl", map_location="cpu", weights_only=False)
            model.load_state_dict(anchor["model_state_dict"], strict=True)
            model.enter_residual_stage()
            append_jsonl(output_dir / "phase_transition.jsonl", {
                "step": step, "selected_global_step": best_global_step,
                "selected_global_source": best_global, "anchor_hash": model.global_anchor_hash,
            })
    save_sirm_res_checkpoint(checkpoints / "last.pkl", model, {
        "step": total_steps - 1, "manifest": manifest, "best_global_source": best_global,
        "best_full_source": best_full, "best_global_source_val": best_global,
        "best_full_source_val": best_full, "best_global_step": best_global_step, "best_full_step": best_full_step,
    })
    selected = torch.load(checkpoints / "best_full.pkl", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True); model.enter_residual_stage(); model.to("cuda")
    source_diagnostics = sirm_res_diagnostics(model, source_val_sets, "cuda", config["eval_batch_size"],
                                               max(0, config["num_workers"] // 2))
    write_json(output_dir / "source_selected_diagnostics.json", source_diagnostics)
    if args.skip_target_eval:
        target_components = None
        target_diagnostics = None
    else:
        target_in, target_out = (
            Subset(eval_dataset[target_env], split[target_env][name])
            for name in ("in", "out")
        )
        target_components = sirm_res_component_accuracies(
            model, target_in, "cuda", config["eval_batch_size"],
            max(0, config["num_workers"] // 2),
        )
        target_diagnostics = sirm_res_diagnostics(
            model, {target_env: target_out}, "cuda",
            config["eval_batch_size"], max(0, config["num_workers"] // 2),
        )
        write_json(output_dir / "target_diagnostics.json", target_diagnostics)
    summary = {
        "run": args.run, "algorithm": "S-IRM-res", "selected_global_step": best_global_step,
        "selected_global_source_val": best_global, "selected_full_step": best_full_step,
        "source_val_accuracy": best_full, "target_component_accuracy": target_components,
        "target_diagnostics": target_diagnostics, "anchor_hash": model.global_anchor_hash,
        "anchor_unchanged": model.anchor_hash() == model.global_anchor_hash,
        "wall_time_seconds": time.time() - start,
        "target_evaluated_after_source_selection": not args.skip_target_eval,
        "target_used_for_checkpoint_selection": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    spec, hparams = resolve_run(config, args.run)
    if spec["algorithm"] == "S-IRM-res":
        return run_sirm_res(args, config, spec, hparams, Path(args.output_dir))
    from domainbed import datasets
    if config["seed"] != 0:
        raise ValueError("fixed protocol requires seed 0")
    steps = int(args.steps or config["steps"])
    checkpoint_freq = int(args.checkpoint_freq or config["checkpoint_freq"])
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()

    seed = int(config["seed"])
    dataset_name = config.get("dataset", "PACS")
    try:
        dataset_class = getattr(datasets, dataset_name)
    except AttributeError as error:
        raise ValueError(f"unknown DomainBed dataset: {dataset_name}") from error
    env_names = list(dataset_class.ENVIRONMENTS)
    num_envs = len(env_names)
    target_env = int(config["target_env"] if args.target_env is None
                     else args.target_env)
    if target_env not in range(num_envs):
        raise ValueError(
            f"{dataset_name} target environment must be in "
            f"[0, {num_envs - 1}]"
        )
    source_envs = [env for env in range(num_envs) if env != target_env]
    if (args.target_env is None and
            env_names[target_env] != config["target_name"]):
        raise RuntimeError(
            f"configured {dataset_name} target name disagrees with dataset"
        )
    print(f"{dataset_name} environments: {env_names}", flush=True)
    print(f"Held-out environment {target_env}: {env_names[target_env]}", flush=True)
    print(f"Training source environments: {[env_names[i] for i in source_envs]}", flush=True)

    seed_everything(seed)
    train_hparams = dict(hparams, data_augmentation=True)
    eval_hparams = dict(hparams, data_augmentation=False)
    train_dataset = dataset_class(args.data_dir, [target_env], train_hparams)
    eval_dataset = dataset_class(args.data_dir, [target_env], eval_hparams)
    split = {}
    for env in range(num_envs):
        in_keys, out_keys = split_keys(
            len(train_dataset[env]), config["holdout_fraction"], seed, env
        )
        split[env] = {"in": in_keys, "out": out_keys}

    schedules = {
        env: make_schedule(split[env]["in"], steps, hparams["batch_size"],
                           seed, env)
        for env in source_envs
    }
    digest = hashlib.sha256()
    for step in range(steps):
        for env in source_envs:
            digest.update(schedules[env][step].tobytes())
    schedule_hash = digest.hexdigest()
    loaders = [DataLoader(
        train_dataset[env], batch_sampler=FixedBatchSampler(schedules[env]),
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    ) for env in source_envs]
    loader_iters = [iter(loader) for loader in loaders]
    source_val_sets = {
        env: Subset(eval_dataset[env], split[env]["out"])
        for env in source_envs
    }
    # Match DomainBed's standard checkpoint logging: evaluate every in/out
    # split. The target metrics are monitoring-only; selection below still
    # depends exclusively on source out-split accuracy. Source-only calibration
    # can retain strict target blindness via --skip-target-eval.
    eval_envs = source_envs if args.skip_target_eval else list(range(num_envs))
    eval_sets = {
        f"env{env}_{split_name}": Subset(
            eval_dataset[env], split[env][split_name]
        )
        for env in eval_envs
        for split_name in ("in", "out")
    }

    algorithm_class = algorithms.get_algorithm_class(spec["algorithm"])
    model = algorithm_class(
        train_dataset.input_shape, train_dataset.num_classes,
        len(source_envs), hparams,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    manifest = {
        "run": args.run,
        "description": spec["description"],
        "algorithm": spec["algorithm"],
        "dataset": dataset_name,
        "environment_order": env_names,
        "target_env": target_env,
        "target_name": env_names[target_env],
        "source_envs": source_envs,
        "source_names": [env_names[i] for i in source_envs],
        "seed": seed,
        "trial_seed": seed,
        "model_seed": seed,
        "data_seed": seed,
        "steps": steps,
        "checkpoint_freq": checkpoint_freq,
        "sample_schedule_sha256": schedule_hash,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hparams": hparams,
        "target_used_during_training": False,
        "target_evaluated_at_checkpoints": not args.skip_target_eval,
        "target_used_for_checkpoint_selection": False,
        "selection": "argmax mean source out-split accuracy",
    }
    write_json(output_dir / "manifest.json", manifest)

    best_source = -1.0
    best_step = None
    checkpoint_records = []
    window = collections.defaultdict(list)
    collapse_guard = config.get("q_collapse_guard", {})
    collapse_streak = 0
    start = time.time()
    for step in range(steps):
        minibatches = []
        for iterator in loader_iters:
            x, y = next(iterator)
            minibatches.append((x.cuda(non_blocking=True), y.cuda(non_blocking=True)))
        values = model.update(minibatches)
        for key, value in values.items():
            if isinstance(value, (int, float)):
                window[key].append(float(value))
        guard_enabled = bool(collapse_guard.get("enabled", False))
        guard_interval = max(1, int(collapse_guard.get("interval", 100)))
        if (guard_enabled and
                step >= int(collapse_guard.get("start_step", 0)) and
                step % guard_interval == 0):
            max_mass = float(values.get("q_capacity_max_mean_mass", 0.0))
            threshold = float(collapse_guard.get("max_mean_mass", 1.0))
            collapse_streak = collapse_streak + 1 if max_mass > threshold else 0
            guard_record = {
                "step": step,
                "q_capacity_max_mean_mass": max_mass,
                "max_mean_mass_threshold": threshold,
                "collapse_streak": collapse_streak,
                "required_consecutive_checks": int(
                    collapse_guard.get("consecutive_checks", 3)
                ),
                "dead_expert_count": float(
                    values.get("dead_expert_count", 0.0)
                ),
                "ess_min": float(values.get("ess_min", 0.0)),
                "routing_entropy": float(values.get("routing_entropy", 0.0)),
            }
            append_jsonl(output_dir / "q_collapse_guard.jsonl", guard_record)
            if collapse_streak >= guard_record["required_consecutive_checks"]:
                write_json(output_dir / "q_collapse.json", guard_record)
                raise RuntimeError(
                    "Q collapse guard triggered: "
                    f"max_mean_mass={max_mass:.4f} > {threshold:.4f} for "
                    f"{collapse_streak} consecutive checks"
                )
        if getattr(model, "last_diagnostics", None) and (
                step % max(1, int(hparams.get("subset_irm_gradient_log_interval", 100))) == 0):
            append_jsonl(output_dir / "diagnostics.jsonl", {
                "step": step, **values, **model.last_diagnostics
            })
        if step % checkpoint_freq != 0 and step != steps - 1:
            continue
        eval_scores = {
            f"{name}_acc": accuracy(
                model, dataset, "cuda", config["eval_batch_size"],
                max(0, config["num_workers"] // 2),
            ) for name, dataset in eval_sets.items()
        }
        source_scores = {
            str(env): eval_scores[f"env{env}_out_acc"]
            for env in source_envs
        }
        source_mean = float(np.mean(list(source_scores.values())))
        record = {
            "step": step,
            "optimizer_steps_completed": step + 1,
            **{key: float(np.mean(vals)) for key, vals in window.items()},
            **eval_scores,
            "source_val_by_env": source_scores,
            "source_val_accuracy": source_mean,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "wall_time_seconds": time.time() - start,
        }
        append_jsonl(output_dir / "train_log.jsonl", record)
        checkpoint_records.append(record)
        if source_mean > best_source:
            best_source = source_mean
            best_step = step
            save_checkpoint(checkpoints / "best.pkl", model, {
                "step": step, "source_val_accuracy": source_mean,
                "manifest": manifest,
            })
        window = collections.defaultdict(list)
        print(json.dumps({"run": args.run, **record}, sort_keys=True), flush=True)

    save_checkpoint(checkpoints / "last.pkl", model, {
        "step": steps - 1, "best_step": best_step,
        "best_source_val_accuracy": best_source, "manifest": manifest,
    })
    selected = torch.load(checkpoints / "best.pkl", map_location="cpu",
                          weights_only=False)
    model.load_state_dict(selected["model_dict"], strict=True)
    model.to("cuda")
    diagnostics = source_expert_diagnostics(
        model, source_val_sets, "cuda", config["eval_batch_size"],
        max(0, config["num_workers"] // 2),
    )
    if diagnostics:
        write_json(output_dir / "source_selected_diagnostics.json", diagnostics)
        rows = [env_names[i] for i in source_envs]
        heatmap(diagnostics["source_val_all_expert_risk"], rows,
                "Source validation expert risk R", output_dir / "R_heatmap.png")
        heatmap(diagnostics["q_matrix"], rows,
                "Source-only assignment Q", output_dir / "Q_heatmap.png")
        if "expert_similarity_labels" in diagnostics:
            labels = diagnostics["expert_similarity_labels"]
            for key, title, filename in (
                ("expert_similarity_feature_cosine_matrix",
                 "Mean feature cosine", "expert_feature_cosine.png"),
                ("expert_similarity_feature_linear_cka_matrix",
                 "Mean batch linear CKA", "expert_feature_cka.png"),
                ("expert_similarity_logit_cosine_matrix",
                 "Mean logit cosine", "expert_logit_cosine.png"),
                ("expert_similarity_prediction_agreement_matrix",
                 "Prediction agreement", "expert_prediction_agreement.png"),
            ):
                heatmap(diagnostics[key], labels, title,
                        output_dir / filename, column_names=labels)

    # The held-out view is materialized/evaluated only after source selection.
    if args.skip_target_eval:
        target_accuracy = None
        target_out_accuracy = None
        target_component_accuracy = None
        target_out_component_accuracy = None
    else:
        target_in = Subset(eval_dataset[target_env], split[target_env]["in"])
        target_out = Subset(eval_dataset[target_env], split[target_env]["out"])
        target_accuracy = accuracy(
            model, target_in, "cuda", config["eval_batch_size"],
            max(0, config["num_workers"] // 2),
        )
        target_out_accuracy = accuracy(
            model, target_out, "cuda", config["eval_batch_size"],
            max(0, config["num_workers"] // 2),
        )
        target_component_accuracy = component_accuracies(
            model, target_in, "cuda", config["eval_batch_size"],
            max(0, config["num_workers"] // 2),
        )
        target_out_component_accuracy = component_accuracies(
            model, target_out, "cuda", config["eval_batch_size"],
            max(0, config["num_workers"] // 2),
        )
    summary = {
        "run": args.run,
        "prediction_mode": hparams["subset_irm_prediction_mode"],
        "topk": hparams["subset_irm_router_topk"],
        "lambda_expert": hparams["subset_irm_lambda_expert"],
        "global_expert_enabled": hparams.get(
            "subset_irm_global_expert_enabled", False
        ),
        "global_training_mode": hparams.get(
            "subset_irm_global_training_mode", "joint"
        ),
        "global_logit_weight": hparams.get(
            "subset_irm_global_logit_weight", 1.0
        ),
        "lambda_global_expert": hparams.get(
            "subset_irm_lambda_global_expert", 0.0
        ),
        "lambda_global_irm": hparams.get(
            "subset_irm_lambda_global_irm", 0.0
        ),
        "lambda_route": hparams["subset_irm_lambda_route"],
        "lambda_q_capacity": hparams.get("subset_irm_lambda_q_capacity", 0.0),
        "q_capacity_rho_max": hparams.get("subset_irm_q_capacity_rho_max"),
        "q_capacity_post_step": hparams.get("subset_irm_q_capacity_post_step"),
        "q_capacity_post_lambda": hparams.get("subset_irm_q_capacity_post_lambda"),
        "q_capacity_decay_start_step": hparams.get(
            "subset_irm_q_capacity_decay_start_step"
        ),
        "q_capacity_decay_end_step": hparams.get(
            "subset_irm_q_capacity_decay_end_step"
        ),
        "lambda_sirm": hparams["subset_irm_lambda_sirm"],
        "lambda_ssi": hparams["lambda_inv"],
        "lambda_sp": hparams["lambda_sp"],
        "lambda_bal": hparams["lambda_bal"],
        "lambda_div": hparams["lambda_div"],
        "selected_step": best_step,
        "source_val_accuracy": best_source,
        "target_env": target_env,
        "target_name": env_names[target_env],
        "target_accuracy": target_accuracy,
        "target_in_accuracy": target_accuracy,
        "target_out_accuracy": target_out_accuracy,
        "target_component_accuracy": target_component_accuracy,
        "target_out_component_accuracy": target_out_component_accuracy,
        "active_sirm_rate": (
            float(model.sirm_active_steps.item() / max(model.subset_step.item(), 1))
            if getattr(model, "subset_enabled", False) else 0.0
        ),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "wall_time_seconds": time.time() - start,
        "target_evaluated_after_source_selection": not args.skip_target_eval,
        "target_evaluated_at_checkpoints": not args.skip_target_eval,
        "target_used_for_checkpoint_selection": False,
    }
    if not args.skip_target_eval:
        summary.update(target_checkpoint_diagnostics(
            checkpoint_records, target_env
        ))
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subset_irm_pacs.json")
    parser.add_argument("--data-dir", default="./domainbed/data")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--target-env", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument("--skip-target-eval", action="store_true")
    parser.add_argument("--resume", help="phase-aware S-IRM-res checkpoint")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
