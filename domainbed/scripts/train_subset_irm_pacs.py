"""GPU-only, source-selected PACS runner for the Subset-IRM ladder."""

import argparse
import collections
import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset

from domainbed import algorithms, datasets
from domainbed.lib import misc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


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
def source_expert_diagnostics(model, source_sets, device, batch_size,
                              num_workers):
    if not getattr(model, "subset_enabled", False):
        return {}
    risks = []
    accuracies = []
    for _, data in source_sets.items():
        ce_sum = torch.zeros(model.num_experts, device=device)
        correct = torch.zeros(model.num_experts, device=device)
        count = 0
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        model.eval()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            expert_logits = model._subset_forward(x)[4]
            for expert in range(model.num_experts):
                ce_sum[expert] += torch.nn.functional.cross_entropy(
                    expert_logits[:, expert], y, reduction="sum"
                )
                correct[expert] += expert_logits[:, expert].argmax(1).eq(y).sum()
            count += y.numel()
        risks.append((ce_sum / count).cpu().tolist())
        accuracies.append((correct / count).cpu().tolist())
    model.train()
    return {
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


def heatmap(matrix, row_names, title, path):
    array = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    image = ax.imshow(array, aspect="auto", cmap="viridis")
    ax.set_xticks(range(array.shape[1]), [f"E{i}" for i in range(array.shape[1])])
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


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if config["seed"] != 0:
        raise ValueError("fixed protocol requires seed 0")
    spec, hparams = resolve_run(config, args.run)
    steps = int(args.steps or config["steps"])
    checkpoint_freq = int(args.checkpoint_freq or config["checkpoint_freq"])
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()

    seed = int(config["seed"])
    target_env = int(config["target_env"] if args.target_env is None
                     else args.target_env)
    if target_env not in range(4):
        raise ValueError("PACS target environment must be one of 0, 1, 2, 3")
    source_envs = [env for env in range(4) if env != target_env]
    env_names = datasets.PACS.ENVIRONMENTS
    if (args.target_env is None and
            env_names[target_env] != config["target_name"]):
        raise RuntimeError("configured PACS target name disagrees with dataset")
    print(f"PACS environments: {env_names}", flush=True)
    print(f"Held-out environment {target_env}: {env_names[target_env]}", flush=True)
    print(f"Training source environments: {[env_names[i] for i in source_envs]}", flush=True)

    seed_everything(seed)
    train_hparams = dict(hparams, data_augmentation=True)
    eval_hparams = dict(hparams, data_augmentation=False)
    train_dataset = datasets.PACS(args.data_dir, [target_env], train_hparams)
    eval_dataset = datasets.PACS(args.data_dir, [target_env], eval_hparams)
    split = {}
    for env in range(4):
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
    eval_envs = source_envs if args.skip_target_eval else list(range(4))
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
        "dataset": "PACS",
        "pacs_environment_order": env_names,
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

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"{args.run}-target-{env_names[target_env]}-seed-{seed}",
        group=args.wandb_group or f"PACS-target-{env_names[target_env]}",
        job_type="train",
        config=manifest,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=["PACS", "SIRM", spec["algorithm"]],
    )
    wandb_run.define_metric("optimizer_step")
    wandb_run.define_metric("train/*", step_metric="optimizer_step")
    wandb_run.define_metric("eval/*", step_metric="optimizer_step")
    wandb_run.define_metric("diagnostics/*", step_metric="optimizer_step")
    manifest["wandb"] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "mode": args.wandb_mode,
        "run_id": wandb_run.id,
        "run_url": wandb_run.url,
    }
    write_json(output_dir / "manifest.json", manifest)

    best_source = -1.0
    best_step = None
    checkpoint_records = []
    window = collections.defaultdict(list)
    start = time.time()
    for step in range(steps):
        minibatches = []
        for iterator in loader_iters:
            x, y = next(iterator)
            minibatches.append((x.cuda(non_blocking=True), y.cuda(non_blocking=True)))
        values = model.update(minibatches)
        wandb_metrics = {
            "optimizer_step": step + 1,
            **{
                f"train/{key}": float(value)
                for key, value in values.items()
                if isinstance(value, (int, float))
            },
        }
        for key, value in values.items():
            if isinstance(value, (int, float)):
                window[key].append(float(value))
        if getattr(model, "last_diagnostics", None) and (
                step % max(1, int(hparams.get("subset_irm_gradient_log_interval", 100))) == 0):
            append_jsonl(output_dir / "diagnostics.jsonl", {
                "step": step, **values, **model.last_diagnostics
            })
            wandb_metrics.update({
                f"diagnostics/{key}": float(value)
                for key, value in model.last_diagnostics.items()
                if isinstance(value, (int, float, bool))
            })
        if step % checkpoint_freq != 0 and step != steps - 1:
            wandb_run.log(wandb_metrics)
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
        wandb_metrics.update({
            **{f"eval/{key}": value for key, value in eval_scores.items()},
            "eval/source_val_accuracy": source_mean,
            "system/peak_gpu_memory_gb": record["peak_gpu_memory_gb"],
            "system/wall_time_seconds": record["wall_time_seconds"],
        })
        for env, value in source_scores.items():
            wandb_metrics[f"eval/source_val_env{env}"] = value
        wandb_run.log(wandb_metrics)
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

    # The held-out view is materialized/evaluated only after source selection.
    if args.skip_target_eval:
        target_accuracy = None
        target_out_accuracy = None
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
    summary = {
        "run": args.run,
        "prediction_mode": hparams["subset_irm_prediction_mode"],
        "feature_skip_enabled": hparams.get(
            "subset_irm_feature_skip_enabled", False
        ),
        "feature_skip_scale": hparams.get("subset_irm_feature_skip_scale", 1.0),
        "topk": hparams["subset_irm_router_topk"],
        "lambda_expert": hparams["subset_irm_lambda_expert"],
        "lambda_route": hparams["subset_irm_lambda_route"],
        "lambda_q_capacity": hparams.get("subset_irm_lambda_q_capacity", 0.0),
        "q_capacity_rho_max": hparams.get("subset_irm_q_capacity_rho_max"),
        "q_capacity_post_step": hparams.get("subset_irm_q_capacity_post_step"),
        "q_capacity_post_lambda": hparams.get("subset_irm_q_capacity_post_lambda"),
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
    wandb_run.summary.update(summary)
    wandb_run.summary["best_source_val_accuracy"] = best_source
    wandb_run.summary["best_source_val_step"] = best_step
    for filename in (
            "manifest.json", "train_log.jsonl", "diagnostics.jsonl",
            "source_selected_diagnostics.json", "R_heatmap.png",
            "Q_heatmap.png", "summary.json"):
        path = output_dir / filename
        if path.exists():
            wandb_run.save(str(path), base_path=str(output_dir), policy="now")
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(f"W&B run: {wandb_run.url}", flush=True)
    wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subset_irm_pacs.json")
    parser.add_argument("--data-dir", default="./domainbed/data")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--target-env", type=int, choices=range(4))
    parser.add_argument("--skip-target-eval", action="store_true")
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "PACS_sweep")
    )
    parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", "hunghn2003")
    )
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
