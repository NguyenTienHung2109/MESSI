"""MESSI expert-output gradient compatibility diagnostic.

For each checkpoint, this script evaluates source-validation samples only and
compares domain/class averages of CE gradients with respect to per-expert
outputs h_stack[:, m, :]. It does not inspect parameter gradients.

Default target run:
    multi_dataset/L_inv_and_L_sp_and_L_bal_and_L_div/
        pacs_gmoe_invmmd_env0_seed0/model.pkl

Usage:
    python -m scripts.messi_grad_compat_diagnostic

    python -m scripts.messi_grad_compat_diagnostic \
        --checkpoint multi_dataset/L_inv_and_L_sp_and_L_bal_and_L_div/pacs_gmoe_invmmd_env0_seed0/model.pkl \
        --output_dir multi_dataset/L_inv_and_L_sp_and_L_bal_and_L_div/grad_compat_env0_seed0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("DOMAINBED_PROJECT_DIR", _REPO_ROOT)

from scripts.messi_selected_nonselected_discrepancy import (  # noqa: E402
    build_source_val_splits,
    load_checkpoint,
)


DEFAULT_RUN_DIR = os.path.join(
    _REPO_ROOT,
    "multi_dataset",
    "L_inv_and_L_sp_and_L_bal_and_L_div",
    "pacs_gmoe_invmmd_env0_seed0",
)
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_RUN_DIR, "model.pkl")
DEFAULT_OUTPUT_DIR = os.path.join(
    _REPO_ROOT,
    "multi_dataset",
    "L_inv_and_L_sp_and_L_bal_and_L_div",
    "grad_compat_env0_seed0",
)

SUMMARY_FIELDS = [
    "step",
    "num_valid_slots",
    "num_selected_slots",
    "gradcos_selected_mean",
    "gradcos_nonselected_mean",
    "gradcos_gap_selected_minus_nonselected",
    "conflict_rate_selected",
    "conflict_rate_nonselected",
    "corr_responsibility_gradcos",
    "gradnorm_selected_mean",
    "gradnorm_nonselected_mean",
]

SLOT_FIELDS = [
    "step",
    "expert_m",
    "domain_i",
    "domain_j",
    "class_c",
    "rho_i_c_m",
    "rho_j_c_m",
    "a_ijc_m",
    "selected_topq",
    "gradcos",
    "conflict",
    "gradnorm_i",
    "gradnorm_j",
    "n_i_c",
    "n_j_c",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="PACS MESSI expert-output gradient compatibility diagnostic"
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint to diagnose. May be passed multiple times. Defaults to "
            "the canonical PACS env0 seed0 final checkpoint."
        ),
    )
    p.add_argument(
        "--checkpoint_dir",
        default=None,
        help=(
            "Optional run directory. If set, diagnose model_step*.pkl in step "
            "order plus model.pkl if present."
        ),
    )
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--q", type=float, default=0.20,
                   help="Top-q fraction of slots by routing responsibility.")
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Override checkpoint data_dir. If omitted and the checkpoint data_dir "
            "does not exist, ./domainbed/data is used when available."
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override routing responsibility temperature. Defaults to checkpoint hparams alpha.",
    )
    return p.parse_args()


def _step_from_path(path: str):
    match = re.search(r"model_step(\d+)\.pkl$", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def discover_checkpoints(args) -> List[str]:
    paths: List[str] = []
    if args.checkpoint_dir:
        step_paths = glob.glob(os.path.join(args.checkpoint_dir, "model_step*.pkl"))
        step_paths.sort(key=lambda p: (_step_from_path(p) is None, _step_from_path(p) or -1))
        paths.extend(step_paths)
        final_path = os.path.join(args.checkpoint_dir, "model.pkl")
        if os.path.exists(final_path):
            paths.append(final_path)
    if args.checkpoint:
        paths.extend(args.checkpoint)
    if not paths:
        paths = [DEFAULT_CHECKPOINT]

    deduped = []
    seen = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            deduped.append(abs_path)
            seen.add(abs_path)
    return deduped


def infer_step(checkpoint_path: str, ckpt: dict) -> int:
    step = _step_from_path(checkpoint_path)
    if step is not None:
        return step

    results_path = os.path.join(os.path.dirname(checkpoint_path), "results.jsonl")
    if os.path.exists(results_path):
        last = None
        with open(results_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last is not None and "step" in last:
            return int(last["step"])

    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("steps") is not None:
        return int(ckpt_args["steps"]) - 1
    return -1


def resolve_data_dir(ckpt_args: dict, override: str | None) -> dict:
    out = dict(ckpt_args)
    if override:
        out["data_dir"] = override
        return out

    data_dir = out.get("data_dir")
    if data_dir and os.path.exists(data_dir):
        return out

    local_data_dir = os.path.join(_REPO_ROOT, "domainbed", "data")
    if os.path.exists(local_data_dir):
        out["data_dir"] = local_data_dir
    return out


def build_eval_splits(ckpt_args: dict, hparams: dict):
    """Build source validation splits with deterministic eval transforms."""
    eval_hparams = dict(hparams)
    eval_hparams["data_augmentation"] = False
    eval_hparams["class_balanced"] = False
    return build_source_val_splits(ckpt_args, eval_hparams)


def adapt_state_dict_for_model(state_dict: dict, model) -> Tuple[dict, List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]], List[str]]:
    """Adapt old checkpoints whose ViT positional embedding lacks dist-token slot."""
    own_state = model.state_dict()
    adapted = {}
    dropped = []
    converted = []
    for key, value in state_dict.items():
        if key not in own_state:
            dropped.append((key, tuple(value.shape), ()))
            continue
        target = own_state[key]
        if target.shape == value.shape:
            adapted[key] = value
            continue

        if (key.endswith("pos_embed") and value.ndim == 3 and target.ndim == 3
                and value.shape[0] == target.shape[0]
                and value.shape[2] == target.shape[2]
                and target.shape[1] == value.shape[1] + 1):
            # Old explicit-head checkpoints stored cls + patch positions. The
            # current DeiT wrapper has an extra dist-token position; duplicate
            # the cls position for that slot and keep all patch positions.
            adapted[key] = torch.cat([value[:, :1, :], value[:, :1, :], value[:, 1:, :]], dim=1)
            converted.append(key)
            continue

        dropped.append((key, tuple(value.shape), tuple(target.shape)))
    return adapted, dropped, converted


def load_checkpoint_compat(checkpoint_path: str, device: str):
    """Load checkpoints robustly across minor ViT wrapper shape drift."""
    try:
        return load_checkpoint(checkpoint_path, device)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        print("[load] strict-compatible load failed; retrying with shape adaptation")

    from domainbed import algorithms

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    hparams = ckpt["model_hparams"]
    algo_cls = getattr(algorithms, ckpt_args["algorithm"])
    model = algo_cls(
        ckpt["model_input_shape"],
        ckpt["model_num_classes"],
        ckpt["model_num_domains"],
        hparams,
    )
    adapted, dropped, converted = adapt_state_dict_for_model(ckpt["model_dict"], model)
    model.load_state_dict(adapted, strict=False)
    if converted:
        print(f"[load] adapted tensors: {converted}")
    if dropped:
        preview = dropped[:5]
        print(f"[load] dropped {len(dropped)} incompatible/unexpected tensors: {preview}")
    model.to(device).eval()
    return model, ckpt, ckpt_args, hparams


def make_loader(subset, batch_size: int, num_workers: int):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x_std = x.std()
    y_std = y.std()
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def selected_mask_by_topq(records: List[dict], q: float) -> np.ndarray:
    n = len(records)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    n_select = max(1, int(math.ceil(float(q) * n)))
    order = np.argsort(-np.array([r["a_ijc_m"] for r in records], dtype=np.float64))
    mask[order[:n_select]] = True
    return mask


def accumulate_gradients(
    model,
    splits,
    num_classes: int,
    num_source_domains: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    num_experts = int(getattr(model, "num_experts", getattr(model, "NUM_EXPERTS", 6)))
    feat_dim = int(getattr(model.moe_head, "expert_dim", model.featurizer.n_outputs))

    grad_sum = torch.zeros(
        num_experts, num_source_domains, num_classes, feat_dim, dtype=torch.float64
    )
    rho_sum = torch.zeros(num_experts, num_source_domains, num_classes, dtype=torch.float64)
    counts = torch.zeros(num_source_domains, num_classes, dtype=torch.long)

    model.eval()
    for original_env_id, source_idx, subset in splits:
        loader = make_loader(subset, batch_size, num_workers)
        print(
            f"  source_idx={source_idx} original_env={original_env_id} "
            f"|out|={len(subset)}"
        )
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits, pi, h_stack = model._forward(x)
                loss = F.cross_entropy(logits, y_device, reduction="sum")
                grads = torch.autograd.grad(
                    loss,
                    h_stack,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                )[0]

            grads_cpu = grads.detach().cpu().to(torch.float64)
            pi_cpu = pi.detach().cpu().to(torch.float64)
            y_cpu = y.detach().cpu()

            for c in range(num_classes):
                mask = (y_cpu == c)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                counts[source_idx, c] += n
                grad_sum[:, source_idx, c, :] += grads_cpu[mask, :, :].sum(dim=0)
                rho_sum[:, source_idx, c] += pi_cpu[mask, :].sum(dim=0)

            del logits, pi, h_stack, loss, grads

    return grad_sum, rho_sum, counts


def compute_slot_records(
    grad_sum: torch.Tensor,
    rho_sum: torch.Tensor,
    counts: torch.Tensor,
    step: int,
    alpha: float,
    min_count: int,
    q: float,
) -> Tuple[List[dict], dict]:
    num_experts, num_source_domains, num_classes, _feat_dim = grad_sum.shape

    denom_grad = counts.to(torch.float64).clamp_min(1).view(1, num_source_domains, num_classes, 1)
    denom_rho = counts.to(torch.float64).clamp_min(1).view(1, num_source_domains, num_classes)
    mean_grad = grad_sum / denom_grad
    rho = rho_sum / denom_rho

    records: List[dict] = []
    eps = 1e-12
    for m in range(num_experts):
        for c in range(num_classes):
            for i in range(num_source_domains):
                n_i = int(counts[i, c].item())
                if n_i < min_count:
                    continue
                gi = mean_grad[m, i, c]
                norm_i = float(torch.linalg.vector_norm(gi).item())
                rho_i = float(rho[m, i, c].item())

                for j in range(i + 1, num_source_domains):
                    n_j = int(counts[j, c].item())
                    if n_j < min_count:
                        continue
                    gj = mean_grad[m, j, c]
                    norm_j = float(torch.linalg.vector_norm(gj).item())
                    rho_j = float(rho[m, j, c].item())

                    denom = max(norm_i * norm_j, eps)
                    gradcos = float(torch.dot(gi, gj).item() / denom)
                    gradcos = max(-1.0, min(1.0, gradcos))
                    a = sigmoid(alpha * rho_i) * sigmoid(alpha * rho_j)

                    records.append({
                        "step": int(step),
                        "expert_m": int(m),
                        "domain_i": int(i),
                        "domain_j": int(j),
                        "class_c": int(c),
                        "rho_i_c_m": rho_i,
                        "rho_j_c_m": rho_j,
                        "a_ijc_m": float(a),
                        "selected_topq": 0,
                        "gradcos": gradcos,
                        "conflict": int(gradcos < 0.0),
                        "gradnorm_i": norm_i,
                        "gradnorm_j": norm_j,
                        "n_i_c": n_i,
                        "n_j_c": n_j,
                    })

    selected = selected_mask_by_topq(records, q)
    for rec, is_selected in zip(records, selected.tolist()):
        rec["selected_topq"] = int(is_selected)

    summary = summarize_records(records, selected, step)
    return records, summary


def summarize_records(records: List[dict], selected: np.ndarray, step: int) -> dict:
    n_valid = len(records)
    if n_valid == 0:
        return {
            "step": int(step),
            "num_valid_slots": 0,
            "num_selected_slots": 0,
            "gradcos_selected_mean": float("nan"),
            "gradcos_nonselected_mean": float("nan"),
            "gradcos_gap_selected_minus_nonselected": float("nan"),
            "conflict_rate_selected": float("nan"),
            "conflict_rate_nonselected": float("nan"),
            "corr_responsibility_gradcos": float("nan"),
            "gradnorm_selected_mean": float("nan"),
            "gradnorm_nonselected_mean": float("nan"),
        }

    gradcos = np.array([r["gradcos"] for r in records], dtype=np.float64)
    conflict = np.array([r["conflict"] for r in records], dtype=np.float64)
    responsibility = np.array([r["a_ijc_m"] for r in records], dtype=np.float64)
    pair_gradnorm = np.array(
        [(r["gradnorm_i"] + r["gradnorm_j"]) / 2.0 for r in records],
        dtype=np.float64,
    )
    nonselected = ~selected

    selected_gradcos = finite_mean(gradcos[selected])
    nonselected_gradcos = finite_mean(gradcos[nonselected])
    return {
        "step": int(step),
        "num_valid_slots": int(n_valid),
        "num_selected_slots": int(selected.sum()),
        "gradcos_selected_mean": selected_gradcos,
        "gradcos_nonselected_mean": nonselected_gradcos,
        "gradcos_gap_selected_minus_nonselected": float(selected_gradcos - nonselected_gradcos),
        "conflict_rate_selected": finite_mean(conflict[selected]),
        "conflict_rate_nonselected": finite_mean(conflict[nonselected]),
        "corr_responsibility_gradcos": pearson_corr(responsibility, gradcos),
        "gradnorm_selected_mean": finite_mean(pair_gradnorm[selected]),
        "gradnorm_nonselected_mean": finite_mean(pair_gradnorm[nonselected]),
    }


def write_csv(path: str, rows: Iterable[dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def diagnose_checkpoint(checkpoint_path: str, args, device: str) -> Tuple[dict, dict]:
    print(f"\n[load] checkpoint = {checkpoint_path}")
    model, ckpt, ckpt_args, hparams = load_checkpoint_compat(checkpoint_path, device)
    ckpt_args = resolve_data_dir(ckpt_args, args.data_dir)

    step = infer_step(checkpoint_path, ckpt)
    alpha = float(args.alpha if args.alpha is not None else hparams.get("alpha", 4.0))
    num_classes = int(ckpt["model_num_classes"])
    num_source_domains = int(ckpt["model_num_domains"])
    test_envs = list(map(int, ckpt_args["test_envs"]))

    print(
        f"[meta] dataset={ckpt_args['dataset']} algorithm={ckpt_args['algorithm']} "
        f"test_envs={test_envs} step={step} alpha={alpha}"
    )
    print("[split] reconstructing source-validation out splits only")
    splits, dataset_obj, _ = build_eval_splits(ckpt_args, hparams)
    if len(splits) != num_source_domains:
        raise RuntimeError(
            f"source split count {len(splits)} != model_num_domains {num_source_domains}"
        )

    print("[grad] accumulating expert-output CE gradients")
    grad_sum, rho_sum, counts = accumulate_gradients(
        model=model,
        splits=splits,
        num_classes=num_classes,
        num_source_domains=num_source_domains,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print("[slots] computing routing responsibility and gradient cosine")
    records, summary = compute_slot_records(
        grad_sum=grad_sum,
        rho_sum=rho_sum,
        counts=counts,
        step=step,
        alpha=alpha,
        min_count=args.min_count,
        q=args.q,
    )

    slot_path = os.path.join(args.output_dir, f"slot_metrics_step_{step}.csv")
    write_csv(slot_path, records, SLOT_FIELDS)
    print(f"[csv] wrote {slot_path}")

    source_mapping = [
        {
            "source_idx": int(source_idx),
            "original_env_id": int(original_env_id),
            "env_name": str(dataset_obj.ENVIRONMENTS[original_env_id]),
            "num_out_samples": int(len(subset)),
        }
        for original_env_id, source_idx, subset in splits
    ]
    metadata = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "step": int(step),
        "dataset": ckpt_args["dataset"],
        "algorithm": ckpt_args["algorithm"],
        "test_envs": test_envs,
        "target_env_names": [str(dataset_obj.ENVIRONMENTS[i]) for i in test_envs],
        "source_mapping": source_mapping,
        "alpha": alpha,
        "q": float(args.q),
        "min_count": int(args.min_count),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "data_dir": ckpt_args.get("data_dir"),
        "slot_metrics_csv": os.path.abspath(slot_path),
    }

    print(
        f"[summary] valid={summary['num_valid_slots']} "
        f"selected={summary['num_selected_slots']} "
        f"gradcos_sel={summary['gradcos_selected_mean']:.6g} "
        f"gradcos_non={summary['gradcos_nonselected_mean']:.6g}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, metadata


def main():
    args = parse_args()
    if not (0.0 < args.q <= 1.0):
        raise ValueError(f"--q must be in (0, 1], got {args.q}")
    if args.min_count < 1:
        raise ValueError(f"--min_count must be >= 1, got {args.min_count}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise RuntimeError("No checkpoints found")

    print(f"[run] device={device}")
    print(f"[run] output_dir={args.output_dir}")
    print(f"[run] checkpoints={len(checkpoints)}")

    summaries = []
    metadata = []
    for checkpoint_path in checkpoints:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        summary, meta = diagnose_checkpoint(checkpoint_path, args, device)
        summaries.append(summary)
        metadata.append(meta)

    summaries.sort(key=lambda row: row["step"])
    summary_path = os.path.join(args.output_dir, "checkpoint_summary.csv")
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    print(f"\n[csv] wrote {summary_path}")

    metadata_path = os.path.join(args.output_dir, "diagnostic_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"checkpoints": metadata}, f, indent=2)
    print(f"[json] wrote {metadata_path}")


if __name__ == "__main__":
    main()
