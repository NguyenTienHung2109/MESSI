"""iWildCam MESSI expert-output gradient compatibility diagnostic.

This script is the WILDSIWildCam/location-domain counterpart of
scripts/messi_grad_compat_diagnostic.py. It intentionally treats source camera
locations inside env_0/train as diagnostic domains, because the official
WILDSIWildCam dataset wrapper exposes only one training env.

It uses held-out env_0_out samples only and never uses val_ood/test_ood/id_val/
id_test samples for gradient diagnostics.
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
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("DOMAINBED_PROJECT_DIR", _REPO_ROOT)

from domainbed import datasets  # noqa: E402
from domainbed.lib import misc  # noqa: E402
from scripts.messi_grad_compat_diagnostic import (  # noqa: E402
    SLOT_FIELDS,
    SUMMARY_FIELDS,
    infer_step,
    load_checkpoint_compat,
    resolve_data_dir,
    selected_mask_by_topq,
    sigmoid,
    summarize_records,
    write_csv,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "WILDSIWildCam location-domain MESSI expert-output gradient "
            "compatibility diagnostic"
        )
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint to diagnose. May be passed multiple times.",
    )
    p.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Optional run directory. Diagnose model_step*.pkl plus model.pkl if present.",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to checkpoint dirname for one checkpoint.",
    )
    p.add_argument("--q", type=float, default=0.20)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_dir", default=None)
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

    deduped = []
    seen = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            deduped.append(abs_path)
            seen.add(abs_path)
    return deduped


def resolve_output_dir(args, checkpoints: List[str]) -> str:
    if args.output_dir:
        return args.output_dir
    if len(checkpoints) == 1:
        return os.path.dirname(checkpoints[0])
    raise ValueError("--output_dir is required when diagnosing multiple checkpoints")


class LocationAnnotatedSubset(torch.utils.data.Dataset):
    """Wrap env_0_out and attach contiguous location-domain metadata."""

    def __init__(
        self,
        subset,
        domain_ids: np.ndarray,
        original_locations: np.ndarray,
        row_ids: np.ndarray,
    ):
        self.subset = subset
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        self.original_locations = np.asarray(original_locations, dtype=np.int64)
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        if not (
            len(self.subset)
            == len(self.domain_ids)
            == len(self.original_locations)
            == len(self.row_ids)
        ):
            raise ValueError("LocationAnnotatedSubset metadata length mismatch")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return (
            x,
            y,
            int(self.domain_ids[idx]),
            int(self.original_locations[idx]),
            int(self.row_ids[idx]),
        )


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def build_iwildcam_location_out_split(
    ckpt_args: dict,
    hparams: dict,
    model_input_shape,
):
    if ckpt_args["dataset"] != "WILDSIWildCam":
        raise ValueError(
            "messi_grad_compat_diagnostic_wild.py only supports WILDSIWildCam; "
            f"got {ckpt_args['dataset']}"
        )

    data_dir = ckpt_args["data_dir"]
    test_envs = list(map(int, ckpt_args["test_envs"]))
    if 0 in test_envs:
        raise ValueError(
            "WILDSIWildCam location diagnostic expects env_0 to be the source train pool"
        )

    eval_hparams = dict(hparams)
    eval_hparams["data_augmentation"] = False
    eval_hparams["class_balanced"] = False
    eval_hparams["cache_eval_in_ram"] = False

    img_size = int(model_input_shape[1])
    original_img_size = datasets.WILDSIWildCam.IMG_SIZE
    original_input_shape = datasets.WILDSIWildCam.INPUT_SHAPE
    try:
        datasets.WILDSIWildCam.IMG_SIZE = img_size
        datasets.WILDSIWildCam.INPUT_SHAPE = (3, img_size, img_size)
        dataset = datasets.WILDSIWildCam(data_dir, test_envs, eval_hparams)
    finally:
        datasets.WILDSIWildCam.IMG_SIZE = original_img_size
        datasets.WILDSIWildCam.INPUT_SHAPE = original_input_shape
    train_env = dataset.datasets[0]

    holdout_fraction = float(ckpt_args.get("holdout_fraction", 0.2))
    trial_seed = int(ckpt_args.get("trial_seed", 0))
    out_split, _in_split = misc.split_dataset(
        train_env,
        int(len(train_env) * holdout_fraction),
        misc.seed_hash(trial_seed, 0),
    )

    if not hasattr(out_split, "keys"):
        raise RuntimeError("Expected misc._SplitDataset with .keys for env_0_out")

    out_local_indices = np.asarray(out_split.keys, dtype=np.int64)
    row_ids = np.asarray(train_env.indices, dtype=np.int64)[out_local_indices]

    metadata = _to_numpy(train_env.dataset.metadata_array)
    loc_array = np.asarray(metadata[:, 0], dtype=np.int64)
    original_locations = loc_array[row_ids]

    unique_locations = sorted(int(x) for x in np.unique(original_locations))
    loc_to_domain = {loc: i for i, loc in enumerate(unique_locations)}
    domain_ids = np.asarray(
        [loc_to_domain[int(loc)] for loc in original_locations],
        dtype=np.int64,
    )

    annotated = LocationAnnotatedSubset(
        out_split,
        domain_ids=domain_ids,
        original_locations=original_locations,
        row_ids=row_ids,
    )

    counts_by_loc = Counter(int(loc) for loc in original_locations)
    domain_mapping = [
        {
            "domain_idx": int(loc_to_domain[loc]),
            "original_location_id": int(loc),
            "num_out_samples": int(counts_by_loc[loc]),
        }
        for loc in unique_locations
    ]

    return annotated, dataset, domain_mapping


def make_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def accumulate_location_gradients(
    model,
    annotated_split,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    num_experts = int(getattr(model, "num_experts", getattr(model, "NUM_EXPERTS", 6)))
    feat_dim = int(getattr(model.moe_head, "expert_dim", model.featurizer.n_outputs))

    stats: Dict[Tuple[int, int], dict] = {}
    loader = make_loader(annotated_split, batch_size, num_workers)

    model.eval()
    print(f"  env_0_out |samples|={len(annotated_split)}")
    for x, y, domain_id, _orig_loc, _row_id in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            z = model.featurizer(x)
        with torch.enable_grad():
            logits, pi, h_stack = model.moe_head(z.detach())
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
        y_cpu = y.detach().cpu().to(torch.long)
        d_cpu = domain_id.detach().cpu().to(torch.long)

        for d in d_cpu.unique().tolist():
            d_mask = d_cpu == int(d)
            for c in y_cpu[d_mask].unique().tolist():
                c = int(c)
                if c < 0 or c >= num_classes:
                    raise ValueError(f"class index {c} outside [0, {num_classes})")
                mask = d_mask & (y_cpu == c)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                key = (int(d), c)
                if key not in stats:
                    stats[key] = {
                        "grad_sum": torch.zeros(
                            num_experts, feat_dim, dtype=torch.float64
                        ),
                        "rho_sum": torch.zeros(num_experts, dtype=torch.float64),
                        "count": 0,
                    }
                stats[key]["count"] += n
                stats[key]["grad_sum"] += grads_cpu[mask, :, :].sum(dim=0)
                stats[key]["rho_sum"] += pi_cpu[mask, :].sum(dim=0)

        del logits, pi, h_stack, loss, grads, z

    return stats, num_experts


def compute_location_slot_records(
    stats: Dict[Tuple[int, int], dict],
    num_experts: int,
    step: int,
    alpha: float,
    min_count: int,
    q: float,
) -> Tuple[List[dict], dict]:
    domains_by_class = defaultdict(list)
    for (domain_id, class_id), item in stats.items():
        if int(item["count"]) >= min_count:
            domains_by_class[int(class_id)].append(int(domain_id))

    records: List[dict] = []
    eps = 1e-12
    for class_id in sorted(domains_by_class):
        domains = sorted(set(domains_by_class[class_id]))
        for pos, i in enumerate(domains):
            item_i = stats[(i, class_id)]
            n_i = int(item_i["count"])
            grad_i_sum = item_i["grad_sum"]
            rho_i_sum = item_i["rho_sum"]
            for j in domains[pos + 1:]:
                item_j = stats[(j, class_id)]
                n_j = int(item_j["count"])
                grad_j_sum = item_j["grad_sum"]
                rho_j_sum = item_j["rho_sum"]

                for m in range(num_experts):
                    gi = grad_i_sum[m] / float(n_i)
                    gj = grad_j_sum[m] / float(n_j)
                    norm_i = float(torch.linalg.vector_norm(gi).item())
                    norm_j = float(torch.linalg.vector_norm(gj).item())
                    denom = max(norm_i * norm_j, eps)
                    gradcos = float(torch.dot(gi, gj).item() / denom)
                    gradcos = max(-1.0, min(1.0, gradcos))

                    rho_i = float((rho_i_sum[m] / float(n_i)).item())
                    rho_j = float((rho_j_sum[m] / float(n_j)).item())
                    a = sigmoid(alpha * rho_i) * sigmoid(alpha * rho_j)

                    records.append({
                        "step": int(step),
                        "expert_m": int(m),
                        "domain_i": int(i),
                        "domain_j": int(j),
                        "class_c": int(class_id),
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


def write_mapping_csv(path: str, rows: Iterable[dict]):
    fieldnames = ["domain_idx", "original_location_id", "num_out_samples"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def diagnose_checkpoint(checkpoint_path: str, args, output_dir: str, device: str):
    print(f"\n[load] checkpoint = {checkpoint_path}")
    model, ckpt, ckpt_args, hparams = load_checkpoint_compat(checkpoint_path, device)
    ckpt_args = resolve_data_dir(ckpt_args, args.data_dir)

    step = infer_step(checkpoint_path, ckpt)
    alpha = float(args.alpha if args.alpha is not None else hparams.get("alpha", 4.0))
    num_classes = int(ckpt["model_num_classes"])
    test_envs = list(map(int, ckpt_args["test_envs"]))

    print(
        f"[meta] dataset={ckpt_args['dataset']} algorithm={ckpt_args['algorithm']} "
        f"test_envs={test_envs} step={step} alpha={alpha}"
    )
    print("[split] reconstructing env_0_out and using location_remapped domains")
    annotated_split, dataset_obj, domain_mapping = build_iwildcam_location_out_split(
        ckpt_args,
        hparams,
        ckpt["model_input_shape"],
    )

    print("[grad] accumulating expert-output CE gradients by location/class")
    stats, num_experts = accumulate_location_gradients(
        model=model,
        annotated_split=annotated_split,
        num_classes=num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print("[slots] computing routing responsibility and gradient cosine")
    records, summary = compute_location_slot_records(
        stats=stats,
        num_experts=num_experts,
        step=step,
        alpha=alpha,
        min_count=args.min_count,
        q=args.q,
    )
    if summary["num_valid_slots"] == 0:
        print(
            "[warn] no valid slots; consider lowering --min_count or checking "
            "location/class support in env_0_out"
        )

    slot_path = os.path.join(output_dir, f"slot_metrics_step_{step}.csv")
    write_csv(slot_path, records, SLOT_FIELDS)
    print(f"[csv] wrote {slot_path}")

    mapping_path = os.path.join(output_dir, f"location_domain_mapping_step_{step}.csv")
    write_mapping_csv(mapping_path, domain_mapping)
    print(f"[csv] wrote {mapping_path}")

    metadata = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "step": int(step),
        "dataset": ckpt_args["dataset"],
        "algorithm": ckpt_args["algorithm"],
        "domain_granularity": "location_remapped",
        "source_env_used": 0,
        "source_env_name": str(dataset_obj.ENVIRONMENTS[0]),
        "excluded_eval_envs": test_envs,
        "excluded_eval_env_names": [str(dataset_obj.ENVIRONMENTS[i]) for i in test_envs],
        "num_location_domains": int(len(domain_mapping)),
        "domain_mapping": domain_mapping,
        "alpha": alpha,
        "q": float(args.q),
        "min_count": int(args.min_count),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "data_dir": ckpt_args.get("data_dir"),
        "slot_metrics_csv": os.path.abspath(slot_path),
        "location_domain_mapping_csv": os.path.abspath(mapping_path),
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

    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise RuntimeError("No checkpoints found. Pass --checkpoint or --checkpoint_dir.")
    output_dir = resolve_output_dir(args, checkpoints)
    os.makedirs(output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[run] device={device}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] checkpoints={len(checkpoints)}")

    summaries = []
    metadata = []
    for checkpoint_path in checkpoints:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        summary, meta = diagnose_checkpoint(checkpoint_path, args, output_dir, device)
        summaries.append(summary)
        metadata.append(meta)

    summaries.sort(key=lambda row: row["step"])
    summary_path = os.path.join(output_dir, "checkpoint_summary.csv")
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    print(f"\n[csv] wrote {summary_path}")

    metadata_path = os.path.join(output_dir, "diagnostic_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"checkpoints": metadata}, f, indent=2)
    print(f"[json] wrote {metadata_path}")


if __name__ == "__main__":
    main()
