"""Multi-method comparison: selected-vs-non-selected pair discrepancy.

Compares MESSI (routing-induced subset alignment) against two controlled
ablations that share the entire architecture/optimizer/schedule:

    Global-MoE         alignment_mode=global       (uniform pair weights)
    Random-Subset-MoE  alignment_mode=random_subset (random subset of pairs)

For each checkpoint we re-use the diagnostic from
`scripts/messi_selected_nonselected_discrepancy.py`:

    1. Reconstruct the held-out source-validation 'out' splits.
    2. Forward-pass to collect (pi, h_stack) for source samples only.
    3. Compute rho[m,k,c] and MESSI-style pair responsibilities
       a[m,i,j,c] = sigmoid(alpha*rho_i)*sigmoid(alpha*rho_j) on the
       trained checkpoint's *own* router (each method gets its own mask
       — expert indices are not comparable across independently-trained
       checkpoints).
    4. Partition pairs by top-q% responsibilities; compute Energy Distance
       on L2-normalized expert features (matches the standalone diagnostic).
    5. Bootstrap SE on each side.

Outputs:
    {output_dir}/alignment_ablation_selected_nonselected.pdf   one-/two-col fig
    {output_dir}/alignment_ablation_selected_nonselected.png   300-dpi preview
    {output_dir}/alignment_ablation_metrics.csv                per-method metrics
    {output_dir}/alignment_ablation_metrics.json               same as JSON

Usage:
    python -m scripts.messi_alignment_ablation_figure \\
        --messi_ckpt   multi_dataset/test_InvMMD/PACS_env0_seed0/model.pkl \\
        --global_ckpt  multi_dataset/test_InvMMD_global/PACS_env0_seed0/model.pkl \\
        --random_ckpt  multi_dataset/test_InvMMD_random/PACS_env0_seed0/model.pkl \\
        --output_dir   multi_dataset/figures/alignment_ablation_pacs_env0_seed0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("DOMAINBED_PROJECT_DIR", _REPO_ROOT)

# Re-use everything from the standalone diagnostic.
from scripts.messi_selected_nonselected_discrepancy import (
    load_checkpoint,
    build_source_val_splits,
    extract_features,
    compute_pair_records,
    bootstrap_se,
)


METHOD_DISPLAY = {
    "global":  "Global-MoE",
    "random":  "Random-Subset-MoE",
    "messi":   "MESSI",
}
METHOD_COLOR_SELECTED = "#4C72B0"
METHOD_COLOR_NONSELECTED = "#C44E52"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--messi_ckpt", required=True, type=str)
    p.add_argument("--global_ckpt", required=True, type=str)
    p.add_argument("--random_ckpt", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--q", type=float, default=0.30)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--max_samples_per_group", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--no_l2_normalize", action="store_true",
                   help="disable per-expert L2 normalization (default: enabled, "
                        "removes per-expert feature-norm confound)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def diagnose_one(method_key: str, ckpt_path: str, args, device: str) -> dict:
    print(f"\n[{method_key.upper()}] checkpoint = {ckpt_path}")
    model, ckpt, ckpt_args, hparams = load_checkpoint(ckpt_path, device)
    alpha = float(hparams.get("alpha", 4.0))
    M = model.NUM_EXPERTS
    num_classes = ckpt["model_num_classes"]
    num_source_domains = ckpt["model_num_domains"]
    actual_mode = hparams.get("alignment_mode", "messi")
    print(f"[{method_key.upper()}] algo={ckpt_args['algorithm']} "
          f"alignment_mode={actual_mode} alpha={alpha} M={M} "
          f"num_classes={num_classes} num_source_domains={num_source_domains}")

    splits, _, _ = build_source_val_splits(ckpt_args, hparams)
    pi, h_stack, y, d = extract_features(
        model, splits, args.batch_size, args.num_workers, device,
    )
    print(f"[{method_key.upper()}] N={pi.shape[0]} pi={pi.shape} h_stack={h_stack.shape}")

    records = compute_pair_records(
        pi, h_stack, y, d, num_classes, num_source_domains,
        alpha, args.min_count, args.max_samples_per_group, args.seed, device,
        l2_normalize=not args.no_l2_normalize,
    )
    n_valid = len(records)
    if n_valid == 0:
        raise RuntimeError(f"[{method_key}] no valid pairs")

    a_arr = np.array([r["a"] for r in records], dtype=np.float64)
    ed_arr = np.array([r["ED"] for r in records], dtype=np.float64)

    if args.tau is not None:
        sel_mask = a_arr > args.tau
        threshold = float(args.tau)
        rule = f"tau={args.tau}"
    else:
        threshold = float(np.quantile(a_arr, 1.0 - args.q))
        sel_mask = a_arr >= threshold
        rule = f"top-{int(args.q*100)}% (a >= {threshold:.4g})"
    non_mask = ~sel_mask

    n_sel = int(sel_mask.sum())
    n_non = int(non_mask.sum())
    delta_sel = float(ed_arr[sel_mask].mean()) if n_sel > 0 else float("nan")
    delta_non = float(ed_arr[non_mask].mean()) if n_non > 0 else float("nan")
    sel_gap = delta_non - delta_sel

    boot_rng = np.random.default_rng(args.seed + 1)
    se_sel = bootstrap_se(ed_arr[sel_mask], args.n_boot, boot_rng)
    se_non = bootstrap_se(ed_arr[non_mask], args.n_boot, boot_rng)

    print(f"[{method_key.upper()}] valid={n_valid} sel={n_sel} non={n_non}  "
          f"Delta_sel={delta_sel:.4f}±{se_sel:.4f}  "
          f"Delta_non={delta_non:.4f}±{se_non:.4f}  SelGap={sel_gap:+.4f}")

    return {
        "method": METHOD_DISPLAY.get(method_key, method_key),
        "method_key": method_key,
        "checkpoint_path": os.path.abspath(ckpt_path),
        "algorithm": ckpt_args["algorithm"],
        "alignment_mode_in_ckpt": actual_mode,
        "dataset": ckpt_args["dataset"],
        "target_env": list(map(int, ckpt_args["test_envs"])),
        "seed": int(ckpt_args["seed"]),
        "trial_seed": int(ckpt_args["trial_seed"]),
        "alpha": alpha,
        "num_experts": int(M),
        "num_source_domains": int(num_source_domains),
        "selection_rule": rule,
        "selection_threshold_on_a": threshold,
        "min_count": int(args.min_count),
        "max_samples_per_group": int(args.max_samples_per_group),
        "feature_normalization": "raw" if args.no_l2_normalize else "l2",
        "distance": "EnergyDistance(Euclidean)",
        "num_valid_pairs": int(n_valid),
        "num_selected": n_sel,
        "num_nonselected": n_non,
        "Delta_selected": delta_sel,
        "Delta_nonselected": delta_non,
        "SelGap": sel_gap,
        "se_selected": se_sel,
        "se_nonselected": se_non,
    }


def make_grouped_bar(results: list, output_dir: str, args):
    """Grouped bar chart: one (Selected, Non-selected) pair per method."""
    methods = [r["method"] for r in results]
    sel_vals = [r["Delta_selected"] for r in results]
    non_vals = [r["Delta_nonselected"] for r in results]
    sel_errs = [r["se_selected"] for r in results]
    non_errs = [r["se_nonselected"] for r in results]

    n = len(methods)
    x = np.arange(n)
    w = 0.36

    fig, ax = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
    b1 = ax.bar(x - w/2, sel_vals, w, yerr=sel_errs,
                label="Selected", color=METHOD_COLOR_SELECTED,
                capsize=3, edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w/2, non_vals, w, yerr=non_errs,
                label="Non-selected", color=METHOD_COLOR_NONSELECTED,
                capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel(r"Class-cond. discrepancy $\downarrow$", fontsize=9)
    ax.set_title("Routing-selected alignment: ablations", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    all_tops = [v + e for v, e in zip(sel_vals + non_vals, sel_errs + non_errs)]
    ax.set_ylim(0, max(all_tops) * 1.20)

    pdf_path = os.path.join(output_dir, "alignment_ablation_selected_nonselected.pdf")
    png_path = os.path.join(output_dir, "alignment_ablation_selected_nonselected.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Run in plot order: ablations first, then MESSI on the right.
    plan = [
        ("global", args.global_ckpt),
        ("random", args.random_ckpt),
        ("messi",  args.messi_ckpt),
    ]
    results = [diagnose_one(k, p, args, device) for k, p in plan]

    pdf, png = make_grouped_bar(results, args.output_dir, args)
    print(f"\n[plot] wrote {pdf}")
    print(f"[plot] wrote {png}")

    json_path = os.path.join(args.output_dir, "alignment_ablation_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "q": None if args.tau is not None else float(args.q),
            "tau": None if args.tau is None else float(args.tau),
            "min_count": int(args.min_count),
            "max_samples_per_group": int(args.max_samples_per_group),
            "feature_normalization": "raw" if args.no_l2_normalize else "l2",
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "results": results,
        }, f, indent=2)
    print(f"[json] wrote {json_path}")

    csv_path = os.path.join(args.output_dir, "alignment_ablation_metrics.csv")
    fieldnames = [
        "method", "alignment_mode_in_ckpt", "dataset", "target_env", "seed",
        "checkpoint_path", "num_valid_pairs", "num_selected", "num_nonselected",
        "Delta_selected", "se_selected",
        "Delta_nonselected", "se_nonselected",
        "SelGap",
        "selection_rule", "min_count", "max_samples_per_group",
        "feature_normalization",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[csv] wrote {csv_path}")

    print("\n=== Summary ===")
    print(f"Selection: {results[0]['selection_rule']}  "
          f"L2-norm: {not args.no_l2_normalize}  "
          f"Diagnostics on held-out SOURCE-validation only (no target data).")
    for r in results:
        print(f"  {r['method']:<20s}  SelGap={r['SelGap']:+.4f}  "
              f"sel={r['Delta_selected']:.4f}±{r['se_selected']:.4f}  "
              f"non={r['Delta_nonselected']:.4f}±{r['se_nonselected']:.4f}")


if __name__ == "__main__":
    main()
