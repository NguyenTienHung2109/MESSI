"""Two-panel figure for the MESSI main experiment section.

Panel (a): pair-responsibility similarity (3x3 heatmap of Pearson corr
           between the per-pair responsibility vectors a_{ijc}^(m) of the
           three methods; Jaccard overlap of selected-slot sets shown
           underneath each cell).
Panel (b): class-conditional feature discrepancy on **MESSI-selected slots**
           — the same (m, c, i, j) tuples are used for all three methods,
           which lets us isolate "alignment quality" from "which slots the
           method chose to align".

Diagnostics live entirely on held-out source-validation splits; no
target-domain samples are touched. L2-normalised expert features are used
to remove the per-expert feature-norm confound.

Usage:
    python -m scripts.messi_alignment_figure_v2 \\
        --messi_ckpt   multi_dataset/test_InvMMD_small/PACS_env0_seed0/model.pkl \\
        --global_ckpt  multi_dataset/test_InvMMD_small_global/PACS_env0_seed0/model.pkl \\
        --random_ckpt  multi_dataset/test_InvMMD_small_random/PACS_env0_seed0/model.pkl \\
        --output_dir   multi_dataset/figures/alignment_v2_pacs_env0_seed0
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

from scripts.messi_selected_nonselected_discrepancy import (
    load_checkpoint, build_source_val_splits, extract_features,
    compute_pair_records, bootstrap_se,
)


METHOD_ORDER = ["global", "random", "messi"]
METHOD_LABEL = {
    "global": "Global-MoE",
    "random": "Random-Subset-MoE",
    "messi":  "MESSI",
}
METHOD_COLOR = {
    "global": "#9aa5b1",   # muted gray-blue
    "random": "#9aa5b1",
    "messi":  "#3b6ea5",   # stronger blue for the proposed method
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--messi_ckpt", required=True)
    p.add_argument("--global_ckpt", required=True)
    p.add_argument("--random_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--q", type=float, default=0.30)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--max_samples_per_group", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def collect_records(ckpt_path: str, args, device: str):
    """Run forward + compute_pair_records on one checkpoint.

    Returns: (records_list, alpha, hparams_alignment_mode_in_ckpt).
    """
    model, ckpt, ckpt_args, hparams = load_checkpoint(ckpt_path, device)
    splits, _, _ = build_source_val_splits(ckpt_args, hparams)
    pi, h_stack, y, d = extract_features(
        model, splits, args.batch_size, args.num_workers, device,
    )
    alpha = float(hparams.get("alpha", 4.0))
    records = compute_pair_records(
        pi, h_stack, y, d, ckpt["model_num_classes"], ckpt["model_num_domains"],
        alpha, args.min_count, args.max_samples_per_group,
        args.seed, device, l2_normalize=True,
    )
    del model
    torch.cuda.empty_cache()
    return records, alpha, hparams.get("alignment_mode", "messi")


def panel_a_similarity(records_per_method: dict, q: float, output_dir: str):
    """3x3 heatmap of Pearson corr; Jaccard overlap annotated underneath."""
    keys_per_method = {
        m: {(r["m"], r["c"], r["i"], r["j"]): r["a"] for r in recs}
        for m, recs in records_per_method.items()
    }
    common_keys = sorted(set.intersection(
        *[set(d.keys()) for d in keys_per_method.values()]
    ))
    a_mats = {
        m: np.array([keys_per_method[m][k] for k in common_keys])
        for m in METHOD_ORDER
    }

    selected_sets = {}
    for m in METHOD_ORDER:
        a = a_mats[m]
        thr = np.quantile(a, 1.0 - q)
        selected_sets[m] = {k for k, av in zip(common_keys, a) if av >= thr}

    n = len(METHOD_ORDER)
    corr = np.eye(n)
    jacc = np.eye(n)
    for i, mi in enumerate(METHOD_ORDER):
        for j, mj in enumerate(METHOD_ORDER):
            if i == j:
                corr[i, j] = 1.0
                jacc[i, j] = 1.0
            else:
                corr[i, j] = float(np.corrcoef(a_mats[mi], a_mats[mj])[0, 1])
                a_set, b_set = selected_sets[mi], selected_sets[mj]
                jacc[i, j] = (len(a_set & b_set) / max(1, len(a_set | b_set)))
    return corr, jacc, selected_sets, common_keys, a_mats


def panel_b_messi_mask(records_per_method, messi_selected, n_boot, seed):
    """Compute mean ED + bootstrap SE for each method on MESSI-selected slots."""
    out = {}
    boot_rng = np.random.default_rng(seed + 1)
    for method, recs in records_per_method.items():
        eds = [r["ED"] for r in recs
               if (r["m"], r["c"], r["i"], r["j"]) in messi_selected]
        eds = np.array(eds, dtype=np.float64)
        out[method] = {
            "n_pairs_on_messi_mask": int(len(eds)),
            "Delta": float(eds.mean()) if len(eds) else float("nan"),
            "se":    bootstrap_se(eds, n_boot, boot_rng) if len(eds) else float("nan"),
            "raw_eds_count": int(len(eds)),
        }
    return out


def render_figure(corr, jacc, panel_b, output_dir):
    fig = plt.figure(figsize=(6.8, 2.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel (a) heatmap ----
    cmap = plt.get_cmap("RdBu_r")
    im = ax_a.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax_a.set_xticks(range(3))
    ax_a.set_yticks(range(3))
    short = ["Global", "Random", "MESSI"]
    ax_a.set_xticklabels(short, fontsize=8)
    ax_a.set_yticklabels(short, fontsize=8)
    ax_a.tick_params(axis="both", length=0)
    for i in range(3):
        for j in range(3):
            v = corr[i, j]
            txt_color = "white" if abs(v) > 0.55 else "black"
            ax_a.text(j, i, f"{v:+.2f}",
                      ha="center", va="center",
                      color=txt_color, fontsize=9, fontweight="bold")
            if i != j:
                ax_a.text(j, i + 0.30, f"J={jacc[i, j]:.2f}",
                          ha="center", va="center",
                          color=txt_color, fontsize=7, alpha=0.85)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson corr", fontsize=8)
    ax_a.set_title("(a) Responsibility pattern similarity", fontsize=9.5, pad=6)

    # ---- Panel (b) bar chart on MESSI-selected slots ----
    methods = METHOD_ORDER
    deltas = [panel_b[m]["Delta"] for m in methods]
    ses    = [panel_b[m]["se"]    for m in methods]
    colors = [METHOD_COLOR[m] for m in methods]
    xs = np.arange(len(methods))
    bars = ax_b.bar(xs, deltas, yerr=ses, color=colors, width=0.62,
                    capsize=3, edgecolor="black", linewidth=0.4)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([METHOD_LABEL[m] for m in methods], fontsize=8)
    ax_b.set_ylabel(r"Class-cond. discrepancy $\downarrow$", fontsize=9)
    ax_b.set_title("(b) Alignment on MESSI-selected slots", fontsize=9.5, pad=6)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(axis="y", labelsize=7)
    ax_b.tick_params(axis="x", labelsize=8, length=0)
    ymax = max(d + s for d, s in zip(deltas, ses)) * 1.22
    ax_b.set_ylim(0, ymax)
    # MESSI annotation: % reduction vs the best baseline.
    baseline_min = min(panel_b["global"]["Delta"], panel_b["random"]["Delta"])
    pct = (baseline_min - panel_b["messi"]["Delta"]) / baseline_min * 100.0
    if pct > 0:
        ax_b.annotate(
            f"~{pct:.0f}% lower",
            xy=(2, panel_b["messi"]["Delta"] + panel_b["messi"]["se"]),
            xytext=(2, panel_b["messi"]["Delta"] + panel_b["messi"]["se"] + ymax * 0.08),
            ha="center", fontsize=7.5, color="#3b6ea5",
            arrowprops=dict(arrowstyle="->", color="#3b6ea5",
                            lw=0.6, shrinkA=1, shrinkB=1),
        )

    pdf_path = os.path.join(output_dir, "alignment_routing_messi_v2.pdf")
    png_path = os.path.join(output_dir, "alignment_routing_messi_v2.png")
    svg_path = os.path.join(output_dir, "alignment_routing_messi_v2.svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path, svg_path


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    records_per_method = {}
    method_meta = {}
    for key, ckpt in [("global", args.global_ckpt),
                      ("random", args.random_ckpt),
                      ("messi",  args.messi_ckpt)]:
        print(f"[{key}] {ckpt}")
        recs, alpha, mode = collect_records(ckpt, args, device)
        records_per_method[key] = recs
        method_meta[key] = {"alpha": alpha, "alignment_mode_in_ckpt": mode,
                            "n_records": len(recs)}
        print(f"[{key}] alpha={alpha}  alignment_mode={mode}  n_records={len(recs)}")

    corr, jacc, selected_sets, common_keys, a_mats = panel_a_similarity(
        records_per_method, args.q, args.output_dir,
    )
    print("\n[panel-a] Pearson corr:")
    for i, m in enumerate(METHOD_ORDER):
        print(f"  {m:<8}  " + "  ".join(
            f"{corr[i, j]:+.3f}" for j in range(len(METHOD_ORDER))
        ))
    print("[panel-a] Jaccard:")
    for i, m in enumerate(METHOD_ORDER):
        print(f"  {m:<8}  " + "  ".join(
            f"{jacc[i, j]:.3f}" for j in range(len(METHOD_ORDER))
        ))

    messi_selected = selected_sets["messi"]
    print(f"\n[panel-b] MESSI selected-slot set size: {len(messi_selected)}")
    panel_b = panel_b_messi_mask(records_per_method, messi_selected,
                                  args.n_boot, args.seed)
    print("[panel-b] Class-cond. discrepancy on MESSI-selected slots:")
    for m in METHOD_ORDER:
        print(f"  {METHOD_LABEL[m]:<22s}  Δ={panel_b[m]['Delta']:.4f}  "
              f"±SE {panel_b[m]['se']:.4f}  "
              f"(n_pairs_on_mask={panel_b[m]['n_pairs_on_messi_mask']})")
    base_min = min(panel_b["global"]["Delta"], panel_b["random"]["Delta"])
    pct = (base_min - panel_b["messi"]["Delta"]) / base_min * 100
    print(f"[panel-b] MESSI vs best baseline on its own slots: {pct:+.1f}% lower")

    pdf, png, svg = render_figure(corr, jacc, panel_b, args.output_dir)
    print(f"\n[plot] {pdf}")
    print(f"[plot] {png}")
    print(f"[plot] {svg}")

    # Persist machine-readable outputs.
    summary = {
        "q": float(args.q),
        "min_count": int(args.min_count),
        "max_samples_per_group": int(args.max_samples_per_group),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "feature_normalization": "l2",
        "distance": "EnergyDistance(Euclidean)",
        "method_meta": method_meta,
        "panel_a": {
            "method_order": METHOD_ORDER,
            "pearson_corr": corr.tolist(),
            "jaccard":      jacc.tolist(),
            "n_common_pair_keys": len(common_keys),
            "selected_set_size_per_method": {
                m: len(selected_sets[m]) for m in METHOD_ORDER
            },
        },
        "panel_b": {
            "messi_selected_set_size": len(messi_selected),
            "metrics_on_messi_mask": {m: panel_b[m] for m in METHOD_ORDER},
            "messi_vs_best_baseline_pct_lower": pct,
        },
    }
    json_path = os.path.join(args.output_dir, "alignment_routing_messi_v2.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] {json_path}")

    csv_path = os.path.join(args.output_dir, "alignment_routing_messi_v2.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "method", "metric", "value"])
        for i, mi in enumerate(METHOD_ORDER):
            for j, mj in enumerate(METHOD_ORDER):
                w.writerow(["a", f"{mi}_vs_{mj}", "pearson_corr", corr[i, j]])
                w.writerow(["a", f"{mi}_vs_{mj}", "jaccard",      jacc[i, j]])
        for m in METHOD_ORDER:
            w.writerow(["b", m, "Delta_on_messi_mask", panel_b[m]["Delta"]])
            w.writerow(["b", m, "se_on_messi_mask",    panel_b[m]["se"]])
            w.writerow(["b", m, "n_pairs_on_messi_mask",
                        panel_b[m]["n_pairs_on_messi_mask"]])
    print(f"[csv]  {csv_path}")


if __name__ == "__main__":
    main()
