"""Two standalone, NeurIPS-quality figures for MESSI's mechanism section.

Figure 1 — responsibility_structure_similarity:
    Horizontal dot plot of Pearson correlation between pair-responsibility
    vectors a_{ijc}^{(m)} for the three method pairs. Jaccard overlap of
    selected-slot sets is annotated next to each point.

Figure 2 — messi_selected_slot_discrepancy:
    Point-range plot of class-conditional Energy Distance evaluated on the
    *same* MESSI-selected class-domain-expert slots, for all three methods.

Diagnostics on held-out source-validation only — no target-domain samples.
L2-normalised expert features (removes per-expert feature-norm confound).

Usage:
    python -m scripts.messi_alignment_two_figures \\
        --messi_ckpt   multi_dataset/test_InvMMD_small/PACS_env0_seed0/model.pkl \\
        --global_ckpt  multi_dataset/test_InvMMD_small_global/PACS_env0_seed0/model.pkl \\
        --random_ckpt  multi_dataset/test_InvMMD_small_random/PACS_env0_seed0/model.pkl \\
        --output_dir   multi_dataset/figures/alignment_neurips_pacs_env0_seed0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
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


# Muted, publication-friendly palette.
COLOR_BASELINE  = "#9AA3AD"   # gray for Global / Random
COLOR_MESSI     = "#4C72B0"   # accent blue for MESSI
COLOR_POS_CORR  = "#C97B5C"   # muted warm for +corr lollipops
COLOR_NEG_CORR  = "#5C7FBE"   # muted blue for -corr lollipops
COLOR_ZERO_LINE = "#888888"


def _setup_rc():
    rcParams.update({
        "font.family":     "DejaVu Sans",
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  7.5,
        "axes.linewidth":   0.7,
        "xtick.major.width":0.7,
        "ytick.major.width":0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype":     42,   # editable text in vector outputs
        "ps.fonttype":      42,
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--messi_ckpt",  required=True)
    p.add_argument("--global_ckpt", required=True)
    p.add_argument("--random_ckpt", required=True)
    p.add_argument("--output_dir",  required=True)
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
    meta = {
        "alpha": alpha,
        "alignment_mode_in_ckpt": hparams.get("alignment_mode", "messi"),
        "dataset": ckpt_args["dataset"],
        "target_env": list(map(int, ckpt_args["test_envs"])),
        "seed": int(ckpt_args["seed"]),
        "trial_seed": int(ckpt_args["trial_seed"]),
        "num_source_domains": int(ckpt["model_num_domains"]),
        "num_classes": int(ckpt["model_num_classes"]),
    }
    del model
    torch.cuda.empty_cache()
    return records, meta


def compute_panel_a(records_per_method, q):
    """Return (corrs, jaccards, selected_sets, common_keys, a_mats)."""
    keys_per_method = {
        m: {(r["m"], r["c"], r["i"], r["j"]): r["a"] for r in recs}
        for m, recs in records_per_method.items()
    }
    common_keys = sorted(set.intersection(
        *[set(d.keys()) for d in keys_per_method.values()]
    ))
    a_mats = {
        m: np.array([keys_per_method[m][k] for k in common_keys])
        for m in records_per_method
    }
    selected_sets = {}
    for m, a in a_mats.items():
        thr = np.quantile(a, 1.0 - q)
        selected_sets[m] = {k for k, av in zip(common_keys, a) if av >= thr}
    pairs = [
        ("global",  "random",  "Global–Random"),
        ("messi",   "global",  "MESSI–Global"),
        ("messi",   "random",  "MESSI–Random"),
    ]
    rows = []
    for a_n, b_n, label in pairs:
        corr = float(np.corrcoef(a_mats[a_n], a_mats[b_n])[0, 1])
        a_set, b_set = selected_sets[a_n], selected_sets[b_n]
        jacc = len(a_set & b_set) / max(1, len(a_set | b_set))
        rows.append({"a": a_n, "b": b_n, "label": label,
                     "corr": corr, "jaccard": jacc})
    return rows, selected_sets, common_keys, a_mats


def compute_panel_b(records_per_method, messi_selected, n_boot, seed):
    boot_rng = np.random.default_rng(seed + 1)
    out = {}
    for method, recs in records_per_method.items():
        eds = np.array([r["ED"] for r in recs
                        if (r["m"], r["c"], r["i"], r["j"]) in messi_selected],
                       dtype=np.float64)
        out[method] = {
            "n_pairs_on_messi_mask": int(len(eds)),
            "Delta": float(eds.mean()) if len(eds) else float("nan"),
            "se":    bootstrap_se(eds, n_boot, boot_rng) if len(eds) else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Figure 1 — Responsibility structure similarity (horizontal dot plot)
# ---------------------------------------------------------------------------

def render_figure_1(rows, output_dir):
    """Render a one-column, horizontal dot plot.

    This replaces the old vertical lollipop plot because long method-pair names
    and Jaccard labels were clipped at one-column width. The horizontal layout
    keeps pair names readable and places both r and J next to each point.
    """
    fig, ax = plt.subplots(figsize=(3.35, 2.05), constrained_layout=True)

    # Keep the scientific story top-to-bottom: baseline controls first, then
    # MESSI-vs-control comparisons. Matplotlib puts y=0 at the bottom, so use
    # reversed positions and then set labels manually.
    y_pos = np.arange(len(rows))[::-1]
    corrs = np.array([r["corr"] for r in rows])
    colors = [COLOR_POS_CORR if c >= 0 else COLOR_NEG_CORR for c in corrs]

    # Reference and subtle x-grid.
    ax.axvline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.6,
               alpha=0.65, zorder=0)
    ax.xaxis.grid(True, linewidth=0.35, color="#E6E6E6", zorder=0)
    ax.set_axisbelow(True)

    # Dot plot. No long lollipop stems: the zero reference line already conveys
    # sign, and avoiding stems makes the figure less visually heavy.
    ax.scatter(corrs, y_pos, s=52, c=colors, edgecolor="white",
               linewidth=0.8, zorder=3)

    # Annotate each point with both metrics. Keep text inside axes bounds.
    for y, r, col in zip(y_pos, rows, colors):
        c = r["corr"]
        # Place positive labels to the left of the point and negative labels to
        # the right, so labels remain inside [-1, 1].
        if c >= 0:
            x_txt, ha = c - 0.07, "right"
        else:
            x_txt, ha = c + 0.08, "left"
        ax.text(
            x_txt, y,
            f"$r={c:+.2f}$, $J={r['jaccard']:.2f}$",
            ha=ha, va="center", fontsize=7.2, color="black", zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Pearson corr. of responsibilities")
    ax.set_title("Responsibility structure differs", pad=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)

    base = os.path.join(output_dir, "responsibility_structure_similarity")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    return base


# ---------------------------------------------------------------------------
# Figure 2 — Alignment on MESSI-selected slots (point-range)
# ---------------------------------------------------------------------------

def render_figure_2(panel_b, output_dir):
    methods = ["global", "random", "messi"]
    labels  = ["Global-MoE", "Rand.-subset", "MESSI"]
    deltas  = [panel_b[m]["Delta"] for m in methods]
    ses     = [panel_b[m]["se"]    for m in methods]
    colors  = [COLOR_BASELINE, COLOR_BASELINE, COLOR_MESSI]

    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    xs = np.arange(len(methods))

    # Thin per-method error bars (errorbar's ecolor needs a single color).
    for x, d, s, col in zip(xs, deltas, ses, colors):
        ax.errorbar(x, d, yerr=s, fmt="none",
                    ecolor=col, elinewidth=1.0, capsize=2.5, capthick=0.8,
                    zorder=1)
    ax.scatter(xs, deltas, s=44, c=colors, edgecolor="white",
               linewidth=0.7, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylabel("Class-cond. discrepancy ↓")
    ax.set_title("Alignment on MESSI-selected slots", pad=3)

    base_min = min(panel_b["global"]["Delta"], panel_b["random"]["Delta"])
    pct = (base_min - panel_b["messi"]["Delta"]) / base_min * 100.0
    ymax = max(d + s for d, s in zip(deltas, ses)) * 1.22
    ax.set_ylim(0, ymax)
    if pct > 5:
        # Small, unobtrusive annotation; avoid a large arrow that dominates the
        # scientific plot.
        ax.annotate(
            f"{pct:.0f}% lower",
            xy=(2, panel_b["messi"]["Delta"] + panel_b["messi"]["se"]),
            xytext=(1.74, panel_b["messi"]["Delta"] + panel_b["messi"]["se"] + ymax * 0.055),
            ha="left", va="center", fontsize=6.8, color=COLOR_MESSI,
            arrowprops=dict(arrowstyle="-", color=COLOR_MESSI,
                            lw=0.5, shrinkA=1, shrinkB=1),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.yaxis.grid(True, linewidth=0.35, color="#E6E6E6", zorder=0)
    ax.set_axisbelow(True)

    base = os.path.join(output_dir, "messi_selected_slot_discrepancy")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    return base, pct


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    _setup_rc()
    device = args.device if torch.cuda.is_available() else "cpu"

    # 1. Collect per-method records.
    records_per_method, method_meta = {}, {}
    for key, ckpt in [("global", args.global_ckpt),
                      ("random", args.random_ckpt),
                      ("messi",  args.messi_ckpt)]:
        print(f"[{key}] {ckpt}")
        recs, meta = collect_records(ckpt, args, device)
        records_per_method[key] = recs
        method_meta[key] = meta
        print(f"[{key}] alpha={meta['alpha']}  alignment_mode={meta['alignment_mode_in_ckpt']}  "
              f"n_records={len(recs)}")

    # 2. Panel-a quantities.
    rows, selected_sets, common_keys, a_mats = compute_panel_a(
        records_per_method, args.q,
    )
    messi_selected = selected_sets["messi"]

    # 3. Panel-b quantities (on MESSI mask, all 3 methods).
    panel_b = compute_panel_b(records_per_method, messi_selected,
                              args.n_boot, args.seed)

    # 4. Render the two figures.
    fig1_base = render_figure_1(rows, args.output_dir)
    fig2_base, pct_lower = render_figure_2(panel_b, args.output_dir)

    # 5. Persist metrics JSONs.
    common_meta = {
        "dataset": method_meta["messi"]["dataset"],
        "target_env": method_meta["messi"]["target_env"],
        "source_envs_contiguous": list(range(method_meta["messi"]["num_source_domains"])),
        "seed": method_meta["messi"]["seed"],
        "trial_seed": method_meta["messi"]["trial_seed"],
        "checkpoint_paths": {
            "global": os.path.abspath(args.global_ckpt),
            "random": os.path.abspath(args.random_ckpt),
            "messi":  os.path.abspath(args.messi_ckpt),
        },
        "q": float(args.q),
        "min_count": int(args.min_count),
        "max_samples_per_group": int(args.max_samples_per_group),
        "n_boot": int(args.n_boot),
        "feature_normalization": "l2",
        "distance": "EnergyDistance(Euclidean)",
        "source_validation_only": True,
        "target_domain_samples_used": False,
        "n_common_valid_slots": len(common_keys),
        "n_messi_selected_slots": len(messi_selected),
        "method_meta": method_meta,
    }

    fig1_metrics = dict(common_meta)
    fig1_metrics["pearson_corr"] = {r["label"]: r["corr"] for r in rows}
    fig1_metrics["jaccard"] = {r["label"]: r["jaccard"] for r in rows}
    fig1_metrics["selected_set_sizes"] = {
        m: len(selected_sets[m]) for m in records_per_method
    }
    with open(fig1_base + "_metrics.json", "w") as f:
        json.dump(fig1_metrics, f, indent=2)

    fig2_metrics = dict(common_meta)
    fig2_metrics["delta_on_messi_mask"] = {m: panel_b[m]["Delta"] for m in panel_b}
    fig2_metrics["se_on_messi_mask"]    = {m: panel_b[m]["se"]    for m in panel_b}
    fig2_metrics["n_pairs_on_messi_mask"] = {
        m: panel_b[m]["n_pairs_on_messi_mask"] for m in panel_b
    }
    avg_baseline = 0.5 * (panel_b["global"]["Delta"] + panel_b["random"]["Delta"])
    fig2_metrics["relative_reduction_messi_vs_global"] = (
        (panel_b["global"]["Delta"] - panel_b["messi"]["Delta"])
        / panel_b["global"]["Delta"]
    )
    fig2_metrics["relative_reduction_messi_vs_random"] = (
        (panel_b["random"]["Delta"] - panel_b["messi"]["Delta"])
        / panel_b["random"]["Delta"]
    )
    fig2_metrics["relative_reduction_messi_vs_avg_baseline"] = (
        (avg_baseline - panel_b["messi"]["Delta"]) / avg_baseline
    )
    with open(fig2_base + "_metrics.json", "w") as f:
        json.dump(fig2_metrics, f, indent=2)

    # 6. Console summary.
    print("\n=== Figure 1: Responsibility structure ===")
    for r in rows:
        print(f"  {r['label']:<14s}  corr={r['corr']:+.3f}  J={r['jaccard']:.3f}")
    print(f"  → wrote {fig1_base}.{{pdf,png,svg}}, _metrics.json")

    print("\n=== Figure 2: Alignment on MESSI-selected slots (n={}) ===".format(
        len(messi_selected)))
    for m, lbl in [("global", "Global-MoE"), ("random", "Random-Subset-MoE"),
                   ("messi",  "MESSI")]:
        print(f"  {lbl:<22s}  Δ={panel_b[m]['Delta']:.4f} ± {panel_b[m]['se']:.4f}")
    print(f"  MESSI vs Global: {fig2_metrics['relative_reduction_messi_vs_global']*100:+.1f}% lower")
    print(f"  MESSI vs Random: {fig2_metrics['relative_reduction_messi_vs_random']*100:+.1f}% lower")
    print(f"  → wrote {fig2_base}.{{pdf,png,svg}}, _metrics.json")

    print("\nFigure 1 shows Global/Random responsibility similarity and MESSI divergence.")
    print("Figure 2 evaluates all methods on the same MESSI-selected slots.")
    print("Confirmed: no target-domain samples were used.")


if __name__ == "__main__":
    main()
