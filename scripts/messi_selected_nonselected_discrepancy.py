"""MESSI mechanistic diagnostic: routing-selected vs non-selected pair discrepancy.

For each (expert m, class c, source-domain pair i<j) on the held-out source-
validation split, we compute:

    rho_{m,k,c} = mean_{x in D_{k,c}} pi_m(x)
    a_{m,i,j,c} = sigmoid(alpha * rho_{m,i,c}) * sigmoid(alpha * rho_{m,j,c})
    ED_{m,i,j,c} = EnergyDistance( {h_m(x): d=i,y=c}, {h_m(x): d=j,y=c} )

Pairs are partitioned into "selected" (top-q by a) vs "non-selected" (rest).
We report Delta_selected, Delta_nonselected, SelGap and a one-column bar plot.

Usage:
    python -m scripts.messi_selected_nonselected_discrepancy \
        --checkpoint multi_dataset/test_InvMMD/PACS_env0_seed0/model.pkl \
        --output_dir multi_dataset/figures/sel_pacs_env0_seed0
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
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("DOMAINBED_PROJECT_DIR", _REPO_ROOT)

from domainbed import algorithms, datasets
from domainbed.lib import misc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--q", type=float, default=0.30,
                   help="top-q fraction of pairs (by routing responsibility) treated as 'selected'")
    p.add_argument("--tau", type=float, default=None,
                   help="if set, override --q with a fixed responsibility threshold")
    p.add_argument("--min_count", type=int, default=5,
                   help="minimum #samples in a (m,k,c) group to be considered valid")
    p.add_argument("--max_samples_per_group", type=int, default=128,
                   help="subsample features in each group to at most this many points")
    p.add_argument("--n_boot", type=int, default=1000,
                   help="bootstrap resamples for standard errors")
    p.add_argument("--l2_normalize", action="store_true",
                   help="L2-normalize each expert's features before ED "
                        "(removes per-expert feature-norm confound)")
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


def load_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    hparams = ckpt["model_hparams"]
    algo_name = ckpt_args["algorithm"]
    algo_cls = getattr(algorithms, algo_name)
    model = algo_cls(
        ckpt["model_input_shape"],
        ckpt["model_num_classes"],
        ckpt["model_num_domains"],
        hparams,
    )
    model.load_state_dict(ckpt["model_dict"], strict=False)
    model.to(device).eval()
    return model, ckpt, ckpt_args, hparams


def build_source_val_splits(ckpt_args: dict, hparams: dict):
    """Reproduce train.py's per-env 'out' split for source environments only."""
    dataset_name = ckpt_args["dataset"]
    data_dir = ckpt_args["data_dir"]
    test_envs = list(ckpt_args["test_envs"])
    holdout_fraction = float(ckpt_args["holdout_fraction"])
    trial_seed = int(ckpt_args["trial_seed"])

    eval_hparams = dict(hparams)
    eval_hparams.setdefault("data_augmentation", False)
    eval_hparams.setdefault("class_balanced", False)

    dataset_cls = getattr(datasets, dataset_name)
    dataset = dataset_cls(data_dir, test_envs, eval_hparams)

    splits = []  # list of (original_env_id, source_idx, out_split)
    source_idx = 0
    for env_i, env in enumerate(dataset):
        if env_i in test_envs:
            continue
        out_split, _in_split = misc.split_dataset(
            env,
            int(len(env) * holdout_fraction),
            misc.seed_hash(trial_seed, env_i),
        )
        splits.append((env_i, source_idx, out_split))
        source_idx += 1

    return splits, dataset, test_envs


@torch.no_grad()
def extract_features(model, splits, batch_size: int, num_workers: int, device: str):
    """Forward pass over each source-val split.

    Returns:
        pi:      (N, M) router probs
        h_stack: (N, M, r) per-expert features
        y:       (N,)  class labels
        d:       (N,)  contiguous source-domain ids in [0, S)
    """
    pis, hs, ys, ds = [], [], [], []
    for original_env_id, source_idx, subset in splits:
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _logits, pi, h_stack = model._forward(x)
            pis.append(pi.detach().cpu().numpy().astype(np.float32))
            hs.append(h_stack.detach().cpu().numpy().astype(np.float32))
            ys.append(y.numpy().astype(np.int64))
            ds.append(np.full(y.shape[0], source_idx, dtype=np.int64))
    pi = np.concatenate(pis, axis=0)
    h_stack = np.concatenate(hs, axis=0)
    y = np.concatenate(ys, axis=0)
    d = np.concatenate(ds, axis=0)
    return pi, h_stack, y, d


def energy_distance(P: np.ndarray, Q: np.ndarray, device: str) -> float:
    """Empirical Energy Distance with Euclidean distance."""
    P_t = torch.from_numpy(P).to(device)
    Q_t = torch.from_numpy(Q).to(device)
    d_PQ = torch.cdist(P_t, Q_t, p=2.0).mean().item()
    if P_t.shape[0] >= 2:
        d_PP = torch.cdist(P_t, P_t, p=2.0).mean().item()
    else:
        d_PP = 0.0
    if Q_t.shape[0] >= 2:
        d_QQ = torch.cdist(Q_t, Q_t, p=2.0).mean().item()
    else:
        d_QQ = 0.0
    ed = 2.0 * d_PQ - d_PP - d_QQ
    return max(ed, 0.0)


def maybe_subsample(idx: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= max_n:
        return idx
    return rng.choice(idx, size=max_n, replace=False)


def compute_pair_records(pi, h_stack, y, d, num_classes, num_source_domains,
                         alpha, min_count, max_samples_per_group, seed, device,
                         l2_normalize=False):
    """Return list of dicts with m,i,j,c,a,ED,nP,nQ for each valid pair."""
    M = pi.shape[1]
    records = []

    # Pre-index sample indices per (k, c)
    group_idx = {}
    for k in range(num_source_domains):
        mask_k = (d == k)
        for c in range(num_classes):
            idx = np.nonzero(mask_k & (y == c))[0]
            if len(idx) >= min_count:
                group_idx[(k, c)] = idx

    base_rng = np.random.default_rng(seed)

    # rho[m, k, c] only for valid (k,c) groups
    for m in range(M):
        # cache per-expert per-group routing mass and feature subsamples
        rho_mc = {}        # (k, c) -> rho
        feats_mc = {}      # (k, c) -> ndarray subsampled
        for (k, c), idx in group_idx.items():
            rho_mc[(k, c)] = float(pi[idx, m].mean())
            sub = maybe_subsample(
                idx, max_samples_per_group,
                np.random.default_rng(int(base_rng.integers(0, 2**31 - 1))),
            )
            feats = h_stack[sub, m, :].astype(np.float32, copy=False)
            if l2_normalize:
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                feats = feats / norms
            feats_mc[(k, c)] = feats

        for c in range(num_classes):
            for i in range(num_source_domains):
                if (i, c) not in rho_mc:
                    continue
                for j in range(i + 1, num_source_domains):
                    if (j, c) not in rho_mc:
                        continue
                    rho_i = rho_mc[(i, c)]
                    rho_j = rho_mc[(j, c)]
                    a = float(
                        1.0 / (1.0 + np.exp(-alpha * rho_i)) *
                        1.0 / (1.0 + np.exp(-alpha * rho_j))
                    )
                    P = feats_mc[(i, c)]
                    Q = feats_mc[(j, c)]
                    ed = energy_distance(P, Q, device)
                    records.append({
                        "m": int(m), "i": int(i), "j": int(j), "c": int(c),
                        "a": a, "ED": ed, "nP": int(P.shape[0]), "nQ": int(Q.shape[0]),
                        "rho_i": rho_i, "rho_j": rho_j,
                    })
    return records


def bootstrap_se(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    if len(values) == 0:
        return float("nan")
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[b] = sample.mean()
    return float(boots.std(ddof=1))


def make_plot(delta_sel, se_sel, delta_non, se_non, output_dir):
    fig, ax = plt.subplots(figsize=(3.2, 2.3), constrained_layout=True)
    xs = [0, 1]
    heights = [delta_sel, delta_non]
    errs = [se_sel, se_non]
    colors = ["#4C72B0", "#C44E52"]
    bars = ax.bar(xs, heights, yerr=errs, color=colors, width=0.6,
                  capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Selected", "Non-selected"], fontsize=9)
    ax.set_ylabel(r"Class-cond. discrepancy $\downarrow$", fontsize=9)
    ax.set_title("Routing-selected alignment", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ymax = max(h + e for h, e in zip(heights, errs)) if heights else 1.0
    ax.set_ylim(0, ymax * 1.25)
    ax.text(0.5, 0.97, "lower on selected pairs",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, alpha=0.7, style="italic")

    pdf_path = os.path.join(output_dir, "selected_nonselected_discrepancy.pdf")
    png_path = os.path.join(output_dir, "selected_nonselected_discrepancy.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[load] checkpoint = {args.checkpoint}")
    model, ckpt, ckpt_args, hparams = load_checkpoint(args.checkpoint, device)
    alpha = float(hparams.get("alpha", 4.0))
    M = model.NUM_EXPERTS
    num_classes = ckpt["model_num_classes"]
    num_source_domains = ckpt["model_num_domains"]
    print(f"[load] algo={ckpt_args['algorithm']}  dataset={ckpt_args['dataset']}  "
          f"test_envs={ckpt_args['test_envs']}  alpha={alpha}  M={M}  "
          f"num_classes={num_classes}  num_source_domains={num_source_domains}")

    print("[split] reconstructing source-validation 'out' splits")
    splits, dataset_obj, test_envs = build_source_val_splits(ckpt_args, hparams)
    assert len(splits) == num_source_domains, \
        f"split count {len(splits)} != model_num_domains {num_source_domains}"
    for original_env_id, source_idx, subset in splits:
        print(f"  source_idx={source_idx}  original_env={original_env_id}  "
              f"|out|={len(subset)}")

    print("[forward] extracting per-sample (pi, h_stack, y, d)")
    pi, h_stack, y, d = extract_features(
        model, splits, args.batch_size, args.num_workers, device,
    )
    assert pi.shape[1] == M, f"router width {pi.shape[1]} != NUM_EXPERTS {M}"
    assert h_stack.shape[1] == M, f"h_stack expert dim {h_stack.shape[1]} != {M}"
    print(f"[forward] N={pi.shape[0]}  pi={pi.shape}  h_stack={h_stack.shape}")

    # Per (env, class) sample counts for the log
    per_group_counts = {}
    for k in range(num_source_domains):
        for c in range(num_classes):
            n = int(((d == k) & (y == c)).sum())
            per_group_counts[f"d={k},c={c}"] = n

    print(f"[pairs] computing pair responsibilities and Energy Distances "
          f"(min_count={args.min_count}, max_samples_per_group={args.max_samples_per_group})")
    records = compute_pair_records(
        pi, h_stack, y, d, num_classes, num_source_domains,
        alpha, args.min_count, args.max_samples_per_group, args.seed, device,
        l2_normalize=args.l2_normalize,
    )
    n_valid = len(records)
    if n_valid == 0:
        print("[ERROR] no valid pairs — try a smaller --min_count")
        sys.exit(1)
    print(f"[pairs] num_valid_pairs = {n_valid}")

    a_arr = np.array([r["a"] for r in records], dtype=np.float64)
    ed_arr = np.array([r["ED"] for r in records], dtype=np.float64)

    # Selection
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
    if not (0 < n_sel < n_valid):
        print(f"[WARNING] degenerate selection: n_sel={n_sel}, n_valid={n_valid}")

    delta_sel = float(ed_arr[sel_mask].mean()) if n_sel > 0 else float("nan")
    delta_non = float(ed_arr[non_mask].mean()) if n_non > 0 else float("nan")
    sel_gap = delta_non - delta_sel

    boot_rng = np.random.default_rng(args.seed + 1)
    se_sel = bootstrap_se(ed_arr[sel_mask], args.n_boot, boot_rng)
    se_non = bootstrap_se(ed_arr[non_mask], args.n_boot, boot_rng)

    print(f"[result] selection rule  : {rule}")
    print(f"[result] num_valid_pairs : {n_valid}")
    print(f"[result] num_selected    : {n_sel}")
    print(f"[result] num_nonselected : {n_non}")
    print(f"[result] Delta_selected    = {delta_sel:.6f}  (SE {se_sel:.6f})")
    print(f"[result] Delta_nonselected = {delta_non:.6f}  (SE {se_non:.6f})")
    print(f"[result] SelGap (non - sel) = {sel_gap:.6f}")
    if not (sel_gap > 0):
        print("[WARNING] SelGap is non-positive on this checkpoint/env")

    pdf_path, png_path = make_plot(delta_sel, se_sel, delta_non, se_non, args.output_dir)
    print(f"[plot] wrote {pdf_path}")
    print(f"[plot] wrote {png_path}")

    summary = {
        "dataset": ckpt_args["dataset"],
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "algorithm": ckpt_args["algorithm"],
        "target_env_id": list(map(int, ckpt_args["test_envs"])),
        "source_env_ids_original": [int(o) for (o, _, _) in splits],
        "source_env_ids_contiguous": [int(s) for (_, s, _) in splits],
        "alpha": alpha,
        "num_experts": int(M),
        "num_classes": int(num_classes),
        "num_source_domains": int(num_source_domains),
        "q": None if args.tau is not None else float(args.q),
        "tau": None if args.tau is None else float(args.tau),
        "selection_threshold_on_a": threshold,
        "min_count": int(args.min_count),
        "max_samples_per_group": int(args.max_samples_per_group),
        "feature_normalization": "l2" if args.l2_normalize else "raw",
        "distance": "EnergyDistance(Euclidean)",
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "num_valid_pairs": int(n_valid),
        "num_selected": int(n_sel),
        "num_nonselected": int(n_non),
        "Delta_selected": delta_sel,
        "Delta_nonselected": delta_non,
        "SelGap": sel_gap,
        "se_selected": se_sel,
        "se_nonselected": se_non,
        "note_on_checkpoint": "final-step model.pkl; training-domain val-best not saved by the training script",
    }
    json_path = os.path.join(args.output_dir, "selected_nonselected_discrepancy.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] wrote {json_path}")

    log_path = os.path.join(args.output_dir, "selected_nonselected_discrepancy.log")
    with open(log_path, "w") as f:
        f.write(f"checkpoint  : {os.path.abspath(args.checkpoint)}\n")
        f.write(f"algorithm   : {ckpt_args['algorithm']}\n")
        f.write(f"dataset     : {ckpt_args['dataset']}\n")
        f.write(f"data_dir    : {ckpt_args['data_dir']}\n")
        f.write(f"test_envs   : {ckpt_args['test_envs']}\n")
        f.write(f"holdout_fraction : {ckpt_args['holdout_fraction']}\n")
        f.write(f"trial_seed       : {ckpt_args['trial_seed']}\n")
        f.write(f"alpha       : {alpha}\n")
        f.write(f"M (experts) : {M}\n")
        f.write(f"feature_dim : {h_stack.shape[2]}\n")
        f.write(f"selection rule : {rule}\n\n")
        f.write("source_env splits (original_id -> contiguous_id, |out|):\n")
        for o, s, sub in splits:
            f.write(f"  {o} -> {s}    |out|={len(sub)}\n")
        f.write("\nper (source_d, class) sample counts in source-validation:\n")
        for key, n in per_group_counts.items():
            f.write(f"  {key} : {n}\n")
        f.write("\nresults:\n")
        f.write(f"  num_valid_pairs   = {n_valid}\n")
        f.write(f"  num_selected      = {n_sel}\n")
        f.write(f"  num_nonselected   = {n_non}\n")
        f.write(f"  Delta_selected    = {delta_sel:.6f}  (SE {se_sel:.6f})\n")
        f.write(f"  Delta_nonselected = {delta_non:.6f}  (SE {se_non:.6f})\n")
        f.write(f"  SelGap            = {sel_gap:.6f}\n")
    print(f"[log] wrote {log_path}")


if __name__ == "__main__":
    main()
