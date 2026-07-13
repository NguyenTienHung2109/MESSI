"""Post-hoc analysis cho M-sweep (number of experts).

Cho mỗi run trong `<sweep_root>/M<M>/<DATASET>_env<E>_seed<S>/`, compute:
  - Params (trainable, từ model thực tế)
  - Train time (sec) — sum step_time × n_steps trong results.jsonl
  - Test acc — training-domain val selection
  - Routing entropy ↓ — E_x[ -Σ_m π_m log π_m ] trên source-val
  - Expert-load std ↓ — std_m( E_x[π_m] ) trên source-val

Aggregate qua (env, seed) per (M, dataset) → CSV và optional LaTeX.

Usage:
    python -m scripts.compute_M_sweep_metrics \\
        --sweep_root multi_dataset/test_InvOT_small_M_sweep \\
        --output_csv multi_dataset/test_InvOT_small_M_sweep/summary.csv \\
        [--latex multi_dataset/test_InvOT_small_M_sweep/table.tex]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("DOMAINBED_PROJECT_DIR", _REPO_ROOT)

# Reuse loaders from the standalone diagnostic script.
from scripts.messi_selected_nonselected_discrepancy import (
    load_checkpoint, build_source_val_splits, extract_features,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="vd: multi_dataset/test_InvOT_small_M_sweep")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--latex", default=None,
                   help="optional: ghi LaTeX tabular ra file")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def best_step_target_acc(results_jsonl, test_env):
    """Training-domain val selection: pick step max mean source env_out_acc."""
    with open(results_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]
    src_envs = sorted({int(k.split("env")[1].split("_")[0])
                       for k in records[0]
                       if k.startswith("env") and k.endswith("_out_acc")}
                      - {test_env})
    best = None
    for r in records:
        score = np.mean([r[f"env{e}_out_acc"] for e in src_envs])
        if best is None or score > best[0]:
            best = (score, r)
    final = records[-1]
    return {
        "best_src_val": float(best[0]),
        "best_step": int(best[1]["step"]),
        "best_tgt_in_acc":  float(best[1].get(f"env{test_env}_in_acc",  np.nan)),
        "best_tgt_out_acc": float(best[1].get(f"env{test_env}_out_acc", np.nan)),
        "final_step": int(final["step"]),
        "final_tgt_in_acc":  float(final.get(f"env{test_env}_in_acc",  np.nan)),
        "final_tgt_out_acc": float(final.get(f"env{test_env}_out_acc", np.nan)),
    }


def train_time_sec(results_jsonl):
    """Sum step_time × (step delta between consecutive log lines)."""
    with open(results_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]
    total = 0.0
    for i, r in enumerate(records):
        step_time = r.get("step_time", 0.0)
        if i == 0:
            n_steps_in_chunk = r["step"] + 1
        else:
            n_steps_in_chunk = r["step"] - records[i - 1]["step"]
        total += step_time * n_steps_in_chunk
    return float(total)


def routing_metrics(pi: np.ndarray, eps=1e-12):
    """
    pi: (N, M)
    Returns (entropy, load_std).
        entropy   = mean over N of -Σ_m π_m log π_m  (nats)
        load_std  = std over m of (mean over N of π_m)
    """
    p = pi.astype(np.float64)
    ent = -(p * np.log(p + eps)).sum(axis=1).mean()
    load = p.mean(axis=0)            # (M,)
    return float(ent), float(load.std())


def model_param_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Walk sweep tree
# ---------------------------------------------------------------------------

RUN_DIR_RE = re.compile(r"^(?P<dataset>[A-Za-z]+)_env(?P<env>\d+)_seed(?P<seed>\d+)$")


def find_runs(sweep_root):
    """Yield (M, dataset, env, seed, run_dir) tuples for every completed run."""
    for m_dir in sorted(os.listdir(sweep_root)):
        m_match = re.match(r"^M(\d+)$", m_dir)
        if not m_match:
            continue
        M = int(m_match.group(1))
        m_path = os.path.join(sweep_root, m_dir)
        for run_name in sorted(os.listdir(m_path)):
            run_path = os.path.join(m_path, run_name)
            done = os.path.join(run_path, "done")
            ckpt = os.path.join(run_path, "model.pkl")
            jsonl = os.path.join(run_path, "results.jsonl")
            if not (os.path.isfile(done) and os.path.isfile(ckpt) and os.path.isfile(jsonl)):
                continue
            m_run = RUN_DIR_RE.match(run_name)
            if not m_run:
                continue
            yield {
                "M": M,
                "dataset": m_run.group("dataset"),
                "env": int(m_run.group("env")),
                "seed": int(m_run.group("seed")),
                "run_dir": run_path,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    rows = []
    for info in find_runs(args.sweep_root):
        run_dir = info["run_dir"]
        test_env = info["env"]
        ckpt_path = os.path.join(run_dir, "model.pkl")
        jsonl_path = os.path.join(run_dir, "results.jsonl")

        print(f"[run] M={info['M']:>2}  {info['dataset']:<11s}  "
              f"env={test_env}  seed={info['seed']}  {run_dir}")

        # 1. Test acc + best step
        sel = best_step_target_acc(jsonl_path, test_env)

        # 2. Train time
        ttime = train_time_sec(jsonl_path)

        # 3. Load checkpoint + forward pass for routing metrics
        model, ckpt, ckpt_args, hparams = load_checkpoint(ckpt_path, device)
        params = model_param_count(model)

        splits, _, _ = build_source_val_splits(ckpt_args, hparams)
        pi, h_stack, y, d = extract_features(
            model, splits, args.batch_size, args.num_workers, device,
        )
        ent, load_std = routing_metrics(pi)
        assert pi.shape[1] == info["M"], (
            f"pi has M={pi.shape[1]} but expected {info['M']} from dir name "
            f"({run_dir})"
        )

        del model
        torch.cuda.empty_cache()

        rows.append({
            "M": info["M"],
            "dataset": info["dataset"],
            "env": info["env"],
            "seed": info["seed"],
            "test_acc_in":  sel["best_tgt_in_acc"],
            "test_acc_out": sel["best_tgt_out_acc"],
            "best_step":    sel["best_step"],
            "train_time_sec": ttime,
            "params": params,
            "routing_entropy": ent,
            "expert_load_std": load_std,
            "run_dir": run_dir,
        })

    if not rows:
        print(f"[ERROR] no completed runs found in {args.sweep_root}")
        sys.exit(1)

    # Aggregate per (M, dataset): mean ± SE over (env, seed)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["M"], r["dataset"])].append(r)

    summary_rows = []
    for (M, dataset), group in sorted(agg.items()):
        accs = np.array([g["test_acc_in"] for g in group])
        times = np.array([g["train_time_sec"] for g in group])
        ents = np.array([g["routing_entropy"] for g in group])
        loads = np.array([g["expert_load_std"] for g in group])
        params = group[0]["params"]   # const within (M, dataset)
        n = len(group)

        def se(x):
            return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

        summary_rows.append({
            "M": M,
            "dataset": dataset,
            "n_runs": n,
            "test_acc_mean":   float(accs.mean()),
            "test_acc_se":     float(se(accs)),
            "params":          params,
            "train_time_sec_mean": float(times.mean()),
            "routing_entropy_mean": float(ents.mean()),
            "routing_entropy_se":   float(se(ents)),
            "expert_load_std_mean": float(loads.mean()),
            "expert_load_std_se":   float(se(loads)),
        })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n[csv] wrote {args.output_csv}  ({len(summary_rows)} rows)")

    # Per-run dump (useful for debugging individual runs)
    per_run_csv = args.output_csv.replace(".csv", "_per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {per_run_csv}  ({len(rows)} per-run rows)")

    # Optional LaTeX (one row per M, columns side-by-side for both datasets)
    if args.latex:
        # Pivot: row = M, columns = (PACS_acc, OH_acc, params, time, ent, load_std)
        # Take avg over datasets for params/time/ent/load_std (since they're per-config),
        # acc kept per-dataset.
        by_M = defaultdict(dict)
        for r in summary_rows:
            by_M[r["M"]][r["dataset"]] = r

        lines = [
            r"\begin{tabular}{rccrrcc}",
            r"\toprule",
            r"$M$ & PACS $\uparrow$ & OfficeHome $\uparrow$ & Params & Train time & "
            r"Routing entropy $\downarrow$ & Expert-load std $\downarrow$ \\",
            r"\midrule",
        ]
        for M in sorted(by_M):
            entry = by_M[M]
            pacs = entry.get("PACS", {})
            oh = entry.get("OfficeHome", {})
            # average ent/load over datasets if both exist
            ents_for_M = [v["routing_entropy_mean"] for v in entry.values()]
            loads_for_M = [v["expert_load_std_mean"] for v in entry.values()]
            times_for_M = [v["train_time_sec_mean"] for v in entry.values()]
            params_for_M = next(iter(entry.values()))["params"]

            pacs_str = (f"{pacs['test_acc_mean']*100:.2f} $\\pm$ {pacs['test_acc_se']*100:.2f}"
                        if pacs else "--")
            oh_str = (f"{oh['test_acc_mean']*100:.2f} $\\pm$ {oh['test_acc_se']*100:.2f}"
                      if oh else "--")
            params_str = f"{params_for_M/1e6:.1f}M"
            time_str = f"{np.mean(times_for_M)/60:.0f} min"
            ent_str = f"{np.mean(ents_for_M):.3f}"
            load_str = f"{np.mean(loads_for_M):.3f}"

            lines.append(
                f"{M} & {pacs_str} & {oh_str} & {params_str} & {time_str} & "
                f"{ent_str} & {load_str} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]

        with open(args.latex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[latex] wrote {args.latex}")

    # Console preview
    print("\n=== Summary (preview) ===")
    print(f"{'M':>3} {'dataset':<11} {'n':>3} {'acc':>14} {'params':>9} "
          f"{'time':>7} {'entropy':>9} {'load_std':>9}")
    for r in summary_rows:
        acc_str = f"{r['test_acc_mean']*100:5.2f} ± {r['test_acc_se']*100:4.2f}"
        print(f"{r['M']:>3} {r['dataset']:<11} {r['n_runs']:>3} {acc_str:>14}  "
              f"{r['params']/1e6:>6.1f}M {r['train_time_sec_mean']/60:>5.0f}m "
              f"{r['routing_entropy_mean']:>9.3f} {r['expert_load_std_mean']:>9.3f}")


if __name__ == "__main__":
    main()
