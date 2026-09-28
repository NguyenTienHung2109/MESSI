#!/usr/bin/env python3
"""Summarize the 4-env x 3-seed full-budget tuned S-IRM-res runs."""

import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for env in range(4):
        for seed in range(3):
            path = root / f"env{env}" / f"seed{seed}" / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            target_out = summary["target_diagnostics"]["by_env"][str(env)]
            rows.append({
                "env": env, "seed": seed,
                "global_source_val": summary["selected_global_source_val"],
                "full_source_val": summary["source_val_accuracy"],
                "global_target_in": summary["target_component_accuracy"]["global_only"],
                "full_target_in": summary["target_component_accuracy"]["full"],
                "global_target_out": target_out["global_accuracy"],
                "full_target_out": target_out["full_accuracy"],
                "anchor_unchanged": summary["anchor_unchanged"],
                "selected_full_step": summary["selected_full_step"],
            })
    if args.require_complete and len(rows) != 12:
        raise RuntimeError(f"found {len(rows)}/12 final runs")
    metrics = {}
    for key in (
        "global_source_val", "full_source_val", "global_target_in",
        "full_target_in", "global_target_out", "full_target_out",
    ):
        metrics[key] = stats([float(row[key]) for row in rows]) if rows else None
    for label, global_key, full_key in (
        ("source_gain", "global_source_val", "full_source_val"),
        ("target_in_gain", "global_target_in", "full_target_in"),
        ("target_out_gain", "global_target_out", "full_target_out"),
    ):
        metrics[label] = stats([
            float(row[full_key]) - float(row[global_key]) for row in rows
        ]) if rows else None
    result = {
        "completed_runs": len(rows), "expected_runs": 12,
        "all_anchors_unchanged": all(row["anchor_unchanged"] for row in rows),
        "metrics": metrics, "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed_runs": len(rows), "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
