#!/usr/bin/env python3
"""Aggregate a source-only S-IRM-res sweep and emit the selected full config."""

import argparse
import json
import statistics
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.root)
    expected = 4 * 3
    candidates = {}
    for run_name, run_spec in sorted(config["runs"].items()):
        rows = []
        for env in range(4):
            for seed in range(3):
                path = root / run_name / f"env{env}" / f"seed{seed}" / "summary.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
                if summary.get("target_component_accuracy") is not None:
                    raise RuntimeError(f"screening run touched target data: {path}")
                rows.append({
                    "env": env,
                    "seed": seed,
                    "source_val": float(summary["source_val_accuracy"]),
                    "global_source_val": float(summary["selected_global_source_val"]),
                    "source_gain": float(
                        summary["source_val_accuracy"]
                        - summary["selected_global_source_val"]
                    ),
                    "anchor_unchanged": bool(summary["anchor_unchanged"]),
                })
        if args.require_complete and len(rows) != expected:
            raise RuntimeError(f"{run_name}: found {len(rows)}/{expected} runs")
        values = [row["source_val"] for row in rows]
        gains = [row["source_gain"] for row in rows]
        candidates[run_name] = {
            "completed_runs": len(rows),
            "expected_runs": expected,
            "source_val_mean": statistics.mean(values) if values else None,
            "source_val_std": statistics.stdev(values) if len(values) > 1 else None,
            "source_gain_mean": statistics.mean(gains) if gains else None,
            "all_anchors_unchanged": all(row["anchor_unchanged"] for row in rows),
            "hparams": run_spec["hparams"],
            "runs": rows,
        }

    eligible = {
        name: result for name, result in candidates.items()
        if result["completed_runs"] == expected and result["all_anchors_unchanged"]
    }
    selected = max(
        eligible, key=lambda name: eligible[name]["source_val_mean"]
    ) if eligible else None
    all_hparam_names = sorted({
        name for spec in config["runs"].values()
        for name in spec["hparams"]
    })
    coverage = {
        name: sorted({
            json.dumps(spec["hparams"][name], sort_keys=True)
            for spec in config["runs"].values() if name in spec["hparams"]
        }) for name in all_hparam_names
    }
    report = {
        "selection_metric": "mean source validation over 4 env x 3 seeds",
        "target_used_for_selection": False,
        "selected_candidate": selected,
        "hyperparameter_coverage": coverage,
        "candidates": candidates,
    }
    output = Path(args.output)
    write_json(output, report)

    if selected is not None:
        selected_hparams = dict(config["runs"][selected]["hparams"])
        selected_hparams["sirm_res_global_steps"] = 2000
        selected_hparams["sirm_res_residual_steps"] = 3000
        final_config = {
            key: value for key, value in config.items() if key != "runs"
        }
        final_config["checkpoint_freq"] = 300
        final_config["runs"] = {
            "S-IRM-res_tuned": {
                "algorithm": "S-IRM-res",
                "description": (
                    f"Full-budget retrain of source-selected {selected}"
                ),
                "hparams": selected_hparams,
            }
        }
        write_json(output.with_name("selected_full_config.json"), final_config)
    print(json.dumps({
        "selected_candidate": selected,
        "completed_candidates": len(eligible),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
