"""Mirror a running Subset-IRM local log into the PACS sweep W&B schema."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import wandb


def load_records(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def canonical_metrics(record, source_envs, previous):
    source = np.asarray([
        float(record[f"env{env}_out_acc"]) for env in source_envs
    ])
    previous_wall = float(previous["wall_time_seconds"]) if previous else 0.0
    previous_step = int(previous["step"]) if previous else -1
    metrics = {
        "step": int(record["step"]),
        "train/loss": float(record["loss"]),
        "train/moe_aux_loss": float(record["loss"] - record["loss_cls"]),
        "train/step_time": (
            (float(record["wall_time_seconds"]) - previous_wall)
            / max(int(record["step"]) - previous_step, 1)
        ),
        "train/mem_gb": float(record["peak_gpu_memory_gb"]),
        "eval/val_avg_acc": float(source.mean()),
        "eval/val_worst_domain_acc": float(source.min()),
        "eval/val_best_domain_acc": float(source.max()),
        "eval/val_std_acc": float(source.std()),
    }
    for key, value in record.items():
        if key.startswith("env") and key.endswith("_acc"):
            split = "in" if "_in_acc" in key else "out"
            metrics[f"eval/{split}/{key}"] = float(value)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="PACS_sweep")
    parser.add_argument("--entity", default="hunghn2003")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        name=args.name,
        resume="allow",
        config=manifest,
        tags=["PACS", "SIRM", "corrected-schema"],
    )
    uploaded = 0
    while True:
        records = load_records(run_dir / "train_log.jsonl")
        for index in range(uploaded, len(records)):
            previous = records[index - 1] if index else None
            metrics = canonical_metrics(records[index], manifest["source_envs"], previous)
            run.log(metrics, step=int(records[index]["step"]))
        uploaded = len(records)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            run.summary.update(summary)
            run.summary["status"] = "finished"
            break
        time.sleep(args.poll_seconds)
    run.finish()


if __name__ == "__main__":
    main()
