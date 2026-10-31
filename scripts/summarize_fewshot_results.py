"""Aggregate target metrics within seed, then mean/SD across support seeds."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/fewshot_ood/runs"
INFEASIBLE = ROOT / "results/fewshot_ood/infeasible"
OUT = ROOT / "results/fewshot_ood/summary"
EXPECTED_FOLDS = {"DomainNet": 6, "PACS": 4, "OfficeHome": 4,
                  "TerraIncognita": 4, "VLCS": 4}


def main():
    by_seed = defaultdict(list)
    per_target = []
    for path in RUNS.rglob("*.json"):
        row = json.loads(path.read_text())
        key = (row["dataset"], row["backbone"], row["method"], row["setting"],
               row["shot_scope"], row["shots"], row["seed"])
        by_seed[key].append(row)
        for target, metrics in row["target_metrics"].items():
            per_target.append({"dataset": row["dataset"], "backbone": row["backbone"],
                               "method": row["method"], "setting": row["setting"],
                               "shot_scope": row["shot_scope"], "shots": row["shots"],
                               "seed": row["seed"], "fold_id": row["fold_id"],
                               "target_domain": target, "accuracy": metrics["accuracy"],
                               "macro_accuracy": metrics["macro_accuracy"],
                               "n_query": metrics["n_query"],
                               "manifest_sha256": row["manifest_sha256"]})
    infeasible_by_config = defaultdict(dict)
    for path in INFEASIBLE.rglob("*.json"):
        row = json.loads(path.read_text())
        key = (row["dataset"], row["setting"], row["shot_scope"],
               row["shots"], row["seed"])
        if row["fold_id"] in infeasible_by_config[key]:
            raise RuntimeError(f"Duplicate infeasible fold: {path}")
        infeasible_by_config[key][row["fold_id"]] = row["reason"]
    seed_scores = defaultdict(dict)
    incomplete = []
    for key, rows in by_seed.items():
        dataset, backbone, method, setting, scope, shots, seed = key
        excluded = infeasible_by_config[(dataset, setting, scope, shots, seed)]
        observed = {row["fold_id"] for row in rows}
        if len(observed) != len(rows) or observed & set(excluded):
            raise RuntimeError(f"Duplicate or contradictory folds: {key}")
        if len(rows) + len(excluded) != EXPECTED_FOLDS[dataset]:
            incomplete.append({"key": key, "completed": len(rows),
                               "infeasible": len(excluded),
                               "expected": EXPECTED_FOLDS[dataset]})
            continue
        # Equal weight to each target within a fold, then each fold within seed.
        fold_means = [statistics.mean(item["accuracy"] for item in row["target_metrics"].values())
                      for row in rows]
        all_targets = [item["accuracy"] for row in rows for item in row["target_metrics"].values()]
        seed_scores[key[:-1]][seed] = {"mean": statistics.mean(fold_means),
                                       "worst": min(all_targets),
                                       "feasible_folds": len(rows),
                                       "infeasible_folds": len(excluded)}
    summary = []
    for key, seeds in sorted(seed_scores.items()):
        means = [seeds[seed]["mean"] for seed in sorted(seeds)]
        worst = [seeds[seed]["worst"] for seed in sorted(seeds)]
        summary.append(dict(zip(("dataset", "backbone", "method", "setting", "shot_scope", "shots"), key),
                            seeds=sorted(seeds), n_seeds=len(seeds),
                            feasible_folds_per_seed={str(seed): seeds[seed]["feasible_folds"] for seed in sorted(seeds)},
                            infeasible_folds_per_seed={str(seed): seeds[seed]["infeasible_folds"] for seed in sorted(seeds)},
                            mean_accuracy=statistics.mean(means),
                            sd_across_seeds=statistics.stdev(means) if len(means) > 1 else None,
                            mean_worst_target_accuracy=statistics.mean(worst),
                            complete_five_seeds=len(seeds) == 5))
    OUT.mkdir(parents=True, exist_ok=True)
    all_infeasible = [
        {"dataset": key[0], "setting": key[1], "shot_scope": key[2],
         "shots": key[3], "seed": key[4], "n_infeasible_folds": len(reasons),
         "reasons_by_fold": reasons}
        for key, reasons in sorted(infeasible_by_config.items())
        if len(reasons) == EXPECTED_FOLDS[key[0]]
    ]
    (OUT / "summary.json").write_text(json.dumps({"aggregation": "equal target within feasible fold, equal feasible fold within seed, mean/SD across seeds",
                                                   "rows": summary, "incomplete_seed_groups": incomplete,
                                                   "infeasible_fold_records": sum(map(len, infeasible_by_config.values())),
                                                   "all_infeasible_seed_configs": all_infeasible}, indent=2) + "\n")
    for name, rows in (("summary.csv", summary), ("per_target.csv", per_target)):
        if not rows:
            continue
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(f"{len(summary)} aggregate rows, {len(per_target)} per-target rows, "
          f"{len(incomplete)} incomplete seed groups")


if __name__ == "__main__":
    main()
