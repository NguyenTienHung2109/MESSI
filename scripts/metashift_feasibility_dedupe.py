"""Phase 0 dedupe-aware feasibility — replaces the raw-count table.

For each candidate (class_set, n) we run the deterministic single-bucket
assignment (same rule used by metashift_build_splits.py) and count how many
contexts have post-dedupe cell-min >= n. Reports max feasible K = (#valid - 2).
"""
from __future__ import annotations

import json
from pathlib import Path

from metashift_build_splits import (
    CLASS_SETS, assign_images, load_subsets, parse_subset, shared_contexts_for,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "metashift" / "feasibility"
N_GRID = [40, 60, 80, 96, 120, 160, 200, 240]


def main() -> None:
    subsets = load_subsets()
    rows = []
    for tag, classes in CLASS_SETS.items():
        if any(c not in {parse_subset(k)[0] for k in subsets if parse_subset(k)} for c in classes):
            continue
        for n in N_GRID:
            ctxs = shared_contexts_for(subsets, classes, n)
            bucket = assign_images(subsets, classes, ctxs, seed=0)
            valid = [
                ctx for ctx in ctxs
                if min(len(bucket.get((cls, ctx), [])) for cls in classes) >= n
            ]
            rows.append({
                "class_set": tag, "classes": list(classes), "n": n,
                "raw_shared_contexts": len(ctxs),
                "post_dedupe_valid_contexts": len(valid),
                "max_feasible_K_post_dedupe": max((K for K in (4, 6, 8, 10, 12) if len(valid) >= K + 2), default=None),
                "valid_contexts_top10": valid[:10],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility_post_dedupe.json").write_text(json.dumps(rows, indent=2))
    # Markdown table.
    lines = ["# Post-dedupe feasibility (the one that actually matters)\n"]
    lines.append("Raw / post-dedupe context counts; max feasible K = post-dedupe - 2.\n")
    lines.append("| class_set | n | raw | post-dedupe | max K |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['class_set']} | {r['n']} | {r['raw_shared_contexts']} | {r['post_dedupe_valid_contexts']} | {r['max_feasible_K_post_dedupe']} |")
    (OUT / "feasibility_post_dedupe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
