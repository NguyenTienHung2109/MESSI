"""Phase 0 feasibility analysis for MetaShift K-domain protocol.

Pickle-only: requires no image download. Loads
data/metashift/meta_data/full-candidate-subsets.pkl and computes, for several
candidate class sets, how many shared contexts have at least n images for every
class in the set, for each (K, n) combination of interest.

Outputs:
  data/metashift/feasibility/feasibility_table.json   - long-form records
  data/metashift/feasibility/feasibility_summary.md   - human-readable matrix
  data/metashift/feasibility/per_class_subsets.json   - top contexts per class
"""
from __future__ import annotations

import json
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "data" / "metashift" / "meta_data" / "full-candidate-subsets.pkl"
OUT = ROOT / "data" / "metashift" / "feasibility"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_SETS = [
    ("cat_dog", ("cat", "dog")),
    ("animals4", ("cat", "dog", "horse", "elephant")),
    ("animals6", ("cat", "dog", "bird", "horse", "elephant", "sheep")),
    ("vehicles4", ("bus", "truck", "car", "motorcycle")),
    ("vehicles6", ("bus", "truck", "car", "motorcycle", "bicycle", "train")),
    ("furniture4", ("chair", "table", "bed", "couch")),
    ("tableware4", ("plate", "cup", "bowl", "glass")),
]
N_GRID = [40, 60, 80, 96, 120, 160, 200, 240]
K_GRID = [4, 6, 8, 10, 12]


def parse(name: str) -> tuple[str, str] | None:
    if "(" not in name or not name.endswith(")"):
        return None
    cls, rest = name.split("(", 1)
    return cls, rest[:-1]


def main() -> None:
    with PKL.open("rb") as f:
        subsets: dict[str, set[int]] = pickle.load(f)

    # cls -> {ctx: count}
    class_to_ctx: dict[str, dict[str, int]] = defaultdict(dict)
    all_image_ids: set[int] = set()
    n_skipped = 0
    for name, ids in subsets.items():
        parsed = parse(name)
        if parsed is None:
            n_skipped += 1
            continue
        cls, ctx = parsed
        class_to_ctx[cls][ctx] = len(ids)
        all_image_ids.update(ids)

    # Recomputed dataset-level stats (vs paper claims).
    counts = [c for ctxs in class_to_ctx.values() for c in ctxs.values()]
    counts_25 = [c for c in counts if c >= 25]
    unique_contexts = {ctx for ctxs in class_to_ctx.values() for ctx in ctxs}
    regen = {
        "n_subsets_total": len(counts),
        "n_subsets_min25": len(counts_25),
        "n_classes": len(class_to_ctx),
        "n_unique_contexts": len(unique_contexts),
        "n_unique_image_ids": len(all_image_ids),
        "median_imgs_per_subset": statistics.median(counts) if counts else 0,
        "mean_imgs_per_subset": (sum(counts) / len(counts)) if counts else 0,
        "skipped_unparseable_keys": n_skipped,
    }

    # Per-class top-contexts dump, useful for split selection.
    per_class = {}
    for cls, ctxs in class_to_ctx.items():
        top = sorted(ctxs.items(), key=lambda kv: kv[1], reverse=True)
        per_class[cls] = {
            "n_subsets_total": len(top),
            "n_subsets_min25": sum(1 for _, c in top if c >= 25),
            "n_subsets_min80": sum(1 for _, c in top if c >= 80),
            "top20": [{"context": k, "n_images": v} for k, v in top[:20]],
        }

    # Feasibility per (class_set, n).
    per_set_n = []
    for tag, S in CLASS_SETS:
        # Verify all classes are present.
        missing = [c for c in S if c not in class_to_ctx]
        if missing:
            per_set_n.append({"class_set": tag, "missing_classes": missing})
            continue
        for n in N_GRID:
            shared = set.intersection(*[
                {ctx for ctx, k in class_to_ctx[cls].items() if k >= n}
                for cls in S
            ])
            # Drop contexts that are themselves a target class (avoids self-co-occur, e.g. chair(table) when both are classes).
            shared = {c for c in shared if c not in S}
            shared_sorted = sorted(shared, key=lambda c: -min(class_to_ctx[cls][c] for cls in S))
            min_counts = {c: min(class_to_ctx[cls][c] for cls in S) for c in shared_sorted}
            per_set_n.append({
                "class_set": tag,
                "classes": list(S),
                "n_per_cell": n,
                "n_shared_contexts": len(shared_sorted),
                "feasible_K": [K for K in K_GRID if len(shared_sorted) >= K + 2],
                "shared_top": [{"context": c, "min_count": min_counts[c]} for c in shared_sorted[:14]],
            })

    table = {
        "regen_stats": regen,
        "per_set_n": per_set_n,
    }
    (OUT / "feasibility_table.json").write_text(json.dumps(table, indent=2))
    (OUT / "per_class_subsets.json").write_text(json.dumps(per_class, indent=2))

    # Markdown summary matrix (rows: class_set, cols: n, cell: max_K_feasible).
    lines = ["# MetaShift Phase 0 feasibility\n"]
    lines.append("## Recomputed dataset stats\n")
    for k, v in regen.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Max feasible K per (class_set, n)\n")
    lines.append("Cell = largest K in {4,6,8,10,12} such that at least K+2 contexts are shared by every class with >= n images. '--' = not feasible at K=4.\n")
    header = "| class_set | " + " | ".join(f"n={n}" for n in N_GRID) + " |"
    sep = "|" + "---|" * (len(N_GRID) + 1)
    lines.append(header)
    lines.append(sep)
    for tag, _ in CLASS_SETS:
        cells = []
        for n in N_GRID:
            row = next((r for r in per_set_n if r.get("class_set") == tag and r.get("n_per_cell") == n), None)
            if row is None or "missing_classes" in row:
                cells.append("?")
            else:
                feas = row["feasible_K"]
                cells.append(str(max(feas)) if feas else "--")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines.append("\n## Per-class subset counts (top 6 classes by total subsets)\n")
    ranked = sorted(per_class.items(), key=lambda kv: kv[1]["n_subsets_total"], reverse=True)[:6]
    for cls, d in ranked:
        top5 = ", ".join(f"{x['context']}({x['n_images']})" for x in d["top20"][:5])
        lines.append(f"- **{cls}**: total={d['n_subsets_total']}, ≥25={d['n_subsets_min25']}, ≥80={d['n_subsets_min80']}; top: {top5}")
    (OUT / "feasibility_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(regen, indent=2))
    print("\nMax feasible K matrix:\n")
    print("\n".join(lines[lines.index(header):lines.index(header) + len(CLASS_SETS) + 2]))


if __name__ == "__main__":
    main()
