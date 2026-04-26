#!/usr/bin/env python3
"""iwildcam_pick_domains.py — pick a balanced source/test domain split for the
iWildCam K-sweep, then emit machine-readable JSON + a human-readable Markdown
justification.

Selection criteria (see plan):
  - Filter candidate locations: #classes >= MIN_CLASSES and #samples >= MIN_SAMPLES
    (drops degenerate locations like loc 187/265 with only 2 species).
  - Pick NUM_TEST test domains from candidates whose sample count is in the
    medium band [2500, 3500], chosen so the source candidates cover ALL test
    classes (hard constraint).
  - Pick NUM_SOURCE source domains greedily: first saturate test-class
    coverage, then minimize coefficient-of-variation of #samples and #classes
    across the pool.
  - Order the source pool by descending #classes so the nested K progression
    (K=4,8,...) consistently picks the richer domains first.

Run once after the iWildCam metadata is downloaded; downstream scripts read
the JSON.
"""

import argparse
import json
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def per_domain_stats(df, loc_col="location_remapped", label_col="y"):
    """Return DataFrame indexed by location with #imgs, #classes, top-1 stats."""
    rows = []
    for loc, sub in df.groupby(loc_col):
        cls_counts = sub[label_col].value_counts()
        rows.append({
            "loc": int(loc),
            "n_imgs": int(len(sub)),
            "n_classes": int(cls_counts.size),
            "top1_class": int(cls_counts.index[0]),
            "top1_pct": float(cls_counts.iloc[0]) / len(sub) * 100.0,
            "classes": frozenset(int(c) for c in cls_counts.index),
        })
    return pd.DataFrame(rows).set_index("loc").sort_index()


def cv(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0 or a.mean() == 0:
        return 0.0
    return float(a.std() / a.mean())


def pick_test_triple(stats, test_band, num_test, candidate_locs):
    """Pick a test triple from `test_band` such that source candidates
    (candidate_locs - triple) cover all test classes.

    Among valid triples, prefer the one that minimizes CV(#samples) of the
    triple (balanced-size test envs)."""
    band_locs = [l for l in test_band if l in candidate_locs]
    best = None  # (cv_samples, triple)
    for triple in combinations(band_locs, num_test):
        src_pool = [l for l in candidate_locs if l not in triple]
        src_classes = set().union(*(stats.loc[l, "classes"] for l in src_pool))
        test_classes = set().union(*(stats.loc[l, "classes"] for l in triple))
        if not test_classes.issubset(src_classes):
            continue
        triple_sizes = [stats.loc[l, "n_imgs"] for l in triple]
        c = cv(triple_sizes)
        if best is None or c < best[0]:
            best = (c, triple)
    if best is None:
        raise RuntimeError(
            "No feasible test triple satisfies test_classes ⊆ source_classes. "
            "Loosen --min-classes / --min-samples or widen the test band."
        )
    return list(best[1])


def pick_source_pool(stats, candidate_locs, test_classes, num_source):
    """Greedy pick of NUM_SOURCE source locations.

    Phase 1: until all test classes are covered by union of picked, pick the
        location adding the most uncovered test classes (tie-break: sample
        count closest to candidate-pool median).
    Phase 2: fill remaining slots picking the location whose #samples is
        closest to the running median (minimizes spread).
    """
    pool = list(candidate_locs)
    picked = []
    covered = set()
    median_size = float(np.median([stats.loc[l, "n_imgs"] for l in pool]))

    # Phase 1: saturate test-class coverage
    while covered != set(test_classes) and len(picked) < num_source:
        def gain(l):
            return len((stats.loc[l, "classes"] & test_classes) - covered)
        scored = [(gain(l), -abs(stats.loc[l, "n_imgs"] - median_size), l)
                  for l in pool if l not in picked]
        scored.sort(reverse=True)
        if scored[0][0] == 0:
            break  # nothing more we can cover; fall through to phase 2
        chosen = scored[0][2]
        picked.append(chosen)
        covered |= stats.loc[chosen, "classes"]

    # Phase 2: minimize CV(#samples)
    while len(picked) < num_source:
        cur_sizes = [stats.loc[l, "n_imgs"] for l in picked] or [median_size]
        target = float(np.median(cur_sizes))
        rest = [l for l in pool if l not in picked]
        rest.sort(key=lambda l: abs(stats.loc[l, "n_imgs"] - target))
        picked.append(rest[0])

    if not set(test_classes).issubset(covered):
        missing = set(test_classes) - covered
        raise RuntimeError(
            f"Source pool fails to cover {len(missing)} test classes: "
            f"{sorted(missing)}. Increase --num-source or relax filters."
        )
    return picked


def order_for_nested_k(stats, source_pool):
    """Order so top-K (K=4,8,...) consistently picks richer domains first.
    Sort by (#classes desc, #imgs desc)."""
    return sorted(
        source_pool,
        key=lambda l: (-stats.loc[l, "n_classes"], -stats.loc[l, "n_imgs"]),
    )


def write_json(path, source_pool, test_locs, total_budget, k_values, stats):
    def loc_row(l):
        return {
            "loc": int(l),
            "n_imgs": int(stats.loc[l, "n_imgs"]),
            "n_classes": int(stats.loc[l, "n_classes"]),
            "top1_class": int(stats.loc[l, "top1_class"]),
            "top1_pct": round(float(stats.loc[l, "top1_pct"]), 2),
        }
    src_rows = [loc_row(l) for l in source_pool]
    test_rows = [loc_row(l) for l in test_locs]
    src_classes = set().union(*(stats.loc[l, "classes"] for l in source_pool))
    test_classes = set().union(*(stats.loc[l, "classes"] for l in test_locs))
    payload = {
        "source_pool": [int(l) for l in source_pool],
        "test_locs": [int(l) for l in test_locs],
        "total_budget": int(total_budget),
        "k_values": list(k_values),
        "stats": {
            "source": src_rows,
            "test": test_rows,
            "n_source_classes": len(src_classes),
            "n_test_classes": len(test_classes),
            "test_classes_not_in_source": sorted(int(c) for c in (test_classes - src_classes)),
            "cv_source_n_imgs": round(cv([r["n_imgs"] for r in src_rows]), 4),
            "cv_source_n_classes": round(cv([r["n_classes"] for r in src_rows]), 4),
            "mean_source_top1_pct": round(float(np.mean([r["top1_pct"] for r in src_rows])), 2),
            "cv_test_n_imgs": round(cv([r["n_imgs"] for r in test_rows]), 4),
            "mean_test_top1_pct": round(float(np.mean([r["top1_pct"] for r in test_rows])), 2),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_md(path, payload, args):
    s = payload["stats"]
    lines = []
    lines.append("# iWildCam K-sweep — domain selection report\n")
    lines.append("Generated by `scripts/iwildcam_pick_domains.py`. Re-run after "
                 "metadata changes; do NOT hand-edit.\n")

    lines.append("## Summary\n")
    lines.append(f"- **Source pool**: {len(payload['source_pool'])} domains "
                 f"(top-K nested for K ∈ {payload['k_values']})")
    lines.append(f"- **Test domains**: {len(payload['test_locs'])} (fixed across all K)")
    lines.append(f"- **Total training budget**: {payload['total_budget']} samples "
                 f"(`max_samples_per_env = budget / K`, stratified by class)")
    lines.append(f"- **Source classes (union)**: {s['n_source_classes']} / 182")
    lines.append(f"- **Test classes (union)**: {s['n_test_classes']}")
    lines.append(f"- **Test classes missing from source pool**: "
                 f"{len(s['test_classes_not_in_source'])} "
                 f"{'✅' if not s['test_classes_not_in_source'] else '❌ ' + str(s['test_classes_not_in_source'])}")
    lines.append("")

    lines.append("## Imbalance metrics (lower CV = more balanced)\n")
    lines.append("| metric | source pool | test |")
    lines.append("|---|---:|---:|")
    lines.append(f"| CV of #samples | {s['cv_source_n_imgs']:.3f} | {s['cv_test_n_imgs']:.3f} |")
    lines.append(f"| CV of #classes | {s['cv_source_n_classes']:.3f} | — |")
    lines.append(f"| Mean top-1 class % | {s['mean_source_top1_pct']:.1f}% | {s['mean_test_top1_pct']:.1f}% |")
    lines.append("")

    def render_table(rows, header="Source pool"):
        lines.append(f"## {header}\n")
        lines.append("| rank | loc | #imgs | #classes | top-1 class | top-1 % |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(rows, 1):
            lines.append(f"| {i} | {r['loc']} | {r['n_imgs']} | {r['n_classes']} | "
                         f"{r['top1_class']} | {r['top1_pct']:.1f}% |")
        lines.append("")

    render_table(s["source"], "Source pool (nested order: K=4 uses ranks 1-4, K=8 uses 1-8, ...)")
    render_table(s["test"], "Test domains")

    lines.append("## Selection criteria\n")
    lines.append(f"1. Filter: `#classes ≥ {args.min_classes}` AND "
                 f"`#samples ≥ {args.min_samples}` (drops single-/few-class locations).")
    lines.append(f"2. Test triple: from medium-band `#samples ∈ "
                 f"[{args.test_band_min}, {args.test_band_max}]`, brute-forced over all triples, "
                 f"kept only those whose classes are fully covered by remaining candidates; "
                 f"tie-broken by smallest CV(#samples) within the triple.")
    lines.append("3. Source pool (greedy, 16 picks):")
    lines.append("   - Phase 1 — pick locations that add the most uncovered test classes.")
    lines.append("   - Phase 2 — fill remaining slots by closest-to-median sample count.")
    lines.append("4. Hard assertion: `test_classes ⊆ source_classes`.")
    lines.append("5. Source order (for nested K): sort by `(#classes desc, #imgs desc)` so K=4 already picks the richest domains.")
    lines.append("")

    lines.append("## Why this matters for MESSI's sub-invariant\n")
    lines.append("With unreachable test classes or single-class source domains, K-sweep gains/losses "
                 "conflate label-support gaps with genuine domain-shift effects. This setup pins down "
                 "label support (test ⊆ source, 100% coverage) and per-domain richness (≥10 classes), "
                 "so any K-dependent change in test accuracy can be attributed to the algorithm's "
                 "behavior under increasing source-domain heterogeneity rather than an artifact of the data split.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="domainbed/data/iwildcam_v2.0/metadata.csv")
    p.add_argument("--out-json", default="domainbed/data/iwildcam_v2.0/k_sweep_setup.json")
    p.add_argument("--out-md", default="domainbed/data/iwildcam_v2.0/k_sweep_setup.md")
    p.add_argument("--min-classes", type=int, default=10)
    p.add_argument("--min-samples", type=int, default=2000)
    p.add_argument("--num-source", type=int, default=16)
    p.add_argument("--num-test", type=int, default=3)
    p.add_argument("--total-budget", type=int, default=16000)
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 10, 12, 14, 16])
    p.add_argument("--test-band-min", type=int, default=2500)
    p.add_argument("--test-band-max", type=int, default=3500)
    args = p.parse_args()

    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata not found: {args.metadata}")

    df = pd.read_csv(args.metadata)
    stats = per_domain_stats(df)

    candidates = stats[(stats["n_classes"] >= args.min_classes)
                       & (stats["n_imgs"] >= args.min_samples)].index.tolist()
    print(f"[picker] {len(candidates)} candidate locations "
          f"(#classes >= {args.min_classes}, #samples >= {args.min_samples})")

    test_band = stats[(stats["n_imgs"] >= args.test_band_min)
                      & (stats["n_imgs"] <= args.test_band_max)
                      & (stats["n_classes"] >= args.min_classes)].index.tolist()
    print(f"[picker] test band has {len(test_band)} candidates "
          f"(samples in [{args.test_band_min}, {args.test_band_max}])")

    test_locs = pick_test_triple(stats, test_band, args.num_test, candidates)
    print(f"[picker] test_locs = {test_locs}")

    src_candidates = [l for l in candidates if l not in test_locs]
    test_classes = set().union(*(stats.loc[l, "classes"] for l in test_locs))
    print(f"[picker] test classes union = {len(test_classes)}")

    source_pool = pick_source_pool(stats, src_candidates, test_classes, args.num_source)
    source_pool = order_for_nested_k(stats, source_pool)
    print(f"[picker] source_pool = {source_pool}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    payload = write_json(args.out_json, source_pool, test_locs,
                         args.total_budget, args.k_values, stats)
    write_md(args.out_md, payload, args)

    print(f"[picker] wrote {args.out_json}")
    print(f"[picker] wrote {args.out_md}")
    print(f"[picker] CV(samples) src={payload['stats']['cv_source_n_imgs']:.3f} "
          f"test={payload['stats']['cv_test_n_imgs']:.3f} "
          f"| mean top-1 src={payload['stats']['mean_source_top1_pct']:.1f}% "
          f"test={payload['stats']['mean_test_top1_pct']:.1f}%")
    print(f"[picker] test classes not in source pool: "
          f"{payload['stats']['test_classes_not_in_source'] or 'NONE ✅'}")


if __name__ == "__main__":
    main()
