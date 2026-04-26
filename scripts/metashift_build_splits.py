"""Phase 2 split builder for MetaShift K-domain protocol.

Reads:
    data/metashift/meta_data/full-candidate-subsets.pkl
    data/metashift/raw/images/<vg_id>.jpg                (Phase 1 output)

Writes:
    data/metashift/splits/<class_set>/K<K>_protoB_seed<seed>.csv
    data/metashift/splits/<class_set>/K<K>_protoB_seed<seed>_summary.json

CSV schema:
    image_path, class_idx, class_name, context, domain_idx, domain_name, split, vg_image_id

split values: in (training), out (validation, in-domain holdout 20%),
              test_near (held-out near context), test_far (held-out far context).

Domain assignment rule:
- For each image_id whose subset memberships intersect the chosen contexts,
  pick exactly one (class, context) bucket via stable hash. Guarantees no
  image lands in two domains.

Distance for near/far selection:
- L2 in 5-D nx.spectral_layout of the overlap-coefficient graph restricted
  to subsets in the chosen class set, sparsified at 0.2 (matches repo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "data" / "metashift" / "meta_data" / "full-candidate-subsets.pkl"
IMG_DIR = ROOT / "data" / "metashift" / "raw" / "images"
SPLITS = ROOT / "data" / "metashift" / "splits"

# Mirror of feasibility's class_sets.
CLASS_SETS = {
    "cat_dog": ("cat", "dog"),
    "cat_dog_horse": ("cat", "dog", "horse"),
    "cat_dog_horse_elephant": ("cat", "dog", "horse", "elephant"),
    "cat_dog_bird": ("cat", "dog", "bird"),
    "cat_dog_horse_bird": ("cat", "dog", "horse", "bird"),
    "cat_dog_horse_elephant_bird": ("cat", "dog", "horse", "elephant", "bird"),
    "furniture4": ("chair", "table", "bed", "couch"),
    "vehicles4": ("bus", "truck", "car", "motorcycle"),
}

PROTOCOL_B_N = {2: 480, 4: 240, 6: 160, 8: 120, 10: 96, 12: 80}
PROTOCOL_A_N = {2: 80, 4: 80, 6: 80, 8: 80, 10: 80, 12: 80}


def parse_subset(name: str) -> tuple[str, str] | None:
    if "(" not in name or not name.endswith(")"):
        return None
    cls, rest = name.split("(", 1)
    return cls, rest[:-1]


def stable_hash(image_id: int, key: tuple[str, str], seed: int) -> int:
    h = hashlib.sha256(f"{image_id}|{key[0]}|{key[1]}|{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def load_subsets() -> dict[str, set[int]]:
    with PKL.open("rb") as f:
        return pickle.load(f)


def shared_contexts_for(subsets: dict[str, set[int]], classes: tuple[str, ...], n_min: int) -> list[str]:
    """Contexts present in every class with at least n_min images, ranked by min count desc.
    Excludes contexts that are themselves in the target class set (avoids self-overlap)."""
    by_class: dict[str, dict[str, int]] = {c: {} for c in classes}
    for name, ids in subsets.items():
        parsed = parse_subset(name)
        if parsed is None or parsed[0] not in by_class:
            continue
        cls, ctx = parsed
        by_class[cls][ctx] = len(ids)
    contexts = set.intersection(*[
        {c for c, k in by_class[cls].items() if k >= n_min} for cls in classes
    ])
    contexts = {c for c in contexts if c not in classes}
    return sorted(contexts, key=lambda c: -min(by_class[cls][c] for cls in classes))


def build_overlap_graph(subsets: dict[str, set[int]], classes: tuple[str, ...]) -> nx.Graph:
    """Overlap-coefficient graph over (class, context) subsets within target classes.
    Mirrors generate_full_MetaShift.py:135 (edge weight = |A intersect B| / min(|A|,|B|))
    with sparsification at 0.2."""
    nodes = {}
    for name, ids in subsets.items():
        parsed = parse_subset(name)
        if parsed and parsed[0] in classes:
            nodes[name] = ids
    G = nx.Graph()
    G.add_nodes_from(nodes)
    keys = list(nodes.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = nodes[a], nodes[b]
            denom = min(len(sa), len(sb))
            if denom == 0:
                continue
            w = len(sa & sb) / denom
            if w >= 0.2:
                G.add_edge(a, b, weight=w)
    return G


def context_centroids(G: nx.Graph, classes: tuple[str, ...], contexts: list[str]) -> dict[str, np.ndarray]:
    """Spectral centroid of each context, averaging over all (class, context) nodes for classes in S."""
    pos = nx.spectral_layout(G, dim=5, weight="weight")
    out: dict[str, np.ndarray] = {}
    for ctx in contexts:
        coords = []
        for cls in classes:
            key = f"{cls}({ctx})"
            if key in pos:
                coords.append(np.asarray(pos[key]))
        if coords:
            out[ctx] = np.mean(coords, axis=0)
    return out


def pick_train_test_contexts_single(subsets: dict[str, set[int]],
                                     classes: tuple[str, ...],
                                     contexts: list[str],
                                     centroids: dict[str, np.ndarray],
                                     K: int, n: int) -> tuple[list[str], str, dict]:
    """Pick K training contexts + 1 test context. Test ctx is the candidate (outside the
    top-K-by-min-count) with largest spectral distance to the train centroid AND that
    leaves every (cls, train_ctx) cell with >= n imgs after exclusion.

    Falls back to closer test contexts if the farthest one depletes training cells.
    Returns (train_ctxs, test_ctx, raw_counts).
    """
    member = raw_membership(subsets, classes, contexts)
    valid = [
        ctx for ctx in contexts
        if min(len(member.get((cls, ctx), set())) for cls in classes) >= n
    ]
    if len(valid) < K + 1:
        raise RuntimeError(
            f"infeasible at n={n}: only {len(valid)} contexts have >= {n} raw imgs for every class; need >= {K+1}"
        )
    valid.sort(key=lambda c: -min(len(member[(cls, c)]) for cls in classes))
    train = valid[:K]
    test_pool = valid[K:]
    train_centroid = np.mean([centroids[c] for c in train if c in centroids], axis=0)
    scored = sorted(
        ((c, float(np.linalg.norm(centroids[c] - train_centroid))) for c in test_pool if c in centroids),
        key=lambda x: -x[1],  # descending: farthest first
    )
    if not scored:
        raise RuntimeError("no test candidate has spectral coords")

    def cells_ok(excluded: set[int]) -> bool:
        for cls in classes:
            for ctx in train:
                if len(member.get((cls, ctx), set()) - excluded) < n:
                    return False
        return True

    for test_ctx, _ in scored:
        excluded = set()
        for cls in classes:
            excluded |= member.get((cls, test_ctx), set())
        if cells_ok(excluded):
            counts = {f"{cls}({ctx})": len(member.get((cls, ctx), set())) for ctx in train + [test_ctx] for cls in classes}
            return train, test_ctx, counts
    raise RuntimeError(f"no test ctx works at K={K}, n={n}; pool size={len(scored)}")


def pick_train_test_contexts(subsets: dict[str, set[int]],
                              classes: tuple[str, ...],
                              contexts: list[str],
                              centroids: dict[str, np.ndarray],
                              K: int, n: int) -> tuple[list[str], str, str, dict]:
    """Pick K training contexts + 1 near + 1 far where every (cls, train_ctx) cell still
    has >= n images AFTER removing the union of test-context image IDs.

    Strategy:
      1) Sort candidates by raw min-count per class (descending).
      2) Greedy: take top-K as training; among the remaining contexts, find the
         ones whose removal still leaves every (cls, train_ctx) cell with >= n
         imgs after exclusion. Among those, pick test_near as the one with
         smallest spectral distance to the train centroid and test_far as the
         one with largest. If the greedy K choice makes no test pair feasible,
         shift: drop the K-th training ctx and retry with a smaller training pool.

    Returns (train_ctxs, near_ctx, far_ctx, raw_counts).
    """
    member = raw_membership(subsets, classes, contexts)
    valid = [
        ctx for ctx in contexts
        if min(len(member.get((cls, ctx), set())) for cls in classes) >= n
    ]
    if len(valid) < K + 2:
        raise RuntimeError(
            f"infeasible at n={n}: only {len(valid)} contexts have >= {n} raw imgs for every class; need >= {K+2}"
        )
    valid.sort(key=lambda c: -min(len(member[(cls, c)]) for cls in classes))

    def cells_ok(train_ctxs: list[str], excluded_ids: set[int]) -> bool:
        for cls in classes:
            for ctx in train_ctxs:
                if len(member.get((cls, ctx), set()) - excluded_ids) < n:
                    return False
        return True

    # Try shrinking the training-pool tier until we find a feasible (train, near, far) triple.
    for slack in range(0, len(valid) - K - 1):
        train = valid[:K]
        candidates = valid[K:] if slack == 0 else valid[K - slack:K] + valid[K:]
        # Actually keep training fixed at top-K and search test pair from the leftover pool.
        if slack > 0:
            # Allow swapping last 'slack' training ctxs out to the test pool.
            train = valid[:K - slack] + valid[K:K + slack]  # keep K total but with reshuffle
            test_pool = valid[K - slack:K] + valid[K + slack:]
        else:
            test_pool = valid[K:]
        # Ensure no duplicates and keep ordering.
        seen = set(); train = [t for t in train if not (t in seen or seen.add(t))][:K]
        test_pool = [c for c in test_pool if c not in train]
        if len(test_pool) < 2:
            continue
        # Score every (near, far) ordered pair by feasibility + spectral preferences.
        train_centroid = np.mean([centroids[c] for c in train if c in centroids], axis=0)
        scored: list[tuple[str, float]] = []
        for c in test_pool:
            if c in centroids:
                scored.append((c, float(np.linalg.norm(centroids[c] - train_centroid))))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[1])
        # Try near = closest, far = farthest. If infeasible, walk inward on each side.
        feas = None
        n_sc = len(scored)
        for ni in range(n_sc):
            for fi in range(n_sc - 1, ni, -1):
                near, far = scored[ni][0], scored[fi][0]
                excluded = (member.get((classes[0], near), set()) | member.get((classes[0], far), set()))
                for cls in classes[1:]:
                    excluded |= member.get((cls, near), set()) | member.get((cls, far), set())
                if cells_ok(train, excluded):
                    feas = (near, far, scored[ni][1], scored[fi][1])
                    break
            if feas is not None:
                break
        if feas is not None:
            near, far, d_near, d_far = feas
            counts = {f"{cls}({ctx})": len(member.get((cls, ctx), set())) for ctx in train + [near, far] for cls in classes}
            return train, near, far, counts
    raise RuntimeError(
        f"could not find K={K}, n={n} configuration with test cells that don't deplete training; "
        f"valid pool size = {len(valid)}"
    )


def assign_images(subsets: dict[str, set[int]], classes: tuple[str, ...],
                  ctxs: list[str], seed: int) -> dict[tuple[str, str], list[int]]:
    """Strict single-bucket assignment (legacy; kept for the dedupe-aware feasibility scan).

    Returns {(cls, ctx): [image_ids]} where every eligible image is in exactly
    one (cls, ctx) bucket, picked by stable hash. This is the most conservative
    rule. The split builder uses raw_membership() instead.
    """
    eligible: dict[int, list[tuple[str, str]]] = defaultdict(list)
    target_set = {(cls, ctx) for cls in classes for ctx in ctxs}
    for name, ids in subsets.items():
        parsed = parse_subset(name)
        if parsed is None:
            continue
        if parsed not in target_set:
            continue
        for img in ids:
            eligible[img].append(parsed)
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for img, candidates in eligible.items():
        chosen = min(candidates, key=lambda p: stable_hash(img, p, seed))
        bucket[chosen].append(img)
    for k in bucket:
        bucket[k].sort()
    return bucket


def raw_membership(subsets: dict[str, set[int]], classes: tuple[str, ...],
                   ctxs: list[str]) -> dict[tuple[str, str], set[int]]:
    """{(cls, ctx): set of image_ids} for cls in classes, ctx in ctxs. Sets overlap freely.

    This is the actual subset membership from the pickle, with no dedupe. Used
    by build_split, which then enforces train/test disjointness as a hard
    constraint while allowing overlap among training domains.
    """
    target_set = {(cls, ctx) for cls in classes for ctx in ctxs}
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for name, ids in subsets.items():
        parsed = parse_subset(name)
        if parsed is None or parsed not in target_set:
            continue
        out[parsed] = set(ids)
    return out


def build_split(class_set: str, K: int, seed: int, protocol: str = "B",
                n_per_cell: int | None = None, single_test: bool = False,
                total_per_class: int | None = None) -> dict:
    if class_set not in CLASS_SETS:
        raise ValueError(f"unknown class_set {class_set}")
    classes = CLASS_SETS[class_set]
    if n_per_cell is not None:
        n = n_per_cell
    elif total_per_class is not None:
        n = total_per_class // K
    elif protocol == "B":
        n = PROTOCOL_B_N[K]
    elif protocol == "A":
        n = PROTOCOL_A_N[K]
    else:
        raise ValueError(f"unknown protocol {protocol}")
    rng = random.Random(seed)

    subsets = load_subsets()
    contexts = shared_contexts_for(subsets, classes, n)
    if len(contexts) < K + 2:
        raise RuntimeError(f"infeasible: need {K+2} shared contexts at n={n}, have {len(contexts)} for {class_set}")

    G = build_overlap_graph(subsets, classes)
    centroids = context_centroids(G, classes, contexts)
    if single_test:
        train_ctx, test_ctx, raw_counts = pick_train_test_contexts_single(
            subsets, classes, contexts, centroids, K, n)
        near_ctx = test_ctx
        far_ctx = test_ctx  # placeholder; only test_near rows are emitted in single-test mode
    else:
        train_ctx, near_ctx, far_ctx, raw_counts = pick_train_test_contexts(
            subsets, classes, contexts, centroids, K, n)
    used_ctx = train_ctx + ([near_ctx] if single_test else [near_ctx, far_ctx])
    member = raw_membership(subsets, classes, used_ctx)

    # Test sets first: union of test-context image IDs (across all classes) is excluded
    # from training.
    test_assignment: dict[int, str] = {}
    if single_test:
        for cls in classes:
            for img in member.get((cls, near_ctx), set()):
                test_assignment[img] = "test"
    else:
        for cls in classes:
            near_ids = member.get((cls, near_ctx), set())
            far_ids = member.get((cls, far_ctx), set())
            for img in near_ids - far_ids:
                test_assignment[img] = "test_near"
            for img in far_ids - near_ids:
                test_assignment[img] = "test_far"
            for img in near_ids & far_ids:
                test_assignment[img] = (
                    "test_near" if stable_hash(img, ("test", "near"), seed) < stable_hash(img, ("test", "far"), seed)
                    else "test_far"
                )
    test_ids = set(test_assignment)

    rows = []
    cell_summary = {}
    overlap_counter = defaultdict(int)  # img_id -> #training cells it appears in
    for d_idx, ctx in enumerate(train_ctx):
        for cls_idx, cls in enumerate(classes):
            ids = sorted(member.get((cls, ctx), set()) - test_ids)
            rng.shuffle(ids)
            if len(ids) < n:
                raise RuntimeError(f"({cls}, {ctx}) has only {len(ids)} imgs after removing test ids (< n={n})")
            chosen = ids[:n]
            n_out = max(1, int(round(0.2 * n)))
            out_ids = set(chosen[:n_out])
            for img in chosen:
                overlap_counter[img] += 1
                rows.append({
                    "image_path": f"data/metashift/raw/images/{img}.jpg",
                    "class_idx": cls_idx,
                    "class_name": cls,
                    "context": ctx,
                    "domain_idx": d_idx,
                    "domain_name": ctx,
                    "split": "out" if img in out_ids else "in",
                    "vg_image_id": img,
                })
            cell_summary[f"{cls}({ctx})"] = {
                "in": n - n_out, "out": n_out,
                "raw_pool": len(member.get((cls, ctx), set())),
                "post_test_excl_pool": len(ids),
            }

    # Test rows.
    if single_test:
        for cls_idx, cls in enumerate(classes):
            for img in sorted(member.get((cls, near_ctx), set())):
                rows.append({
                    "image_path": f"data/metashift/raw/images/{img}.jpg",
                    "class_idx": cls_idx,
                    "class_name": cls,
                    "context": near_ctx,
                    "domain_idx": K,
                    "domain_name": "test",
                    "split": "test",
                    "vg_image_id": img,
                })
        for cls in classes:
            cell_summary[f"{cls}({near_ctx})/test"] = {"raw": len(member.get((cls, near_ctx), set()))}
    else:
        for cls_idx, cls in enumerate(classes):
            for img in sorted(member.get((cls, near_ctx), set()) | member.get((cls, far_ctx), set())):
                tag = test_assignment.get(img)
                if tag is None:
                    continue
                ctx = near_ctx if tag == "test_near" else far_ctx
                rows.append({
                    "image_path": f"data/metashift/raw/images/{img}.jpg",
                    "class_idx": cls_idx,
                    "class_name": cls,
                    "context": ctx,
                    "domain_idx": K if tag == "test_near" else K + 1,
                    "domain_name": tag,
                    "split": tag,
                    "vg_image_id": img,
                })
        for cls in classes:
            cell_summary[f"{cls}({near_ctx})/test_near"] = {"raw": len(member.get((cls, near_ctx), set()))}
            cell_summary[f"{cls}({far_ctx})/test_far"] = {"raw": len(member.get((cls, far_ctx), set()))}

    overlap_dist = defaultdict(int)
    for c in overlap_counter.values():
        overlap_dist[c] += 1

    df = pd.DataFrame(rows)
    out_dir = SPLITS / class_set
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"K{K}_N{n*K if single_test else n}_seed{seed}" if single_test else f"K{K}_proto{protocol}_seed{seed}"
    csv_path = out_dir / f"{suffix}.csv"
    df.to_csv(csv_path, index=False)

    train_centroid_np = np.mean([centroids[c] for c in train_ctx if c in centroids], axis=0)
    summary = {
        "class_set": class_set,
        "classes": list(classes),
        "K": K,
        "n_per_cell": n,
        "protocol": protocol,
        "single_test": single_test,
        "seed": seed,
        "train_contexts": train_ctx,
        "test_context": near_ctx if single_test else None,
        "test_near_context": None if single_test else near_ctx,
        "test_far_context": None if single_test else far_ctx,
        "spectral_distance_train_to_test": float(np.linalg.norm(centroids[near_ctx] - train_centroid_np)) if single_test else None,
        "spectral_distance_train_to_near": None if single_test else float(np.linalg.norm(centroids[near_ctx] - train_centroid_np)),
        "spectral_distance_train_to_far": None if single_test else float(np.linalg.norm(centroids[far_ctx] - train_centroid_np)),
        "cells": cell_summary,
        "n_rows": len(df),
        "n_train_rows": int((df.split == "in").sum()),
        "n_val_rows": int((df.split == "out").sum()),
        "n_test_rows": int((df.split == "test").sum()) if single_test else None,
        "n_test_near_rows": None if single_test else int((df.split == "test_near").sum()),
        "n_test_far_rows": None if single_test else int((df.split == "test_far").sum()),
        "n_unique_image_ids": int(df.vg_image_id.nunique()),
        "training_overlap_distribution": dict(sorted(overlap_dist.items())),
        "n_unique_training_imgs": int(df[df.split.isin(["in","out"])].vg_image_id.nunique()),
    }
    (out_dir / f"{suffix}_summary.json").write_text(json.dumps(summary, indent=2))

    # Sanity: no image in both train and any test split.
    train_ids = set(df[df.split.isin(["in", "out"])].vg_image_id)
    test_ids = set(df[df.split.isin(["test_near", "test_far"])].vg_image_id)
    overlap = train_ids & test_ids
    assert not overlap, f"LEAK: {len(overlap)} image_ids in both train and test"

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-set", required=True, choices=sorted(CLASS_SETS))
    p.add_argument("--K", type=int, required=True, choices=sorted(PROTOCOL_B_N))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--protocol", choices=["A", "B"], default="B")
    p.add_argument("--n-per-cell", type=int, default=None,
                   help="override per-cell sample count; ignores --protocol if set")
    p.add_argument("--total-per-class", type=int, default=None,
                   help="set n = total_per_class // K (Protocol B-style with custom budget)")
    p.add_argument("--single-test", action="store_true",
                   help="emit one held-out test domain instead of test_near + test_far")
    args = p.parse_args()

    summary = build_split(args.class_set, args.K, args.seed, args.protocol,
                          n_per_cell=args.n_per_cell,
                          total_per_class=args.total_per_class,
                          single_test=args.single_test)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
