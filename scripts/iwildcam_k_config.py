#!/usr/bin/env python3
"""DEPRECATED (2026-05-01): superseded by the WILDS-official-split refactor.
WILDSIWildCam now exposes 5 envs by metadata.csv split, so the location IDs
emitted by this helper no longer reference valid env indices. Use
scripts/run_iwildcam_paper.sh instead.

iwildcam_k_config.py — emit CLI fragments for the iWildCam K-sweep experiment.

Reads `domainbed/data/iwildcam_v2.0/k_sweep_setup.json` (produced by
`scripts/iwildcam_pick_domains.py`) so the source pool and test domains are
the result of a documented selection process, not hand-picked constants.

Usage:
    python scripts/iwildcam_k_config.py --K 8 --emit source_envs
    python scripts/iwildcam_k_config.py --K 8 --emit test_envs
    python scripts/iwildcam_k_config.py --K 8 --emit max_samples

Used by run_iwildcam_k_sweep.sh to compose the train.py CLI.
"""

import argparse
import json
import os

_SETUP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "domainbed", "data", "iwildcam_v2.0", "k_sweep_setup.json",
)
_SETUP = None


def _load():
    global _SETUP
    if _SETUP is None:
        if not os.path.exists(_SETUP_PATH):
            raise FileNotFoundError(
                f"Missing {_SETUP_PATH}. Run "
                "`python scripts/iwildcam_pick_domains.py` first."
            )
        with open(_SETUP_PATH) as f:
            _SETUP = json.load(f)
    return _SETUP


def get_source_envs(K):
    """Return the K source location indices (top-K from pool)."""
    s = _load()
    if K not in s["k_values"]:
        raise ValueError(f"K={K} not in supported k_values={s['k_values']}")
    return s["source_pool"][:K]


def get_test_envs(K):
    """Return test_envs flag value: test locations only (3 of them)."""
    return list(_load()["test_locs"])


def get_max_samples(K):
    """Return max_samples_per_env = total_budget // K."""
    return _load()["total_budget"] // K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--emit", choices=["source_envs", "test_envs", "max_samples", "all"],
                   default="all")
    args = p.parse_args()

    if args.emit == "source_envs":
        print(" ".join(str(x) for x in get_source_envs(args.K)))
    elif args.emit == "test_envs":
        print(" ".join(str(x) for x in get_test_envs(args.K)))
    elif args.emit == "max_samples":
        print(get_max_samples(args.K))
    else:
        print(f"K={args.K}")
        print(f"  test_envs        = {get_test_envs(args.K)}")
        print(f"  source_envs (K={args.K}) = {get_source_envs(args.K)}")
        print(f"  max_samples_per_env = {get_max_samples(args.K)}")


if __name__ == "__main__":
    main()
