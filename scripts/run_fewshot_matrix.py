"""Resume the declared few-shot matrix without changing support IDs.

Each invocation is an independent fixed-configuration run. Failed jobs retain
their log and stop the controller so missing combinations cannot be hidden.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from domainbed.fewshot.protocols import folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["PACS", "OfficeHome", "TerraIncognita", "DomainNet", "VLCS"])
    parser.add_argument("--backbones", nargs="+", default=["dinov3_vits16", "clip_vit_b16"])
    parser.add_argument("--methods", nargs="+", default=["ncm"])
    parser.add_argument("--settings", nargs="+", default=["multi_source", "single_source"])
    parser.add_argument("--scopes", nargs="+", default=["total", "per_domain"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--proker-rff-on-large", action="store_true",
                        help="Use distinct RFF method key when exact solver estimated working set exceeds 48 GiB")
    parser.add_argument("--rff-device", default="cpu")
    parser.add_argument("--rff-components", type=int, default=4096)
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--ecalp-device", default="cpu")
    args = parser.parse_args()
    log_dir = ROOT / "results/fewshot_ood/matrix_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / "results/fewshot_ood/data_audit.json").read_text())
    for dataset, backbone, method, setting, scope, shots, seed in product(
            args.datasets, args.backbones, args.methods, args.settings,
            args.scopes, args.shots, args.seeds):
        if args.proker_rff_on_large and method == "proker_imagenet_fixed":
            view_path = ROOT / "results/fewshot_ood/support_views" / dataset / backbone / "complete.json"
            views = json.loads(view_path.read_text())["n_views"]
            source_count = 1 if setting == "single_source" else len(audit[dataset]["domains"]) - 1
            budget = shots * (source_count if scope == "per_domain" else 1)
            n_keys = len(audit[dataset]["canonical_classes"]) * budget * views
            if n_keys ** 2 * 8 * 3 > 48 * 1024**3:
                method = "proker_rff_imagenet_fixed"
                print(f"Using ProKeR RFF for {dataset} K={shots}: exact dense solve exceeds 48 GiB estimate", flush=True)
        result_dir = (ROOT / "results/fewshot_ood/runs" / dataset / backbone /
                      setting / scope / str(shots) / str(seed) / method)
        expected = {fold.identifier for fold in folds(sorted(audit[dataset]["domains"]), setting)}
        infeasible_dir = (ROOT / "results/fewshot_ood/infeasible" / dataset /
                          setting / scope / str(shots) / str(seed))

        def coverage():
            completed = {path.stem for path in result_dir.glob("*.json")}
            excluded = {path.stem for path in infeasible_dir.glob("*.json")}
            if completed & excluded or (completed | excluded) - expected:
                raise RuntimeError(f"Contradictory fold coverage: {result_dir}")
            return completed | excluded

        if coverage() == expected:
            continue
        command = [sys.executable, "-m", "domainbed.scripts.run_fewshot_ood",
                   "--dataset", dataset, "--backbone", backbone, "--method", method,
                   "--setting", setting, "--shot_scope", scope, "--shots", str(shots),
                   "--seed", str(seed)]
        if method == "proker_rff_imagenet_fixed":
            command += ["--rff-device", args.rff_device,
                        "--rff-components", str(args.rff_components)]
        if method in {"taskres_frozen", "clip_adapter_frozen", "tip_adapter_f_imagenet_fixed", "ape_t_imagenet_fixed", "clap_frozen", "lp_pp_frozen", "cross_modal_probe_frozen", "cross_modal_lp_clap"}:
            command += ["--training-device", args.training_device]
        if method == "ecalp_inductive":
            command += ["--ecalp-device", args.ecalp_device]
        log_path = log_dir / f"{dataset}_{backbone}_{method}_{setting}_{scope}_K{shots}_S{seed}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Run failed (exit {result.returncode}): {log_path}")
        if coverage() != expected:
            raise RuntimeError(f"Missing folds after run: {result_dir}; "
                               f"missing={sorted(expected - coverage())}")
        print(dataset, backbone, method, setting, scope, shots, seed, flush=True)


if __name__ == "__main__":
    main()
