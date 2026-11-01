"""Run strict frozen-feature inductive OOD classification folds.

Source support is sampled once per experiment key and saved independently of
method. Queries are streamed in chunks; target labels are used only after
predict_scores returns.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time

import numpy as np

from domainbed.fewshot.methods import (APETrainingFree, APET, CALIPFixed, CLIPZeroShot, CLIPAdapter, CLAP, CrossModalLinearProbe, CrossModalCLAP, ECALPInductive, GDA, GDAClip, KNN, NCM, LogisticProbe, LPPlusPlus, ProKeR, ProKeROrthogonalRFF, TaskRes, TipAdapterF,
                                       Ridge, SimpleShotSupport, TIMOFixed, TipAdapter, TipAdapterFull, VisualCache)
from domainbed.fewshot.methods.proker import IMAGENET_HYPERPARAMETERS
from domainbed.fewshot.protocols import SourceRecord, build_manifest, candidate_pool, folds, save_manifest


ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = {"PACS": "PACS", "OfficeHome": "office_home",
             "TerraIncognita": "terra_incognita", "DomainNet": "domain_net", "VLCS": "VLCS"}
METHODS = {"ncm": NCM, "knn": KNN, "ridge": Ridge,
           "simpleshot_support": SimpleShotSupport, "visual_cache": VisualCache,
           "clip_zeroshot": CLIPZeroShot, "tip_adapter_single_view": TipAdapter,
           "tip_adapter_full": TipAdapterFull, "gda": GDA, "gda_clip": GDAClip,
           "ape_training_free": APETrainingFree, "ape_t_imagenet_fixed": APET,
           "calip_fixed_beta": CALIPFixed,
           "timo_fixed": TIMOFixed, "proker_imagenet_fixed": ProKeR,
           "proker_rff_imagenet_fixed": ProKeROrthogonalRFF,
           "taskres_frozen": TaskRes, "clip_adapter_frozen": CLIPAdapter,
           "tip_adapter_f_imagenet_fixed": TipAdapterF,
           "clap_frozen": CLAP, "logistic_probe_c0316": LogisticProbe,
           "lp_pp_frozen": LPPlusPlus,
           "cross_modal_probe_frozen": CrossModalLinearProbe,
           "cross_modal_lp_clap": CrossModalCLAP,
           "ecalp_inductive": ECALPInductive}
TEXT_METHODS = {"clip_zeroshot", "tip_adapter_single_view", "tip_adapter_full", "gda_clip", "ape_training_free", "ape_t_imagenet_fixed", "calip_fixed_beta", "timo_fixed", "proker_imagenet_fixed", "proker_rff_imagenet_fixed", "taskres_frozen", "clip_adapter_frozen", "tip_adapter_f_imagenet_fixed", "clap_frozen", "lp_pp_frozen", "cross_modal_probe_frozen", "cross_modal_lp_clap", "ecalp_inductive"}
VIEW_METHODS = {"tip_adapter_full", "gda", "gda_clip", "ape_training_free", "ape_t_imagenet_fixed", "timo_fixed", "proker_imagenet_fixed", "proker_rff_imagenet_fixed", "taskres_frozen", "clip_adapter_frozen", "tip_adapter_f_imagenet_fixed", "clap_frozen", "cross_modal_probe_frozen", "cross_modal_lp_clap", "lp_pp_frozen"}
ZERO_SUPPORT_METHODS = {"clip_zeroshot", "calip_fixed_beta"}
ADAPTED_FIXED_METHODS = {"visual_cache", "tip_adapter_single_view", "tip_adapter_full",
                         "gda_clip", "ape_training_free", "ape_t_imagenet_fixed", "calip_fixed_beta", "timo_fixed",
                         "proker_imagenet_fixed", "proker_rff_imagenet_fixed", "taskres_frozen",
                         "clip_adapter_frozen", "tip_adapter_f_imagenet_fixed", "clap_frozen",
                         "logistic_probe_c0316", "lp_pp_frozen", "cross_modal_probe_frozen",
                         "ecalp_inductive", "cross_modal_lp_clap"}


def write_immutable_json(path: Path, payload: dict) -> None:
    """Publish shared support metadata atomically across method processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        for _ in range(3):
            if path.read_text() == encoded:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Existing immutable metadata changed: {path}")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        stream.write(encoded)
        temp = Path(stream.name)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_text() != encoded:
        raise RuntimeError(f"Concurrent immutable metadata mismatch: {path}")


def eligible_mask(sample_ids, audit: dict, dataset: str) -> np.ndarray:
    excluded = {Path(item).relative_to(DATA_DIRS[dataset]).as_posix()
                for item in audit.get("leakage_exclusions", [])}
    return np.asarray([str(sample) not in excluded for sample in sample_ids], dtype=bool)


def source_candidates(dataset: str, fold, seed: int, ids, labels, domain_ids,
                      names, allowed, num_classes: int, audit_sha: str):
    path = (ROOT / "results/fewshot_ood/candidate_pools" / dataset /
            fold.setting / fold.identifier / f"seed{seed}.json")
    if path.exists():
        payload = json.loads(path.read_text())
        if (payload["dataset_manifest_sha256"] != audit_sha or
                payload["source_domains"] != list(fold.source_domains)):
            raise RuntimeError(f"Candidate pool provenance mismatch: {path}")
        return [SourceRecord(**item) for item in payload["records"]]
    source_domain_ids = [names.index(x) for x in fold.source_domains]
    indices = np.flatnonzero(allowed & np.isin(domain_ids, source_domain_ids))
    all_source = [SourceRecord(str(ids[i]), names[int(domain_ids[i])], int(labels[i]))
                  for i in indices]
    top = candidate_pool(all_source, fold.source_domains, num_classes, seed)
    payload = {"dataset_manifest_sha256": audit_sha, "source_domains": list(fold.source_domains),
               "records": [vars(item) for item in top], "selection": "sha256_top16_v1"}
    write_immutable_json(path, payload)
    return top


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes() -> dict[str, str]:
    files = ["domainbed/scripts/run_fewshot_ood.py", "domainbed/fewshot/protocols.py",
             "domainbed/fewshot/methods/primitives.py", "domainbed/fewshot/methods/gda.py",
             "domainbed/fewshot/methods/ape.py", "domainbed/fewshot/methods/ape_t.py",
             "domainbed/fewshot/methods/calip.py",
             "domainbed/fewshot/methods/timo.py", "domainbed/fewshot/methods/proker.py",
             "domainbed/fewshot/methods/taskres.py", "domainbed/fewshot/methods/clip_adapter.py",
             "domainbed/fewshot/methods/tip_adapter_f.py", "domainbed/fewshot/methods/clap.py",
             "domainbed/fewshot/methods/logistic_probe.py", "domainbed/fewshot/methods/lp_pp.py",
             "domainbed/fewshot/methods/cross_modal_probe.py",
             "domainbed/fewshot/methods/cross_modal_clap.py",
             "domainbed/fewshot/methods/ecalp_inductive.py"]
    return {name: sha(ROOT / name) for name in files}


def load_bank(dataset: str, backbone: str):
    if backbone == "clip_vit_b16":
        path = ROOT / "results/fewshot_ood/features" / dataset / "clip_vit_b16_224"
        meta = json.loads((path / "complete.json").read_text())
        features = np.load(path / "image_features.npy", mmap_mode="r")
        names = meta["domains"]
        classes = meta["classes"]
    elif backbone == "dinov3_vits16":
        path = ROOT / "results/fewshot_ood/features" / dataset / "dinov3_vits16_224"
        if (path / "complete.json").exists():
            meta = json.loads((path / "complete.json").read_text())
            features = np.load(path / "true_cls.npy", mmap_mode="r")
            names, classes = meta["domains"], meta["classes"]
        else:
            path = ROOT / "outputs/paver/multidataset" / dataset / "auxiliary224"
            meta = json.loads((path / "complete.json").read_text())
            features = np.load(path / "true_cls.npy", mmap_mode="r")
            audit = json.loads((ROOT / "results/fewshot_ood/data_audit.json").read_text())[dataset]
            names = sorted(audit["domains"])
            classes = audit["canonical_classes"]
            lineage = (meta.get("checkpoint_sha256") or
                       json.loads((ROOT / "outputs/paver/cache/pretrained224/complete.json").read_text())
                       ["provenance"]["checkpoint_sha256"])
            actual = sha(ROOT / "results/representation_study/checkpoints/dinov3_s16.safetensors")
            if lineage != actual or meta.get("representation") != "exact_pretrained_initialization":
                raise RuntimeError("Existing DINO cache provenance mismatch")
            meta = dict(meta, checkpoint_sha256=lineage)
    else:
        raise ValueError(backbone)
    inv = np.load(path / "inventory.npz", allow_pickle=False)
    sample_ids = inv["sample_id"]
    labels = inv["label"].astype(np.int64)
    domains = inv["environment"].astype(np.int64)
    audit = json.loads((ROOT / "results/fewshot_ood/data_audit.json").read_text())[dataset]
    if len(features) != audit["sample_count"] or len(sample_ids) != len(features):
        raise RuntimeError(f"Incomplete feature cache for {dataset}: {len(features)} vs {audit['sample_count']}")
    if not np.array_equal(np.unique(labels), np.arange(len(classes))):
        raise RuntimeError("Class index mapping mismatch")
    if len(names) != len(audit["domains"]) or len(classes) != len(audit["canonical_classes"]):
        raise RuntimeError("Domain/class mapping mismatch")
    if names != sorted(audit["domains"]) or classes != audit["canonical_classes"]:
        raise RuntimeError("Cache domain/class names differ from audited dataset")
    return path, meta, features, sample_ids, labels, domains, names, classes, audit


def cache_timing(meta: dict) -> dict:
    """Describe the recorded cache-building time without counting reused rows twice."""
    ranks = meta.get("rank_progress")
    seconds = max(item["elapsed_seconds"] for item in ranks) if ranks else meta.get("elapsed_seconds")
    rows = meta.get("newly_extracted_rows", meta.get("n"))
    return {"feature_extraction_seconds_recorded": seconds,
            "feature_extraction_rows_recorded": rows,
            "feature_extraction_wall_time_semantics":
            "maximum concurrent rank time; excludes model load and cache hashing" if ranks
            else "source cache recorded elapsed time"}


def score_target(classifier, features, labels, query_indices, batch_size, num_classes,
                 patch_features=None):
    correct = np.zeros(num_classes, dtype=np.int64)
    totals = np.zeros(num_classes, dtype=np.int64)
    start = time.perf_counter()
    for offset in range(0, len(query_indices), batch_size):
        indices = query_indices[offset:offset + batch_size]
        batch = np.asarray(features[indices], dtype=np.float32)
        inputs = ((batch, np.asarray(patch_features[indices], dtype=np.float32))
                  if patch_features is not None else batch)
        scores = classifier.predict_scores(inputs)
        if scores.shape != (len(indices), num_classes) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid method scores")
        predictions = scores.argmax(axis=1)
        truth = labels[indices]  # Target labels enter only the metric computation.
        totals += np.bincount(truth, minlength=num_classes)
        correct += np.bincount(truth[predictions == truth], minlength=num_classes)
    elapsed = time.perf_counter() - start
    return {"n_query": int(totals.sum()), "accuracy": float(correct.sum() / totals.sum()),
            "macro_accuracy": float(np.mean(correct[totals > 0] / totals[totals > 0])),
            "per_class_correct": correct.tolist(), "per_class_total": totals.tolist(),
            "seconds": elapsed, "seconds_per_query": elapsed / int(totals.sum())}


def zero_support_metric_path(dataset: str, backbone: str, method: str, target: str,
                             config: dict, text_hash: str, patch_hash: str | None,
                             audit_hash: str, query_batch: int) -> Path:
    relevant = {"dataset": dataset, "backbone": backbone, "method": method,
                "target": target, "text_sha256": text_hash,
                "patch_sha256": patch_hash, "dataset_manifest_sha256": audit_hash,
                "query_batch_size": query_batch,
                "calip_beta2": config["calip_beta2"] if method == "calip_fixed_beta" else None,
                "calip_beta3": config["calip_beta3"] if method == "calip_fixed_beta" else None}
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
    return ROOT / "results/fewshot_ood/zero_shot_metrics" / dataset / backbone / method / digest / f"{target}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATA_DIRS)
    parser.add_argument("--setting", required=True, choices=("multi_source", "single_source"))
    parser.add_argument("--shot_scope", required=True, choices=("total", "per_domain"))
    parser.add_argument("--shots", required=True, type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--backbone", required=True, choices=("dinov3_vits16", "clip_vit_b16"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--query-batch", type=int, default=1024)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ape-feature-count", type=int, default=400)
    parser.add_argument("--calip-beta2", type=float, default=2.0)
    parser.add_argument("--calip-beta3", type=float, default=0.1)
    parser.add_argument("--timo-prompt-gamma", type=float, default=50.0)
    parser.add_argument("--timo-alpha", type=float, default=100.0)
    parser.add_argument("--rff-components", type=int, default=4096)
    parser.add_argument("--rff-device", default="cpu")
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--taskres-scale", type=float, default=0.5)
    parser.add_argument("--taskres-epochs", type=int)
    parser.add_argument("--taskres-lr", type=float, default=0.002)
    parser.add_argument("--taskres-warmup-lr", type=float, default=1e-5)
    parser.add_argument("--clip-adapter-ratio", type=float, default=0.2)
    parser.add_argument("--clip-adapter-epochs", type=int, default=300)
    parser.add_argument("--clip-adapter-lr", type=float, default=0.1)
    parser.add_argument("--clip-adapter-optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--tip-f-epochs", type=int, default=20)
    parser.add_argument("--tip-f-lr", type=float, default=0.001)
    parser.add_argument("--ape-t-epochs", type=int, default=20)
    parser.add_argument("--ape-t-lr", type=float, default=0.0001)
    parser.add_argument("--ape-t-feature-count", type=int, default=500)
    parser.add_argument("--clap-epochs", type=int, default=300)
    parser.add_argument("--clap-lr", type=float, default=0.1)
    parser.add_argument("--clap-warmup-lr", type=float, default=1e-5)
    parser.add_argument("--clap-reference-views", type=int, default=20)
    parser.add_argument("--logistic-c", type=float, default=0.316)
    parser.add_argument("--lp-pp-epochs", type=int, default=300)
    parser.add_argument("--cross-modal-iterations", type=int, default=12800)
    parser.add_argument("--cross-modal-batch-size", type=int, default=8)
    parser.add_argument("--cross-modal-lr", type=float, default=0.001)
    parser.add_argument("--cross-modal-weight-decay", type=float, default=0.0)
    parser.add_argument("--cross-modal-clap-epochs", type=int, default=300)
    parser.add_argument("--cross-modal-clap-lr", type=float, default=0.1)
    parser.add_argument("--ecalp-device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.method in TEXT_METHODS and args.backbone != "clip_vit_b16":
        parser.error(f"{args.method} requires CLIP")
    if args.query_batch < 1:
        parser.error("query-batch must be positive")

    path, meta, features, ids, labels, domain_ids, names, classes, audit = load_bank(args.dataset, args.backbone)
    allowed = eligible_mask(ids, audit, args.dataset)
    patch_features = None
    patch_metadata = None
    if args.method == "calip_fixed_beta":
        patch_path = ROOT / "results/fewshot_ood/clip_tokens" / args.dataset
        patch_metadata = json.loads((patch_path / "complete.json").read_text())
        token_inventory = np.load(patch_path / "inventory.npz", allow_pickle=False)
        if (patch_metadata["checkpoint_sha256"] != meta["checkpoint_sha256"] or
                not np.array_equal(token_inventory["sample_id"], ids)):
            raise RuntimeError("CALIP token-cache provenance or row alignment mismatch")
        patch_features = np.load(patch_path / "patch_features.npy", mmap_mode="r")
        if patch_features.shape != (len(ids), 196, 512):
            raise RuntimeError("CALIP patch cache shape mismatch")
    from domainbed.fewshot.protocols import validate_domainnet_domains, validate_result_targets
    validate_domainnet_domains(args.dataset, names)
    fold_list = folds(names, args.setting)
    if args.fold is not None:
        fold_list = [fold_list[args.fold]]
    dataset_manifest_sha = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "torch": importlib.metadata.version("torch"),
               "torchvision": importlib.metadata.version("torchvision")}
    source_hashes = code_hashes()
    lookup = None
    for fold in fold_list:
        start_support = time.perf_counter()
        records = source_candidates(args.dataset, fold, args.seed, ids, labels,
                                    domain_ids, names, allowed, len(classes), dataset_manifest_sha)
        try:
            manifest = build_manifest(args.dataset, fold, records, len(classes),
                                      args.shots, args.shot_scope, args.seed)
        except ValueError as exc:
            if not str(exc).startswith("Infeasible:"):
                raise
            exclusion = {"dataset": args.dataset, "setting": args.setting,
                         "shot_scope": args.shot_scope, "shots": args.shots,
                         "seed": args.seed, "fold_id": fold.identifier,
                         "source_domains": fold.source_domains,
                         "target_domains": fold.target_domains,
                         "dataset_manifest_sha256": dataset_manifest_sha,
                         "reason": str(exc)}
            destination = (ROOT / "results/fewshot_ood/infeasible" / args.dataset /
                           args.setting / args.shot_scope / str(args.shots) /
                           str(args.seed) / f"{fold.identifier}.json")
            write_immutable_json(destination, exclusion)
            print(f"{fold.identifier}: {exc}", flush=True)
            continue
        manifest_path = (ROOT / "results/fewshot_ood/manifests" / args.dataset /
                         args.setting / args.shot_scope / str(args.shots) / str(args.seed) /
                         f"{fold.identifier}.json")
        save_manifest(manifest_path, manifest)
        support_seconds = time.perf_counter() - start_support
        print(f"{fold.identifier}: support={len(manifest['support'])} "
              f"sha={manifest['manifest_sha256'][:12]}", flush=True)
        if args.dry_run:
            continue
        destination = (ROOT / "results/fewshot_ood/runs" / args.dataset / args.backbone /
                       args.setting / args.shot_scope / str(args.shots) / str(args.seed) /
                       args.method / f"{fold.identifier}.json")
        if destination.exists():
            existing = json.loads(destination.read_text())
            validate_result_targets(existing, fold)
            if existing["manifest_sha256"] != manifest["manifest_sha256"]:
                raise RuntimeError(f"Existing result uses a different support set: {destination}")
            if args.method == "proker_rff_imagenet_fixed" and (
                    existing["method_config"].get("rff_components") != args.rff_components or
                    existing["method_config"].get("rff_seed") != args.seed):
                raise RuntimeError(f"Existing RFF result uses different parameters: {destination}")
            taskres_epochs = (args.taskres_epochs if args.taskres_epochs is not None
                              else (100 if args.shots <= 4 else 200))
            if args.method == "taskres_frozen" and (
                    existing["method_config"].get("taskres_scale") != args.taskres_scale or
                    existing["method_config"].get("taskres_epochs") != taskres_epochs or
                    existing["method_config"].get("taskres_lr") != args.taskres_lr or
                    existing["method_config"].get("taskres_warmup_lr") != args.taskres_warmup_lr):
                raise RuntimeError(f"Existing TaskRes result uses different parameters: {destination}")
            if args.method == "clip_adapter_frozen" and (
                    existing["method_config"].get("clip_adapter_ratio") != args.clip_adapter_ratio or
                    existing["method_config"].get("clip_adapter_epochs") != args.clip_adapter_epochs or
                    existing["method_config"].get("clip_adapter_lr") != args.clip_adapter_lr or
                    existing["method_config"].get("clip_adapter_optimizer") != args.clip_adapter_optimizer):
                raise RuntimeError(f"Existing CLIP-Adapter result uses different parameters: {destination}")
            if args.method == "tip_adapter_f_imagenet_fixed" and (
                    existing["method_config"].get("tip_f_epochs") != args.tip_f_epochs or
                    existing["method_config"].get("tip_f_lr") != args.tip_f_lr):
                raise RuntimeError(f"Existing Tip-Adapter-F result uses different parameters: {destination}")
            if args.method == "ape_t_imagenet_fixed" and (
                    existing["method_config"].get("ape_t_epochs") != args.ape_t_epochs or
                    existing["method_config"].get("ape_t_lr") != args.ape_t_lr or
                    existing["method_config"].get("ape_t_feature_count") != args.ape_t_feature_count):
                raise RuntimeError(f"Existing APE-T result uses different parameters: {destination}")
            if args.method == "clap_frozen" and (
                    existing["method_config"].get("clap_epochs") != args.clap_epochs or
                    existing["method_config"].get("clap_lr") != args.clap_lr or
                    existing["method_config"].get("clap_warmup_lr") != args.clap_warmup_lr or
                    existing["method_config"].get("clap_reference_views") != args.clap_reference_views or
                    existing["method_config"].get("clap_epoch_batches") != "source_loader"):
                raise RuntimeError(f"Existing CLAP result uses different parameters: {destination}")
            if args.method == "logistic_probe_c0316" and (
                    existing["method_config"].get("logistic_c") != args.logistic_c):
                raise RuntimeError(f"Existing logistic probe uses different C: {destination}")
            if args.method == "lp_pp_frozen" and (
                    existing["method_config"].get("lp_pp_epochs") != args.lp_pp_epochs or
                    existing["method_config"].get("lp_pp_selection") != "source_views"):
                raise RuntimeError(f"Existing LP++ result uses different epochs: {destination}")
            if args.method == "cross_modal_probe_frozen" and (
                    existing["method_config"].get("cross_modal_iterations") != args.cross_modal_iterations or
                    existing["method_config"].get("training_batch_size") != args.cross_modal_batch_size or
                    existing["method_config"].get("cross_modal_lr") != args.cross_modal_lr or
                    existing["method_config"].get("cross_modal_weight_decay") != args.cross_modal_weight_decay or
                    existing["method_config"].get("cross_modal_init") != "text"):
                raise RuntimeError(f"Existing cross-modal result uses different parameters: {destination}")
            if args.method == "ecalp_inductive" and (
                    existing["method_config"].get("ecalp_device") != args.ecalp_device):
                raise RuntimeError(f"Existing ECALP result uses different device: {destination}")
            if args.method == "cross_modal_lp_clap" and (
                    existing["method_config"].get("cross_modal_clap_epochs") != args.cross_modal_clap_epochs or
                    existing["method_config"].get("cross_modal_clap_lr") != args.cross_modal_clap_lr):
                raise RuntimeError(f"Existing CrossModal CLAP result uses different parameters: {destination}")
            print(f"  existing result: {destination}", flush=True)
            continue
        support_read_start = time.perf_counter()
        if lookup is None:
            lookup = {str(sample): i for i, sample in enumerate(ids)}
        support_indices = np.array([lookup[item["sample_id"]] for item in manifest["support"]])
        x_support = np.asarray(features[support_indices], dtype=np.float32)
        y_support = np.array([item["class_id"] for item in manifest["support"]])
        support_feature_read_seconds = time.perf_counter() - support_read_start
        input_load_start = time.perf_counter()
        beta = args.beta if args.beta is not None else (1.0 if args.method == "ape_training_free" else 5.5)
        config = {"num_classes": len(classes), "metric": args.metric, "k": args.k,
                  "alpha": args.alpha, "beta": beta, "gamma": args.gamma,
                  "ape_feature_count": args.ape_feature_count,
                  "ape_criterion_weights": [0.5, 0.5],
                  "calip_beta2": args.calip_beta2, "calip_beta3": args.calip_beta3,
                  "timo_prompt_gamma": args.timo_prompt_gamma,
                  "timo_alpha": args.timo_alpha,
                  "hyperparameter_policy": ("fixed_ood_adaptation" if args.method in ADAPTED_FIXED_METHODS
                                            else "fixed_reference")}
        if args.method in {"proker_imagenet_fixed", "proker_rff_imagenet_fixed"}:
            config["proker_beta"], config["proker_lambda"] = IMAGENET_HYPERPARAMETERS[args.shots]
        if args.method == "proker_rff_imagenet_fixed":
            config.update(rff_components=args.rff_components, rff_device=args.rff_device,
                          rff_seed=args.seed)
        if args.method == "taskres_frozen":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed,
                          taskres_scale=args.taskres_scale,
                          taskres_epochs=(args.taskres_epochs if args.taskres_epochs is not None
                                          else (100 if args.shots <= 4 else 200)),
                          taskres_lr=args.taskres_lr,
                          taskres_warmup_lr=args.taskres_warmup_lr)
        if args.method == "clip_adapter_frozen":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed,
                          clip_adapter_ratio=args.clip_adapter_ratio,
                          clip_adapter_epochs=args.clip_adapter_epochs,
                          clip_adapter_lr=args.clip_adapter_lr,
                          clip_adapter_optimizer=args.clip_adapter_optimizer)
        if args.method == "tip_adapter_f_imagenet_fixed":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed, tip_f_beta=1.0, tip_f_alpha=1.17,
                          tip_f_epochs=args.tip_f_epochs, tip_f_lr=args.tip_f_lr)
        if args.method == "ape_t_imagenet_fixed":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed, ape_t_beta=1.0, ape_t_alpha=2.0,
                          ape_t_epochs=args.ape_t_epochs, ape_t_lr=args.ape_t_lr,
                          ape_t_eps=0.001, ape_t_feature_count=args.ape_t_feature_count)
        if args.method == "clap_frozen":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed, clap_epochs=args.clap_epochs,
                          clap_lr=args.clap_lr,
                          clap_warmup_lr=args.clap_warmup_lr,
                          clap_reference_views=args.clap_reference_views,
                          clap_epoch_batches="source_loader")
        if args.method == "logistic_probe_c0316":
            config["logistic_c"] = args.logistic_c
        if args.method == "lp_pp_frozen":
            config.update(training_device=args.training_device, training_seed=args.seed,
                          lp_pp_epochs=args.lp_pp_epochs,
                          lp_pp_selection="source_views", lp_pp_validation_view=0)
        if args.method == "cross_modal_probe_frozen":
            config.update(training_device=args.training_device,
                          training_batch_size=args.cross_modal_batch_size,
                          training_seed=args.seed,
                          cross_modal_iterations=args.cross_modal_iterations,
                          cross_modal_lr=args.cross_modal_lr,
                          cross_modal_weight_decay=args.cross_modal_weight_decay,
                          cross_modal_init="text", cross_modal_selection="final")
        if args.method == "cross_modal_lp_clap":
            config.update(training_device=args.training_device,
                          training_batch_size=args.training_batch_size,
                          training_seed=args.seed,
                          cross_modal_clap_epochs=args.cross_modal_clap_epochs,
                          cross_modal_clap_lr=args.cross_modal_clap_lr)
        if args.method == "ecalp_inductive":
            config.update(ecalp_device=args.ecalp_device,
                          ecalp_k_text=3, ecalp_k_fewshot=8,
                          ecalp_gamma=10.0, ecalp_alpha=1.0,
                          ecalp_iterations=3,
                          ecalp_protocol="fresh_two_query_graph")
        view_metadata = None
        if args.method in VIEW_METHODS:
            view_path = (ROOT / "results/fewshot_ood/support_views" /
                         args.dataset / args.backbone)
            view_metadata = json.loads((view_path / "complete.json").read_text())
            if view_metadata["checkpoint_sha256"] != meta["checkpoint_sha256"]:
                raise RuntimeError("Support view checkpoint provenance mismatch")
            view_inventory = np.load(view_path / "inventory.npz", allow_pickle=False)
            view_lookup = {str(sample): i for i, sample in enumerate(view_inventory["sample_id"])}
            try:
                view_rows = [view_lookup[item["sample_id"]] for item in manifest["support"]]
            except KeyError as exc:
                raise RuntimeError("Selected support is missing from view cache") from exc
            config["support_views"] = np.asarray(
                np.load(view_path / "support_views.npy", mmap_mode="r")[view_rows],
                dtype=np.float32)
        if args.method in TEXT_METHODS:
            text_meta = json.loads((path / "text_metadata.json").read_text())
            if text_meta["classes"] != classes or text_meta["checkpoint_sha256"] != meta["checkpoint_sha256"]:
                raise RuntimeError("CLIP text cache provenance mismatch")
            config["text_features"] = np.load(path / "text_features.npy")
            text_hash = text_meta["text_features_sha256"]
        raw_text_meta = None
        if args.method in {"taskres_frozen", "clap_frozen", "cross_modal_lp_clap"}:
            raw_text_meta = json.loads((path / "raw_text_metadata.json").read_text())
            if (raw_text_meta["classes"] != classes or
                    raw_text_meta["checkpoint_sha256"] != meta["checkpoint_sha256"]):
                raise RuntimeError("Raw text cache provenance mismatch")
            config["raw_text_features"] = np.load(path / "raw_text_features.npy")
        if args.method in {"timo_fixed", "cross_modal_probe_frozen", "cross_modal_lp_clap"}:
            prompt_meta = json.loads((path / "prompt_metadata.json").read_text())
            if (prompt_meta["classes"] != classes or
                    prompt_meta["checkpoint_sha256"] != meta["checkpoint_sha256"]):
                raise RuntimeError("TIMO prompt cache provenance mismatch")
            config["prompt_features"] = np.load(path / "prompt_features.npy")
        fit_start = time.perf_counter()
        method_input_load_seconds = fit_start - input_load_start
        classifier = METHODS[args.method]().fit(x_support, y_support, config)
        fit_seconds = time.perf_counter() - fit_start
        target_metrics = {}
        metrics_reused = {}
        query_batch = min(args.query_batch, 128) if patch_features is not None else args.query_batch
        for target in fold.target_domains:
            query_indices = np.flatnonzero(allowed & (domain_ids == names.index(target)))
            if not len(query_indices) or np.intersect1d(query_indices, support_indices).size:
                raise RuntimeError("Empty target or support/query contamination")
            metric_path = (zero_support_metric_path(
                args.dataset, args.backbone, args.method, target, config, text_hash,
                patch_metadata["patch_features_sha256"] if patch_metadata else None,
                dataset_manifest_sha, query_batch) if args.method in ZERO_SUPPORT_METHODS else None)
            if metric_path is not None and metric_path.exists():
                metrics = json.loads(metric_path.read_text())
                if metrics["n_query"] != len(query_indices):
                    raise RuntimeError("Cached zero-support target count changed")
                metrics_reused[target] = True
            else:
                metrics = score_target(classifier, features, labels, query_indices,
                                       query_batch, len(classes), patch_features)
                if metric_path is not None:
                    write_immutable_json(metric_path, metrics)
                metrics_reused[target] = False
            target_metrics[target] = metrics
        result = {"dataset": args.dataset, "setting": args.setting, "shot_scope": args.shot_scope,
                  "shots": args.shots, "seed": args.seed, "fold_id": fold.identifier,
                  "source_domains": fold.source_domains, "target_domains": fold.target_domains,
                  "method": args.method, "backbone": args.backbone,
                  "method_config": {k: v for k, v in config.items()
                                    if k not in {"text_features", "raw_text_features", "support_views", "prompt_features"}},
                  "raw_text_cache_sha256": (raw_text_meta["raw_text_features_sha256"]
                                            if raw_text_meta else None),
                  "prompt_cache_sha256": (prompt_meta["prompt_features_sha256"]
                                          if args.method in {"timo_fixed", "cross_modal_probe_frozen", "cross_modal_lp_clap"} else None),
                  "support_view_cache": (str(view_path.relative_to(ROOT)) if view_metadata else None),
                  "support_view_cache_sha256": (view_metadata["support_views_sha256"]
                                                if view_metadata else None),
                  "support_views_per_image": (view_metadata["n_views"] if view_metadata else None),
                  "support_view_extraction_seconds_recorded": (
                      max(item["elapsed_seconds"] for item in view_metadata["rank_progress"])
                      if view_metadata else None),
                  "clip_token_cache": (str(patch_path.relative_to(ROOT)) if patch_metadata else None),
                  "clip_token_cache_sha256": (patch_metadata["patch_features_sha256"]
                                               if patch_metadata else None),
                  "clip_token_extraction_seconds_recorded": (
                      max(item["elapsed_seconds"] for item in patch_metadata["rank_progress"])
                      if patch_metadata else None),
                  "manifest_sha256": manifest["manifest_sha256"],
                  "dataset_manifest_sha256": dataset_manifest_sha,
                  "checkpoint_sha256": meta["checkpoint_sha256"], "git_commit": commit,
                  "runtime_environment": runtime, "source_code_sha256": source_hashes,
                  "feature_cache_path": str(path.relative_to(ROOT)),
                  "feature_cache_inventory_sha256": meta["inventory_sha256"],
                  **cache_timing(meta),
                  "support_construction_seconds": support_seconds,
                  "support_feature_read_seconds": support_feature_read_seconds,
                  "method_input_load_seconds": method_input_load_seconds,
                  "fit_seconds": fit_seconds,
                  "query_batch_size": query_batch,
                  "target_metrics": target_metrics,
                  "target_metrics_reused": metrics_reused,
                  "mean_target_accuracy": float(np.mean([v["accuracy"] for v in target_metrics.values()])),
                  "worst_target_accuracy": float(min(v["accuracy"] for v in target_metrics.values())),
                  "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        write_immutable_json(destination, result)
        print(f"  mean={100*result['mean_target_accuracy']:.2f}% "
              f"worst={100*result['worst_target_accuracy']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
