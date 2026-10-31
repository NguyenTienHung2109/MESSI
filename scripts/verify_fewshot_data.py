"""Audit the image-folder datasets used by the few-shot OOD benchmark.

The audit uses paths for stable sample IDs. ``--hash-content`` additionally
checks byte-identical images across domains; this is intentionally optional
because DomainNet has hundreds of thousands of files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


DATASETS = {
    "PACS": ("PACS", 4, 7),
    "OfficeHome": ("office_home", 4, 65),
    "TerraIncognita": ("terra_incognita", 4, 10),
    "DomainNet": ("domain_net", 6, 345),
    "VLCS": ("VLCS", 4, 5),
}
REFERENCE_COUNTS = {"PACS": 9991, "OfficeHome": 15588,
                    "TerraIncognita": 24788, "DomainNet": 586575, "VLCS": 10729}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(root: Path, name: str, hash_content: bool = False) -> dict:
    subdir, expected_domains, expected_classes = DATASETS[name]
    path = root / subdir
    report = {"dataset": name, "path": str(path.resolve()), "present": path.is_dir()}
    if not path.is_dir():
        report["errors"] = ["dataset directory missing"]
        return report

    domains = sorted(p for p in path.iterdir() if p.is_dir())
    errors: list[str] = []
    if len(domains) != expected_domains:
        errors.append(f"expected {expected_domains} domains, found {len(domains)}")
    per_domain: dict[str, dict] = {}
    all_classes: set[str] = set()
    ids: set[str] = set()
    content: dict[str, tuple[str, str]] = {}
    duplicate_content: list[dict] = []
    warnings: list[str] = []
    total = 0
    for domain in domains:
        classes = sorted(p for p in domain.iterdir() if p.is_dir())
        class_counts: Counter[str] = Counter()
        all_classes.update(p.name for p in classes)
        for class_dir in classes:
            for image in class_dir.rglob("*"):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                sample_id = image.relative_to(root).as_posix()
                if sample_id in ids:
                    errors.append(f"duplicate sample ID: {sample_id}")
                ids.add(sample_id)
                class_counts[class_dir.name] += 1
                total += 1
                if hash_content:
                    value = digest(image)
                    previous = content.setdefault(value, (domain.name, sample_id))
                    if previous[0] != domain.name and len(duplicate_content) < 100:
                        duplicate_content.append({"first": previous[1], "second": sample_id})
        per_domain[domain.name] = {
            "samples": sum(class_counts.values()),
            "classes": len(class_counts),
            "class_counts": dict(sorted(class_counts.items())),
            "classes_below_16": {k: v for k, v in sorted(class_counts.items()) if v < 16},
        }
        if len(class_counts) != expected_classes:
            errors.append(f"{domain.name}: expected {expected_classes} nonempty classes, found {len(class_counts)}")
    if len(all_classes) != expected_classes:
        errors.append(f"expected {expected_classes} canonical classes, found {len(all_classes)}")
    if total != REFERENCE_COUNTS[name]:
        errors.append(f"expected {REFERENCE_COUNTS[name]} reference images, found {total}")
    for domain, detail in per_domain.items():
        missing = sorted(all_classes - detail["class_counts"].keys())
        if missing:
            errors.append(f"{domain}: missing classes {missing}")
    if duplicate_content:
        warnings.append(f"{len(duplicate_content)} cross-domain byte duplicates; exclude second paths from benchmark")
    report.update({
        "domains": per_domain,
        "canonical_classes": sorted(all_classes),
        "sample_count": total,
        "unique_sample_ids": len(ids),
        "content_hash_checked": hash_content,
        "cross_domain_byte_duplicates": duplicate_content,
        "leakage_exclusions": [item["second"] for item in duplicate_content],
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("domainbed/data"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--hash-content", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/fewshot_ood/data_audit.json"))
    args = parser.parse_args()
    reports = {name: audit_dataset(args.data_dir, name, args.hash_content) for name in args.datasets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    for name, result in reports.items():
        print(f"{name}: {result.get('sample_count', 0)} images; "
              f"{len(result.get('domains', {}))} domains; "
              f"{'PASS' if result.get('valid') else 'FAIL'}")
        for error in result.get("errors", []):
            print(f"  {error}")
        for warning in result.get("warnings", []):
            print(f"  WARNING: {warning}")
    if any(not report.get("valid") for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
