"""Domain folds and source-only deterministic K-shot support manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable


SHOTS = (1, 2, 4, 8, 16)
SEEDS = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    domain: str
    class_id: int


@dataclass(frozen=True)
class Fold:
    setting: str
    source_domains: tuple[str, ...]
    target_domains: tuple[str, ...]

    @property
    def identifier(self) -> str:
        return f"{self.setting}__{'-'.join(self.source_domains)}__to__{'-'.join(self.target_domains)}"


def folds(domains: Iterable[str], setting: str) -> list[Fold]:
    names = tuple(domains)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Need at least two distinct domains")
    if setting == "multi_source":
        return [Fold(setting, tuple(x for x in names if x != target), (target,))
                for target in names]
    if setting == "single_source":
        return [Fold(setting, (source,), tuple(x for x in names if x != source))
                for source in names]
    raise ValueError(f"Unknown setting: {setting}")


def _integer(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _permuted(records: list[SourceRecord], *key: object) -> list[SourceRecord]:
    return sorted(records, key=lambda r: (_integer(*key, r.sample_id), r.sample_id))


def candidate_pool(source_records: Iterable[SourceRecord], source_domains: Iterable[str],
                   num_classes: int, seed: int) -> list[SourceRecord]:
    """Keep the 16 earliest source IDs per class/domain under the sampler hash.

    All supported budgets are at most 16. This is exactly equivalent to
    sampling from the full source inventory and never reads a target record.
    """
    domains = tuple(source_domains)
    grouped: dict[tuple[int, str], list[SourceRecord]] = defaultdict(list)
    for record in source_records:
        if record.domain not in domains or not 0 <= record.class_id < num_classes:
            raise ValueError("Candidate pool received a non-source record")
        grouped[(record.class_id, record.domain)].append(record)
    result = []
    for class_id in range(num_classes):
        for domain in domains:
            key = lambda r: (_integer(seed, class_id, domain, r.sample_id), r.sample_id)
            result.extend(heapq.nsmallest(16, grouped[(class_id, domain)], key=key))
    return result


def sample_support(source_records: Iterable[SourceRecord], source_domains: Iterable[str],
                   num_classes: int, shots: int, shot_scope: str, seed: int) -> list[SourceRecord]:
    """Use source records only; no target record or target statistic is accepted.

    S1 rotates the domain allocation across classes. With K below the number
    of sources, selected domains are distinct. If a domain has insufficient
    examples, its allocation deterministically spills into the remaining
    eligible source domains; the total remains exactly K per class.
    """
    if shots not in SHOTS or shot_scope not in {"total", "per_domain"}:
        raise ValueError("Unsupported shot configuration")
    domains = tuple(source_domains)
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("Invalid source domains")
    grouped: dict[tuple[int, str], list[SourceRecord]] = defaultdict(list)
    seen_ids: set[str] = set()
    for record in source_records:
        if record.domain not in domains or not 0 <= record.class_id < num_classes:
            raise ValueError("Source record outside declared source domains/classes")
        if record.sample_id in seen_ids:
            raise ValueError(f"Duplicate source sample ID: {record.sample_id}")
        seen_ids.add(record.sample_id)
        grouped[(record.class_id, record.domain)].append(record)
    result: list[SourceRecord] = []
    for class_id in range(num_classes):
        pools = {domain: _permuted(grouped[(class_id, domain)], seed, class_id, domain)
                 for domain in domains}
        if shot_scope == "per_domain":
            for domain in domains:
                if len(pools[domain]) < shots:
                    raise ValueError(f"Infeasible: class {class_id}, domain {domain}, "
                                     f"need {shots}, have {len(pools[domain])}")
                result.extend(pools[domain][:shots])
            continue
        if sum(map(len, pools.values())) < shots:
            raise ValueError(f"Infeasible: class {class_id}, need {shots}, "
                             f"have {sum(map(len, pools.values()))} across sources")
        start = (class_id + seed) % len(domains)
        order = domains[start:] + domains[:start]
        chosen: Counter[str] = Counter()
        while sum(chosen.values()) < shots:
            progressed = False
            for domain in order:
                if chosen[domain] < len(pools[domain]):
                    chosen[domain] += 1
                    progressed = True
                    if sum(chosen.values()) == shots:
                        break
            if not progressed:
                raise AssertionError("Capacity check and allocation disagree")
        for domain in domains:
            result.extend(pools[domain][:chosen[domain]])
    expected = num_classes * shots * (len(domains) if shot_scope == "per_domain" else 1)
    if len(result) != expected or len({r.sample_id for r in result}) != expected:
        raise AssertionError("Support budget or uniqueness violated")
    return result


def build_manifest(dataset: str, fold: Fold, records: Iterable[SourceRecord],
                   num_classes: int, shots: int, shot_scope: str, seed: int) -> dict:
    support = sample_support(records, fold.source_domains, num_classes, shots, shot_scope, seed)
    allocation: dict[int, Counter[str]] = defaultdict(Counter)
    for record in support:
        allocation[record.class_id][record.domain] += 1
    payload = {"dataset": dataset, "fold_id": fold.identifier, "setting": fold.setting,
               "shot_scope": shot_scope, "shots": shots, "seed": seed,
               "source_domains": list(fold.source_domains),
               "target_domains": list(fold.target_domains), "num_classes": num_classes,
               "support": [vars(record) for record in support],
               "source_allocation": {str(class_id): dict(sorted(allocation[class_id].items()))
                                     for class_id in range(num_classes)}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        # A writer from a previously launched process may still be completing
        # its old non-atomic write. Never accept a genuinely changed manifest.
        for _ in range(3):
            time.sleep(0.05)
            if path.read_text() == encoded:
                return
        raise RuntimeError(f"Existing manifest changed: {path}")
    if path.exists():
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        stream.write(encoded)
        temp = Path(stream.name)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_text() != encoded:
        raise RuntimeError(f"Concurrent manifest mismatch: {path}")
