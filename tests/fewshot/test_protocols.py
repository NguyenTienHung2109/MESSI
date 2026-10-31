from collections import Counter

import pytest

from domainbed.fewshot.protocols import SourceRecord, candidate_pool, folds, sample_support


DOMAINS = ("A", "C", "P", "S")


def records(domains=DOMAINS, classes=7, per_cell=20):
    return [SourceRecord(f"{domain}/{klass}/{index}", domain, klass)
            for domain in domains for klass in range(classes) for index in range(per_cell)]


def test_fold_coverage_and_separation():
    for setting in ("multi_source", "single_source"):
        assignments = folds(DOMAINS, setting)
        assert len(assignments) == 4
        for fold in assignments:
            assert set(fold.source_domains).isdisjoint(fold.target_domains)
            assert set(fold.source_domains) | set(fold.target_domains) == set(DOMAINS)
            assert len(fold.source_domains) == (3 if setting == "multi_source" else 1)


@pytest.mark.parametrize("shots", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("scope", ("total", "per_domain"))
def test_exact_support_budgets_and_target_exclusion(shots, scope):
    fold = folds(DOMAINS, "multi_source")[0]
    source = [r for r in records() if r.domain in fold.source_domains]
    chosen = sample_support(source, fold.source_domains, 7, shots, scope, 42)
    expected = shots * (3 if scope == "per_domain" else 1)
    assert Counter(r.class_id for r in chosen) == {i: expected for i in range(7)}
    assert len({r.sample_id for r in chosen}) == len(chosen)
    assert not any(r.domain in fold.target_domains for r in chosen)
    assert chosen == sample_support(reversed(source), fold.source_domains, 7, shots, scope, 42)
    if shots == 1 and scope == "total":
        assert len(chosen) == 7


def test_single_source_scopes_identical():
    source = [r for r in records() if r.domain == "P"]
    for shots in (1, 2, 4, 8, 16):
        assert sample_support(source, ("P",), 7, shots, "total", 42) == \
            sample_support(source, ("P",), 7, shots, "per_domain", 42)


def test_infeasible_is_explicit():
    source = records(domains=("A", "C"), classes=2, per_cell=1)
    with pytest.raises(ValueError, match="Infeasible"):
        sample_support(source, ("A", "C"), 2, 4, "total", 42)
    with pytest.raises(ValueError, match="Infeasible"):
        sample_support(source, ("A", "C"), 2, 2, "per_domain", 42)


def test_bad_source_record_rejected():
    with pytest.raises(ValueError, match="outside"):
        sample_support(records(domains=("A", "C")), ("A",), 7, 1, "total", 42)


def test_source_candidate_pool_is_exact_for_all_budgets():
    source = records(domains=("A", "C", "P"), classes=7, per_cell=37)
    for seed in (42, 46):
        top = candidate_pool(source, ("A", "C", "P"), 7, seed)
        for shots in (1, 2, 4, 8, 16):
            for scope in ("total", "per_domain"):
                assert sample_support(top, ("A", "C", "P"), 7, shots, scope, seed) == \
                    sample_support(source, ("A", "C", "P"), 7, shots, scope, seed)
