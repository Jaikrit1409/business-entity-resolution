"""Construct reproducible labeled training pairs from blocked candidates only."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .blocking import BlockingResult, ground_truth_pairs
except ImportError:  # pragma: no cover - supports direct script execution
    from blocking import BlockingResult, ground_truth_pairs


@dataclass(frozen=True)
class LabeledCandidatePair:
    """One model input pair, labeled strictly from supplied ground truth."""

    source1_entity_id: str
    candidate_entity_id: str
    candidate_source: str
    label: int


@dataclass(frozen=True)
class PairSplit:
    """A Source 1 group split, safe for macro-per-entity evaluation."""

    train_source1_ids: tuple[str, ...]
    validation_source1_ids: tuple[str, ...]
    train_pairs: tuple[LabeledCandidatePair, ...]
    validation_pairs: tuple[LabeledCandidatePair, ...]


@dataclass(frozen=True)
class PairConstructionReport:
    """Class balance, candidate recall, and Source 1 cardinality diagnostics."""

    positive_count: int
    negative_count: int
    positive_rate: float
    blocking_recall: float | None
    true_match_count: int
    recovered_true_match_count: int
    source1_entity_count: int
    source1_with_candidates_count: int
    source1_with_labeled_pairs_count: int
    singleton_count: int
    one_match_count: int
    multiple_match_count: int
    lost_true_matches: dict[str, tuple[str, ...]]


def _entity_id(record: Mapping[str, Any]) -> str:
    entity_id = "" if record.get("entity_id") is None else str(record["entity_id"]).strip()
    if not entity_id:
        raise ValueError("Source records require a non-empty entity_id")
    return entity_id


def _source1_ids(source1_records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    ids = tuple(_entity_id(record) for record in source1_records)
    if len(ids) != len(set(ids)):
        raise ValueError("Source 1 entity IDs must be unique")
    return ids


def construct_labeled_pairs(
    candidates: BlockingResult,
    source1_records: Sequence[Mapping[str, Any]],
    source2_records: Sequence[Mapping[str, Any]],
    source3_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
) -> tuple[LabeledCandidatePair, ...]:
    """Label only candidate pairs emitted by the actual blocking pipeline.

    A true pair missed by blocking is deliberately *not* fabricated as a
    training row; it is captured by :func:`pair_construction_report` as a
    blocking-recall loss instead.
    """
    source1_ids = _source1_ids(source1_records)
    source1_id_set = set(source1_ids)
    true_pairs = ground_truth_pairs(
        ground_truth_records, source1_records, source2_records, source3_records
    )
    rows: list[LabeledCandidatePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source1_id in source1_ids:
        for target_source, target_ids in (("S2", candidates.s1_to_s2.get(source1_id, ())), ("S3", candidates.s1_to_s3.get(source1_id, ()) )):
            for target_id in target_ids:
                if not target_id.startswith(f"{target_source}-"):
                    raise ValueError(f"{target_id} is not a {target_source} candidate for {source1_id}")
                pair = (source1_id, target_id)
                if pair in seen_pairs:
                    raise ValueError(f"Duplicate candidate pair: {pair}")
                seen_pairs.add(pair)
                rows.append(LabeledCandidatePair(source1_id, target_id, target_source, int(pair in true_pairs)))
    unknown_truth_source1 = {source1_id for source1_id, _ in true_pairs} - source1_id_set
    if unknown_truth_source1:
        raise ValueError(f"Ground truth references unknown Source 1 IDs: {sorted(unknown_truth_source1)}")
    return tuple(rows)


def split_source1_entities(
    source1_ids: Sequence[str], validation_fraction: float = 0.2, random_seed: int = 42
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a deterministic Source 1-level train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    unique_ids = sorted(set(source1_ids))
    if len(unique_ids) != len(source1_ids):
        raise ValueError("source1_ids must be unique")
    if len(unique_ids) < 2:
        return tuple(unique_ids), ()
    shuffled = unique_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    validation_ids = tuple(sorted(shuffled[:validation_count]))
    train_ids = tuple(sorted(shuffled[validation_count:]))
    return train_ids, validation_ids


def split_labeled_pairs(
    pairs: Sequence[LabeledCandidatePair],
    source1_ids: Sequence[str],
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> PairSplit:
    """Split pairs by Source 1, retaining all of an entity's pairs together."""
    train_ids, validation_ids = split_source1_entities(source1_ids, validation_fraction, random_seed)
    train_set, validation_set = set(train_ids), set(validation_ids)
    if train_set & validation_set:
        raise AssertionError("Source 1 leakage detected in split construction")
    unknown_pair_ids = {pair.source1_entity_id for pair in pairs} - (train_set | validation_set)
    if unknown_pair_ids:
        raise ValueError(f"Pairs reference Source 1 IDs outside the split: {sorted(unknown_pair_ids)}")
    return PairSplit(
        train_source1_ids=train_ids,
        validation_source1_ids=validation_ids,
        train_pairs=tuple(pair for pair in pairs if pair.source1_entity_id in train_set),
        validation_pairs=tuple(pair for pair in pairs if pair.source1_entity_id in validation_set),
    )


def pair_construction_report(
    candidates: BlockingResult,
    source1_records: Sequence[Mapping[str, Any]],
    source2_records: Sequence[Mapping[str, Any]],
    source3_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    labeled_pairs: Sequence[LabeledCandidatePair] | None = None,
) -> PairConstructionReport:
    """Compute class balance and macro-F0.5-relevant Source 1 diagnostics."""
    source1_ids = _source1_ids(source1_records)
    true_pairs = ground_truth_pairs(
        ground_truth_records, source1_records, source2_records, source3_records
    )
    pairs = tuple(labeled_pairs) if labeled_pairs is not None else construct_labeled_pairs(
        candidates, source1_records, source2_records, source3_records, ground_truth_records
    )
    positives = sum(pair.label for pair in pairs)
    negatives = len(pairs) - positives
    generated_pairs = {(pair.source1_entity_id, pair.candidate_entity_id) for pair in pairs}
    recovered = true_pairs & generated_pairs
    lost: dict[str, list[str]] = defaultdict(list)
    for source1_id, candidate_id in sorted(true_pairs - generated_pairs):
        lost[source1_id].append(candidate_id)

    true_match_counts = Counter(source1_id for source1_id, _ in true_pairs)
    singleton_count = sum(true_match_counts[source1_id] == 0 for source1_id in source1_ids)
    one_match_count = sum(true_match_counts[source1_id] == 1 for source1_id in source1_ids)
    multiple_match_count = sum(true_match_counts[source1_id] > 1 for source1_id in source1_ids)
    candidate_map = candidates.combined()
    return PairConstructionReport(
        positive_count=positives,
        negative_count=negatives,
        positive_rate=(positives / len(pairs)) if pairs else 0.0,
        blocking_recall=(len(recovered) / len(true_pairs)) if true_pairs else None,
        true_match_count=len(true_pairs),
        recovered_true_match_count=len(recovered),
        source1_entity_count=len(source1_ids),
        source1_with_candidates_count=sum(bool(candidate_map.get(source1_id)) for source1_id in source1_ids),
        source1_with_labeled_pairs_count=len({pair.source1_entity_id for pair in pairs}),
        singleton_count=singleton_count,
        one_match_count=one_match_count,
        multiple_match_count=multiple_match_count,
        lost_true_matches={source1_id: tuple(ids) for source1_id, ids in lost.items()},
    )
