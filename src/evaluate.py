"""Evaluate entity-resolution predictions against labeled pairs."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _pairs(
    records: Iterable[Mapping[str, Any]],
    source1_key: str = "source1_entity_id",
    matched_key: str = "matched_entity_ids",
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()

    for record in records:
        source1 = str(record.get(source1_key, "")).strip()
        matched = str(record.get(matched_key, "")).strip()

        if not source1 or not matched:
            continue

        for candidate in matched.split(","):
            candidate = candidate.strip()
            if candidate:
                result.add((source1, candidate))

    return result


def macro_f0_5(
    source1_ids: Sequence[str],
    true_pairs: set[tuple[str, str]],
    predicted_pairs: set[tuple[str, str]],
) -> float:
    """Calculate entity-level macro F0.5."""
    if not source1_ids:
        return 0.0

    true_by: dict[str, set[str]] = defaultdict(set)
    pred_by: dict[str, set[str]] = defaultdict(set)

    for s1, candidate in true_pairs:
        true_by[s1].add(candidate)
    for s1, candidate in predicted_pairs:
        pred_by[s1].add(candidate)

    scores = []

    for s1 in source1_ids:
        truth = true_by[s1]
        pred = pred_by[s1]

        if not truth and not pred:
            scores.append(1.0)
            continue

        if not truth or not pred:
            scores.append(0.0)
            continue

        tp = len(truth & pred)
        precision = tp / len(pred)
        recall = tp / len(truth)

        scores.append(
            (1.25 * precision * recall) / (0.25 * precision + recall)
            if precision + recall
            else 0.0
        )

    return sum(scores) / len(scores)


def evaluate_predictions(
    source1_ids: Sequence[str],
    truth_records: Iterable[Mapping[str, Any]],
    predicted_records: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return precision, recall and macro F0.5 diagnostics."""
    truth = _pairs(truth_records)
    predicted = _pairs(predicted_records)

    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    return {
        "true_pair_count": len(truth),
        "predicted_pair_count": len(predicted),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "pair_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
        "macro_f0_5": macro_f0_5(source1_ids, truth, predicted),
    }
