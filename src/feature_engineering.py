"""Numerical pairwise features for candidates produced by the blocking stage.

This module intentionally receives only blocked candidate pairs. It never
constructs an all-record Cartesian product and uses no external data or APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from collections.abc import Mapping, Sequence
from typing import Any

from rapidfuzz import fuzz

try:
    from .preprocess import (
        normalize_address,
        normalize_business_name,
        normalize_business_name_core,
        normalize_country,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from preprocess import (
        normalize_address,
        normalize_business_name,
        normalize_business_name_core,
        normalize_country,
    )


FEATURE_COLUMNS = (
    "name_exact",
    "name_character_similarity",
    "name_levenshtein_similarity",
    "name_token_similarity",
    "name_token_set_similarity",
    "name_token_sort_similarity",
    "name_ngram_similarity",
    "name_length_difference",
    "address_exact",
    "address_character_similarity",
    "address_token_similarity",
    "address_token_set_similarity",
    "address_ngram_similarity",
    "address_numeric_token_overlap",
    "address_length_difference",
    "country_equal",
    "name_left_missing",
    "name_right_missing",
    "name_any_missing",
    "address_left_missing",
    "address_right_missing",
    "address_any_missing",
    "country_left_missing",
    "country_right_missing",
    "country_any_missing",
    "target_is_source2",
    "target_is_source3",
    "name_address_exact_both",
    "name_address_character_mean",
    "name_address_character_min",
    "name_address_evidence_product",
    "country_and_name_exact",
    "country_and_address_exact",
)


@dataclass(frozen=True)
class PreparedRecord:
    """One record's reusable normalized representations."""

    entity_id: str
    source: str
    name: str
    name_core: str
    address: str
    country: str
    name_tokens: tuple[str, ...] = ()
    name_core_tokens: tuple[str, ...] = ()
    address_tokens: tuple[str, ...] = ()
    name_ngrams: frozenset[str] = frozenset()
    address_ngrams: frozenset[str] = frozenset()
    address_numeric_tokens: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FeatureMatrix:
    """Metadata and purely numerical values for blocked candidate pairs."""

    source1_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def as_dict_rows(self) -> list[dict[str, float]]:
        """Return values as named numeric feature dictionaries."""
        return [dict(zip(self.feature_names, row, strict=True)) for row in self.values]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_field(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _source(record: Mapping[str, Any], entity_id: str) -> str:
    value = _text(record.get("source")).casefold()
    if value in {"s2", "2", "source2", "source_2"} or entity_id.startswith("S2-"):
        return "S2"
    if value in {"s3", "3", "source3", "source_3"} or entity_id.startswith("S3-"):
        return "S3"
    if entity_id.startswith("S1-"):
        return "S1"
    return "UNKNOWN"


def prepare_record(record: Mapping[str, Any]) -> PreparedRecord:
    """Normalize one raw record once for repeated candidate-pair feature work."""
    entity_id = _text(record.get("entity_id"))
    if not entity_id:
        raise ValueError("Feature generation requires a non-empty entity_id")
    raw_name = _raw_field(record, "name", "business_name")
    name = normalize_business_name(raw_name)
    name_core = normalize_business_name_core(raw_name)
    address = normalize_address(
        _raw_field(record, "address", "business_address")
    )

    return PreparedRecord(
        entity_id=entity_id,
        source=_source(record, entity_id),
        name=name,
        name_core=name_core,
        address=address,
        country=normalize_country(record.get("country")),
        name_tokens=tuple(_tokens(name)),
        name_core_tokens=tuple(_tokens(name_core)),
        address_tokens=tuple(_tokens(address)),
        name_ngrams=frozenset(_ngrams(name)),
        address_ngrams=frozenset(_ngrams(address)),
        address_numeric_tokens=frozenset(_numeric_tokens(address)),
    )


def _character_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(fuzz.ratio(left, right) / 100.0)


def _levenshtein_similarity(left: str, right: str) -> float:
    """Return normalized Levenshtein similarity using RapidFuzz."""
    if not left or not right:
        return 0.0
    return float(fuzz.ratio(left, right) / 100.0)


def _tokens(value: str) -> list[str]:
    return value.split() if value else []


def _token_similarity(left: str, right: str) -> float:
    """Position-aware token-sequence similarity."""
    if not left or not right:
        return 0.0
    return float(fuzz.ratio(left, right) / 100.0)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _token_set_similarity(left: str, right: str) -> float:
    return _jaccard(set(_tokens(left)), set(_tokens(right)))


def _token_sort_similarity(left: str, right: str) -> float:
    left_sorted, right_sorted = " ".join(sorted(_tokens(left))), " ".join(sorted(_tokens(right)))
    return _levenshtein_similarity(left_sorted, right_sorted)


def _ngrams(value: str, size: int = 3) -> set[str]:
    compact = value.replace(" ", "")
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _ngram_similarity(left: str, right: str) -> float:
    return _jaccard(_ngrams(left), _ngrams(right))


def _numeric_tokens(value: str) -> set[str]:
    return {token for token in _tokens(value) if any(character.isdigit() for character in token)}


def _exact(left: str, right: str) -> float:
    """Empty values never count as matching evidence."""
    return float(bool(left and right and left == right))


def pair_features(source1: PreparedRecord | Mapping[str, Any], candidate: PreparedRecord | Mapping[str, Any]) -> dict[str, float]:
    """Create named numeric features for one already-blocked S1/candidate pair."""
    left = source1 if isinstance(source1, PreparedRecord) else prepare_record(source1)
    right = candidate if isinstance(candidate, PreparedRecord) else prepare_record(candidate)
    if left.source not in {"S1", "UNKNOWN"}:
        raise ValueError("The left record must be a Source 1 record")

    name_character = _character_similarity(left.name, right.name)
    address_character = _character_similarity(left.address, right.address)
    name_exact = _exact(left.name, right.name)
    address_exact = _exact(left.address, right.address)
    country_equal = _exact(left.country, right.country)
    name_token_set = _jaccard(
        set(left.name_core_tokens),
        set(right.name_core_tokens),
    )
    name_token_sort = _levenshtein_similarity(
        " ".join(sorted(left.name_core_tokens)),
        " ".join(sorted(right.name_core_tokens)),
    )
    name_ngram = _jaccard(left.name_ngrams, right.name_ngrams)
    address_token_set = _jaccard(
        set(left.address_tokens),
        set(right.address_tokens),
    )
    address_ngram = _jaccard(left.address_ngrams, right.address_ngrams)
    address_numeric_overlap = _jaccard(
        left.address_numeric_tokens,
        right.address_numeric_tokens,
    )

    features = {
        "name_exact": name_exact,
        "name_character_similarity": name_character,
        "name_levenshtein_similarity": _levenshtein_similarity(left.name, right.name),
        "name_token_similarity": _token_similarity(left.name, right.name),
        "name_token_set_similarity": name_token_set,
        "name_token_sort_similarity": name_token_sort,
        "name_ngram_similarity": name_ngram,
        "name_length_difference": float(abs(len(left.name) - len(right.name))),
        "address_exact": address_exact,
        "address_character_similarity": address_character,
        "address_token_similarity": _token_similarity(left.address, right.address),
        "address_token_set_similarity": address_token_set,
        "address_ngram_similarity": address_ngram,
        "address_numeric_token_overlap": address_numeric_overlap,
        "address_length_difference": float(abs(len(left.address) - len(right.address))),
        "country_equal": country_equal,
        "name_left_missing": float(not left.name),
        "name_right_missing": float(not right.name),
        "name_any_missing": float(not left.name or not right.name),
        "address_left_missing": float(not left.address),
        "address_right_missing": float(not right.address),
        "address_any_missing": float(not left.address or not right.address),
        "country_left_missing": float(not left.country),
        "country_right_missing": float(not right.country),
        "country_any_missing": float(not left.country or not right.country),
        "target_is_source2": float(right.source == "S2"),
        "target_is_source3": float(right.source == "S3"),
        "name_address_exact_both": name_exact * address_exact,
        "name_address_character_mean": (name_character + address_character) / 2,
        "name_address_character_min": min(name_character, address_character),
        "name_address_evidence_product": name_character * address_character,
        "country_and_name_exact": country_equal * name_exact,
        "country_and_address_exact": country_equal * address_exact,
    }
    return {name: float(features[name]) for name in FEATURE_COLUMNS}


def build_feature_matrix(
    source1_records: Sequence[Mapping[str, Any]],
    source2_records: Sequence[Mapping[str, Any]],
    source3_records: Sequence[Mapping[str, Any]],
    candidates: Any,
) -> FeatureMatrix:
    """Build a numerical matrix only for final candidates returned by blocking.

    ``candidates`` must provide ``s1_to_s2`` and ``s1_to_s3`` mappings, such as
    :class:`src.blocking.BlockingResult`. Missing or cross-source IDs are errors,
    preventing silent divergence between candidate generation and feature work.
    """
    def prepare_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, PreparedRecord]:
        prepared_by_id: dict[str, PreparedRecord] = {}
        for record in records:
            prepared = prepare_record(record)
            if prepared.entity_id in prepared_by_id:
                raise ValueError(f"Duplicate entity_id: {prepared.entity_id}")
            prepared_by_id[prepared.entity_id] = prepared
        return prepared_by_id

    source1_by_id = prepare_by_id(source1_records)
    source2_by_id = prepare_by_id(source2_records)
    source3_by_id = prepare_by_id(source3_records)
    source1_ids: list[str] = []
    candidate_ids: list[str] = []
    values: list[tuple[float, ...]] = []

    for source1_id in sorted(source1_by_id):
        left = source1_by_id[source1_id]
        for target_id in candidates.s1_to_s2.get(source1_id, ()):
            if target_id not in source2_by_id:
                raise KeyError(f"Candidate {target_id} is not in Source 2")
            features = pair_features(left, source2_by_id[target_id])
            source1_ids.append(source1_id)
            candidate_ids.append(target_id)
            values.append(tuple(features[name] for name in FEATURE_COLUMNS))
        for target_id in candidates.s1_to_s3.get(source1_id, ()):
            if target_id not in source3_by_id:
                raise KeyError(f"Candidate {target_id} is not in Source 3")
            features = pair_features(left, source3_by_id[target_id])
            source1_ids.append(source1_id)
            candidate_ids.append(target_id)
            values.append(tuple(features[name] for name in FEATURE_COLUMNS))
    return FeatureMatrix(tuple(source1_ids), tuple(candidate_ids), FEATURE_COLUMNS, tuple(values))
