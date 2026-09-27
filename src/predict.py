"""Score blocked candidate record pairs using a trained LightGBM model."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .blocking import MultiStrategyBlocker
from .feature_engineering import build_feature_matrix


def load_model(model_path: str | Path):
    """Load a LightGBM Booster from a saved model."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("Install lightgbm before scoring.") from error

    return lgb.Booster(model_file=str(model_path))


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def score_candidates(
    source1_records: Sequence[Mapping[str, Any]],
    source2_records: Sequence[Mapping[str, Any]],
    source3_records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
    config_path: str | Path,
) -> tuple[Any, Any]:
    """Generate candidates and return their feature matrix plus probabilities."""
    config = load_model_config(config_path)
    model = load_model(model_path)

    blocker = MultiStrategyBlocker(
        config=config["blocking_config"]
    ) if False else None

    from .blocking import BlockingConfig

    blocking_config = BlockingConfig(**config["blocking_config"])
    candidates = MultiStrategyBlocker(blocking_config).generate(
        source1_records,
        source2_records,
        source3_records,
    )

    matrix = build_feature_matrix(
        source1_records,
        source2_records,
        source3_records,
        candidates,
    )

    if not matrix.values:
        return matrix, []

    import numpy as np

    features = np.asarray(matrix.values, dtype=float)
    probabilities = model.predict(features)

    return matrix, probabilities.tolist()


def select_matches(
    matrix: Any,
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, list[str]]:
    """Convert candidate probabilities into S1 -> matched IDs."""
    results: dict[str, list[str]] = {}

    for source1_id, candidate_id, probability in zip(
        matrix.source1_ids,
        matrix.candidate_ids,
        probabilities,
        strict=True,
    ):
        if probability >= threshold:
            results.setdefault(source1_id, []).append(candidate_id)

    return results
