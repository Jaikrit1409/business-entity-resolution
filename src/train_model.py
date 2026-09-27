"""Train and evaluate a precision-oriented LightGBM entity-matching baseline.

The model is trained only on candidates emitted by the project's blocker. Its
decision threshold is selected using macro F0.5 per Source 1 entity, matching
the challenge objective rather than ordinary binary classification accuracy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .blocking import BlockingConfig, BlockingResult, MultiStrategyBlocker, ground_truth_pairs
    from .feature_engineering import build_feature_matrix
    from .pair_generator import construct_labeled_pairs, split_source1_entities
except ImportError:  # pragma: no cover - supports direct script execution
    from blocking import BlockingConfig, BlockingResult, MultiStrategyBlocker, ground_truth_pairs
    from feature_engineering import build_feature_matrix
    from pair_generator import construct_labeled_pairs, split_source1_entities


DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True)
class ThresholdMetrics:
    """Pair diagnostics plus the challenge-style entity-level result."""

    threshold: float
    pair_precision: float
    pair_recall: float
    pair_predicted_positive_count: int
    macro_f0_5: float
    validation_source1_count: int


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Paths and validation measurements needed to reproduce a training run."""

    model_path: Path
    config_path: Path
    thresholds_path: Path
    selected_threshold: float
    threshold_metrics: tuple[ThresholdMetrics, ...]
    train_pair_count: int
    validation_pair_count: int
    train_source1_count: int
    validation_source1_count: int


def _entity_id(record: Mapping[str, Any]) -> str:
    value = "" if record.get("entity_id") is None else str(record["entity_id"]).strip()
    if not value:
        raise ValueError("Every source record must have a non-empty entity_id")
    return value


def macro_f0_5(
    source1_ids: Sequence[str],
    true_pairs: set[tuple[str, str]],
    predicted_pairs: set[tuple[str, str]],
) -> float:
    """Compute challenge-style macro F0.5, including singleton Source 1 entities."""
    if not source1_ids:
        return 0.0
    true_by_source1: dict[str, set[str]] = defaultdict(set)
    predicted_by_source1: dict[str, set[str]] = defaultdict(set)
    for source1_id, candidate_id in true_pairs:
        true_by_source1[source1_id].add(candidate_id)
    for source1_id, candidate_id in predicted_pairs:
        predicted_by_source1[source1_id].add(candidate_id)

    scores: list[float] = []
    for source1_id in source1_ids:
        truth, predicted = true_by_source1[source1_id], predicted_by_source1[source1_id]
        if not truth and not predicted:
            scores.append(1.0)
            continue
        if not truth or not predicted:
            scores.append(0.0)
            continue
        true_positive = len(truth & predicted)
        precision = true_positive / len(predicted)
        recall = true_positive / len(truth)
        scores.append((1.25 * precision * recall) / (0.25 * precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def pair_precision_recall(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> tuple[float, float, int]:
    """Return pair-level diagnostics; these are not used as the optimization target."""
    predictions = [probability >= threshold for probability in probabilities]
    true_positive = sum(label == 1 and predicted for label, predicted in zip(labels, predictions, strict=True))
    false_positive = sum(label == 0 and predicted for label, predicted in zip(labels, predictions, strict=True))
    false_negative = sum(label == 1 and not predicted for label, predicted in zip(labels, predictions, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return precision, recall, sum(predictions)


def evaluate_thresholds(
    validation_source1_ids: Sequence[str],
    validation_pair_source1_ids: Sequence[str],
    validation_candidate_ids: Sequence[str],
    validation_labels: Sequence[int],
    validation_probabilities: Sequence[float],
    true_pairs: set[tuple[str, str]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[ThresholdMetrics, ...]:
    """Evaluate thresholds against pair diagnostics and macro Source 1 F0.5."""
    if not (len(validation_pair_source1_ids) == len(validation_candidate_ids) == len(validation_labels) == len(validation_probabilities)):
        raise ValueError("Validation labels, probabilities, and pair IDs must have equal lengths")
    reports: list[ThresholdMetrics] = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between zero and one")
        precision, recall, predicted_count = pair_precision_recall(validation_labels, validation_probabilities, threshold)
        predicted_pairs = {
            (source1_id, candidate_id)
            for source1_id, candidate_id, probability in zip(validation_pair_source1_ids, validation_candidate_ids, validation_probabilities, strict=True)
            if probability >= threshold
        }
        reports.append(ThresholdMetrics(
            threshold=float(threshold),
            pair_precision=precision,
            pair_recall=recall,
            pair_predicted_positive_count=predicted_count,
            macro_f0_5=macro_f0_5(validation_source1_ids, true_pairs, predicted_pairs),
            validation_source1_count=len(validation_source1_ids),
        ))
    return tuple(reports)


def _select_threshold(reports: Sequence[ThresholdMetrics]) -> ThresholdMetrics:
    if not reports:
        raise ValueError("At least one threshold is required")
    # Precision breaks an F0.5 tie; a higher threshold then favors safer merges.
    return max(reports, key=lambda report: (report.macro_f0_5, report.pair_precision, report.threshold))


def train_baseline_model(
    source1_records: Sequence[Mapping[str, Any]],
    source2_records: Sequence[Mapping[str, Any]],
    source3_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "models",
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    blocking_config: BlockingConfig | None = None,
) -> BaselineTrainingResult:
    """Fit LightGBM on blocked training candidates and persist model/configuration.

    LightGBM is MIT licensed. Importing it happens here, so scoring utilities and
    unit tests do not require the optional training dependency to be installed.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install the training dependency with `python -m pip install lightgbm`.") from error

    source1_ids = tuple(_entity_id(record) for record in source1_records)
    if len(source1_ids) != len(set(source1_ids)):
        raise ValueError("Source 1 entity IDs must be unique")
    train_source1_ids, validation_source1_ids = split_source1_entities(source1_ids, validation_fraction, random_seed)
    if not train_source1_ids or not validation_source1_ids:
        raise ValueError("At least two Source 1 entities are required for train/validation modeling")

    config = blocking_config or BlockingConfig()
    candidates = MultiStrategyBlocker(config).generate(source1_records, source2_records, source3_records)
    labeled_pairs = construct_labeled_pairs(candidates, source1_records, source2_records, source3_records, ground_truth_records)
    label_by_pair = {(pair.source1_entity_id, pair.candidate_entity_id): pair.label for pair in labeled_pairs}
    matrix = build_feature_matrix(source1_records, source2_records, source3_records, candidates)
    if not matrix.values:
        raise ValueError("Blocking produced no training candidates; cannot train a supervised model")

    labels = np.asarray([
        label_by_pair[(source1_id, candidate_id)]
        for source1_id, candidate_id in zip(matrix.source1_ids, matrix.candidate_ids, strict=True)
    ], dtype=int)
    features = np.asarray(matrix.values, dtype=float)
    train_set, validation_set = set(train_source1_ids), set(validation_source1_ids)
    train_mask = np.asarray([source1_id in train_set for source1_id in matrix.source1_ids])
    validation_mask = np.asarray([source1_id in validation_set for source1_id in matrix.source1_ids])
    if len(set(labels[train_mask])) < 2:
        raise ValueError("Training candidates need at least one positive and one negative pair")

    positives, negatives = int(labels[train_mask].sum()), int((labels[train_mask] == 0).sum())
    scale_pos_weight = negatives / positives if positives else 1.0
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(features[train_mask], labels[train_mask])
    validation_probabilities = model.predict_proba(features[validation_mask])[:, 1]
    true_pairs = ground_truth_pairs(ground_truth_records, source1_records, source2_records, source3_records)
    validation_indices = np.flatnonzero(validation_mask)
    reports = evaluate_thresholds(
        validation_source1_ids,
        [matrix.source1_ids[index] for index in validation_indices],
        [matrix.candidate_ids[index] for index in validation_indices],
        labels[validation_mask].tolist(),
        validation_probabilities.tolist(),
        true_pairs,
        thresholds,
    )
    selected = _select_threshold(reports)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "lightgbm.txt"
    config_path = output_path / "baseline_config.json"
    thresholds_path = output_path / "thresholds.json"
    model.booster_.save_model(str(model_path))
    reproducibility = {
        "model": "LightGBM LGBMClassifier (MIT license)",
        "random_seed": random_seed,
        "validation_fraction": validation_fraction,
        "feature_names": list(matrix.feature_names),
        "blocking_config": asdict(config),
        "preprocessing": {
            "module": "src.preprocess",
            "fields": ["name_normalized", "name_core_normalized", "address_normalized", "country_normalized"],
            "external_data": False,
        },
        "selected_threshold": selected.threshold,
        "train_source1_count": len(train_source1_ids),
        "validation_source1_count": len(validation_source1_ids),
        "train_pair_count": int(train_mask.sum()),
        "validation_pair_count": int(validation_mask.sum()),
    }
    config_path.write_text(json.dumps(reproducibility, indent=2, sort_keys=True), encoding="utf-8")
    thresholds_path.write_text(json.dumps([asdict(report) for report in reports], indent=2), encoding="utf-8")
    return BaselineTrainingResult(
        model_path=model_path,
        config_path=config_path,
        thresholds_path=thresholds_path,
        selected_threshold=selected.threshold,
        threshold_metrics=reports,
        train_pair_count=int(train_mask.sum()),
        validation_pair_count=int(validation_mask.sum()),
        train_source1_count=len(train_source1_ids),
        validation_source1_count=len(validation_source1_ids),
    )


def train_large_model_from_tsv(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    ground_truth_path: str | Path,
    candidate_s2_path: str | Path,
    candidate_s3_path: str | Path,
    output_dir: str | Path = "models",
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    sample_per_source1: int = 50,
    chunk_size: int = 100_000,
) -> BaselineTrainingResult:
    """Train LightGBM from large TSV datasets without materializing all pairs.

    Candidate files must be the final candidate sets emitted by the large-data
    blocker. Training samples candidates per Source 1 entity so the feature
    matrix remains bounded in memory.
    """
    try:
        import duckdb
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Large-data training requires duckdb, pandas, numpy and lightgbm."
        ) from error

    source1_path = Path(source1_path)
    source2_path = Path(source2_path)
    source3_path = Path(source3_path)
    ground_truth_path = Path(ground_truth_path)
    candidate_s2_path = Path(candidate_s2_path)
    candidate_s3_path = Path(candidate_s3_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA preserve_insertion_order=false")

    # DuckDB owns the large joins. Python only receives bounded chunks.
    def register_tsv(name: str, path: Path) -> None:
        escaped = str(path.resolve()).replace("\\", "/").replace("'", "''")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT *
            FROM read_csv(
                '{escaped}',
                delim='\\t',
                header=true,
                quote='',
                escape='',
                nullstr=''
            )
            """
        )

    register_tsv("s1", source1_path)
    register_tsv("s2", source2_path)
    register_tsv("s3", source3_path)
    register_tsv("truth", ground_truth_path)
    register_tsv("cand_s2", candidate_s2_path)
    register_tsv("cand_s3", candidate_s3_path)

    # Candidate files contain pair-per-row records.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE all_candidates AS
        SELECT source1_id, target_id, 'S2' AS target_source
        FROM cand_s2

        UNION ALL

        SELECT source1_id, target_id, 'S3' AS target_source
        FROM cand_s3
        """
    )

    # Split Source 1 deterministically. This avoids leaking the same entity
    # into both training and validation.
    s1_ids = [
        row[0]
        for row in con.execute(
            "SELECT entity_id FROM s1 ORDER BY entity_id"
        ).fetchall()
    ]

    if len(s1_ids) < 2:
        raise ValueError("At least two Source 1 entities are required.")

    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(s1_ids, dtype=object)
    rng.shuffle(shuffled)

    validation_count = max(
        1,
        int(round(len(shuffled) * validation_fraction))
    )
    validation_ids = set(shuffled[:validation_count].tolist())
    train_ids = set(shuffled[validation_count:].tolist())

    if not train_ids or not validation_ids:
        raise ValueError("Train/validation split produced an empty partition.")

    # Ground truth is converted to a compact lookup table inside DuckDB.
    # Empty/null match lists produce no positive rows.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE truth_pairs AS
        SELECT
            source1_entity_id AS source1_id,
            trim(target_id) AS target_id
        FROM truth,
        UNNEST(
            string_split(
                coalesce(matched_entity_ids, ''),
                ','
            )
        ) AS u(target_id)
        WHERE trim(target_id) <> ''
        """
    )

    # Mark candidates as positive/negative inside DuckDB.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE labeled_candidates AS
        SELECT
            c.source1_id,
            c.target_id,
            c.target_source,
            CASE
                WHEN t.source1_id IS NOT NULL THEN 1
                ELSE 0
            END AS label
        FROM all_candidates c
        LEFT JOIN truth_pairs t
          ON t.source1_id = c.source1_id
         AND t.target_id = c.target_id
        """
    )

    # Sample negatives per Source 1 entity while retaining every available
    # positive candidate. This is especially important because the challenge
    # is precision-heavy and raw blocked sets are highly imbalanced.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE sampled_train AS
        WITH positives AS (
            SELECT *
            FROM labeled_candidates
            WHERE source1_id IN (
                SELECT * FROM UNNEST(?)
            )
            AND label = 1
        ),
        negatives AS (
            SELECT *
            FROM labeled_candidates
            WHERE source1_id IN (
                SELECT * FROM UNNEST(?)
            )
            AND label = 0
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY source1_id
                ORDER BY hash(target_id)
            ) <= {int(sample_per_source1)}
        )
        SELECT * FROM positives
        UNION ALL
        SELECT * FROM negatives
        """,
        [list(train_ids), list(train_ids)],
    )

    # Validation keeps the complete candidate set. It is smaller than the
    # training universe after the Source-1 split and is required for proper
    # entity-level threshold selection.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE sampled_validation AS
        SELECT *
        FROM labeled_candidates
        WHERE source1_id IN (
            SELECT * FROM UNNEST(?)
        )
        """,
        [list(validation_ids)],
    )

    # Load records needed by the current chunk only.
    def feature_chunk(query: str):
        frame = con.execute(query).fetch_df()

        if frame.empty:
            return None

        left_records = []
        s2_records = []
        s3_records = []
        s1_to_s2 = defaultdict(list)
        s1_to_s3 = defaultdict(list)

        for row in frame.itertuples(index=False):
            left_records.append({
                "entity_id": row.source1_id,
                "business_name": row.s1_name,
                "business_address": row.s1_address,
                "country": row.s1_country,
            })

            target_record = {
                "entity_id": row.target_id,
                "business_name": row.target_name,
                "business_address": row.target_address,
                "country": row.target_country,
            }

            if row.target_source == "S2":
                s2_records.append(target_record)
                s1_to_s2[row.source1_id].append(row.target_id)
            else:
                s3_records.append(target_record)
                s1_to_s3[row.source1_id].append(row.target_id)

        left_records = list({
            r["entity_id"]: r for r in left_records
        }.values())

        s2_records = list({
            r["entity_id"]: r for r in s2_records
        }.values())

        s3_records = list({
            r["entity_id"]: r for r in s3_records
        }.values())

        candidates = BlockingResult(
            s1_to_s2={k: tuple(v) for k, v in s1_to_s2.items()},
            s1_to_s3={k: tuple(v) for k, v in s1_to_s3.items()},
        )

        matrix = build_feature_matrix(
            left_records,
            s2_records,
            s3_records,
            candidates,
        )

        return frame, matrix

    def joined_query(table: str, limit: int, last_source1_id=None, last_target_id=None, last_target_source=None) -> str:
        where = ""
        if last_source1_id is not None:
            escaped_s1 = str(last_source1_id).replace("'", "''")
            escaped_target = str(last_target_id).replace("'", "''")
            escaped_source = str(last_target_source).replace("'", "''")
            where = f"""
        WHERE (
            c.source1_id > '{escaped_s1}'
            OR (
                c.source1_id = '{escaped_s1}'
                AND c.target_id > '{escaped_target}'
            )
            OR (
                c.source1_id = '{escaped_s1}'
                AND c.target_id = '{escaped_target}'
                AND c.target_source > '{escaped_source}'
            )
        )
        """

        return f"""
        SELECT
            c.source1_id,
            c.target_id,
            c.target_source,
            c.label,
            s1.business_name AS s1_name,
            s1.business_address AS s1_address,
            s1.country AS s1_country,
            CASE
                WHEN c.target_source = 'S2' THEN s2.business_name
                ELSE s3.business_name
            END AS target_name,
            CASE
                WHEN c.target_source = 'S2' THEN s2.business_address
                ELSE s3.business_address
            END AS target_address,
            CASE
                WHEN c.target_source = 'S2' THEN s2.country
                ELSE s3.country
            END AS target_country
        FROM {table} c
        JOIN s1
          ON s1.entity_id = c.source1_id
        LEFT JOIN s2
          ON c.target_source = 'S2'
         AND s2.entity_id = c.target_id
        LEFT JOIN s3
          ON c.target_source = 'S3'
         AND s3.entity_id = c.target_id
        {where}
        ORDER BY c.source1_id, c.target_id, c.target_source
        LIMIT {limit}
        """

    def iterate_joined(table: str, total_count: int):
        last_source1_id = None
        last_target_id = None
        last_target_source = None

        while True:
            query = joined_query(
                table,
                chunk_size,
                last_source1_id,
                last_target_id,
                last_target_source,
            )
            frame = con.execute(query).fetch_df()

            if frame.empty:
                break

            yield frame

            last_source1_id = str(frame.iloc[-1]["source1_id"])
            last_target_id = str(frame.iloc[-1]["target_id"])
            last_target_source = str(frame.iloc[-1]["target_source"])

            if len(frame) < chunk_size:
                break

    # Collect a bounded training matrix.
    train_frames = []
    train_feature_matrices = []
    train_labels = []

    train_count = con.execute(
        "SELECT COUNT(*) FROM sampled_train"
    ).fetchone()[0]

    def feature_frame(frame):
        if frame.empty:
            return None

        left_records = []
        s2_records = []
        s3_records = []
        s1_to_s2 = defaultdict(list)
        s1_to_s3 = defaultdict(list)

        for row in frame.itertuples(index=False):
            left_records.append({
                "entity_id": row.source1_id,
                "business_name": row.s1_name,
                "business_address": row.s1_address,
                "country": row.s1_country,
            })

            target_record = {
                "entity_id": row.target_id,
                "business_name": row.target_name,
                "business_address": row.target_address,
                "country": row.target_country,
            }

            if row.target_source == "S2":
                s2_records.append(target_record)
                s1_to_s2[row.source1_id].append(row.target_id)
            else:
                s3_records.append(target_record)
                s1_to_s3[row.source1_id].append(row.target_id)

        left_records = list({r["entity_id"]: r for r in left_records}.values())
        s2_records = list({r["entity_id"]: r for r in s2_records}.values())
        s3_records = list({r["entity_id"]: r for r in s3_records}.values())

        candidates = BlockingResult(
            s1_to_s2={k: tuple(v) for k, v in s1_to_s2.items()},
            s1_to_s3={k: tuple(v) for k, v in s1_to_s3.items()},
        )

        matrix = build_feature_matrix(
            left_records,
            s2_records,
            s3_records,
            candidates,
        )

        return matrix

    for frame in iterate_joined("sampled_train", train_count):
        matrix = feature_frame(frame)
        if matrix is None:
            continue

        train_frames.append(matrix)
        train_feature_matrices.append(matrix.values)

        label_lookup = {
            (str(row.source1_id), str(row.target_id), str(row.target_source)): int(row.label)
            for row in frame.itertuples(index=False)
        }

        source_by_pair = {
            (str(row.source1_id), str(row.target_id)): str(row.target_source)
            for row in frame.itertuples(index=False)
        }

        train_labels.extend(
            label_lookup[
                (
                    str(s1_id),
                    str(tid),
                    source_by_pair[(str(s1_id), str(tid))],
                )
            ]
            for s1_id, tid in zip(matrix.source1_ids, matrix.candidate_ids)
        )

    if not train_feature_matrices:
        raise ValueError("No training candidates were produced.")

    features = np.asarray(
        [row for chunk in train_feature_matrices for row in chunk],
        dtype=np.float32,
    )
    labels = np.asarray(train_labels, dtype=np.int8)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Training candidates need at least one positive and one negative pair."
        )

    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    scale_pos_weight = negatives / positives if positives else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=1,
        verbosity=-1,
    )

    model.fit(features, labels)

    # Validation is evaluated in chunks so it never becomes a giant feature
    # matrix.
    validation_count = con.execute(
        "SELECT COUNT(*) FROM sampled_validation"
    ).fetchone()[0]

    validation_source_ids = list(validation_ids)
    validation_pair_source1_ids = []
    validation_candidate_ids = []
    validation_labels = []
    validation_probabilities = []

    for frame in iterate_joined("sampled_validation", validation_count):
        result = feature_frame(frame)
        if result is None:
            continue

        matrix = result
        chunk_features = np.asarray(matrix.values, dtype=np.float32)
        probabilities = model.predict_proba(chunk_features)[:, 1]

        # Align predictions with build_feature_matrix() deterministic order.
        frame_lookup = {
            (
                str(row.source1_id),
                str(row.target_id),
                str(row.target_source),
            ): int(row.label)
            for row in frame.itertuples(index=False)
        }

        source_lookup = {
            (str(row.source1_id), str(row.target_id)): str(row.target_source)
            for row in frame.itertuples(index=False)
        }

        for s1_id, target_id, probability in zip(
            matrix.source1_ids,
            matrix.candidate_ids,
            probabilities,
            strict=True,
        ):
            pair = (str(s1_id), str(target_id))
            target_source = source_lookup[pair]
            validation_pair_source1_ids.append(str(s1_id))
            validation_candidate_ids.append(str(target_id))
            validation_labels.append(
                frame_lookup[(str(s1_id), str(target_id), target_source)]
            )
            validation_probabilities.append(float(probability))

    # Build the true validation pairs directly from the compact truth table.
    true_rows = con.execute(
        """
        SELECT source1_id, target_id
        FROM truth_pairs
        WHERE source1_id IN (
            SELECT * FROM UNNEST(?)
        )
        """,
        [list(validation_ids)],
    ).fetchall()

    true_pairs = {
        (str(source1_id), str(target_id))
        for source1_id, target_id in true_rows
    }

    reports = evaluate_thresholds(
        validation_source_ids,
        validation_pair_source1_ids,
        validation_candidate_ids,
        validation_labels,
        validation_probabilities,
        true_pairs,
        thresholds,
    )

    selected = _select_threshold(reports)

    model_path = output_path / "lightgbm.txt"
    config_path = output_path / "baseline_config.json"
    thresholds_path = output_path / "thresholds.json"

    model.booster_.save_model(str(model_path))

    reproducibility = {
        "model": "LightGBM LGBMClassifier (MIT license)",
        "random_seed": random_seed,
        "validation_fraction": validation_fraction,
        "sample_per_source1": sample_per_source1,
        "chunk_size": chunk_size,
        "feature_names": list(train_frames[0].feature_names),
        "blocking_config": asdict(BlockingConfig(
            max_posting_size=25,
            max_candidates_per_source=200,
        )),
        "preprocessing": {
            "module": "src.preprocess",
            "fields": [
                "name_normalized",
                "name_core_normalized",
                "address_normalized",
                "country_normalized",
            ],
            "external_data": False,
        },
        "selected_threshold": selected.threshold,
        "train_source1_count": len(train_ids),
        "validation_source1_count": len(validation_ids),
        "train_pair_count": int(len(labels)),
        "validation_pair_count": int(validation_count),
    }

    config_path.write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    thresholds_path.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )

    con.close()

    return BaselineTrainingResult(
        model_path=model_path,
        config_path=config_path,
        thresholds_path=thresholds_path,
        selected_threshold=selected.threshold,
        threshold_metrics=reports,
        train_pair_count=len(labels),
        validation_pair_count=int(validation_count),
        train_source1_count=len(train_ids),
        validation_source1_count=len(validation_ids),
    )
