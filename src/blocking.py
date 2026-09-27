"""Scalable deterministic candidate generation for business entity resolution.

The blocker supports two modes:

1. ``generate(...)`` for in-memory records and unit tests.
2. ``generate_from_tsv(...)`` for the real multi-million-row challenge data.

The large-data path uses DuckDB and SQL-based blocking rather than creating
millions of Python ``set`` objects.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

try:
    from .preprocess import (
        normalize_address,
        normalize_business_name,
        normalize_country,
    )
except ImportError:  # pragma: no cover
    from preprocess import (
        normalize_address,
        normalize_business_name,
        normalize_country,
    )


@dataclass(frozen=True)
class BlockingConfig:
    """Parameters controlling recall / candidate-volume trade-off."""

    ngram_size: int = 3
    min_token_length: int = 3
    max_posting_size: int = 25
    max_candidates_per_source: int = 40
    exact_max_posting_size: int = 100

    # Large-data blocking controls.
    rare_key_max_posting: int = 250
    max_address_tokens: int = 3
    max_name_tokens: int = 2
    max_name_ngrams: int = 3
    large_data_chunk_size: int = 100_000


@dataclass(frozen=True)
class BlockingResult:
    """Final S1-to-S2 and S1-to-S3 candidate IDs."""

    s1_to_s2: dict[str, tuple[str, ...]]
    s1_to_s3: dict[str, tuple[str, ...]]

    def combined(self) -> dict[str, tuple[str, ...]]:
        return {
            source1_id: tuple(
                sorted(
                    set(self.s1_to_s2[source1_id])
                    | set(self.s1_to_s3[source1_id])
                )
            )
            for source1_id in self.s1_to_s2
        }


@dataclass(frozen=True)
class BlockingValidation:
    true_match_recall: float | None
    true_match_count: int
    recovered_true_match_count: int
    average_candidates_per_source1: float
    median_candidates_per_source1: float
    maximum_candidates_per_source1: int
    candidate_reduction_ratio: float
    lost_true_matches: dict[str, tuple[str, ...]]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _field(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _entity_id(record: Mapping[str, Any]) -> str:
    entity_id = _text(record.get("entity_id"))
    if not entity_id:
        raise ValueError(
            "Every record used for blocking must have a non-empty entity_id"
        )
    return entity_id


def _tokens(value: str, min_length: int) -> set[str]:
    return {
        token
        for token in value.split()
        if len(token) >= min_length
    }


def _character_ngrams(value: str, ngram_size: int) -> set[str]:
    compact = value.replace(" ", "")
    if len(compact) < ngram_size:
        return set()
    return {
        compact[index:index + ngram_size]
        for index in range(len(compact) - ngram_size + 1)
    }


def _numeric_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if any(character.isdigit() for character in token)
    }


def _record_features(
    record: Mapping[str, Any],
    config: BlockingConfig,
) -> dict[str, Any]:
    name = normalize_business_name(
        _field(record, "name", "business_name")
    )
    address = normalize_address(
        _field(record, "address", "business_address")
    )
    country = normalize_country(record.get("country"))

    return {
        "id": _entity_id(record),
        "name": name,
        "address": address,
        "country": country,
        "name_tokens": _tokens(name, config.min_token_length),
        "address_tokens": _tokens(address, config.min_token_length),
        "name_ngrams": _character_ngrams(
            name,
            config.ngram_size,
        ),
        "numeric_address_tokens": _numeric_tokens(address),
    }


def _add(
    index: dict[Any, set[str]],
    key: Any,
    entity_id: str,
) -> None:
    if key:
        index[key].add(entity_id)


class MultiStrategyBlocker:
    """Candidate generator with a small-data and DuckDB large-data path."""

    def __init__(
        self,
        config: BlockingConfig | None = None,
    ) -> None:
        self.config = config or BlockingConfig()

        if (
            self.config.ngram_size < 2
            or self.config.max_posting_size < 1
            or self.config.max_candidates_per_source < 1
            or self.config.rare_key_max_posting < 1
        ):
            raise ValueError(
                "BlockingConfig limits must be positive; "
                "ngram_size must be at least 2"
            )

    # ------------------------------------------------------------------
    # Small-data compatibility implementation
    # ------------------------------------------------------------------

    def _build_indexes(
        self,
        target_records: Sequence[Mapping[str, Any]],
    ) -> tuple[
        dict[str, dict[Any, set[str]]],
        dict[str, dict[str, Any]],
    ]:
        indexes: dict[str, dict[Any, set[str]]] = {
            "exact_name": defaultdict(set),
            "country_name": defaultdict(set),
            "exact_address": defaultdict(set),
            "country_address": defaultdict(set),
            "name_token": defaultdict(set),
            "country_name_token": defaultdict(set),
            "name_ngram": defaultdict(set),
            "address_token": defaultdict(set),
            "numeric_address": defaultdict(set),
        }

        features_by_id: dict[str, dict[str, Any]] = {}

        for record in target_records:
            features = _record_features(record, self.config)
            entity_id = features["id"]

            if entity_id in features_by_id:
                raise ValueError(
                    f"Duplicate target entity_id: {entity_id}"
                )

            features_by_id[entity_id] = features

            _add(indexes["exact_name"], features["name"], entity_id)
            _add(
                indexes["country_name"],
                (features["country"], features["name"]),
                entity_id,
            )
            _add(
                indexes["exact_address"],
                features["address"],
                entity_id,
            )
            _add(
                indexes["country_address"],
                (features["country"], features["address"]),
                entity_id,
            )

            for token in features["name_tokens"]:
                _add(indexes["name_token"], token, entity_id)
                _add(
                    indexes["country_name_token"],
                    (features["country"], token),
                    entity_id,
                )

            for ngram in features["name_ngrams"]:
                _add(
                    indexes["name_ngram"],
                    ngram,
                    entity_id,
                )

            for token in features["address_tokens"]:
                _add(
                    indexes["address_token"],
                    token,
                    entity_id,
                )

            for token in features["numeric_address_tokens"]:
                _add(
                    indexes["numeric_address"],
                    token,
                    entity_id,
                )

        return indexes, features_by_id

    def _usable_posting(
        self,
        posting: set[str] | None,
        *,
        exact: bool = False,
    ) -> set[str]:
        if posting is None:
            return set()

        limit = (
            self.config.exact_max_posting_size
            if exact
            else self.config.max_posting_size
        )

        if len(posting) > limit:
            return set()

        return posting

    def _retrieve_one(
        self,
        record: Mapping[str, Any],
        indexes: dict[str, dict[Any, set[str]]],
    ) -> tuple[str, ...]:
        features = _record_features(record, self.config)
        evidence: dict[str, set[str]] = defaultdict(set)

        def collect(
            strategy: str,
            key: Any,
            *,
            exact: bool = False,
        ) -> None:
            posting = self._usable_posting(
                indexes[strategy].get(key),
                exact=exact,
            )
            for entity_id in posting:
                evidence[entity_id].add(strategy)

        collect(
            "exact_name",
            features["name"],
            exact=True,
        )
        collect(
            "country_name",
            (features["country"], features["name"]),
            exact=True,
        )
        collect(
            "exact_address",
            features["address"],
            exact=True,
        )
        collect(
            "country_address",
            (features["country"], features["address"]),
            exact=True,
        )

        for token in features["name_tokens"]:
            collect("name_token", token)
            collect(
                "country_name_token",
                (features["country"], token),
            )

        for ngram in features["name_ngrams"]:
            collect("name_ngram", ngram)

        for token in features["address_tokens"]:
            collect("address_token", token)

        for token in features["numeric_address_tokens"]:
            collect("numeric_address", token)

        weights = {
            "exact_name": 20,
            "country_name": 18,
            "exact_address": 20,
            "country_address": 18,
            "numeric_address": 10,
            "address_token": 8,
            "name_token": 4,
            "country_name_token": 3,
            "name_ngram": 2,
        }

        def score(entity_id: str) -> tuple[int, int, str]:
            signals = evidence[entity_id]
            return (
                sum(weights[s] for s in signals),
                len(signals),
                entity_id,
            )

        ranked = sorted(
            evidence,
            key=lambda entity_id: (
                -score(entity_id)[0],
                -score(entity_id)[1],
                entity_id,
            ),
        )

        return tuple(
            ranked[:self.config.max_candidates_per_source]
        )

    def generate_pair_candidates(
        self,
        source1_records: Sequence[Mapping[str, Any]],
        target_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[str, ...]]:
        indexes, _ = self._build_indexes(target_records)

        candidates: dict[str, tuple[str, ...]] = {}

        for record in source1_records:
            source1_id = _entity_id(record)

            if source1_id in candidates:
                raise ValueError(
                    f"Duplicate Source 1 entity_id: {source1_id}"
                )

            candidates[source1_id] = self._retrieve_one(
                record,
                indexes,
            )

        return candidates

    def generate(
        self,
        source1_records: Sequence[Mapping[str, Any]],
        source2_records: Sequence[Mapping[str, Any]],
        source3_records: Sequence[Mapping[str, Any]],
    ) -> BlockingResult:
        """Compatibility API for in-memory datasets."""

        return BlockingResult(
            s1_to_s2=self.generate_pair_candidates(
                source1_records,
                source2_records,
            ),
            s1_to_s3=self.generate_pair_candidates(
                source1_records,
                source3_records,
            ),
        )

    # ------------------------------------------------------------------
    # Large-data DuckDB implementation
    # ------------------------------------------------------------------

    def _connect(self, database_path: str | Path | None = None):
        if duckdb is None:
            raise RuntimeError(
                "DuckDB is required for large-data blocking. "
                "Install it with: pip install duckdb"
            )

        if database_path is None:
            return duckdb.connect(":memory:")

        return duckdb.connect(str(database_path))

    @staticmethod
    def _quote_path(path: str | Path) -> str:
        value = str(Path(path).resolve()).replace("'", "''")
        return f"'{value}'"

    def _create_raw_table(
        self,
        con,
        table_name: str,
        path: str | Path,
    ) -> None:
        path_sql = self._quote_path(path)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,
                CAST(business_name AS VARCHAR) AS business_name,
                CAST(business_address AS VARCHAR) AS business_address,
                CAST(country AS VARCHAR) AS country
            FROM read_csv(
                {path_sql},
                delim='\\t',
                header=true,
                quote='\"',
                escape='\"',
                null_padding=true,
                ignore_errors=false,
                auto_detect=true
            )
            """
        )

    def _normalize_table(
        self,
        con,
        source_table: str,
        output_table: str,
    ) -> None:
        """Normalize using SQL-safe string operations.

        The preprocessing module is deliberately retained for the Python
        compatibility path. For the large path we use equivalent conservative
        normalization operations in DuckDB to avoid materializing millions
        of Python dictionaries.
        """

        # The preprocessing rules are simple enough to express directly in
        # SQL: Unicode normalization itself is not guaranteed by every DuckDB
        # build, so lower/case-fold-compatible operations are used here.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                entity_id,

                lower(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                coalesce(business_name, ''),
                                '&',
                                ' and ',
                                'g'
                            ),
                            '[^[:alnum:]]+',
                            ' ',
                            'g'
                        ),
                        '\\\\s+',
                        ' ',
                        'g'
                    )
                ) AS name,

                lower(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                coalesce(business_address, ''),
                                '&',
                                ' and ',
                                'g'
                            ),
                            '[^[:alnum:]]+',
                            ' ',
                            'g'
                        ),
                        '\\\\s+',
                        ' ',
                        'g'
                    )
                ) AS address,

                lower(
                    regexp_replace(
                        regexp_replace(
                            coalesce(country, ''),
                            '[^[:alnum:]]+',
                            ' ',
                            'g'
                        ),
                        '\\\\s+',
                        ' ',
                        'g'
                    )
                ) AS country

            FROM {source_table}
            """
        )

    def _make_keys(
        self,
        con,
        input_table: str,
        output_table: str,
    ) -> None:
        """Create a compact blocking-key table.

        Only entity_id, strategy and key are stored.  The previous
        implementation repeated name/address/country on every key row,
        creating a very large intermediate relation on the full dataset.
        """

        min_len = self.config.min_token_length
        max_addr = self.config.max_address_tokens
        max_name = self.config.max_name_tokens
        rare_cap = self.config.rare_key_max_posting

        # Exact keys are cheap and remain useful for high-confidence matches.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                entity_id,
                'exact_name' AS strategy,
                name AS key
            FROM {input_table}
            WHERE name <> ''

            UNION ALL

            SELECT
                entity_id,
                'country_name' AS strategy,
                country || '|' || name AS key
            FROM {input_table}
            WHERE name <> '' AND country <> ''

            UNION ALL

            SELECT
                entity_id,
                'exact_address' AS strategy,
                address AS key
            FROM {input_table}
            WHERE address <> ''

            UNION ALL

            SELECT
                entity_id,
                'country_address' AS strategy,
                country || '|' || address AS key
            FROM {input_table}
            WHERE address <> '' AND country <> ''
            """
        )

        # Address tokens: first calculate frequencies, then retain only rare
        # tokens and at most max_addr tokens per entity.
        address_freq = f"{output_table}_address_freq"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {address_freq} AS
            SELECT
                token,
                COUNT(*) AS freq
            FROM (
                SELECT
                    entity_id,
                    unnest(
                        regexp_split_to_array(
                            regexp_replace(trim(address), '\\s+', ' ', 'g'),
                            ' '
                        )
                    ) AS token
                FROM {input_table}
                WHERE address <> ''
            ) x
            WHERE length(token) >= {min_len}
            GROUP BY token
            HAVING COUNT(*) <= {rare_cap}
            """
        )

        address_tokens = f"{output_table}_address_tokens"

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {address_tokens} AS
            WITH exploded AS (
                SELECT
                    entity_id,
                    unnest(
                        regexp_split_to_array(
                            regexp_replace(trim(address), '\\s+', ' ', 'g'),
                            ' '
                        )
                    ) AS token
                FROM {input_table}
                WHERE address <> ''
            ),
            rare AS (
                SELECT
                    e.entity_id,
                    e.token,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.entity_id
                        ORDER BY f.freq, e.token
                    ) AS rn
                FROM exploded e
                JOIN {address_freq} f
                  ON e.token = f.token
                WHERE length(e.token) >= {min_len}
            )
            SELECT entity_id, token
            FROM rare
            WHERE rn <= {max_addr}
            """
        )

        con.execute(
            f"""
            INSERT INTO {output_table}
            SELECT
                entity_id,
                'address_token',
                token
            FROM {address_tokens}

            UNION ALL

            SELECT
                a.entity_id,
                'country_address_token',
                i.country || '|' || a.token
            FROM {address_tokens} a
            JOIN {input_table} i USING (entity_id)
            WHERE i.country <> ''
            """
        )

        # Name tokens.
        name_freq = f"{output_table}_name_freq"

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {name_freq} AS
            SELECT
                token,
                COUNT(*) AS freq
            FROM (
                SELECT
                    entity_id,
                    unnest(
                        regexp_split_to_array(
                            regexp_replace(trim(name), '\\s+', ' ', 'g'),
                            ' '
                        )
                    ) AS token
                FROM {input_table}
                WHERE name <> ''
            ) x
            WHERE length(token) >= {min_len}
            GROUP BY token
            HAVING COUNT(*) <= {rare_cap}
            """
        )

        name_tokens = f"{output_table}_name_tokens"

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {name_tokens} AS
            WITH exploded AS (
                SELECT
                    entity_id,
                    unnest(
                        regexp_split_to_array(
                            regexp_replace(trim(name), '\\s+', ' ', 'g'),
                            ' '
                        )
                    ) AS token
                FROM {input_table}
                WHERE name <> ''
            ),
            rare AS (
                SELECT
                    e.entity_id,
                    e.token,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.entity_id
                        ORDER BY f.freq, e.token
                    ) AS rn
                FROM exploded e
                JOIN {name_freq} f
                  ON e.token = f.token
                WHERE length(e.token) >= {min_len}
            )
            SELECT entity_id, token
            FROM rare
            WHERE rn <= {max_name}
            """
        )

        con.execute(
            f"""
            INSERT INTO {output_table}
            SELECT
                entity_id,
                'name_token',
                token
            FROM {name_tokens}

            UNION ALL

            SELECT
                n.entity_id,
                'country_name_token',
                i.country || '|' || n.token
            FROM {name_tokens} n
            JOIN {input_table} i USING (entity_id)
            WHERE i.country <> ''
            """
        )

        # Temporary helper tables are no longer needed.
        for table in (
            address_freq,
            address_tokens,
            name_freq,
            name_tokens,
        ):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    def _generate_large_direction(
        self,
        con,
        source1_table: str,
        target_table: str,
        output_table: str,
    ) -> None:
        """Generate bounded candidates using one blocking strategy at a time."""

        target_cap = min(
            self.config.max_posting_size,
            self.config.rare_key_max_posting,
        )

        per_strategy_cap = max(
            8,
            self.config.max_candidates_per_source // 3,
        )

        # Keep only the compact key columns needed by each strategy.
        strategies = [
            ("exact_name", "name", 20),
            ("country_name", "country || '|' || name", 18),
            ("exact_address", "address", 20),
            ("country_address", "country || '|' || address", 18),
        ]

        token_strategies = [
            ("address_token", 8),
            ("country_address_token", 10),
            ("name_token", 4),
            ("country_name_token", 3),
            ("name_ngram", 6),
        ]

        accumulator = f"{output_table}_accumulator"

        # Start with an empty accumulator. This lets every strategy follow
        # exactly the same bounded path.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {accumulator} (
                source1_id VARCHAR,
                target_id VARCHAR,
                score DOUBLE,
                signal_count INTEGER
            )
            """
        )

        def merge_strategy(result_table: str) -> None:
            """Merge one already-bounded strategy into the accumulator."""

            next_table = f"{output_table}_merge"

            con.execute(
                f"""
                CREATE OR REPLACE TABLE {next_table} AS
                WITH combined AS (
                    SELECT
                        source1_id,
                        target_id,
                        score,
                        signal_count
                    FROM {accumulator}

                    UNION ALL

                    SELECT
                        source1_id,
                        target_id,
                        score,
                        1 AS signal_count
                    FROM {result_table}
                ),
                aggregated AS (
                    SELECT
                        source1_id,
                        target_id,
                        SUM(score) AS score,
                        SUM(signal_count) AS signal_count
                    FROM combined
                    GROUP BY source1_id, target_id
                )
                SELECT
                    source1_id,
                    target_id,
                    score,
                    signal_count
                FROM aggregated
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY source1_id
                    ORDER BY score DESC,
                             signal_count DESC,
                             target_id
                ) <= {self.config.max_candidates_per_source}
                """
            )

            con.execute(f"DROP TABLE {accumulator}")
            con.execute(
                f"ALTER TABLE {next_table} RENAME TO {accumulator}"
            )

        # ---------------------------------------------------------------
        # Exact-field strategies
        # ---------------------------------------------------------------
        for strategy, expression, weight in strategies:
            target_keys = f"{output_table}_target_keys"

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {target_keys} AS
                SELECT
                    entity_id AS target_id,
                    {expression} AS block_key
                FROM {target_table}
                WHERE {expression} <> ''
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY {expression}
                    ORDER BY entity_id
                ) <= {target_cap}
                """
            )

            result_table = f"{output_table}_strategy_result"

            # QUALIFY applies the per-S1 limit immediately after the join.
            # No separate unbounded "joined" CTE is materialized.
            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {result_table} AS
                SELECT
                    s.entity_id AS source1_id,
                    t.target_id,
                    CAST(
                        {weight}
                        + CASE
                            WHEN s.country <> ''
                             AND s.country = split_part(t.block_key, '|', 1)
                            THEN 2
                            ELSE 0
                          END
                        AS DOUBLE
                    ) AS score
                FROM {source1_table} s
                JOIN {target_keys} t
                  ON {expression} = t.block_key
                WHERE {expression} <> ''
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.entity_id
                    ORDER BY
                        score DESC,
                        t.target_id
                ) <= {per_strategy_cap}
                """
            )

            merge_strategy(result_table)

            con.execute(f"DROP TABLE {result_table}")
            con.execute(f"DROP TABLE {target_keys}")

        # ---------------------------------------------------------------
        # Token strategies
        # ---------------------------------------------------------------
        for strategy, weight in token_strategies:
            target_keys = f"{output_table}_target_keys"

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {target_keys} AS
                SELECT
                    entity_id AS target_id,
                    key AS block_key
                FROM {target_table}_keys
                WHERE strategy = '{strategy}'
                  AND key <> ''
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY key
                    ORDER BY entity_id
                ) <= {target_cap}
                """
            )

            result_table = f"{output_table}_strategy_result"

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {result_table} AS
                SELECT
                    s.entity_id AS source1_id,
                    t.target_id,
                    CAST({weight} AS DOUBLE) AS score
                FROM {source1_table}_keys s
                JOIN {target_keys} t
                  ON s.key = t.block_key
                WHERE s.strategy = '{strategy}'
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.entity_id
                    ORDER BY t.target_id
                ) <= {per_strategy_cap}
                """
            )

            merge_strategy(result_table)

            con.execute(f"DROP TABLE {result_table}")
            con.execute(f"DROP TABLE {target_keys}")

        # Final output is deliberately pair-per-row internally.
        # generate_large_to_tsv converts this into the submission format.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                source1_id,
                target_id
            FROM {accumulator}
            """
        )

        con.execute(f"DROP TABLE {accumulator}")
    def generate_large_to_tsv(
        self,
        source1_path: str | Path,
        source2_path: str | Path,
        source3_path: str | Path,
        output_dir: str | Path = "output/candidates",
        *,
        database_path: str | Path | None = None,
    ) -> dict[str, Path]:
        """Generate large candidate files with compact bounded memory."""
        import csv
        from collections import defaultdict

        import numpy as np
        import pandas as pd

        del database_path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        s1_path = Path(source1_path)
        s2_path = Path(source2_path)
        s3_path = Path(source3_path)

        s2_out = output_dir / "candidate_pairs_s2.tsv"
        s3_out = output_dir / "candidate_pairs_s3.tsv"

        for output_path in (s2_out, s3_out):
            if output_path.exists():
                output_path.unlink()

        s2_fh = s2_out.open("w", encoding="utf-8", newline="")
        s3_fh = s3_out.open("w", encoding="utf-8", newline="")

        s2_writer = csv.writer(s2_fh, delimiter="\t")
        s3_writer = csv.writer(s3_fh, delimiter="\t")

        header = ["source1_id", "target_id", "target_source"]
        s2_writer.writerow(header)
        s3_writer.writerow(header)

        block_size = 500_000
        top_k = self.config.max_candidates_per_source
        posting_limit = self.config.rare_key_max_posting

        def normalize_tokens(value):
            return [
                token
                for token in str(value).split()
                if len(token) >= self.config.min_token_length
            ]

        def build_indexes(s1_block):
            indexes = {
                "exact_name": defaultdict(list),
                "exact_address": defaultdict(list),
                "country_name": defaultdict(list),
                "country_address": defaultdict(list),
                "name_token": defaultdict(list),
                "country_name_token": defaultdict(list),
                "address_token": defaultdict(list),
                "country_address_token": defaultdict(list),
            }

            s1_ids = []

            for local_idx, row in enumerate(
                s1_block.to_dict("records")
            ):
                sid = str(row["entity_id"])
                s1_ids.append(sid)

                name = normalize_business_name(
                    row.get("business_name", "")
                )
                address = normalize_address(
                    row.get("business_address", "")
                )
                country = normalize_country(
                    row.get("country", "")
                )

                if name:
                    indexes["exact_name"][name].append(local_idx)

                    if country:
                        indexes["country_name"][
                            (country, name)
                        ].append(local_idx)

                if address:
                    indexes["exact_address"][address].append(
                        local_idx
                    )

                    if country:
                        indexes["country_address"][
                            (country, address)
                        ].append(local_idx)

                for token in normalize_tokens(name)[
                    : self.config.max_name_tokens
                ]:
                    indexes["name_token"][token].append(local_idx)

                    if country:
                        indexes["country_name_token"][
                            (country, token)
                        ].append(local_idx)

                for token in normalize_tokens(address)[
                    : self.config.max_address_tokens
                ]:
                    indexes["address_token"][token].append(local_idx)

                    if country:
                        indexes["country_address_token"][
                            (country, token)
                        ].append(local_idx)

            # Convert postings to compact tuples and remove common keys.
            compact = {}

            for index_name, index in indexes.items():
                compact[index_name] = {
                    key: tuple(values[:posting_limit])
                    for key, values in index.items()
                    if len(values) <= posting_limit
                }

            return s1_ids, compact

        def process_target_file(
            target_path,
            target_source,
            writer,
            s1_ids,
            indexes,
        ):
            n_s1 = len(s1_ids)

            # -1 means no candidate stored.
            candidate_ids = np.full(
                (n_s1, top_k),
                -1,
                dtype=np.int32,
            )

            candidate_scores = np.zeros(
                (n_s1, top_k),
                dtype=np.uint8,
            )

            target_row_number = 0
            chunk_number = 0

            # Keep target IDs only while processing the current chunk.
            for target_chunk in pd.read_csv(
                target_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=self.config.large_data_chunk_size,
            ):
                chunk_number += 1

                for row in target_chunk.to_dict("records"):
                    tid_row = target_row_number
                    target_row_number += 1

                    name = normalize_business_name(
                        row.get("business_name", "")
                    )
                    address = normalize_address(
                        row.get("business_address", "")
                    )
                    country = normalize_country(
                        row.get("country", "")
                    )

                    hits = defaultdict(int)

                    if name:
                        for local_idx in indexes[
                            "exact_name"
                        ].get(name, ()):
                            hits[local_idx] += 3

                        if country:
                            for local_idx in indexes[
                                "country_name"
                            ].get((country, name), ()):
                                hits[local_idx] += 3

                    if address:
                        for local_idx in indexes[
                            "exact_address"
                        ].get(address, ()):
                            hits[local_idx] += 3

                        if country:
                            for local_idx in indexes[
                                "country_address"
                            ].get((country, address), ()):
                                hits[local_idx] += 3

                    for token in normalize_tokens(name)[
                        : self.config.max_name_tokens
                    ]:
                        for local_idx in indexes[
                            "name_token"
                        ].get(token, ()):
                            hits[local_idx] += 1

                        if country:
                            for local_idx in indexes[
                                "country_name_token"
                            ].get((country, token), ()):
                                hits[local_idx] += 1

                    for token in normalize_tokens(address)[
                        : self.config.max_address_tokens
                    ]:
                        for local_idx in indexes[
                            "address_token"
                        ].get(token, ()):
                            hits[local_idx] += 2

                        if country:
                            for local_idx in indexes[
                                "country_address_token"
                            ].get((country, token), ()):
                                hits[local_idx] += 2

                    for local_idx, score in hits.items():
                        row_scores = candidate_scores[local_idx]

                        min_pos = int(np.argmin(row_scores))
                        min_score = int(row_scores[min_pos])

                        if score > min_score:
                            candidate_scores[
                                local_idx,
                                min_pos,
                            ] = min(score, 255)

                            candidate_ids[
                                local_idx,
                                min_pos,
                            ] = tid_row

                if chunk_number % 10 == 0:
                    print(
                        f"    {target_source}: "
                        f"processed {chunk_number} chunks"
                    )

            # Read only the target IDs after ranking is complete.
            target_ids = pd.read_csv(
                target_path,
                sep="\t",
                usecols=["entity_id"],
                dtype=str,
                keep_default_na=False,
            )["entity_id"].tolist()

            for local_idx, sid in enumerate(s1_ids):
                row_scores = candidate_scores[local_idx]
                row_ids = candidate_ids[local_idx]

                valid = row_ids >= 0

                if not np.any(valid):
                    continue

                pairs = [
                    (
                        int(row_ids[pos]),
                        int(row_scores[pos]),
                    )
                    for pos in np.flatnonzero(valid)
                ]

                pairs.sort(
                    key=lambda item: (
                        -item[1],
                        item[0],
                    )
                )

                for target_pos, _score in pairs:
                    writer.writerow(
                        [
                            sid,
                            target_ids[target_pos],
                            target_source,
                        ]
                    )

            del target_ids
            del candidate_ids
            del candidate_scores

        try:
            s1_iter = pd.read_csv(
                s1_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=block_size,
            )

            block_number = 0

            for s1_block in s1_iter:
                block_number += 1

                print(
                    f"Starting Source-1 block {block_number} "
                    f"({len(s1_block):,} entities)"
                )

                s1_ids, indexes = build_indexes(s1_block)

                process_target_file(
                    s2_path,
                    "source2",
                    s2_writer,
                    s1_ids,
                    indexes,
                )

                s2_fh.flush()

                process_target_file(
                    s3_path,
                    "source3",
                    s3_writer,
                    s1_ids,
                    indexes,
                )

                s3_fh.flush()

                del indexes
                del s1_ids
                del s1_block

                print(
                    f"Completed Source-1 block {block_number}"
                )

        finally:
            s2_fh.close()
            s3_fh.close()

        print("Bounded-memory candidate generation complete.")

        return {
            "s2": s2_out,
            "s3": s3_out,
        }


