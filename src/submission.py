"""Format entity-resolution predictions into the competition schema."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
import csv


def write_matching_results(
    source1_ids: Sequence[str],
    matches: Mapping[str, Sequence[str]],
    output_path: str | Path,
) -> Path:
    """Write one row for every Source 1 entity."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for source1_id in source1_ids:
            unique_ids = list(dict.fromkeys(str(x) for x in matches.get(source1_id, [])))
            writer.writerow([source1_id, ",".join(unique_ids)])

    return output_path


def write_candidate_pairs(
    candidates: Mapping[str, Sequence[str]],
    output_path: str | Path,
) -> Path:
    """Write the final candidate set as one row per Source 1 entity."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for source1_id, candidate_ids in candidates.items():
            unique_ids = list(dict.fromkeys(str(x) for x in candidate_ids))
            writer.writerow([source1_id, ",".join(unique_ids)])

    return output_path
