"""Load entity-resolution datasets from TSV files."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator, Mapping
from typing import Any
import csv


REQUIRED_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def load_tsv(path: str | Path) -> list[dict[str, str]]:
    """Load a TSV file into a list of record dictionaries."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing and "ground_truth" not in path.name.lower():
            raise ValueError(f"{path} is missing required columns: {missing}")

        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


def iter_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream records from a TSV without loading the complete file."""
    path = Path(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")

        for row in reader:
            yield {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
            }


def load_training_data(data_dir: str | Path = "data") -> dict[str, list[dict[str, str]]]:
    """Load all training TSVs."""
    root = Path(data_dir) / "train"
    return {
        "source1": load_tsv(root / "train_source1.tsv"),
        "source2": load_tsv(root / "train_source2.tsv"),
        "source3": load_tsv(root / "train_source3.tsv"),
        "ground_truth": load_tsv(root / "train_ground_truth.tsv"),
    }


def load_test_data(data_dir: str | Path = "data") -> dict[str, list[dict[str, str]]]:
    """Load all test TSVs."""
    root = Path(data_dir) / "test"
    return {
        "source1": load_tsv(root / "test_source1.tsv"),
        "source2": load_tsv(root / "test_source2.tsv"),
        "source3": load_tsv(root / "test_source3.tsv"),
    }


def dataset_paths(data_dir: str | Path = "data", split: str = "train") -> dict[str, Path]:
    """Return canonical paths without reading the files."""
    root = Path(data_dir) / split
    if split == "train":
        return {
            "source1": root / "train_source1.tsv",
            "source2": root / "train_source2.tsv",
            "source3": root / "train_source3.tsv",
            "ground_truth": root / "train_ground_truth.tsv",
        }
    if split == "test":
        return {
            "source1": root / "test_source1.tsv",
            "source2": root / "test_source2.tsv",
            "source3": root / "test_source3.tsv",
        }
    raise ValueError("split must be 'train' or 'test'")
