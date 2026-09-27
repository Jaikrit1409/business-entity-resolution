#!/usr/bin/env python3
"""Validation utility for Amazon ML Challenge 2026 submission files.

Strictly uses Python standard library only (no external dependencies).
Checks matching_results.tsv and candidate_pairs.tsv against all competition rules:
  1. File existence and valid TSV encoding.
  2. Correct header columns.
  3. Exactly one row per test Source-1 entity (no missing, no duplicate).
  4. Valid entity ID prefixes (S2-, S3- only; no S1- self matches).
  5. No duplicate IDs within a single entity's match/candidate list.
  6. Strict subset rule: Every matched ID must appear in candidate_pairs.tsv.
  7. Verification against test source records (if --test-dir is provided).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_tsv_ids(id_str: str) -> list[str]:
    """Parse comma-separated IDs, handling empty lists properly."""
    cleaned = id_str.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def validate(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    # 1. Existence Checks
    if not matching_path.exists():
        errors.append(f"Matching file not found: {matching_path}")
    if not candidate_path.exists():
        errors.append(f"Candidate file not found: {candidate_path}")
    if errors:
        return False, errors

    # Load test IDs if test directory is provided
    test_s1_ids: set[str] | None = None
    valid_target_ids: set[str] | None = None

    if test_dir is not None and test_dir.exists():
        s1_file = test_dir / "test_source1.tsv"
        s2_file = test_dir / "test_source2.tsv"
        s3_file = test_dir / "test_source3.tsv"

        if s1_file.exists():
            with s1_file.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)
                id_col = 0
                if header and "entity_id" in header:
                    id_col = header.index("entity_id")
                test_s1_ids = {row[id_col].strip() for row in reader if row and row[id_col].strip()}

        targets: set[str] = set()
        for t_file in (s2_file, s3_file):
            if t_file.exists():
                with t_file.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader, None)
                    id_col = 0
                    if header and "entity_id" in header:
                        id_col = header.index("entity_id")
                    targets.update(row[id_col].strip() for row in reader if row and row[id_col].strip())
        if targets:
            valid_target_ids = targets

    # 2. Validate candidate_pairs.tsv
    candidate_by_s1: dict[str, list[str]] = {}
    candidate_s1_seen: set[str] = set()

    with candidate_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            errors.append(f"{candidate_path.name}: Missing or invalid TSV header")
        elif (header[0].strip(), header[1].strip()) != ("source1_entity_id", "candidate_entity_ids"):
            errors.append(
                f"{candidate_path.name}: Expected header ['source1_entity_id', 'candidate_entity_ids'], "
                f"got [{header[0].strip()}, {header[1].strip()}]"
            )

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                errors.append(f"{candidate_path.name}: Line {line_num} has empty source1_entity_id")
                continue
            if s1_id in candidate_s1_seen:
                errors.append(f"{candidate_path.name}: Duplicate row for entity {s1_id} at line {line_num}")
            candidate_s1_seen.add(s1_id)

            cand_str = row[1] if len(row) > 1 else ""
            cands = parse_tsv_ids(cand_str)

            # Check duplicate candidate IDs
            if len(cands) != len(set(cands)):
                errors.append(f"{candidate_path.name}: Duplicate candidate IDs for entity {s1_id}")

            # Check valid prefix
            for cid in cands:
                if not (cid.startswith("S2-") or cid.startswith("S3-")):
                    errors.append(f"{candidate_path.name}: Invalid candidate ID prefix in '{cid}' for {s1_id}")
                if valid_target_ids is not None and cid not in valid_target_ids:
                    errors.append(f"{candidate_path.name}: Candidate ID '{cid}' does not exist in test set")

            candidate_by_s1[s1_id] = cands

    # 3. Validate matching_results.tsv
    matching_by_s1: dict[str, list[str]] = {}
    matching_s1_seen: set[str] = set()

    with matching_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            errors.append(f"{matching_path.name}: Missing or invalid TSV header")
        elif (header[0].strip(), header[1].strip()) != ("source1_entity_id", "matched_entity_ids"):
            errors.append(
                f"{matching_path.name}: Expected header ['source1_entity_id', 'matched_entity_ids'], "
                f"got [{header[0].strip()}, {header[1].strip()}]"
            )

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                errors.append(f"{matching_path.name}: Line {line_num} has empty source1_entity_id")
                continue
            if s1_id in matching_s1_seen:
                errors.append(f"{matching_path.name}: Duplicate row for entity {s1_id} at line {line_num}")
            matching_s1_seen.add(s1_id)

            match_str = row[1] if len(row) > 1 else ""
            matches = parse_tsv_ids(match_str)

            # Check duplicate matched IDs
            if len(matches) != len(set(matches)):
                errors.append(f"{matching_path.name}: Duplicate matched IDs for entity {s1_id}")

            # Check valid prefix
            for mid in matches:
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    errors.append(f"{matching_path.name}: Invalid matched ID prefix in '{mid}' for {s1_id}")
                if valid_target_ids is not None and mid not in valid_target_ids:
                    errors.append(f"{matching_path.name}: Matched ID '{mid}' does not exist in test set")

            matching_by_s1[s1_id] = matches

    # 4. Cross-file checks: Subset constraint
    for s1_id, matches in matching_by_s1.items():
        if s1_id not in candidate_by_s1:
            errors.append(f"Entity {s1_id} exists in matching_results but missing from candidate_pairs")
            continue
        cands_set = set(candidate_by_s1[s1_id])
        for mid in matches:
            if mid not in cands_set:
                errors.append(
                    f"Pipeline violation: Matched ID '{mid}' for {s1_id} was never generated as a candidate"
                )

    # 5. Entity completeness check against test set
    if test_s1_ids is not None:
        missing_in_matching = test_s1_ids - matching_s1_seen
        extra_in_matching = matching_s1_seen - test_s1_ids
        if missing_in_matching:
            sample = sorted(missing_in_matching)[:5]
            errors.append(f"{len(missing_in_matching)} test Source-1 entities missing from matching_results (e.g. {sample})")
        if extra_in_matching:
            sample = sorted(extra_in_matching)[:5]
            errors.append(f"{len(extra_in_matching)} unknown entities present in matching_results (e.g. {sample})")

        missing_in_cands = test_s1_ids - candidate_s1_seen
        if missing_in_cands:
            sample = sorted(missing_in_cands)[:5]
            errors.append(f"{len(missing_in_cands)} test Source-1 entities missing from candidate_pairs (e.g. {sample})")

    return (len(errors) == 0), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission files for Amazon ML Challenge 2026")
    parser.add_argument("--matching", type=Path, default=Path("output/matching_results.tsv"), help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"), help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=Path, default=None, help="Directory containing test_source1.tsv, etc.")

    args = parser.parse_args()

    passed, errors = validate(args.matching, args.candidate, args.test_dir)

    if passed:
        print("PASS: All submission constraints satisfied. Files are valid for submission.")
        sys.exit(0)
    else:
        print(f"FAIL: Found {len(errors)} issues with submission files:")
        for idx, err in enumerate(errors[:20], 1):
            print(f"  {idx}. {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
