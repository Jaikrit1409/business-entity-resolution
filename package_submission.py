#!/usr/bin/env python3
"""Automated packaging script for Amazon ML Challenge 2026 final submission.

Generates the required submission zip structure:
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def create_submission_zip(team_name: str, root_dir: Path, output_zip_path: Path) -> Path:
    output_dir = root_dir / "output"
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"
    doc_file = root_dir / "Documentation_template.md"
    readme_file = root_dir / "README.md"
    req_file = root_dir / "requirements.txt"
    src_dir = root_dir / "src"

    # Pre-flight check
    missing = []
    for p in (matching_file, candidate_file, doc_file, readme_file, req_file):
        if not p.exists():
            missing.append(str(p.relative_to(root_dir)))
    if missing:
        print(f"WARNING: The following expected files are missing before packaging:\n  " + "\n  ".join(missing))
        confirm = input("Do you still want to package without them? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(1)

    print(f"Building submission zip: {output_zip_path.name}...")

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. output/
        if matching_file.exists():
            zipf.write(matching_file, "output/matching_results.tsv")
            print("  + output/matching_results.tsv")
        if candidate_file.exists():
            zipf.write(candidate_file, "output/candidate_pairs.tsv")
            print("  + output/candidate_pairs.tsv")

        # 2. Documentation_template.md
        if doc_file.exists():
            zipf.write(doc_file, "Documentation_template.md")
            print("  + Documentation_template.md")

        # 3. code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution"
        if readme_file.exists():
            zipf.write(readme_file, f"{code_prefix}/README.md")
            print(f"  + {code_prefix}/README.md")
        if req_file.exists():
            zipf.write(req_file, f"{code_prefix}/requirements.txt")
            print(f"  + {code_prefix}/requirements.txt")

        # Recursively add src/ excluding __pycache__ and binaries
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo", ".DS_Store"}:
                    continue
                rel_path = file_path.relative_to(root_dir)
                archive_path = f"{code_prefix}/{rel_path}"
                zipf.write(file_path, archive_path)
                print(f"  + {archive_path}")

    print(f"\nSUCCESS! Created submission package: {output_zip_path} ({output_zip_path.stat().st_size / 1024:.1f} KB)")
    return output_zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge 2026 final submission")
    parser.add_argument("--team-name", type=str, default="amazon_team", help="Team name for zip filename")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory")
    args = parser.parse_args()

    clean_team_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.team_name.strip())
    zip_name = f"{clean_team_name}_submission.zip"
    output_path = args.root_dir / zip_name

    create_submission_zip(clean_team_name, args.root_dir, output_path)


if __name__ == "__main__":
    main()
