"""Tests for submission formatting."""

import tempfile
import unittest
from pathlib import Path

from src.submission import write_candidate_pairs, write_matching_results


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_matching_results_formats_singletons_and_multiple_matches(self):
        output_file = self.temp_path / "matching_results.tsv"
        source1_ids = ["S1-1", "S1-2", "S1-3"]
        matches = {
            "S1-1": ["S2-1", "S3-1"],
            "S1-2": [],  # Singleton / no matches
            "S1-3": ["S2-5", "S2-5"],  # Duplicates should be deduplicated
        }

        result_path = write_matching_results(source1_ids, matches, output_file)
        self.assertTrue(result_path.exists())

        lines = result_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)  # Header + 3 entities
        self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
        self.assertEqual(lines[1], "S1-1\tS2-1,S3-1")
        self.assertEqual(lines[2], "S1-2\t")
        self.assertEqual(lines[3], "S1-3\tS2-5")

    def test_write_matching_results_handles_missing_keys_as_empty(self):
        output_file = self.temp_path / "sub" / "matching_results.tsv"
        source1_ids = ["S1-1", "S1-unmatched"]
        matches = {"S1-1": ["S2-1"]}

        result_path = write_matching_results(source1_ids, matches, output_file)
        lines = result_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[1], "S1-1\tS2-1")
        self.assertEqual(lines[2], "S1-unmatched\t")

    def test_write_candidate_pairs_formats_correctly(self):
        output_file = self.temp_path / "candidate_pairs.tsv"
        candidates = {
            "S1-1": ["S2-1", "S3-1", "S2-1"],
            "S1-2": [],
        }

        result_path = write_candidate_pairs(candidates, output_file)
        lines = result_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
        self.assertEqual(lines[1], "S1-1\tS2-1,S3-1")
        self.assertEqual(lines[2], "S1-2\t")



if __name__ == "__main__":
    unittest.main()
