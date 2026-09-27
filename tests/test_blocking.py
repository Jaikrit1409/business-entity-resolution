"""Tests for deterministic, inverted-index candidate generation."""

import unittest

from src.blocking import BlockingConfig, MultiStrategyBlocker, final_candidate_rows, validate_blocking


S1 = [
    {"entity_id": "S1-1", "name": "Acme & Sons", "address": "12 Main Road, Paris 75001", "country": "France"},
    {"entity_id": "S1-2", "name": "Café Central", "address": "8 Market Lane", "country": "India"},
]
S2 = [
    {"entity_id": "S2-1", "name": "Acme and Sons Inc", "address": "12 Main Rd, Paris 75001", "country": "France"},
    {"entity_id": "S2-2", "name": "Other Business", "address": "99 Remote Avenue", "country": "France"},
]
S3 = [
    {"entity_id": "S3-1", "name": "Cafe Central Pvt Ltd", "address": "8 Market Ln", "country": "India"},
    {"entity_id": "S3-2", "name": "Noise Shop", "address": "1 Elsewhere Road", "country": "India"},
]
GROUND_TRUTH = [
    {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
    {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-1"},
]


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.blocker = MultiStrategyBlocker(BlockingConfig(max_posting_size=20, max_candidates_per_source=20))

    def test_generates_candidates_per_target_source_without_country_hard_coding(self):
        result = self.blocker.generate(S1, S2, S3)
        self.assertIn("S2-1", result.s1_to_s2["S1-1"])
        self.assertIn("S3-1", result.s1_to_s3["S1-2"])
        self.assertTrue(all(entity_id.startswith("S2-") for ids in result.s1_to_s2.values() for entity_id in ids))
        self.assertTrue(all(entity_id.startswith("S3-") for ids in result.s1_to_s3.values() for entity_id in ids))

    def test_generation_is_deterministic_and_final_rows_match_model_input(self):
        first = self.blocker.generate(S1, S2, S3)
        second = self.blocker.generate(S1, S2, S3)
        self.assertEqual(first, second)
        rows = final_candidate_rows(first)
        self.assertEqual({row["source1_entity_id"] for row in rows}, {"S1-1", "S1-2"})
        self.assertIn("S2-1", next(row for row in rows if row["source1_entity_id"] == "S1-1")["candidate_entity_ids"])

    def test_validation_reports_recall_reduction_and_lost_matches(self):
        result = self.blocker.generate(S1, S2, S3)
        metrics = validate_blocking(result, S1, S2, S3, GROUND_TRUTH)
        self.assertEqual(metrics.true_match_recall, 1.0)
        self.assertEqual(metrics.lost_true_matches, {})
        self.assertGreater(metrics.candidate_reduction_ratio, 0.0)
        self.assertLessEqual(metrics.maximum_candidates_per_source1, 4)

    def test_validation_identifies_a_true_match_lost_by_blocking(self):
        result = self.blocker.generate(S1, S2, S3)
        altered = type(result)(s1_to_s2={"S1-1": (), "S1-2": result.s1_to_s2["S1-2"]}, s1_to_s3=result.s1_to_s3)
        metrics = validate_blocking(altered, S1, S2, S3, GROUND_TRUTH)
        self.assertEqual(metrics.true_match_recall, 0.5)
        self.assertEqual(metrics.lost_true_matches, {"S1-1": ("S2-1",)})


if __name__ == "__main__":
    unittest.main()
