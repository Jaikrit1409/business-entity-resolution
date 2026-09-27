"""Tests for candidate-only pair labeling and Source 1 group splitting."""

import unittest

from src.blocking import BlockingResult
from src.pair_generator import construct_labeled_pairs, pair_construction_report, split_labeled_pairs


S1 = [
    {"entity_id": "S1-1"},
    {"entity_id": "S1-2"},
    {"entity_id": "S1-3"},
]
S2 = [{"entity_id": "S2-1"}, {"entity_id": "S2-2"}]
S3 = [{"entity_id": "S3-1"}]
GROUND_TRUTH = [
    {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
    {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-2,S3-1"},
]
CANDIDATES = BlockingResult(
    s1_to_s2={"S1-1": ("S2-1",), "S1-2": ("S2-2",), "S1-3": ()},
    s1_to_s3={"S1-1": (), "S1-2": (), "S1-3": ("S3-1",)},
)


class PairGeneratorTests(unittest.TestCase):
    def test_labels_only_blocked_candidates(self):
        pairs = construct_labeled_pairs(CANDIDATES, S1, S2, S3, GROUND_TRUTH)
        self.assertEqual([(pair.source1_entity_id, pair.candidate_entity_id, pair.label) for pair in pairs], [
            ("S1-1", "S2-1", 1),
            ("S1-2", "S2-2", 0),
            ("S1-3", "S3-1", 1),
        ])
        self.assertNotIn("S2-2", [pair.candidate_entity_id for pair in pairs if pair.source1_entity_id == "S1-3"])

    def test_report_separates_blocking_recall_and_class_balance(self):
        pairs = construct_labeled_pairs(CANDIDATES, S1, S2, S3, GROUND_TRUTH)
        report = pair_construction_report(CANDIDATES, S1, S2, S3, GROUND_TRUTH, pairs)
        self.assertEqual(report.positive_count, 2)
        self.assertEqual(report.negative_count, 1)
        self.assertAlmostEqual(report.positive_rate, 2 / 3)
        self.assertAlmostEqual(report.blocking_recall, 2 / 3)
        self.assertEqual(report.source1_entity_count, 3)
        self.assertEqual((report.singleton_count, report.one_match_count, report.multiple_match_count), (1, 1, 1))
        self.assertEqual(report.lost_true_matches, {"S1-3": ("S2-2",)})

    def test_source1_split_is_reproducible_and_prevents_pair_leakage(self):
        pairs = construct_labeled_pairs(CANDIDATES, S1, S2, S3, GROUND_TRUTH)
        first = split_labeled_pairs(pairs, ["S1-1", "S1-2", "S1-3"], validation_fraction=1 / 3, random_seed=7)
        second = split_labeled_pairs(pairs, ["S1-1", "S1-2", "S1-3"], validation_fraction=1 / 3, random_seed=7)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_source1_ids) & set(first.validation_source1_ids))
        self.assertTrue(all(pair.source1_entity_id in first.train_source1_ids for pair in first.train_pairs))
        self.assertTrue(all(pair.source1_entity_id in first.validation_source1_ids for pair in first.validation_pairs))


if __name__ == "__main__":
    unittest.main()
