"""Edge-case tests for blocked-pair numerical features."""

import unittest

from src.blocking import BlockingResult
from src.feature_engineering import FEATURE_COLUMNS, build_feature_matrix, pair_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_identical_records_have_strong_evidence(self):
        left = {"entity_id": "S1-1", "name": "Acme & Sons", "address": "12 Main Rd, 75001", "country": "France"}
        right = {"entity_id": "S2-1", "name": "ACME and Sons", "address": "12 Main Road 75001", "country": "France"}
        features = pair_features(left, right)
        self.assertEqual(features["name_exact"], 1.0)
        self.assertEqual(features["address_exact"], 1.0)
        self.assertEqual(features["country_equal"], 1.0)
        self.assertEqual(features["address_numeric_token_overlap"], 1.0)
        self.assertEqual(features["name_address_exact_both"], 1.0)
        self.assertEqual(features["target_is_source2"], 1.0)

    def test_missing_and_empty_fields_do_not_create_false_exact_matches(self):
        left = {"entity_id": "S1-1", "name": None, "address": "", "country": ""}
        right = {"entity_id": "S3-1", "name": "", "address": None, "country": None}
        features = pair_features(left, right)
        self.assertEqual(features["name_exact"], 0.0)
        self.assertEqual(features["address_exact"], 0.0)
        self.assertEqual(features["country_equal"], 0.0)
        self.assertEqual(features["name_any_missing"], 1.0)
        self.assertEqual(features["address_any_missing"], 1.0)
        self.assertEqual(features["country_any_missing"], 1.0)
        self.assertEqual(features["target_is_source3"], 1.0)

    def test_different_records_have_weak_name_address_evidence(self):
        left = {"entity_id": "S1-1", "name": "Blue Harbor Bakery", "address": "101 Ocean Road", "country": "France"}
        right = {"entity_id": "S2-2", "name": "Quantum Auto Parts", "address": "900 Industrial Avenue", "country": "India"}
        features = pair_features(left, right)
        self.assertEqual(features["name_exact"], 0.0)
        self.assertEqual(features["address_exact"], 0.0)
        self.assertEqual(features["country_equal"], 0.0)
        self.assertLess(features["name_character_similarity"], 0.5)
        self.assertEqual(features["address_numeric_token_overlap"], 0.0)

    def test_matrix_contains_only_the_provided_blocked_candidates(self):
        s1 = [{"entity_id": "S1-1", "name": "Acme", "address": "12 Main Rd", "country": "France"}]
        s2 = [{"entity_id": "S2-1", "name": "Acme", "address": "12 Main Road", "country": "France"}]
        s3 = [{"entity_id": "S3-1", "name": "Unrelated", "address": "9 Other Rd", "country": "France"}]
        candidates = BlockingResult(s1_to_s2={"S1-1": ("S2-1",)}, s1_to_s3={"S1-1": ()})
        matrix = build_feature_matrix(s1, s2, s3, candidates)
        self.assertEqual(matrix.source1_ids, ("S1-1",))
        self.assertEqual(matrix.candidate_ids, ("S2-1",))
        self.assertEqual(matrix.feature_names, FEATURE_COLUMNS)
        self.assertEqual(len(matrix.values[0]), len(FEATURE_COLUMNS))
        self.assertTrue(all(isinstance(value, float) for value in matrix.values[0]))


if __name__ == "__main__":
    unittest.main()
