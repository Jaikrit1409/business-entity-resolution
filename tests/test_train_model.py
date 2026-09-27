"""Tests for challenge-style threshold evaluation without fitting a model."""

import unittest

from src.train_model import _select_threshold, evaluate_thresholds, macro_f0_5, pair_precision_recall


class TrainModelMetricsTests(unittest.TestCase):
    def test_macro_f0_5_rewards_correct_singletons_and_penalizes_false_merges(self):
        source1_ids = ["S1-singleton", "S1-match", "S1-missed"]
        truth = {("S1-match", "S2-1"), ("S1-missed", "S3-1")}
        predictions = {("S1-match", "S2-1")}
        self.assertAlmostEqual(macro_f0_5(source1_ids, truth, predictions), 2 / 3)
        false_merge = predictions | {("S1-singleton", "S2-9")}
        self.assertLess(macro_f0_5(source1_ids, truth, false_merge), 2 / 3)

    def test_pair_diagnostics_handle_zero_predicted_matches(self):
        precision, recall, predicted = pair_precision_recall([1, 0], [0.2, 0.1], 0.9)
        self.assertEqual((precision, recall, predicted), (0.0, 0.0, 0))

    def test_threshold_selection_uses_macro_f0_5_not_accuracy(self):
        reports = evaluate_thresholds(
            validation_source1_ids=["S1-singleton", "S1-match", "S1-missed"],
            validation_pair_source1_ids=["S1-match", "S1-match", "S1-missed"],
            validation_candidate_ids=["S2-1", "S2-2", "S3-1"],
            validation_labels=[1, 0, 1],
            validation_probabilities=[0.9, 0.6, 0.4],
            true_pairs={("S1-match", "S2-1"), ("S1-missed", "S3-1")},
            thresholds=[0.5, 0.7, 0.95],
        )
        selected = _select_threshold(reports)
        self.assertEqual(selected.threshold, 0.7)
        self.assertGreater(selected.macro_f0_5, next(report.macro_f0_5 for report in reports if report.threshold == 0.5))

    def test_macro_f0_5_weights_precision_twice_as_much_as_recall(self):
        source1_ids = ["S1-multi"]
        truth = {("S1-multi", "S2-1"), ("S1-multi", "S3-1")}

        # 1 TP, 0 FP: precision = 1.0, recall = 0.5 -> F0.5 = 1.25 * 0.5 / 0.75 = 5/6 ~= 0.8333
        pred_miss = {("S1-multi", "S2-1")}
        score_miss = macro_f0_5(source1_ids, truth, pred_miss)
        self.assertAlmostEqual(score_miss, 5 / 6)

        # 1 TP, 1 FP: precision = 0.5, recall = 0.5 -> F0.5 = 1.25 * 0.25 / (0.125 + 0.5) = 0.50
        pred_false = {("S1-multi", "S2-1"), ("S1-multi", "S2-wrong")}
        score_false = macro_f0_5(source1_ids, truth, pred_false)
        self.assertAlmostEqual(score_false, 0.50)

        # False positive hurts significantly more than false negative
        self.assertGreater(score_miss, score_false)

    def test_macro_f0_5_deduplicates_source1_ids_safely(self):
        source1_ids = ["S1-1", "S1-1", "S1-2"]
        truth = {("S1-1", "S2-1")}
        pred = {("S1-1", "S2-1")}
        # S1-1 is perfect (1.0), S1-2 is correct singleton (1.0) -> macro average 1.0
        self.assertEqual(macro_f0_5(source1_ids, truth, pred), 1.0)

    def test_macro_f0_5_empty_returns_zero(self):
        self.assertEqual(macro_f0_5([], set(), set()), 0.0)


class TrainBaselineModelIntegrationTests(unittest.TestCase):
    def test_end_to_end_training_on_synthetic_data_with_noisy_records(self):
        import tempfile
        from pathlib import Path
        from src.blocking import BlockingConfig
        from src.predict import score_candidates, select_matches
        from src.train_model import train_baseline_model

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            s1 = [
                {"entity_id": "S1-1", "name": "Global Logistics Corp", "address": "100 Broadway St, Ste 400", "country": "Kenya"},
                {"entity_id": "S1-2", "name": "Northern Star Brewery", "address": "45 Harbour Lane", "country": "Iceland"},
                {"entity_id": "S1-3", "name": "Lone Mountain Bakery", "address": "12 Peak Rd", "country": "Peru"},
                {"entity_id": "S1-4", "name": "Apex Clean Energy LLC", "address": "88 Solar Way", "country": "Vietnam"},
            ]
            s2 = [
                {"entity_id": "S2-1", "name": "Global Logistics Corporation", "address": "100 Broadway Street Suite 400", "country": "Kenya"},
                {"entity_id": "S2-2", "name": "Northern Star Brewing Co", "address": "45 Harbour Ln", "country": "Iceland"},
                {"entity_id": "S2-3", "name": "Random Unrelated Company", "address": "999 Nowhere St", "country": "Kenya"},
            ]
            s3 = [
                {"entity_id": "S3-1", "name": "Apex Clean Energy", "address": "88 Solar Way", "country": "Vietnam"},
                {"entity_id": "S3-2", "name": "Northern Star Beers", "address": "45 Harbour Lane", "country": "Iceland"},
            ]
            truth = [
                {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
                {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2,S3-2"},
                {"source1_entity_id": "S1-3", "matched_entity_ids": ""},  # true singleton
                {"source1_entity_id": "S1-4", "matched_entity_ids": "S3-1"},
            ]

            blocking_config = BlockingConfig(min_token_length=2, max_posting_size=50, max_candidates_per_source=20)
            result = train_baseline_model(
                source1_records=s1,
                source2_records=s2,
                source3_records=s3,
                ground_truth_records=truth,
                output_dir=out_path,
                validation_fraction=0.5,
                random_seed=42,
                blocking_config=blocking_config,
                scale_pos_weight=1.0,
            )

            self.assertTrue(result.model_path.exists())
            self.assertTrue(result.config_path.exists())
            self.assertGreater(result.selected_threshold, 0.0)

            # Test prediction scoring
            matrix, probabilities = score_candidates(
                source1_records=s1,
                source2_records=s2,
                source3_records=s3,
                model_path=result.model_path,
                config_path=result.config_path,
            )
            self.assertEqual(len(matrix.source1_ids), len(probabilities))

            matches = select_matches(matrix, probabilities, threshold=result.selected_threshold)
            self.assertIsInstance(matches, dict)
            # Ensure candidate IDs don't contain duplicates
            for s1_id, candidate_ids in matches.items():
                self.assertEqual(len(candidate_ids), len(set(candidate_ids)))


if __name__ == "__main__":
    unittest.main()

