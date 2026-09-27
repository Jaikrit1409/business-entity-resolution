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


if __name__ == "__main__":
    unittest.main()
