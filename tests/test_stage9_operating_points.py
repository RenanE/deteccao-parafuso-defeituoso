"""Testes dos limiares de operação com decisão estrita."""

import unittest

import numpy as np

from src.stage9_operating_points import determine_thresholds, recall_threshold


class RecallThresholdTests(unittest.TestCase):
    def test_95_percent_requires_56_of_58(self):
        scores = np.arange(58, dtype=float)[::-1]
        threshold = recall_threshold(scores, 95)
        self.assertEqual(threshold, np.nextafter(2., -np.inf))
        self.assertEqual(np.sum(scores > threshold), 56)
        self.assertEqual(np.sum(scores > np.nextafter(threshold, np.inf)), 55)

    def test_entire_tie_at_cutoff_is_detected(self):
        scores = np.r_[0., np.repeat(1., 4), np.arange(2., 55.)]
        self.assertEqual(len(scores), 58)
        threshold = recall_threshold(scores, 95)
        self.assertEqual(np.sum(scores > threshold), 57)
        self.assertEqual(np.sum(scores > np.nextafter(threshold, np.inf)), 53)

    def test_100_percent_includes_minimum_even_if_tied_or_negative(self):
        scores = np.array([-3., -3., -1.])
        threshold = recall_threshold(scores, 100)
        self.assertEqual(threshold, np.nextafter(-3., -np.inf))
        self.assertTrue(np.all(scores > threshold))
        self.assertFalse(np.all(scores > np.nextafter(threshold, np.inf)))

    def test_original_quantile_keeps_equal_scores_normal(self):
        cal = np.r_[np.arange(61.), [60., 100., 101.]]
        thresholds = determine_thresholds(cal, np.arange(58.))
        self.assertEqual(thresholds["original"], 60.)
        self.assertEqual(np.sum(cal > thresholds["original"]), 2)
        self.assertEqual(np.sum(np.arange(58.) > thresholds["selection_95"]), 56)
        self.assertEqual(np.sum(np.arange(58.) > thresholds["selection_100"]), 58)

    def test_rejects_invalid_scores_or_targets(self):
        for scores in ([], [np.nan], [np.inf], [[1.]]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                recall_threshold(scores, 95)
        for percent in (0, 101, 95.5, True):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                recall_threshold([1.], percent)
