import unittest

import numpy as np

from src.stage9_threshold_cost import fit_thresholds, nondominated


class ThresholdCostTests(unittest.TestCase):
    def test_cost_tie_prefers_lowest_fnr_and_handles_equal_scores(self):
        limits = fit_thresholds([1., 1.], [1., 1.])
        self.assertEqual(limits[1], np.nextafter(1., -np.inf))
        self.assertTrue(all(1. > threshold for threshold in limits.values()))

    def test_weight_changes_the_optimal_threshold(self):
        limits = fit_thresholds([2., 3., 4.], [1., 5., 6.])
        self.assertEqual(limits[1], 4.)
        self.assertEqual(limits[10], np.nextafter(1., -np.inf))

    def test_pareto_keeps_tradeoffs_and_drops_dominated_points(self):
        rows = [{"fn": fn, "fp": fp} for fn, fp in [(1, 2), (1, 2), (0, 3), (2, 0), (2, 2)]]
        self.assertEqual(nondominated(rows), {(1, 2), (0, 3), (2, 0)})
