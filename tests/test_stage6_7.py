"""Testes sintéticos do LOF (etapa 6) e das métricas da avaliação (etapa 8)."""

import unittest

import numpy as np

from src.evaluate_test import metricas
from src.final_evaluation import post_hoc
from src.lof import LOFAnomalyDetector


class LOFTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.fit = rng.normal(size=(60, 5))
        self.cal = rng.normal(size=(40, 5))
        self.fit_ids = [f"f{i}" for i in range(60)]
        self.cal_ids = [f"c{i}" for i in range(40)]

    def test_outlier_scores_higher_and_is_rejected(self):
        model = LOFAnomalyDetector(n_neighbors=10).fit(self.fit, sample_ids=self.fit_ids)
        model.calibrate(self.cal, sample_ids=self.cal_ids)
        far = np.full((1, 5), 8.0)
        self.assertGreater(model.anomaly_score(far)[0], model.anomaly_score(self.cal).max())
        self.assertEqual(model.predict(far)[0], -1)
        self.assertLessEqual(model.calibration_fpr_, 0.05)

    def test_rejects_invalid_neighbors_and_leakage(self):
        with self.assertRaises(ValueError):
            LOFAnomalyDetector(n_neighbors=60).fit(self.fit)
        model = LOFAnomalyDetector(n_neighbors=5).fit(self.fit, sample_ids=self.fit_ids)
        with self.assertRaises(ValueError):
            model.calibrate(self.fit[:5], sample_ids=self.fit_ids[:5])

    def test_predict_requires_calibration(self):
        model = LOFAnomalyDetector(n_neighbors=5).fit(self.fit)
        with self.assertRaises(Exception):
            model.predict(self.cal)


class MetricTests(unittest.TestCase):
    def test_false_negative_is_defect_predicted_good(self):
        y_true = np.array([1, 1, -1, -1, -1])
        y_pred = np.array([1, -1, 1, -1, -1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
        classes = ["good", "good", "scratch_head", "scratch_head", "thread_top"]
        m = metricas(y_true, y_pred, scores, classes)
        self.assertEqual((m["tp"], m["fn"], m["tn"], m["fp"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["por_categoria"]["scratch_head"]["recall"], 0.5)

    def test_post_hoc_zero_fn_operating_point(self):
        y_true = np.array([1] * 20 + [-1] * 3)
        scores = np.concatenate([np.arange(20) / 20, [0.5, 2, 3]])
        r = post_hoc(y_true, scores)
        self.assertEqual(r["fp_para_zero_fn"], 10)  # boas com score >= 0,5


if __name__ == "__main__":
    unittest.main()
