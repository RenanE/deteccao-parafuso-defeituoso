"""Testes sintéticos da etapa 9 (descritores locais e divisão do teste)."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.defect_split import make_split
from src.local_features import LocalConfig, PatchAnomalyDetector, extrair_lote
from src.mahalanobis import NumericalFitError
from src.stage9_experiments import DISTANCE_MODEL_GRID, select


def fake_roi(rng, defect=False):
    roi = np.full((128, 384), 200, np.uint8)
    mask = np.zeros((128, 384), np.uint8)
    mask[44:84, 20:364] = 1
    roi[mask > 0] = 90
    for x in range(150, 360, 18):            # "rosca" periódica
        roi[40:88, x:x + 6] = 60
    roi = np.clip(roi.astype(int) + rng.normal(0, 4, roi.shape), 0, 255).astype(np.uint8)
    if defect:
        roi[50:70, 60:80] = 250               # risco claro local no corpo
    return roi, mask


class LocalDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        goods = [fake_roi(rng) for _ in range(24)]
        cls.cfg = LocalConfig(patch=32, neighbors=1)
        cls.F_fit = extrair_lote(np.stack([g[0] for g in goods[:20]]), np.stack([g[1] for g in goods[:20]]), cls.cfg, n_jobs=1)
        cls.F_good = extrair_lote(np.stack([g[0] for g in goods[20:]]), np.stack([g[1] for g in goods[20:]]), cls.cfg, n_jobs=1)
        bad = [fake_roi(rng, defect=True) for _ in range(4)]
        cls.F_bad = extrair_lote(np.stack([b[0] for b in bad]), np.stack([b[1] for b in bad]), cls.cfg, n_jobs=1)

    def test_grid_shape(self):
        self.assertEqual(self.F_fit.shape[:3], (20, 7, 23))

    def test_local_defect_scores_higher_for_all_detectors(self):
        for model, params in (("lof", {"n_neighbors": 10}), ("isolation_forest", {"n_estimators": 50}),
                              ("svm", {"nu": 0.1})):
            det = PatchAnomalyDetector(model, params, self.cfg, n_jobs=1)
            good_m, bad_m = det.fit_score(self.F_fit, [self.F_good, self.F_bad])
            self.assertGreater(det.aggregate(bad_m).min(), det.aggregate(good_m).max(), model)

    def test_defect_location_in_patch_map(self):
        det = PatchAnomalyDetector("lof", {"n_neighbors": 10}, self.cfg, n_jobs=1)
        m = det.fit_score(self.F_fit, [self.F_bad[:1]])[0][0]
        # Patches que cobrem o risco (y 50-70, x 60-80) ficam entre os 5% mais anômalos do mapa.
        self.assertGreater(m[2:5, 3:5].max(), np.quantile(m, 0.95))


class DistanceModelTests(unittest.TestCase):
    def test_local_distances_rank_perturbed_patches_higher(self):
        rng = np.random.default_rng(42)
        fit = rng.normal(size=(80, 2, 2, 3))
        good = rng.normal(size=(8, 2, 2, 3))
        bad = good.copy()
        bad[:, 0, 0] += 12
        config = LocalConfig(neighbors=1)
        for model, params in DISTANCE_MODEL_GRID:
            with self.subTest(model=model, params=params):
                detector = PatchAnomalyDetector(model, params, config, n_jobs=1)
                good_maps, bad_maps = detector.fit_score(fit, [good, bad])
                self.assertEqual(bad_maps.shape, (8, 2, 2))
                self.assertTrue(np.isfinite(bad_maps).all())
                np.testing.assert_array_equal(bad_maps[:, 1:, :], good_maps[:, 1:, :])
                self.assertTrue(np.all(detector.aggregate(bad_maps) > detector.aggregate(good_maps)))

    def test_singular_elliptic_patch_raises_without_fallback(self):
        fit = np.arange(80, dtype=float).reshape(80, 1, 1, 1)
        fit = np.repeat(fit, 2, axis=-1)
        detector = PatchAnomalyDetector(
            "mahalanobis", {"covariance": "elliptic"}, LocalConfig(), n_jobs=1,
        )
        with self.assertRaisesRegex(NumericalFitError, r"Patch \(0, 0\): Elliptic inviável"):
            detector.fit_score(fit, [fit[:2]])

    def test_selection_logs_infeasible_rows_without_using_holdout(self):
        fit = np.arange(120, dtype=float).reshape(120, 1, 1, 1)
        features = np.repeat(fit, 2, axis=-1)
        idx = {"fit": np.arange(100), "calibration": np.arange(100, 104),
               "sel_defects": np.arange(104, 108)}
        cache = {"rois": None, "masks": None}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selection"
            with (
                patch("src.stage9_experiments.load", return_value=(cache, [], idx, {"signature": "split"})),
                patch("src.stage9_experiments.FEATURE_GRID", [LocalConfig()]),
                patch("src.stage9_experiments.extrair_lote", return_value=features),
                patch("src.stage9_experiments.file_hash", return_value="hash"),
                patch("src.stage9_experiments.PatchAnomalyDetector",
                      side_effect=lambda *args: PatchAnomalyDetector(*args, n_jobs=1)),
                patch("builtins.print"),
            ):
                select(out, model_set="knn-mahalanobis")

            selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(set(selection["selected"]), {"knn", "mahalanobis"})
            self.assertEqual(json.loads(selection["selected"]["mahalanobis"]["params"]),
                             {"covariance": "ledoit_wolf"})
            with (out / "runs.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            skipped = [row for row in rows if row["status"] == "skipped"]
            self.assertEqual({r["aggregation"] for r in skipped}, {"max", "top3"})
            self.assertTrue(all("Elliptic inviável" in row["reason"] and not row["auc_sel"] for row in skipped))


class SplitTests(unittest.TestCase):
    def test_stratified_disjoint_split(self):
        inv = [{"caminho": f"t/{c}/{i}.png", "classe": c} for c in ("good", "scratch_head") for i in range(7)]
        ids = [d["caminho"] for d in inv]
        s = make_split(inv, ids)
        self.assertFalse(set(s["selection_ids"]) & set(s["holdout_ids"]))
        self.assertEqual(len(s["selection_ids"]) + len(s["holdout_ids"]), 14)
        self.assertEqual(sum(s["classes"][p] == "scratch_head" for p in s["selection_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
