import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from src.features import (
    HOGConfig, HOG_CONFIGS, criar_banco_gabor, extrair_todas_caracteristicas,
    extrair_hog, extrair_gabor, extrair_bordas, extrair_atributos_geometricos,
)
from src.representations import FeatureRepresentation, InvalidPCADimension


class FeatureConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.roi = np.random.default_rng(42).integers(0, 256, (80, 240), dtype=np.uint8)
        self.mask = np.zeros_like(self.roi)
        self.mask[10:70, 20:220] = 255
        self.bank = criar_banco_gabor()

    def test_dimensoes_das_quatro_configuracoes_e_ordem_dos_blocos(self):
        auxiliary = np.hstack((extrair_gabor(self.roi, self.mask, self.bank),
                               extrair_bordas(self.roi, self.mask),
                               extrair_atributos_geometricos(self.mask))).astype(np.float32)
        for name, (hog_size, total) in {"A": (25380, 25413), "B": (5796, 5829),
                                       "C": (3864, 3897), "D": (1188, 1221)}.items():
            with self.subTest(config=name):
                config = HOG_CONFIGS[name]
                vector = extrair_todas_caracteristicas(self.roi, self.mask, self.bank, hog_config=config)
                self.assertEqual(vector.shape, (total,))
                self.assertEqual(config.hog_dimension, hog_size)
                self.assertEqual(vector.dtype, np.float32)
                self.assertTrue(np.isfinite(vector).all())
                np.testing.assert_array_equal(vector[hog_size:], auxiliary)
                self.assertEqual(config.metadata()["blocks"]["geometria"], [total - 6, total])

    def test_baseline_default_igual_a_parametros_originais(self):
        legacy = np.hstack((
            extrair_hog(self.roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),
            extrair_gabor(self.roi, self.mask, self.bank), extrair_bordas(self.roi, self.mask),
            extrair_atributos_geometricos(self.mask),
        )).astype(np.float32)
        default = extrair_todas_caracteristicas(self.roi, self.mask, self.bank)
        explicit = extrair_todas_caracteristicas(self.roi, self.mask, self.bank, hog_config=HOG_CONFIGS["A"])
        np.testing.assert_array_equal(default, legacy)
        np.testing.assert_array_equal(default, explicit)
        self.assertEqual(HOGConfig().metadata()["block_norm"], "L1")

    def test_configuracoes_invalidas(self):
        for kwargs in ({"orientations": 0}, {"orientations": True}, {"pixels_per_cell": (7, 7)},
                       {"pixels_per_cell": (128, 384)}, {"pixels_per_cell": [8, 8]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HOGConfig(**kwargs)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=1)
        self.addCleanup(self.limits.restore_original_limits)
        self.X = np.random.default_rng(42).normal(size=(130, 143))
        self.validation = np.random.default_rng(43).normal(size=(7, 143)) + 100

    def test_dimensoes_pca_full_e_hog_10_25_50_100(self):
        for scope in ("full", "hog"):
            for n in (10, 25, 50, 100):
                with self.subTest(scope=scope, n=n):
                    model = FeatureRepresentation(110, scope, n).fit(self.X)
                    result = model.transform(self.validation)
                    self.assertEqual(result.shape, (7, n + (33 if scope == "hog" else 0)))
                    self.assertGreaterEqual(model.effective_rank_, n)
                    if scope == "hog":
                        np.testing.assert_allclose(result[:, n:], model.scaler_.transform(self.validation)[:, 110:])

    def test_sem_pca_preserva_ordem_e_nao_ajusta_com_validacao(self):
        model = FeatureRepresentation(110).fit(self.X)
        expected_mean = self.X.mean(axis=0)
        np.testing.assert_allclose(model.scaler_.mean_, expected_mean)
        result = model.transform(self.validation)
        self.assertEqual(result.shape, (7, 143))
        np.testing.assert_allclose(result, (self.validation - expected_mean) / self.X.std(axis=0))
        self.assertEqual(model.scaler_.n_samples_seen_, 130)

    def test_reprodutibilidade_e_pca_sem_vazamento(self):
        first = FeatureRepresentation(110, "hog", 10).fit(self.X)
        second = FeatureRepresentation(110, "hog", 10).fit(self.X)
        components = first.pca_.components_.copy()
        first.transform(self.validation * 1000)
        np.testing.assert_array_equal(first.pca_.components_, components)
        np.testing.assert_allclose(first.transform(self.validation), second.transform(self.validation))
        self.assertEqual(first.pca_.n_samples_, 130)
        self.assertGreater(first.state_nbytes(), 0)

    def test_limites_de_tamanho_e_posto(self):
        for X, n in ((self.X[:5], 10), (np.zeros((20, 143)), 10),
                     (np.tile(np.arange(20)[:, None], (1, 143)), 2)):
            with self.subTest(shape=X.shape, n=n):
                model = FeatureRepresentation(110, "full", n)
                with self.assertRaises(InvalidPCADimension):
                    model.fit(X)
                with self.assertRaises(NotFittedError):
                    model.transform(self.validation)

    def test_entradas_invalidas(self):
        for args in ((110, "invalid", 10), (110, "none", 10), (110, "hog", 0), (110, "full", True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FeatureRepresentation(*args).fit(self.X)
        with self.assertRaises(ValueError):
            FeatureRepresentation(111).fit(self.X)


if __name__ == "__main__":
    unittest.main()
