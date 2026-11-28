from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.knn import KNNAnomalyDetector


class KNNTests(unittest.TestCase):
    def model(self, **kwargs):
        return KNNAnomalyDetector(n_neighbors=1, target_fpr=0.5, **kwargs)

    def test_distancias_internas_excluem_a_propria_amostra(self):
        detector = self.model().fit([[0], [10], [20], [30]])
        np.testing.assert_allclose(detector.training_scores_, [10, 10, 10, 10])
        for i, vizinhos in enumerate(detector.training_neighbor_indices_):
            self.assertNotIn(i, vizinhos)
        self.assertFalse(hasattr(detector, "threshold_"))

    def test_zeros_reais_de_amostras_distintas_nao_sao_removidos(self):
        detector = self.model().fit([[0], [0], [10]])
        np.testing.assert_allclose(detector.training_scores_, [0, 0, 10])
        for i, vizinhos in enumerate(detector.training_neighbor_indices_):
            self.assertNotIn(i, vizinhos)

    def test_calibracao_quantitativa_e_convencao_de_predicao(self):
        detector = self.model().fit([[0], [10], [20], [30]])
        detector.calibrate([[1], [2], [3], [4]])
        np.testing.assert_allclose(detector.calibration_scores_, [1, 2, 3, 4])
        self.assertEqual(detector.threshold_, 3)
        self.assertEqual(detector.calibration_fpr_, 0.25)
        np.testing.assert_array_equal(detector.predict([[31], [33], [34]]), [1, 1, -1])
        np.testing.assert_allclose(detector.decision_function([[31], [33], [34]]), [2, 0, -1])
        np.testing.assert_allclose(detector.score_samples([[31], [33], [34]]), [-1, -3, -4])

    def test_limite_independe_das_distancias_internas(self):
        detector = self.model().fit([[0], [10], [20], [30]])
        detector.calibrate([[1], [2], [3], [4]])
        self.assertEqual(detector.threshold_, 3)
        detector.calibrate([[40], [50], [60], [70]])
        self.assertEqual(detector.threshold_, 30)
        np.testing.assert_allclose(detector.training_scores_, [10] * 4)

    def test_empates_no_limiar_sao_normais(self):
        detector = self.model().fit([[0], [10]])
        detector.calibrate([[1], [9]])
        self.assertEqual(detector.calibration_fpr_, 0)
        np.testing.assert_array_equal(detector.predict([[1], [9]]), [1, 1])

    def test_fit_e_refit_exigem_calibracao(self):
        detector = self.model()
        with self.assertRaises(NotFittedError):
            detector.predict([[1]])
        detector.fit([[0], [10]])
        with self.assertRaises(NotFittedError):
            detector.predict([[1]])
        detector.calibrate([[1], [2]])
        detector.fit([[50], [100]])
        with self.assertRaises(NotFittedError):
            detector.predict([[60]])

    def test_ids_bloqueiam_reuso_de_amostras(self):
        detector = self.model().fit([[0], [10]], sample_ids=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Vazamento"):
            detector.calibrate([[1], [2]], sample_ids=["a", "c"])
        with self.assertRaisesRegex(ValueError, "Forneça sample_ids"):
            detector.calibrate([[1], [2]])
        detector.calibrate([[1], [2]], sample_ids=["c", "d"])
        self.assertEqual(detector.calibration_ids_, ("c", "d"))

    def test_parametros_e_entradas_invalidas(self):
        for k in (0, -1, 1.5, True, 2):
            with self.subTest(k=k), self.assertRaises(ValueError):
                KNNAnomalyDetector(n_neighbors=k).fit([[0], [10]])
        for alpha in (0, 1, -0.1, np.nan, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                KNNAnomalyDetector(n_neighbors=1, target_fpr=alpha).fit([[0], [10]])
        for X in ([], [1, 2], [[np.nan], [1]], [[np.inf], [0]], [[0]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                self.model().fit(X)
        with self.assertRaisesRegex(ValueError, "duplicados"):
            self.model().fit([[0], [10]], sample_ids=["a", "a"])
        with self.assertRaises(ValueError):
            self.model().fit([[0], [10]], y=[1, -1])

    def test_calibracao_pequena_dimensoes_e_ids_invalidos(self):
        detector = self.model().fit([[0], [10]])
        for X in ([[1]], [], [[1, 2], [3, 4]], [[1], [np.inf]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                detector.calibrate(X)
        with self.assertRaises(ValueError):
            detector.calibrate([[1], [2]], sample_ids=["c"])
        with self.assertRaises(ValueError):
            detector.calibrate([[1], [2]], y=[1, -1])
        with self.assertWarnsRegex(UserWarning, "1/n"):
            detector.calibrate([[1], [2]], target_fpr=0.05)

    def test_clone_e_serializacao_preservam_contrato(self):
        detector = self.model().fit([[0], [10]]).calibrate([[1], [2]])
        novo = clone(detector)
        self.assertFalse(hasattr(novo, "nn_"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.joblib"
            joblib.dump(detector, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict([[1], [5]]), detector.predict([[1], [5]]))


if __name__ == "__main__":
    unittest.main()
