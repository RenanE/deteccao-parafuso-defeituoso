"""Testes sintéticos dos modelos neurais (experimento separado do escopo clássico)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch não instalado")
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        cls.paths = []
        for i in range(10):
            image = np.full((64, 64, 3), 200, np.uint8)
            cv2.rectangle(image, (20, 28), (44, 36), (60, 60, 60), -1)
            image = np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)
            path = root / f"img_{i}.png"
            cv2.imwrite(str(path), image)
            cls.paths.append(str(path))
        defect = cv2.imread(cls.paths[0])
        cv2.circle(defect, (10, 10), 6, (0, 0, 0), -1)
        cls.defect = str(root / "defect.png")
        cv2.imwrite(cls.defect, defect)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_autoencoder_fit_calibrate_predict(self):
        from neural_networks.neural import AutoencoderDetector
        model = AutoencoderDetector(size=32, epochs=2, batch_size=4)
        model.fit(self.paths[:6], sample_ids=self.paths[:6])
        model.calibrate(self.paths[6:], sample_ids=self.paths[6:])
        self.assertEqual(len(model.loss_history_), 2)
        self.assertEqual(model.predict([self.defect]).shape, (1,))
        with self.assertRaises(ValueError):
            model.calibrate(self.paths[:3], sample_ids=self.paths[:3])

    def test_patchcore_flags_synthetic_defect(self):
        from neural_networks.neural import PatchCoreDetector
        model = PatchCoreDetector(resize=64, crop=64, coreset_ratio=0.5)
        try:
            model.fit(self.paths[:6], sample_ids=self.paths[:6])
        except Exception as exc:  # pesos ImageNet indisponíveis offline
            self.skipTest(f"backbone indisponível: {exc}")
        model.calibrate(self.paths[6:], sample_ids=self.paths[6:])
        self.assertGreater(model.anomaly_score([self.defect])[0], model.threshold_)


if __name__ == "__main__":
    unittest.main()
