import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.feature_experiments import (
    audit_dataset, compare_study, extract_study, file_hash, load_features, load_study, prepare_study,
)
from src.features import HOG_CONFIGS


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data"
        for split, n in (("train", 30), ("test/good", 6)):
            folder = self.data / split
            folder.mkdir(parents=True)
            for i in range(n):
                image = np.full((16, 32, 3), i + (30 if split == "train" else 100), dtype=np.uint8)
                ok, encoded = cv2.imencode(".png", image)
                self.assertTrue(ok)
                (folder / f"{i:03}.png").write_bytes(encoded.tobytes())
        self.study = self.root / "study"
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        prepare_study(self.data, self.study, train_flat_is_good=True)

    def fake_extract(self, path, bank, *, hog_config):
        rng = np.random.default_rng(int(path.stem))
        return rng.normal(size=hog_config.total_dimension).astype(np.float32), True

    def full_cache(self):
        cache = self.root / "cache"
        with patch("src.feature_experiments.processar_imagem", side_effect=self.fake_extract):
            extract_study(self.study, cache, configs=["D"], full=True)
        return cache

    def test_manifestos_reprodutiveis_e_reservas_intocadas(self):
        inventory, outer, inner = load_study(self.study)
        second = self.root / "second"
        prepare_study(self.data, second, train_flat_is_good=True)
        self.assertEqual(outer, load_study(second)[1])
        self.assertEqual(inner, load_study(second)[2])
        forbidden = set(outer["calibration_ids"]) | set(outer["test_ids"])
        for fold in inner["folds"]:
            fit, threshold, val = (set(fold[k]) for k in ("fit_ids", "threshold_ids", "validation_ids"))
            self.assertFalse(fit & threshold or fit & val or threshold & val)
            self.assertEqual(fit | threshold | val, set(outer["fit_ids"]))
            self.assertFalse((fit | threshold | val) & forbidden)
        self.assertEqual(len(inventory), 36)

    def test_piloto_le_apenas_ajuste_e_preserva_arquivos(self):
        before = {p: (file_hash(p), p.stat().st_mtime_ns) for p in self.data.rglob("*.png")}
        cache = self.root / "pilot"
        with patch("src.feature_experiments.processar_imagem", side_effect=self.fake_extract) as process:
            extract_study(self.study, cache, configs=["A", "B", "C", "D"], sample_size=3)
        inventory, outer, _ = load_study(self.study)
        self.assertEqual(process.call_count, 12)
        self.assertTrue({str(call.args[0]) for call in process.call_args_list} <= set(outer["fit_ids"]))
        for name, config in HOG_CONFIGS.items():
            with np.load(cache / f"features_{name}.npz") as arrays:
                self.assertEqual(arrays["features"].shape, (3, config.total_dimension))
        self.assertEqual(before, {p: (file_hash(p), p.stat().st_mtime_ns) for p in self.data.rglob("*.png")})
        with self.assertRaisesRegex(ValueError, "amostra piloto"):
            load_features(cache, "A", inventory, outer)

    def test_recusa_sobrescrita_e_imagem_alterada(self):
        with self.assertRaises(FileExistsError):
            prepare_study(self.data, self.study, train_flat_is_good=True)
        cache = self.full_cache()
        before = file_hash(cache / "features_D.npz")
        with self.assertRaises(FileExistsError):
            extract_study(self.study, cache, full=True)
        self.assertEqual(file_hash(cache / "features_D.npz"), before)
        path = Path(load_study(self.study)[1]["fit_ids"][0])
        path.write_bytes(b"alterado apenas no fixture")
        with self.assertRaisesRegex(ValueError, "Imagem mudou"):
            extract_study(self.study, self.root / "changed", full=True)

    def test_detecta_csv_antigo_sem_reutilizar(self):
        old = self.data / "processed/features_train.csv"
        old.parent.mkdir()
        old.write_text("classe,feat_0\ngood,1\n")
        original = old.read_bytes()

        def audit_fixture(data_dir, inventory):
            with patch("src.feature_experiments.ROOT", self.root / "project"):
                return audit_dataset(data_dir, inventory)

        with patch("src.feature_experiments.audit_dataset", side_effect=audit_fixture):
            prepare_study(self.data, self.root / "with_old", train_flat_is_good=True)
        audit = json.loads((self.root / "with_old/audit.json").read_text())
        self.assertEqual(len(audit["historical_csvs"]), 1)
        self.assertEqual(audit["historical_csvs"][0]["path"], str(old.resolve()))
        self.assertEqual(audit["historical_csvs"][0]["feature_columns"], 1)
        self.assertEqual(old.read_bytes(), original)

    def test_comparacao_sintetica_reprodutivel_sem_vencedor(self):
        cache = self.full_cache()
        reports = []
        for i in range(2):
            report = compare_study(self.study, cache, self.root / f"comparison{i}", configs=["D"],
                                   components=[1, 1000], target_fpr=0.5)
            reports.append(report)
            self.assertIsNone(report["winner"])
            self.assertEqual(sum(r["status"] == "skipped" for r in report["rows"]), 6)
            self.assertEqual(sum(r["status"] == "ok" for r in report["rows"]), 9)
            for row in report["rows"]:
                if row["status"] == "ok":
                    self.assertIn("transformer_state_bytes", row)
                    self.assertGreaterEqual(row["normal_rejection_rate"], 0)
                    self.assertLessEqual(row["normal_rejection_rate"], 1)
        def metrics(report):
            return [(r["status"], r.get("normal_rejection_rate"), r.get("threshold")) for r in report["rows"]]
        self.assertEqual(metrics(reports[0]), metrics(reports[1]))

    def test_cache_nao_aceita_calibracao_final_nem_teste(self):
        cache = self.full_cache()
        inventory, outer, _ = load_study(self.study)
        path = cache / "features_D.npz"
        with np.load(path) as arrays:
            X, ids, hashes = arrays["features"], arrays["ids"], arrays["image_hashes"]
        for forbidden in (outer["calibration_ids"][0], outer["test_ids"][0]):
            # Simula cache indevido dentro do diretório temporário do teste.
            altered = ids.astype("U1024")
            altered[0] = forbidden
            with path.open("wb") as stream:
                np.savez_compressed(stream, features=X, ids=altered, image_hashes=hashes)
            meta_path = cache / "features_D.json"
            meta = json.loads(meta_path.read_text())
            meta["cache_sha256"] = file_hash(path)
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "IDs indevidos"):
                load_features(cache, "D", inventory, outer)

    def test_confirmacao_real_ainda_e_exigida_na_preparacao(self):
        with self.assertRaisesRegex(ValueError, "sem confirmação"):
            prepare_study(self.data, self.root / "blocked", train_flat_is_good=False)


if __name__ == "__main__":
    unittest.main()
