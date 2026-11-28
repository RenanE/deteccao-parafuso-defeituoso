import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from src.knn import KNNAnomalyDetector
from src.validation import (
    adicionar_hashes, carregar_divisao, gerar_divisao, normalizar_caminhos,
    obter_ou_criar_divisao, preparar_caracteristicas, salvar_divisao,
    transformar_teste, validar_divisao,
)


def inventario(base, n=20, n_test=3):
    rows = []
    for split, total in (("train", n), ("test", n_test)):
        for i in range(total):
            rows.append({
                "caminho": str(base / split / f"{i:04}.png"), "split": split,
                "classe": "good" if split == "train" or i == 0 else "scratch_head",
                "sha256": f"{i + (10000 if split == 'test' else 0):064x}",
                "feat_0": float(i), "feat_1": float(i**2), "feat_2": 1.0,
            })
    return pd.DataFrame(rows)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()

    def test_reprodutivel_independente_da_ordem_de_linhas(self):
        df = inventario(self.base)
        primeiro = gerar_divisao(df)
        self.assertEqual(primeiro, gerar_divisao(df.sample(frac=1, random_state=7)))
        self.assertNotEqual(primeiro["fit_ids"], gerar_divisao(df, seed=123)["fit_ids"])
        self.assertEqual(len(primeiro["fit_ids"]), 16)
        self.assertEqual(len(primeiro["calibration_ids"]), 4)

    def test_ausencia_de_sobreposicao_e_test_nao_afeta_particao(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        fit, cal, test = (set(manifest[k]) for k in ("fit_ids", "calibration_ids", "test_ids"))
        self.assertFalse(fit & cal or (fit | cal) & test)
        self.assertEqual(len(fit | cal), 20)
        sem_test = gerar_divisao(df[df.split == "train"])
        self.assertEqual(manifest["fit_ids"], sem_test["fit_ids"])
        self.assertEqual(manifest["calibration_ids"], sem_test["calibration_ids"])

    def test_grupos_transitivos_de_peca_e_aquisicao(self):
        df = inventario(self.base, 24)
        # Cada bloco de seis fica ligado transitivamente pelas duas colunas.
        df["peca_id"] = [f"p{i // 2}" for i in range(24)] + [f"testp{i}" for i in range(3)]
        df["aquisicao_id"] = [f"a{i // 6}-{[0, 1, 1, 2, 2, 3][i % 6]}" for i in range(24)] + [f"testa{i}" for i in range(3)]
        manifest = gerar_divisao(df, calibration_fraction=0.25)
        self.assertEqual(len(manifest["calibration_ids"]), 6)
        fit = df.set_index("caminho").loc[manifest["fit_ids"]]
        cal = df.set_index("caminho").loc[manifest["calibration_ids"]]
        for col in ("peca_id", "aquisicao_id"):
            self.assertFalse(set(fit[col]) & set(cal[col]))
        validar_divisao(manifest, df)

    def test_hashes_iguais_permanecem_na_mesma_particao(self):
        df = inventario(self.base)
        df.loc[1, "sha256"] = df.loc[0, "sha256"]
        manifest = gerar_divisao(df)
        ids = set(df.loc[:1, "caminho"])
        self.assertTrue(ids <= set(manifest["fit_ids"]) or ids <= set(manifest["calibration_ids"]))

    def test_duplicatas_ou_grupos_compartilhados_com_test_sao_rejeitados(self):
        df = inventario(self.base)
        df.loc[20, "sha256"] = df.loc[0, "sha256"]
        with self.assertRaisesRegex(ValueError, "train e test"):
            gerar_divisao(df)
        df = inventario(self.base)
        df["grupo"] = [f"g{i}" for i in range(len(df))]
        df.loc[20, "grupo"] = df.loc[0, "grupo"]
        with self.assertRaisesRegex(ValueError, "train e test"):
            gerar_divisao(df)

    def test_invalidos_e_amostras_insuficientes(self):
        for n in (1, 2, 5, 7):
            with self.subTest(n=n), self.assertRaises(ValueError):
                gerar_divisao(inventario(self.base, n))
        for config in ({"calibration_fraction": 0}, {"calibration_fraction": 1}, {"seed": None},
                       {"seed": True}, {"min_fit": 0}, {"min_calibration": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                gerar_divisao(inventario(self.base), **config)
        df = inventario(self.base)
        df["grupo"] = "unico"
        with self.assertRaises(ValueError):
            gerar_divisao(df[df.split == "train"])

    def test_rejeita_rotulos_caminhos_e_grupos_invalidos(self):
        for col, valor in (("classe", "scratch_head"), ("classe", None), ("split", "valid"),
                           ("caminho", ""), ("sha256", "hash_incompleto")):
            df = inventario(self.base)
            df.loc[0, col] = valor
            with self.subTest(col=col, valor=valor), self.assertRaises(ValueError):
                gerar_divisao(df)
        df = inventario(self.base)
        df.loc[1, "caminho"] = df.loc[0, "caminho"]
        with self.assertRaisesRegex(ValueError, "duplicados"):
            gerar_divisao(df)
        df = inventario(self.base)
        df["peca_id"] = None
        with self.assertRaisesRegex(ValueError, "grupo ausente"):
            gerar_divisao(df)

    def test_grupo_personalizado_e_minimos_configuraveis(self):
        df = inventario(self.base, 8, 0)
        df["lote"] = [f"g{i // 2}" for i in range(8)]
        manifest = gerar_divisao(df, group_columns=["lote"], calibration_fraction=0.5, min_fit=4)
        self.assertEqual(len(manifest["fit_ids"]), 4)
        self.assertEqual(manifest["group_columns"], ["lote"])

    def test_manifesto_reutilizado_sem_sobrescrever(self):
        df = inventario(self.base)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            manifest = obter_ou_criar_divisao(df, path)
            original = (path.read_bytes(), path.stat().st_mtime_ns)
            self.assertEqual(manifest, obter_ou_criar_divisao(df.sample(frac=1), path))
            self.assertEqual(manifest, carregar_divisao(path, df))
            with self.assertRaises(FileExistsError):
                salvar_divisao(manifest, path)
            with self.assertRaisesRegex(ValueError, "diferente"):
                obter_ou_criar_divisao(df, path, seed=123)
            self.assertEqual(original, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_manifesto_detecta_edicao_e_mudanca_de_inventario(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        alterado = copy.deepcopy(manifest)
        alterado["calibration_ids"][0] = alterado["fit_ids"][0]
        with self.assertRaisesRegex(ValueError, "integridade"):
            validar_divisao(alterado, df)
        for col, value in (("sha256", "f" * 64), ("caminho", str(self.base / "novo.png"))):
            atual = df.copy()
            atual.loc[0, col] = value
            with self.assertRaisesRegex(ValueError, "Inventário mudou"):
                validar_divisao(manifest, atual)

    def test_scaler_ignora_calibracao_e_teste(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        calibration = df.caminho.isin(manifest["calibration_ids"])
        self.assertEqual(int(calibration.sum()), len(manifest["calibration_ids"]))
        df.loc[calibration, "feat_0"] = 10000
        # Nem mesmo valores inválidos no teste entram no ajuste ou calibração.
        df.loc[df.split == "test", "feat_0"] = np.nan
        prepared = preparar_caracteristicas(df, manifest, ["feat_0", "feat_1"])
        raw_fit = df.set_index("caminho").loc[manifest["fit_ids"], ["feat_0", "feat_1"]].to_numpy()
        scaler = prepared.preprocessor.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, raw_fit.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(raw_fit))
        np.testing.assert_allclose(prepared.X_fit.mean(axis=0), 0, atol=1e-12)
        self.assertGreater(prepared.X_calibration[:, 0].mean(), 100)

    def test_seletor_e_pca_clonados_e_ajustados_so_no_fit(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        calibration = df.caminho.isin(manifest["calibration_ids"])
        self.assertEqual(int(calibration.sum()), len(manifest["calibration_ids"]))
        df.loc[calibration, "feat_2"] = 999
        selector, pca = VarianceThreshold(), PCA(n_components=1)
        prepared = preparar_caracteristicas(df, manifest, ["feat_0", "feat_1", "feat_2"], selector=selector, pca=pca)
        self.assertFalse(hasattr(selector, "variances_"))
        self.assertFalse(hasattr(pca, "components_"))
        np.testing.assert_array_equal(prepared.preprocessor.named_steps["selector"].get_support(), [True, True, False])
        self.assertEqual(prepared.preprocessor.named_steps["pca"].n_samples_, 16)
        self.assertEqual(prepared.X_fit.shape, (16, 1))
        self.assertEqual(prepared.X_calibration.shape, (4, 1))

    def test_transformar_teste_nao_reajusta(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        prepared = preparar_caracteristicas(df, manifest, ["feat_0", "feat_1"])
        mean = prepared.preprocessor.named_steps["scaler"].mean_.copy()
        df.loc[df.split == "test", "feat_0"] = 999999
        result = transformar_teste(df, manifest, prepared)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_equal(prepared.preprocessor.named_steps["scaler"].mean_, mean)
        with self.assertRaisesRegex(ValueError, "outra divisão"):
            transformar_teste(df, gerar_divisao(df, seed=123), prepared)

    def test_features_invalidas_e_integracao_knn_sintetica(self):
        df = inventario(self.base)
        manifest = gerar_divisao(df)
        for cols in ([], ["classe"], ["feat_0", "feat_0"], ["feat_99"]):
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                preparar_caracteristicas(df, manifest, cols)
        prepared = preparar_caracteristicas(df, manifest, ["feat_0", "feat_1"])
        detector = KNNAnomalyDetector(n_neighbors=2, target_fpr=0.5)
        detector.fit(prepared.X_fit, sample_ids=prepared.fit_ids)
        detector.calibrate(prepared.X_calibration, sample_ids=prepared.calibration_ids)
        self.assertLessEqual(detector.calibration_fpr_, 0.5)
        self.assertEqual(detector.n_calibration_, 4)

    def test_hashes_de_arquivos_temporarios_detectam_mudanca(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = [Path(tmp) / f"{i}.png" for i in range(10)]
            for i, path in enumerate(files):
                path.write_bytes(f"fixture{i}".encode())
            df = pd.DataFrame({"caminho": [str(p) for p in files], "classe": "good", "split": "train"})
            hashed = adicionar_hashes(df)
            manifest = gerar_divisao(hashed)
            files[0].write_bytes(b"conteudo alterado")
            with self.assertRaisesRegex(ValueError, "Inventário mudou"):
                validar_divisao(manifest, adicionar_hashes(df))

    def test_caminhos_relativos_com_base_explicita(self):
        df = pd.DataFrame({"caminho": ["train/a.png"]})
        normalized = normalizar_caminhos(df, base_dir=self.base)
        self.assertEqual(normalized["caminho"].tolist(), [str(self.base / "train" / "a.png")])


if __name__ == "__main__":
    unittest.main()
