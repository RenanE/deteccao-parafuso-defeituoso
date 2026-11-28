import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src import extract_features as extracao
from src.dataset import DatasetInvalidoError


class ExtractionSafetyTests(unittest.TestCase):
    def setUp(self):
        ambiente = patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": ""})
        ambiente.start()
        self.addCleanup(ambiente.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relativo, tom in (("train/good/a.png", 70), ("test/scratch_head/b.png", 90)):
            caminho = self.root / relativo
            caminho.parent.mkdir(parents=True)
            imagem = np.full((48, 96, 3), 240, dtype=np.uint8)
            imagem[16:32, 12:84] = tom
            ok, dados = cv2.imencode(".png", imagem)
            self.assertTrue(ok)
            caminho.write_bytes(dados.tobytes())
        self.vetor = np.zeros(extracao.TAMANHO_VETOR_ESPERADO, dtype=np.float32)

    def executar_simulada(self, **kwargs):
        with patch.object(extracao, "criar_banco_gabor", return_value=[]), \
             patch.object(extracao, "processar_imagem", **kwargs) as processar, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            destinos = extracao.executar_extracao_lote(self.root)
        return destinos, processar

    def test_usa_carregador_e_preserva_classes_e_schema(self):
        with patch.object(extracao, "carregar_caminhos_dataset", wraps=extracao.carregar_caminhos_dataset) as carregar:
            destinos, processar = self.executar_simulada(return_value=(self.vetor, True))
        carregar.assert_called_once_with(self.root, train_flat_is_good=None)
        self.assertEqual(processar.call_count, 2)
        for split, classe in (("train", "good"), ("test", "scratch_head")):
            df = pd.read_csv(destinos[split])
            self.assertEqual(df.shape, (1, 25416))
            self.assertEqual(df["classe"].tolist(), [classe])
            self.assertEqual(df["split"].tolist(), [split])
            self.assertEqual(df.columns[-1], "feat_25412")

    def test_recusa_cada_csv_existente_antes_de_processar(self):
        pasta = self.root / "processed"
        pasta.mkdir()
        for split in ("train", "test"):
            with self.subTest(split=split):
                existente = pasta / f"features_{split}.csv"
                existente.write_bytes(b"conteudo anterior")
                with patch.object(extracao, "processar_imagem") as processar:
                    with self.assertRaisesRegex(FileExistsError, "nada será sobrescrito"):
                        extracao.executar_extracao_lote(self.root)
                    processar.assert_not_called()
                self.assertEqual(existente.read_bytes(), b"conteudo anterior")
                self.assertEqual(list(pasta.iterdir()), [existente])
                existente.unlink()

    def test_treino_plano_bloqueia_antes_de_extrair_e_aceita_declaracao(self):
        # Reorganiza apenas o fixture temporário criado por este teste.
        imagem = self.root / "train/good/a.png"
        dados = imagem.read_bytes()
        imagem.unlink()
        imagem.parent.rmdir()
        (self.root / "train/a.png").write_bytes(dados)
        with patch.object(extracao, "processar_imagem") as processar:
            with self.assertRaisesRegex(DatasetInvalidoError, "sem confirmação"):
                extracao.executar_extracao_lote(self.root)
            processar.assert_not_called()
        self.assertFalse((self.root / "processed").exists())
        with patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": "true"}):
            destinos, _ = self.executar_simulada(return_value=(self.vetor, True))
        self.assertEqual(pd.read_csv(destinos["train"])["classe"].tolist(), ["good"])
        self.assertEqual(pd.read_csv(destinos["test"])["classe"].tolist(), ["scratch_head"])

    def test_cli_encaminha_declaracao_sem_extrair(self):
        for flag, esperado in (("--train-flat-is-good", True), ("--no-train-flat-is-good", False)):
            with patch.object(extracao, "executar_extracao_lote") as lote:
                self.assertEqual(extracao.main(["--data-dir", str(self.root), flag]), 0)
                lote.assert_called_once_with(self.root, train_flat_is_good=esperado)

    def test_imagem_ilegivel_nao_processa_nem_cria_csv(self):
        (self.root / "test/scratch_head/b.png").write_bytes(b"corrompida")
        with patch.object(extracao, "processar_imagem") as processar:
            with self.assertRaisesRegex(DatasetInvalidoError, "Imagem ilegível"):
                extracao.executar_extracao_lote(self.root)
            processar.assert_not_called()
        self.assertFalse((self.root / "processed").exists())

    def test_extracao_respeita_ambiente_e_prioridade_sem_processar(self):
        # A proteção de CSV comprova o destino escolhido antes de qualquer lote.
        pasta = self.root / "processed"
        pasta.mkdir()
        (pasta / "features_train.csv").write_bytes(b"preservar")
        with patch.object(extracao, "processar_imagem") as processar:
            with patch.dict(os.environ, {"DATASET_DIR": str(self.root)}):
                with self.assertRaises(FileExistsError):
                    extracao.executar_extracao_lote()
            with patch.dict(os.environ, {"DATASET_DIR": str(self.root / "ausente")}):
                with self.assertRaises(FileExistsError):
                    extracao.executar_extracao_lote(self.root)
            processar.assert_not_called()
        self.assertEqual((pasta / "features_train.csv").read_bytes(), b"preservar")

    def test_falha_na_segunda_imagem_nao_publica_csv_de_treino(self):
        with self.assertRaisesRegex(DatasetInvalidoError, "b.png"):
            self.executar_simulada(side_effect=[(self.vetor, True), ValueError("sem componente")])
        self.assertFalse((self.root / "processed").exists())

    def test_rejeita_vetores_invalidos_sem_csv(self):
        for vetor, ok in ((None, False), (np.zeros(3), True), (np.full(25413, np.nan), True)):
            with self.subTest(ok=ok, tamanho=None if vetor is None else vetor.size):
                with self.assertRaises(DatasetInvalidoError):
                    self.executar_simulada(return_value=(vetor, ok))
                self.assertFalse((self.root / "processed").exists())

    def test_falha_de_escrita_remove_apenas_csvs_novos(self):
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disco cheio")):
            with self.assertRaisesRegex(OSError, "disco cheio"):
                self.executar_simulada(return_value=(self.vetor, True))
        self.assertEqual(list((self.root / "processed").glob("*.csv")), [])
        self.assertTrue((self.root / "train/good/a.png").is_file())

    def test_csv_criado_durante_processamento_nao_e_sobrescrito(self):
        destino = self.root / "processed/features_test.csv"

        def simular_concorrencia(*args):
            destino.parent.mkdir(exist_ok=True)
            destino.write_bytes(b"outra execucao")
            return self.vetor, True

        with self.assertRaises(FileExistsError):
            self.executar_simulada(side_effect=simular_concorrencia)
        self.assertEqual(destino.read_bytes(), b"outra execucao")
        self.assertFalse((destino.parent / "features_train.csv").exists())

    def test_pipeline_real_em_uma_imagem_sintetica(self):
        # Smoke test de compatibilidade; não executa lote nem treina modelos.
        vetor, ok = extracao.processar_imagem(
            self.root / "train/good/a.png", extracao.criar_banco_gabor()
        )
        self.assertTrue(ok)
        self.assertEqual(vetor.shape, (25413,))
        self.assertTrue(np.isfinite(vetor).all())


if __name__ == "__main__":
    unittest.main()
