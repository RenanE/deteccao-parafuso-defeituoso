import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.dataset import (
    CATEGORIAS_TESTE,
    DATA_DIR_PADRAO,
    DatasetInvalidoError,
    carregar_caminhos_dataset,
    inspecionar_dataset,
    main,
    resolver_data_dir,
    resolver_confirmacao_treino,
)


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        ambiente = patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": ""})
        ambiente.start()
        self.addCleanup(ambiente.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dados com acentuação"
        self.train = self.imagem("train/good/b.png", 80)
        self.test = self.imagem("test/good/a.png", 100)

    def imagem(self, relativo, tom=80):
        caminho = self.root / relativo
        caminho.parent.mkdir(parents=True, exist_ok=True)
        imagem = np.full((48, 96, 3), 240, dtype=np.uint8)
        imagem[16:32, 12:84] = tom
        ok, dados = cv2.imencode(".png", imagem)
        self.assertTrue(ok)
        caminho.write_bytes(dados.tobytes())
        return caminho

    def treino_plano(self):
        # Apenas o fixture temporário; nunca modifica o dataset real.
        self.train.unlink()
        self.train.parent.rmdir()
        return self.imagem("train/sem_rotulo.png")

    def test_plano_rejeitado_sem_declaracao_explicita(self):
        self.treino_plano()
        with self.assertRaisesRegex(DatasetInvalidoError, "Treino plano sem confirmação"):
            carregar_caminhos_dataset(self.root)
        relatorio = inspecionar_dataset(self.root)
        self.assertTrue(relatorio["estrutura_valida"])
        self.assertEqual(relatorio["layout_treino"], "train plano")
        self.assertIn("PENDENTE", relatorio["rotulos_treino"])
        self.assertIsNone(relatorio["dataset"])
        self.assertEqual(relatorio["imagens_legiveis"], 2)

    def test_plano_confirmado_preserva_seis_categorias_e_conteudo(self):
        treino = self.treino_plano()
        for categoria in CATEGORIAS_TESTE:
            if categoria != "good":
                self.imagem(f"test/{categoria}/a.png", 100)
        antes = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*.png")}
        df = carregar_caminhos_dataset(self.root, train_flat_is_good=True)
        self.assertEqual(df.loc[df.split == "train", "classe"].tolist(), ["good"])
        self.assertEqual(df.loc[df.split == "train", "caminho"].tolist(), [str(treino)])
        self.assertEqual(set(df.loc[df.split == "test", "classe"]), set(CATEGORIAS_TESTE))
        self.assertEqual((df.loc[df.split == "test", "classe"] != "good").sum(), 5)
        relatorio = inspecionar_dataset(self.root, train_flat_is_good=True)
        self.assertEqual(relatorio["erros"], [])
        self.assertEqual(relatorio["categorias_teste_ausentes"], [])
        self.assertIn("explicitamente", relatorio["rotulos_treino"])
        depois = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*.png")}
        self.assertEqual(antes, depois)

    def test_declaracao_ambiente_e_override_negativo(self):
        self.treino_plano()
        with patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": "true"}):
            self.assertEqual(len(carregar_caminhos_dataset(self.root)), 2)
            with self.assertRaisesRegex(DatasetInvalidoError, "sem confirmação"):
                carregar_caminhos_dataset(self.root, train_flat_is_good=False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(self.root), "--no-train-flat-is-good"]), 1)
        with patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": "false"}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(self.root), "--train-flat-is-good"]), 0)

    def test_configuracoes_ambiguas_nao_habilitam_treino(self):
        self.treino_plano()
        for valor in ("1", "yes", "good", "tru"):
            with self.subTest(valor=valor), patch.dict(os.environ, {"TRAIN_FLAT_IS_GOOD": valor}):
                with self.assertRaisesRegex(DatasetInvalidoError, "deve ser"):
                    carregar_caminhos_dataset(self.root)
                self.assertIsNone(inspecionar_dataset(self.root)["dataset"])
        for valor in ("false", "true", 1, 0):
            with self.subTest(valor=valor), self.assertRaises(DatasetInvalidoError):
                resolver_confirmacao_treino(valor)

    def test_confirmacao_nao_permite_layout_misto(self):
        self.imagem("train/solta.png")
        for confirmado in (False, True):
            with self.subTest(confirmado=confirmado):
                with self.assertRaisesRegex(DatasetInvalidoError, "Treino misto"):
                    carregar_caminhos_dataset(self.root, train_flat_is_good=confirmado)

    def test_confirmacao_nao_permite_classe_inesperada(self):
        self.treino_plano()
        self.imagem("train/scratch_head/a.png")
        with self.assertRaisesRegex(DatasetInvalidoError, "Classe inesperada"):
            carregar_caminhos_dataset(self.root, train_flat_is_good=True)

    def test_plano_confirmado_rejeita_treino_vazio_corrompido_e_formato_invalido(self):
        treino = self.treino_plano()
        treino.unlink()
        with self.assertRaisesRegex(DatasetInvalidoError, "Categoria vazia"):
            carregar_caminhos_dataset(self.root, train_flat_is_good=True)
        treino.write_bytes(b"invalida")
        with self.assertRaisesRegex(DatasetInvalidoError, "Imagem ilegível"):
            carregar_caminhos_dataset(self.root, train_flat_is_good=True)
        relatorio = inspecionar_dataset(self.root, train_flat_is_good=True)
        self.assertTrue(relatorio["estrutura_valida"])
        self.assertEqual(len(relatorio["erros_leitura"]), 1)
        self.assertIsNone(relatorio["dataset"])
        self.imagem("train/outro.jpg")
        with self.assertRaisesRegex(DatasetInvalidoError, "Formato de imagem não aceito"):
            carregar_caminhos_dataset(self.root, train_flat_is_good=True)

    def test_train_good_independe_de_declaracao_para_layout_plano(self):
        for confirmado in (False, True):
            relatorio = inspecionar_dataset(self.root, train_flat_is_good=confirmado)
            self.assertEqual(relatorio["erros"], [])
            self.assertEqual(relatorio["layout_treino"], "train/good")
            self.assertIn("indicado pela pasta", relatorio["rotulos_treino"])

    def test_carrega_rotulos_ordem_estavel_e_png_maiusculo(self):
        primeiro = self.imagem("train/good/a.PNG")
        self.imagem("test/scratch_head/c.png")
        df = carregar_caminhos_dataset(self.root)
        self.assertEqual(list(df.columns), ["caminho", "classe", "split"])
        self.assertEqual(len(df), 4)
        self.assertEqual(df.iloc[0]["caminho"], str(primeiro))
        self.assertEqual(set(df.loc[df.split == "train", "classe"]), {"good"})
        self.assertEqual(set(df.loc[df.split == "test", "classe"]), {"good", "scratch_head"})

    def test_variavel_ambiente_consultada_em_cada_chamada(self):
        with patch.dict(os.environ, {"DATASET_DIR": str(self.root)}):
            self.assertEqual(resolver_data_dir(), self.root)
            self.assertEqual(len(carregar_caminhos_dataset()), 2)
            self.assertEqual(inspecionar_dataset()["imagens_legiveis"], 2)
        with patch.dict(os.environ, {"DATASET_DIR": str(self.root / "ausente")}):
            with self.assertRaises(DatasetInvalidoError):
                carregar_caminhos_dataset()

    def test_caminho_explicito_tem_prioridade_sobre_ambiente(self):
        with patch.dict(os.environ, {"DATASET_DIR": str(self.root / "ausente")}):
            self.assertEqual(len(carregar_caminhos_dataset(self.root)), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(self.root)]), 0)

    def test_padrao_sem_variavel_ou_com_variavel_vazia(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolver_data_dir(), DATA_DIR_PADRAO)
        with patch.dict(os.environ, {"DATASET_DIR": ""}):
            self.assertEqual(resolver_data_dir(), DATA_DIR_PADRAO)

    def test_expande_til_em_variavel_e_argumento(self):
        with patch.dict(os.environ, {"DATASET_DIR": "~/Downloads/data"}):
            esperado = Path.home() / "Downloads/data"
            self.assertEqual(resolver_data_dir(), esperado)
            self.assertEqual(resolver_data_dir("~/Downloads/data"), esperado)

    def test_cli_utiliza_variavel_ambiente(self):
        with patch.dict(os.environ, {"DATASET_DIR": str(self.root)}):
            saida = io.StringIO()
            with contextlib.redirect_stdout(saida):
                self.assertEqual(main([]), 0)
            self.assertIn(str(self.root), saida.getvalue())

    def test_inspeciona_leitura_mesmo_sem_train_good(self):
        self.train.unlink()
        self.train.parent.rmdir()
        self.imagem("train/sem_rotulo.png")
        (self.root / "train/corrompida.png").write_bytes(b"invalida")
        relatorio = inspecionar_dataset(self.root)
        self.assertIsNone(relatorio["dataset"])
        self.assertEqual(relatorio["imagens_legiveis"], 2)
        self.assertEqual(len(relatorio["erros_leitura"]), 1)
        self.assertIn("corrompida.png", relatorio["erros_leitura"][0])
        self.assertEqual(relatorio["quantidades"][("train", "(fora de categoria)")], 2)
        self.assertNotIn(("train", "good"), relatorio["quantidades"])

    def test_diretorios_obrigatorios_ausentes(self):
        for relativo in ("ausente", "somente_train", "sem_good", "sem_test"):
            with self.subTest(caso=relativo):
                raiz = Path(self.temp.name) / relativo
                if relativo != "ausente":
                    (raiz / "train").mkdir(parents=True)
                if relativo == "sem_test":
                    (raiz / "train/good").mkdir()
                with self.assertRaisesRegex(DatasetInvalidoError, "obrigatório"):
                    carregar_caminhos_dataset(raiz)

    def test_rejeita_classe_inesperada_no_treino_mesmo_vazia(self):
        (self.root / "train/scratch_head").mkdir()
        with self.assertRaisesRegex(DatasetInvalidoError, "Classe inesperada no treino"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_imagem_solto_no_treino(self):
        self.imagem("train/solta.png")
        with self.assertRaisesRegex(DatasetInvalidoError, "fora de train/good"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_categoria_desconhecida_no_teste(self):
        self.imagem("test/desconhecida/a.png")
        with self.assertRaisesRegex(DatasetInvalidoError, "Categoria de teste inesperada"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_imagem_solto_no_teste(self):
        self.imagem("test/solta.png")
        with self.assertRaisesRegex(DatasetInvalidoError, "fora de test/<categoria>"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_categoria_vazia(self):
        for caminho in (self.train, self.test):
            with self.subTest(pasta=caminho.parent):
                original = caminho.read_bytes()
                caminho.unlink()
                with self.assertRaisesRegex(DatasetInvalidoError, "Categoria vazia"):
                    carregar_caminhos_dataset(self.root)
                caminho.write_bytes(original)

    def test_rejeita_teste_sem_categorias(self):
        self.test.unlink()
        self.test.parent.rmdir()
        with self.assertRaisesRegex(DatasetInvalidoError, "Teste vazio"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_subdiretorio_na_categoria(self):
        self.imagem("train/good/aninhada/a.png")
        with self.assertRaisesRegex(DatasetInvalidoError, "Subdiretório inesperado"):
            carregar_caminhos_dataset(self.root)

    def test_rejeita_png_vazio_ou_corrompido_em_ambos_splits(self):
        for caminho in (self.train, self.test):
            original = caminho.read_bytes()
            for conteudo in (b"", b"nao e uma imagem"):
                with self.subTest(caminho=caminho, conteudo=conteudo):
                    caminho.write_bytes(conteudo)
                    with self.assertRaisesRegex(DatasetInvalidoError, "Imagem ilegível") as erro:
                        carregar_caminhos_dataset(self.root)
                    self.assertIn(str(caminho), str(erro.exception))
            caminho.write_bytes(original)

    def test_inspecao_conta_extensoes_e_duplicatas_sem_modificar(self):
        self.test.write_bytes(self.train.read_bytes())
        (self.root / "train/good/notas.txt").write_text("auxiliar")
        antes = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        relatorio = inspecionar_dataset(self.root)
        self.assertEqual(relatorio["erros"], [])
        self.assertEqual(relatorio["quantidades"], {("train", "good"): 1, ("test", "good"): 1})
        self.assertEqual(relatorio["extensoes"][("train", ".txt")], 1)
        self.assertEqual(len(relatorio["duplicatas"]), 1)
        grupo = next(iter(relatorio["duplicatas"].values()))
        self.assertEqual({i["split"] for i in grupo}, {"train", "test"})
        depois = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(antes, depois)
        saida = io.StringIO()
        with contextlib.redirect_stdout(saida):
            self.assertEqual(main(["--data-dir", str(self.root)]), 0)
        self.assertIn("ENTRE TREINO E TESTE", saida.getvalue())

    def test_inspecao_informa_formato_nao_aceito_e_falha(self):
        self.imagem("test/good/outro.jpg")
        relatorio = inspecionar_dataset(self.root)
        self.assertEqual(relatorio["extensoes"][("test", ".jpg")], 1)
        self.assertIn("Formato de imagem não aceito", relatorio["erros"][0])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data-dir", str(self.root)]), 1)

    def test_cli_em_outro_diretorio_e_dataset_ausente(self):
        script = Path(__file__).resolve().parents[1] / "src/dataset.py"
        for raiz, codigo in ((self.root, 0), (self.root / "ausente", 1)):
            with self.subTest(raiz=raiz):
                resultado = subprocess.run(
                    [sys.executable, str(script), "--data-dir", str(raiz)],
                    cwd=self.temp.name, capture_output=True, text=True, check=False,
                )
                self.assertEqual(resultado.returncode, codigo, resultado.stderr)
                self.assertNotIn("Traceback", resultado.stderr)


if __name__ == "__main__":
    unittest.main()
