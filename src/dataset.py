"""Descoberta, validação e inspeção somente de leitura do dataset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CATEGORIAS_TESTE = [
    "good", "manipulated_front", "scratch_head", "scratch_neck",
    "thread_side", "thread_top",
]
DATA_DIR_PADRAO = Path(__file__).resolve().parent.parent / "data"
# A extração continua aceitando PNG; outros formatos são apenas inspecionados.
EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


class DatasetInvalidoError(ValueError):
    """Dataset ausente, fora da estrutura prevista ou com imagem ilegível."""


def resolver_data_dir(data_dir=None):
    """Prioridade: argumento explícito > DATASET_DIR não vazia > data/ local.

    Consulta o ambiente a cada chamada. Caminhos relativos explícitos ou da
    variável são relativos ao diretório de execução; o padrão é fixo no projeto.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATASET_DIR") or DATA_DIR_PADRAO
    return Path(data_dir).expanduser()


def resolver_confirmacao_treino(train_flat_is_good=None):
    """Declaração explícita: bool na API > TRAIN_FLAT_IS_GOOD > False.

    Não comprova a origem nem os rótulos reais das imagens.
    """
    if train_flat_is_good is not None:
        if type(train_flat_is_good) is not bool:
            raise DatasetInvalidoError("train_flat_is_good deve ser bool (True ou False).")
        return train_flat_is_good
    valor = os.environ.get("TRAIN_FLAT_IS_GOOD", "").strip().lower()
    if valor not in ("", "true", "false"):
        raise DatasetInvalidoError("TRAIN_FLAT_IS_GOOD deve ser 'true' ou 'false'; vazio equivale a false.")
    return valor == "true"


def adicionar_argumento_confirmacao(parser):
    parser.add_argument(
        "--train-flat-is-good", action=argparse.BooleanOptionalAction, default=None,
        help="Declara que TODOS os PNGs de train/ são good. Sobrescreve TRAIN_FLAT_IS_GOOD; padrão: não confirmado.",
    )


def ler_imagem(caminho):
    """Decodifica BGR preservando caminhos Unicode, sem modificar a imagem."""
    caminho = Path(caminho)
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
    except (OSError, cv2.error) as exc:
        raise DatasetInvalidoError(f"Imagem ilegível: {caminho}: {exc}") from exc
    if imagem is None:
        raise DatasetInvalidoError(f"Imagem ilegível ou vazia: {caminho}")
    return imagem


def _exigir_diretorio(caminho):
    if not caminho.is_dir():
        raise DatasetInvalidoError(f"Diretório obrigatório ausente ou inválido: {caminho}")


def _imagens_categoria(pasta):
    imagens = []
    for caminho in sorted(pasta.iterdir()):
        if caminho.is_dir():
            raise DatasetInvalidoError(f"Subdiretório inesperado: {caminho}. As imagens devem estar diretamente na categoria.")
        extensao = caminho.suffix.lower()
        if extensao in EXTENSOES_IMAGEM and extensao != ".png":
            raise DatasetInvalidoError(f"Formato de imagem não aceito: {caminho}. A extração atual utiliza apenas PNG.")
        if extensao == ".png":
            imagens.append(caminho)
    if not imagens:
        raise DatasetInvalidoError(f"Categoria vazia ou sem imagens PNG: {pasta}")
    return imagens


def _descobrir_estrutura(data_path):
    """Valida pastas/extensões sem decodificar nem rotular o treino plano."""
    try:
        train = data_path / "train"
        test = data_path / "test"
        for pasta in (data_path, train, test):
            _exigir_diretorio(pasta)

        imagens_soltas = []
        for caminho in sorted(train.iterdir()):
            if caminho.is_dir() and caminho.name != "good":
                raise DatasetInvalidoError(f"Classe inesperada no treino: {caminho}. Somente 'good' é permitida.")
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM:
                imagens_soltas.append(caminho)
        good = train / "good"
        if good.exists():
            _exigir_diretorio(good)
            if imagens_soltas:
                raise DatasetInvalidoError("Treino misto: imagem fora de train/good junto com a pasta good. Escolha um único layout no mapeamento; não há mesclagem automática.")
            layout = "train/good"
            grupos = [("train", "good", _imagens_categoria(good))]
        else:
            layout = "train plano"
            grupos = [("train", None, _imagens_categoria(train))]

        categorias = []
        for caminho in sorted(test.iterdir()):
            if caminho.is_dir():
                if caminho.name not in CATEGORIAS_TESTE:
                    raise DatasetInvalidoError(f"Categoria de teste inesperada: {caminho}")
                categorias.append(caminho)
            elif caminho.suffix.lower() in EXTENSOES_IMAGEM:
                raise DatasetInvalidoError(f"Imagem fora de test/<categoria>: {caminho}")
        if not categorias:
            raise DatasetInvalidoError(f"Teste vazio: {test}. É necessária pelo menos uma categoria conhecida.")

        grupos.extend(("test", pasta.name, _imagens_categoria(pasta)) for pasta in categorias)
        return layout, grupos
    except OSError as exc:
        raise DatasetInvalidoError(f"Não foi possível acessar o dataset em {data_path}: {exc}") from exc


def _exigir_rotulos(layout, confirmado):
    if layout == "train plano" and not confirmado:
        raise DatasetInvalidoError(
            "Treino plano sem confirmação dos rótulos. Após verificar TODAS as imagens, "
            "declare TRAIN_FLAT_IS_GOOD=true ou --train-flat-is-good. "
            "A formulação One-Class, sozinha, não confirma que os arquivos são good."
        )


def carregar_caminhos_dataset(data_dir=None, *, train_flat_is_good=None):
    """Valida estrutura, declaração dos rótulos e leitura dos PNGs.

    Aceita train/good ou train plano, este apenas mediante declaração explícita.
    Não mistura layouts nem transforma outras classes de treino em good.
    """
    layout, grupos = _descobrir_estrutura(resolver_data_dir(data_dir))
    confirmado = resolver_confirmacao_treino(train_flat_is_good)
    _exigir_rotulos(layout, confirmado)
    dados = []
    for split, classe, imagens in grupos:
        # None só ocorre no treino plano, cuja declaração já foi exigida acima.
        classe = "good" if classe is None else classe
        for imagem in imagens:
            ler_imagem(imagem)
            dados.append({"caminho": str(imagem), "classe": classe, "split": split})
    return pd.DataFrame(dados, columns=["caminho", "classe", "split"])


def resumo_dataset(df):
    """Contagens das imagens validadas por categoria e split."""
    if df.empty:
        return "Nenhuma imagem validada."
    return pd.crosstab(df["classe"], df["split"], margins=True)


def inspecionar_dataset(data_dir=None, *, train_flat_is_good=None):
    """Inventaria train/test e calcula SHA-256 dos arquivos de imagem.

    Duplicatas são igualdade de bytes, não similaridade visual. O inventário é
    exibido mesmo se a validação falhar; processed/ e ground_truth/ não entram.
    """
    raiz = resolver_data_dir(data_dir)
    extensoes = Counter()
    quantidades = Counter()
    hashes = defaultdict(list)
    erros = []
    legiveis = 0
    erros_leitura = []
    for split in ("train", "test"):
        pasta = raiz / split
        if not pasta.is_dir():
            continue
        try:
            for caminho in sorted(pasta.rglob("*")):
                if not caminho.is_file():
                    continue
                extensao = caminho.suffix.lower() or "(sem extensão)"
                extensoes[(split, extensao)] += 1
                if extensao not in EXTENSOES_IMAGEM:
                    continue
                relativo = caminho.relative_to(pasta)
                categoria = relativo.parts[0] if len(relativo.parts) > 1 else "(fora de categoria)"
                quantidades[(split, categoria)] += 1
                # Verifica todas as imagens, mesmo quando a estrutura é inválida.
                # Isso não atribui rótulos nem libera o dataset para extração.
                try:
                    ler_imagem(caminho)
                    legiveis += 1
                except DatasetInvalidoError as exc:
                    erros_leitura.append(str(exc))
                digest = hashlib.sha256()
                try:
                    with caminho.open("rb") as arquivo:
                        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
                            digest.update(bloco)
                    hashes[digest.hexdigest()].append({"split": split, "caminho": str(caminho)})
                except OSError as exc:
                    erros.append(f"Falha ao calcular hash de {caminho}: {exc}")
        except OSError as exc:
            erros.append(f"Falha ao inspecionar {pasta}: {exc}")

    estrutura_valida = False
    layout = None
    rotulos = "não verificados (estrutura inválida)"
    categorias_ausentes = list(CATEGORIAS_TESTE)
    df = None
    try:
        layout, grupos = _descobrir_estrutura(raiz)
        estrutura_valida = True
        presentes = {classe for split, classe, _ in grupos if split == "test"}
        categorias_ausentes = sorted(set(CATEGORIAS_TESTE) - presentes)
        rotulos = "configuração inválida"
        confirmado = resolver_confirmacao_treino(train_flat_is_good)
        rotulos = (
            "good indicado pela pasta train/good; conteúdo não certificado"
            if layout == "train/good" else
            "good declarado explicitamente na configuração; conteúdo não certificado"
            if confirmado else
            "PENDENTE: nenhuma declaração de que todas as imagens de train/ são good"
        )
        _exigir_rotulos(layout, confirmado)
        if not erros_leitura:
            df = carregar_caminhos_dataset(raiz, train_flat_is_good=confirmado)
    except DatasetInvalidoError as exc:
        erros.append(str(exc))

    return {
        "dataset": df,
        "quantidades": dict(quantidades),
        "extensoes": dict(extensoes),
        "duplicatas": {h: itens for h, itens in hashes.items() if len(itens) > 1},
        "imagens_legiveis": legiveis,
        "erros_leitura": erros_leitura,
        "estrutura_valida": estrutura_valida,
        "layout_treino": layout,
        "rotulos_treino": rotulos,
        "categorias_teste_ausentes": categorias_ausentes,
        "erros": erros + erros_leitura,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspeciona o dataset sem modificar imagens ou gerar CSVs.")
    parser.add_argument("--data-dir", type=Path, help="Pasta com train/ e test/. Prioridade: --data-dir > DATASET_DIR > data/ na raiz do projeto.")
    adicionar_argumento_confirmacao(parser)
    args = parser.parse_args(argv)
    data_dir = resolver_data_dir(args.data_dir)
    relatorio = inspecionar_dataset(data_dir, train_flat_is_good=args.train_flat_is_good)
    print(f"Dataset: {data_dir}")
    print(f"Estrutura: {'VÁLIDA' if relatorio['estrutura_valida'] else 'INVÁLIDA'} (layout: {relatorio['layout_treino'] or 'não identificado'})")
    print(f"Rótulos do treino: {relatorio['rotulos_treino']}")
    print("Confirmação documental: não verificada automaticamente pela inspeção.")
    if relatorio["estrutura_valida"]:
        print(f"Categorias de teste ausentes: {', '.join(relatorio['categorias_teste_ausentes']) or 'nenhuma (6/6 presentes)'}")
    print("\nArquivos de imagem encontrados (antes da validação):")
    for (split, categoria), total in sorted(relatorio["quantidades"].items()):
        print(f"  {split}/{categoria}: {total}")
    print("\nExtensões de todos os arquivos em train/test:")
    for (split, extensao), total in sorted(relatorio["extensoes"].items()):
        print(f"  {split} {extensao}: {total}")
    print(f"\nLeitura das imagens encontradas: {relatorio['imagens_legiveis']} legíveis; {len(relatorio['erros_leitura'])} ilegíveis.")
    print("\nPossíveis duplicatas por SHA-256 (igualdade de bytes):")
    if not relatorio["duplicatas"]:
        print("  Nenhum grupo encontrado entre os arquivos lidos.")
    for digest, itens in sorted(relatorio["duplicatas"].items()):
        aviso = " [ENTRE TREINO E TESTE: possível vazamento]" if len({i["split"] for i in itens}) > 1 else ""
        print(f"  {digest}{aviso}")
        for item in itens:
            print(f"    {item['caminho']}")
    if relatorio["erros"]:
        print("\nVALIDAÇÃO FALHOU:")
        for erro in relatorio["erros"]:
            print(f"  {erro}")
        return 1
    print("\nImagens PNG validadas por categoria/split:")
    print(resumo_dataset(relatorio["dataset"]))
    print("\nValidação concluída. Revise eventuais duplicatas antes de prosseguir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
