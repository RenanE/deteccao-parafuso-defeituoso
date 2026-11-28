import argparse
from contextlib import ExitStack
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suporta tanto python -m src.extract_features quanto execução como script.
if __package__:
    from .dataset import DatasetInvalidoError, adicionar_argumento_confirmacao, carregar_caminhos_dataset, ler_imagem, resolver_data_dir
    from .features import criar_banco_gabor, extrair_todas_caracteristicas
    from .preprocessing import aplicar_clahe, converter_para_cinza, remover_ruido
    from .segmentation import aplicar_morfologia, binarizar_otsu, extrair_maior_componente_e_roi
else:
    from dataset import DatasetInvalidoError, adicionar_argumento_confirmacao, carregar_caminhos_dataset, ler_imagem, resolver_data_dir
    from features import criar_banco_gabor, extrair_todas_caracteristicas
    from preprocessing import aplicar_clahe, converter_para_cinza, remover_ruido
    from segmentation import aplicar_morfologia, binarizar_otsu, extrair_maior_componente_e_roi


TAMANHO_VETOR_ESPERADO = 25413

def processar_imagem(
    caminho_img: Path,
    banco_gabor: list[np.ndarray],
    *,
    hog_config=None,
) -> tuple[np.ndarray, bool]:
    """
    Aplica o pipeline completo da Milestone 2 (pré-processamento/ROI) e extrai o vetor unificado de características.
    """
    
    # Mesmo decodificador usado pela validação, incluindo caminhos Unicode.
    img_bgr = ler_imagem(caminho_img)

    # Pipeline Milestone 2
    cinza = converter_para_cinza(img_bgr)
    suavizada = remover_ruido(cinza, metodo="gaussian", ksize=(5, 5))
    _, mascara_otsu = binarizar_otsu(suavizada, inverter=True)

    mascara_limpa = aplicar_morfologia(
        mascara_otsu,
        tamanho_abertura=3,
        tamanho_fechamento=11,
        preencher_lacunas=True,
    )

    roi_cinza, roi_mascara, sucesso = extrair_maior_componente_e_roi(
        cinza,
        mascara_limpa,
        alinhar_eixo=True
    )

    if roi_cinza is None or roi_mascara is None:
        return None, False

    # Aplica CLAHE na ROI recortada
    roi_clahe = aplicar_clahe(
        roi_cinza,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    )

    # Extração Milestone 3
    vetor = extrair_todas_caracteristicas(
        roi_clahe,
        roi_mascara,
        banco_gabor=banco_gabor,
        hog_config=hog_config,
    )

    return vetor, True

def executar_extracao_lote(data_dir=None, *, train_flat_is_good=None):
    """Valida ambos os splits antes de processar e nunca sobrescreve CSVs.

    Nenhum CSV é publicado se uma imagem falhar ou produzir vetor inválido.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = resolver_data_dir(data_dir).resolve()
    out_dir = data_dir / "processed"
    destinos = {split: out_dir / f"features_{split}.csv" for split in ("train", "test")}
    for destino in destinos.values():
        # Inclui links simbólicos quebrados para impedir substituição acidental.
        if destino.exists() or destino.is_symlink():
            raise FileExistsError(f"CSV já existente; nada será sobrescrito: {destino}")

    dataset = carregar_caminhos_dataset(data_dir, train_flat_is_good=train_flat_is_good)
    if dataset.empty or set(dataset["split"]) != {"train", "test"}:
        raise DatasetInvalidoError("Treino e teste devem conter imagens validadas.")
    if not dataset.loc[dataset["split"] == "train", "classe"].eq("good").all():
        raise DatasetInvalidoError("Classe inesperada no treino. Somente 'good' é permitida.")

    banco_gabor = criar_banco_gabor(orientations=4, lambdas=[8.0, 12.0, 16.0])
    linhas = {"train": [], "test": []}
    for amostra in tqdm(dataset.itertuples(index=False), total=len(dataset), desc="Extração"):
        img_path = Path(amostra.caminho)
        try:
            vetor, ok = processar_imagem(img_path, banco_gabor)
        except (ValueError, OSError, cv2.error) as exc:
            raise DatasetInvalidoError(f"Falha ao processar {img_path}: {exc}") from exc
        if not ok or vetor is None:
            raise DatasetInvalidoError(f"Falha ao extrair características: {img_path}")
        vetor = np.asarray(vetor)
        if vetor.shape != (TAMANHO_VETOR_ESPERADO,) or not np.isfinite(vetor).all():
            raise DatasetInvalidoError(f"Vetor inválido em {img_path}: esperado {TAMANHO_VETOR_ESPERADO} atributos finitos.")
        caminho = img_path.relative_to(base_dir) if img_path.is_relative_to(base_dir) else img_path
        linha = {"caminho": str(caminho), "classe": amostra.classe, "split": amostra.split}
        linha.update({f"feat_{idx}": val for idx, val in enumerate(vetor)})
        linhas[amostra.split].append(linha)

    tabelas = {split: pd.DataFrame(itens) for split, itens in linhas.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    criados = []
    try:
        with ExitStack() as stack:
            arquivos = {}
            for split, destino in destinos.items():
                # Criação exclusiva também protege contra arquivos surgidos durante a extração.
                arquivos[split] = stack.enter_context(destino.open("x", encoding="utf-8", newline=""))
                criados.append(destino)
            for split, tabela in tabelas.items():
                tabela.to_csv(arquivos[split], index=False)
    except BaseException:
        # Remove apenas saídas novas desta execução, nunca imagens ou CSVs anteriores.
        for destino in criados:
            destino.unlink()
        raise
    for split, destino in destinos.items():
        print(f"Características salvas: {destino} ({len(tabelas[split])} amostras)")
    return destinos


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extrai características de PNGs validados, sem sobrescrever CSVs.")
    parser.add_argument("--data-dir", type=Path, help="Prioridade: --data-dir > DATASET_DIR > data/ na raiz do projeto.")
    adicionar_argumento_confirmacao(parser)
    args = parser.parse_args(argv)
    try:
        executar_extracao_lote(args.data_dir, train_flat_is_good=args.train_flat_is_good)
    except (DatasetInvalidoError, OSError) as exc:
        parser.exit(1, f"Extração interrompida: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
