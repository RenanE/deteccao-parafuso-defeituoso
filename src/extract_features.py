from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importações dos módulos do projeto
from features import criar_banco_gabor, extrair_todas_caracteristicas
from preprocessing import aplicar_clahe, converter_para_cinza, remover_ruido
from segmentation import aplicar_morfologia, binarizar_otsu, extrair_maior_componente_e_roi

def processar_imagem(
    caminho_img: Path,
    banco_gabor: list[np.ndarray]
) -> tuple[np.ndarray, bool]:
    """Aplica o pipeline completo da Milestone 2 (pré-processamento/ROI) e extrai o vetor unificado de características.
    """
    dados_img = np.fromfile(str(caminho_img), dtype=np.uint8)
    img_bgr = cv2.imdecode(dados_img, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, False

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
        banco_gabor=banco_gabor
    )

    return vetor, True

def executar_extracao_lote():
    # Diretórios base
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    out_dir = data_dir / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Inicializa o banco de Gabor global (reutilizado em todas as imagens)
    banco_gabor = criar_banco_gabor(
        orientations=4,
        lambdas=[8.0, 12.0, 16.0]
    )

    print("==================================================")
    print(" EXTRAÇÃO EM LOTE DE CARACTERÍSTICAS ")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. PROCESSAMENTO DO TREINO (data/train/*.png -> classe 'good')
    # ---------------------------------------------------------
    train_dir = data_dir / "train"
    imagens_train = sorted(list(train_dir.glob("*.png")))
    print(f"[1/2] Processando pasta de Treino: {len(imagens_train)} imagens (.png)")

    linhas_train = []
    for img_path in tqdm(imagens_train, desc="Treino (good)"):
        vetor, ok = processar_imagem(img_path, banco_gabor)

        if ok:
            dict_linha = {
                "caminho": str(img_path.relative_to(base_dir)),
                "classe": "good",
                "split": "train",
            }

            for idx, val in enumerate(vetor):
                dict_linha[f"feat_{idx}"] = val
            linhas_train.append(dict_linha)

    df_train = pd.DataFrame(linhas_train)
    caminho_csv_train = out_dir / "features_train.csv"
    df_train.to_csv(caminho_csv_train, index=False)
    print( f"-> Treino salvo em: {caminho_csv_train} ({len(df_train)} amostras)\n")


    # ---------------------------------------------------------
    # 2. PROCESSAMENTO DO TESTE (data/test/<categoria>/*.png)
    # ---------------------------------------------------------
    test_dir = data_dir / "test"
    imagens_test = sorted(list(test_dir.rglob("*.png")))
    print(f"[2/2] Processando pasta de Teste: {len(imagens_test)} imagens")

    linhas_test = []
    for img_path in tqdm(imagens_test, desc="Teste (todas as categorias)"):
        categoria = img_path.parent.name
        vetor, ok = processar_imagem(img_path, banco_gabor)

        if ok:
            dict_linha = {
                "caminho": str(img_path.relative_to(base_dir)),
                "classe": categoria,
                "split": "test",
            }

            for idx, val in enumerate(vetor):
                dict_linha[f"feat_{idx}"] = val
            linhas_test.append(dict_linha)

    df_test = pd.DataFrame(linhas_test)
    caminho_csv_test = out_dir / "features_test.csv"
    df_test.to_csv(caminho_csv_test, index=False)
    print(f"-> Teste salvo em: {caminho_csv_test} ({len(df_test)} amostras)\n")

    print("==================================================")
    print(" EXTRAÇÃO CONCLUÍDA! ARQUIVOS SALVOS EM data/processed/")
    print("==================================================")


if __name__ == "__main__":
    executar_extracao_lote()