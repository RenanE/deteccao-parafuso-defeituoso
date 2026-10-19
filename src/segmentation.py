import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate
from scipy.ndimage import binary_fill_holes

def binarizar_otsu(
    imagem_cinza: np.ndarray, 
    inverter: bool = True
) -> tuple[float, np.ndarray]:
    """
    Aplica a limiarização automática de Otsu.
    Por padrão, inverte (inverter=True) para que o parafuso (objeto escuro no 
    fundo claro) fique em branco (255) e o fundo em preto (0).
    """
    tipo_limiar = cv2.THRESH_BINARY_INV if inverter else cv2.THRESH_BINARY
    limiar, mascara = cv2.threshold(
        imagem_cinza, 0, 255, tipo_limiar + cv2.THRESH_OTSU
    )
    return limiar, mascara


def preencher_furos(mascara_binaria: np.ndarray) -> np.ndarray:
    """
    Preenche buracos e lacunas internas na máscara binária causados por 
    reflexos metálicos, preservando intactas as bordas externas da peça.
    """
    mascara_bool = mascara_binaria > 0
    mascara_preenchida = binary_fill_holes(mascara_bool)
    return (mascara_preenchida * 255).astype(np.uint8)

def aplicar_morfologia(
    mascara_binaria: np.ndarray,
    tamanho_abertura: int = 3,
    tamanho_fechamento: int = 11,
    preencher_lacunas: bool = True,
) -> np.ndarray:
    """
    Aplica Abertura para eliminar ruídos do fundo, Fechamento e 
    preenchimento topológico de furos.
    """
    kernel_abertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_abertura, tamanho_abertura)
    )
    kernel_fechamento = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_fechamento, tamanho_fechamento)
    )
    
    # 1. Abertura: Elimina sujeiras isoladas do fundo
    mascara_limpa = cv2.morphologyEx(
        mascara_binaria, cv2.MORPH_OPEN, kernel_abertura
    )
    
    # 2. Fechamento: Preenche pequenas falhas na borda
    mascara_fechada = cv2.morphologyEx(
        mascara_limpa, cv2.MORPH_CLOSE, kernel_fechamento
    )
    
    # 3. Preenchimento de furos internos (solução para o reflexo no corpo metálico)
    if preencher_lacunas:
        mascara_fechada = preencher_furos(mascara_fechada)
        
    return mascara_fechada

def extrair_maior_componente_e_roi(
    imagem_cinza: np.ndarray,
    mascara_binaria: np.ndarray,
    alinhar_eixo: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Identifica a ROI do parafuso, rotaciona para a posição horizontal padronizada

    (cabeça para a esquerda) e retorna o recorte da imagem e da máscara.
    """
    rotulada = label(mascara_binaria > 0)
    regioes = regionprops(rotulada)

    if not regioes:
        raise ValueError("Nenhuma componente conexa encontrada na máscara.")

    maior_regiao = max(regioes, key=lambda r: r.area)

    if alinhar_eixo:
        # 1. Ângulo da skimage é em relação ao eixo VERTICAL (Y).
        # Para alinhar na HORIZONTAL (X), subtraímos de 90 graus.
        orientacao_rad = maior_regiao.orientation
        angulo_horizontal = 90.0 - np.degrees(orientacao_rad)

        # 2. Rotaciona a imagem e a máscara para a horizontal
        # cval/mode mantêm o tom claro de fundo ao expandir as bordas
        tom_fundo = (
            float(np.median(imagem_cinza[mascara_binaria == 0])) / 255.0
        )
        imagem_rot = rotate(
            imagem_cinza, angulo_horizontal, resize=True, cval=tom_fundo
        )
        mascara_rot = rotate(
            mascara_binaria, angulo_horizontal, resize=True, cval=0
        )

        # Converte de volta para uint8 (0..255)
        imagem_rot = (
            (imagem_rot * 255).astype(np.uint8)
            if imagem_rot.max() <= 1.0
            else imagem_rot.astype(np.uint8)
        )
        mascara_rot = (
            (mascara_rot * 255).astype(np.uint8)
            if mascara_rot.max() <= 1.0
            else mascara_rot.astype(np.uint8)
        )

        # Recalcula a caixa delimitadora (bbox) na imagem alinhada
        rotulada_rot = label(mascara_rot > 0)
        regioes_rot = regionprops(rotulada_rot)

        if regioes_rot:
            maior_regiao_rot = max(regioes_rot, key=lambda r: r.area)
            minr, minc, maxr, maxc = maior_regiao_rot.bbox
            roi_imagem = imagem_rot[minr:maxr, minc:maxc]
            roi_mascara = mascara_rot[minr:maxr, minc:maxc]

            # 3. Orientação Padrão: Garante que a cabeça (parte mais larga) fique na ESQUERDA
            largura_colunas = np.sum(roi_mascara > 0, axis=0)
            meio = len(largura_colunas) // 2
            area_esquerda = np.sum(largura_colunas[:meio])
            area_direita = np.sum(largura_colunas[meio:])

            # Se a direita for mais larga que a esquerda, gira 180 graus
            if area_direita > area_esquerda:
                roi_imagem = np.rot90(roi_imagem, 2)
                roi_mascara = np.rot90(roi_mascara, 2)

            return roi_imagem, roi_mascara, (minr, minc, maxr, maxc)

    # Caso sem alinhamento
    minr, minc, maxr, maxc = maior_regiao.bbox
    return (
        imagem_cinza[minr:maxr, minc:maxc],
        mascara_binaria[minr:maxr, minc:maxc],
        (minr, minc, maxr, maxc),
    )

