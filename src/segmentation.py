import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate

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

def aplicar_morfologia(
    mascara_binaria: np.ndarray,
    tamanho_abertura: int = 3,
    tamanho_fechamento: int = 11,
) -> np.ndarray:
    """
    Aplica Abertura para eliminar pequenos ruídos do fundo e Fechamento para 
    preencher falhas de reflexo metálico na superfície do parafuso.
    """
    kernel_abertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_abertura, tamanho_abertura)
    )
    kernel_fechamento = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_fechamento, tamanho_fechamento)
    )
    
    # 1. Abertura: Elimina pequenas sujeiras ou ruídos isolados no fundo
    mascara_limpa = cv2.morphologyEx(
        mascara_binaria, cv2.MORPH_OPEN, kernel_abertura
    )
    
    # 2. Fechamento: Preenche lacunas/buracos causados pelo reflexo do brilho metálico
    mascara_fechada = cv2.morphologyEx(
        mascara_limpa, cv2.MORPH_CLOSE, kernel_fechamento
    )
    return mascara_fechada

def extrair_maior_componente_e_roi(
    imagem_cinza: np.ndarray,
    mascara_binaria: np.ndarray,
    alinhar_eixo: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Identifica a maior componente conexa (o parafuso), calcula a orientação e 
    retorna a ROI recortada e alinhada.
    """
    rotulada = label(mascara_binaria > 0)
    regioes = regionprops(rotulada)
    
    if not regioes:
        raise ValueError("Nenhuma componente conexa encontrada na máscara.")
        
    # Seleciona a maior região por área (o parafuso)
    maior_regiao = max(regioes, key=lambda r: r.area)
    
    if alinhar_eixo:
        # Ângulo do eixo principal em graus
        angulo_graus = np.degrees(maior_regiao.orientation)
        
        # Rotaciona a imagem e a máscara para alinhar o parafuso
        imagem_rot = rotate(imagem_cinza, angulo_graus, resize=True)
        mascara_rot = rotate(mascara_binaria, angulo_graus, resize=True)
        
        # Converte de volta para uint8 no intervalo [0, 255]
        imagem_rot = (imagem_rot * 255).astype(np.uint8)
        mascara_rot = (mascara_rot * 255).astype(np.uint8)
        
        # Recalcula a caixa delimitadora (bbox) na imagem alinhada
        rotulada_rot = label(mascara_rot > 0)
        regioes_rot = regionprops(rotulada_rot)
        
        if regioes_rot:
            maior_regiao = max(regioes_rot, key=lambda r: r.area)
            minr, minc, maxr, maxc = maior_regiao.bbox
            roi_imagem = imagem_rot[minr:maxr, minc:maxc]
            roi_mascara = mascara_rot[minr:maxr, minc:maxc]
            return roi_imagem, roi_mascara, (minr, minc, maxr, maxc)
            
    # Recorte simples sem rotação (caso alinhar_eixo=False ou falha na rotação)
    minr, minc, maxr, maxc = maior_regiao.bbox
    roi_imagem = imagem_cinza[minr:maxr, minc:maxc]
    roi_mascara = mascara_binaria[minr:maxr, minc:maxc]
    return roi_imagem, roi_mascara, (minr, minc, maxr, maxc)
