import cv2
import numpy as np
from skimage.feature import hog
from skimage.measure import label, regionprops


def extrair_hog(
    roi_imagem: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    visualize: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Extrai o vetor de características HoG (Histogram of Oriented Gradients) de uma ROI."""
    
    # Redimensiona a ROI para uma resolução padrão fixa (ex: 128x384)
    # para garantir que todas as ROIs gerem vetores HoG de mesmo tamanho
    roi_redimensionada = cv2.resize(
        roi_imagem, (384, 128), interpolation=cv2.INTER_AREA
    )

    if visualize:
        vetor_caracteristicas, imagem_hog = hog(
            roi_redimensionada,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L1",
            visualize=True,
        )
        return vetor_caracteristicas, imagem_hog

    vetor_caracteristicas = hog(
        roi_redimensionada,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L1",
        visualize=False,
    )
    return vetor_caracteristicas

def criar_banco_gabor(
    orientations: int = 4,
    lambdas: list[float] = [8.0, 12.0, 16.0],  # Escala adequada para os sulcos
    gamma: float = 0.5,
    psi: float = 0.0,
) -> list[np.ndarray]:
    """Cria banco de filtros de Gabor com sigma dinâmico proporcional ao comprimento de onda (lambda)."""
    kernels = []
    # Orientações: 0°, 45°, 90°, 135°
    thetas = np.linspace(0, np.pi, orientations, endpoint=False)
    for theta in thetas:
        for lam in lambdas:
            # Sigma proporcional a lambda (largura de banda de ~1 oitava)
            sigma = 0.56 * lam
            # Kernel proporcional ao sigma
            ksize = int(6 * sigma) | 1  # Garante tamanho ímpar
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                lam,
                gamma,
                psi,
                ktype=cv2.CV_32F,
            )
            kernels.append(kernel)
    return kernels


def extrair_gabor(
    roi_imagem: np.ndarray,
    roi_mascara: np.ndarray = None,
    kernels: list[np.ndarray] = None,
    visualize: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]] | np.ndarray:
    """Aplica o banco de Gabor na ROI, calcula a MAGNITUDE da resposta e aplica a máscara para isolar o fundo."""
    if kernels is None:
        kernels = criar_banco_gabor()

    roi_red = cv2.resize(roi_imagem, (384, 128), interpolation=cv2.INTER_AREA)

    mascara_red = None
    if roi_mascara is not None:
        mascara_red = (
            cv2.resize(
                roi_mascara, (384, 128), interpolation=cv2.INTER_NEAREST
            )
            > 0
        )

    caracteristicas = []
    imagens_magnitude = []

    for kernel in kernels:
        # 1. Convolução
        fimg = cv2.filter2D(roi_red, cv2.CV_32F, kernel)

        # 2. MAGNITUDE (elimina as franjas/listras zebradas e gera mapa de energia)
        magnitude = np.abs(fimg)

        # 3. Zera o fundo fora do parafuso se a máscara for fornecida
        if mascara_red is not None:
            magnitude = magnitude * mascara_red

        if visualize:
            imagens_magnitude.append(magnitude)

        # Extrai estatísticas da magnitude sobre a superfície do parafuso
        if mascara_red is not None and np.any(mascara_red):
            valores_validos = magnitude[mascara_red]
            media = float(np.mean(valores_validos))
            desvio = float(np.std(valores_validos))
        else:
            media = float(np.mean(magnitude))
            desvio = float(np.std(magnitude))

        caracteristicas.extend([media, desvio])

    vetor_caracteristicas = np.array(caracteristicas, dtype=np.float32)

    if visualize:
        return vetor_caracteristicas, imagens_magnitude

    return vetor_caracteristicas


def extrair_bordas(
    roi_imagem: np.ndarray,
    roi_mascara: np.ndarray = None,
    threshold1: int = 50,
    threshold2: int = 150,
    visualize: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Extrai mapa de bordas com Canny e gradientes Sobel, retornando
    estatísticas de densidade de bordas e energia de gradiente.
    """
    roi_red = cv2.resize(roi_imagem, (384, 128), interpolation=cv2.INTER_AREA)
    mascara_red = None

    if roi_mascara is not None:
        mascara_red = (
            cv2.resize(roi_mascara, (384, 128), interpolation=cv2.INTER_NEAREST)
            > 0
        )

    # 1. Bordas com Canny
    bordas_canny = cv2.Canny(roi_red, threshold1, threshold2)

    # 2. Gradientes direcionais com Sobel
    sobel_x = cv2.Sobel(roi_red, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi_red, cv2.CV_32F, 0, 1, ksize=3)
    magnitude_sobel = cv2.magnitude(sobel_x, sobel_y)

    if mascara_red is not None:
        bordas_canny = bordas_canny * mascara_red
        magnitude_sobel = magnitude_sobel * mascara_red

    # Estatísticas numéricas
    if mascara_red is not None and np.any(mascara_red):
        area_peca = np.sum(mascara_red)
        densidade_canny = float(np.sum(bordas_canny > 0)) / float(area_peca)
        sobel_medio = float(np.mean(magnitude_sobel[mascara_red]))
        sobel_std = float(np.std(magnitude_sobel[mascara_red]))
    else:
        densidade_canny = float(np.mean(bordas_canny > 0))
        sobel_medio = float(np.mean(magnitude_sobel))
        sobel_std = float(np.std(magnitude_sobel))

    vetor_caracteristicas = np.array(
        [densidade_canny, sobel_medio, sobel_std], dtype=np.float32
    )

    if visualize:
        return vetor_caracteristicas, bordas_canny

    return vetor_caracteristicas


def extrair_atributos_geometricos(roi_mascara: np.ndarray) -> np.ndarray:
    """Extrai atributos geométricos e morfológicos da máscara binária
    alinhada da ROI calculados em sua escala original (antes do resize).
    """
    # Trabalha diretamente com a máscara na escala original (sem cv2.resize)
    mascara_bool = roi_mascara > 0
    rotulada = label(mascara_bool)
    regioes = regionprops(rotulada)

    if not regioes:
        return np.zeros(6, dtype=np.float32)

    maior_regiao = max(regioes, key=lambda r: r.area)

    # Extrai métricas na proporção física real da ROI
    area = float(maior_regiao.area)
    perimetro = float(maior_regiao.perimeter)
    solidez = float(maior_regiao.solidity)
    excentricidade = float(maior_regiao.eccentricity)
    extent = float(maior_regiao.extent)

    # Bounding box real
    minr, minc, maxr, maxc = maior_regiao.bbox
    altura = maxr - minr
    largura = maxc - minc
    razao_aspecto = float(largura) / float(altura) if altura > 0 else 0.0

    return np.array(
        [area, perimetro, solidez, excentricidade, extent, razao_aspecto],
        dtype=np.float32,
    )

def extrair_todas_caracteristicas(
    roi_clahe: np.ndarray,
    roi_mascara: np.ndarray,
    banco_gabor: list[np.ndarray] = None,
) -> np.ndarray:
    """Concatena todas as características extraídas em um único vetor 1D:
    1. HoG (estrutura de gradientes locais)
    2. Filtros de Gabor (estatísticas de magnitude da textura da rosca)
    3. Bordas (densidade Canny, média e desvio Sobel)
    4. Atributos Geométricos (área, perímetro, solidez, excentricidade, extent, razão de aspecto)
    """
    if banco_gabor is None:
        banco_gabor = criar_banco_gabor(
            orientations=4,
            lambdas=[8.0, 12.0, 16.0]
        )

    # 1. HoG
    vetor_hog = extrair_hog(
        roi_clahe,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )

    # 2. Gabor (com aplicação da máscara de fundo)
    vetor_gabor = extrair_gabor(
        roi_clahe,
        roi_mascara=roi_mascara,
        kernels=banco_gabor,
        visualize=False
    )

    # 3. Bordas (Canny + Sobel)
    vetor_bordas = extrair_bordas(
        roi_clahe,
        roi_mascara=roi_mascara,
        visualize=False
    )

    # 4. Atributos Geométricos (calculados na resolução original da máscara)
    vetor_geom = extrair_atributos_geometricos(roi_mascara)

    # Concatenação final em um único vetor 1D
    vetor_completo = np.hstack(
        [vetor_hog, vetor_gabor, vetor_bordas, vetor_geom]
    ).astype(np.float32)

    return vetor_completo