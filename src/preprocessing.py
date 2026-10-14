import cv2
import numpy as np

def converter_para_cinza(imagem_bgr: np.ndarray) -> np.ndarray:
    """Converte uma imagem no formato BGR para escala de cinza."""
    if len(imagem_bgr.shape) == 2:
        return imagem_bgr
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

def aplicar_clahe(
    imagem_cinza: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Aplica o CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    para normalizar a iluminação e destacar detalhes na superfície metálica.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(imagem_cinza)

def remover_ruido(
    imagem_cinza: np.ndarray,
    metodo: str = "bilateral",
    d: int = 9,
    sigma_color: float = 20.0,
    sigma_space: float = 20.0,
    ksize: tuple = (5, 5),
) -> np.ndarray:
    """
    Aplica filtro para redução de ruído.
    'bilateral': Preserva bordas nítidas (ideal para roscas e rachaduras).
    'gaussian': Suavização gaussiana tradicional.
    """
    if metodo == "bilateral":
        return cv2.bilateralFilter(
            imagem_cinza, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
        )
    elif metodo == "gaussian":
        return cv2.GaussianBlur(imagem_cinza, ksize, 0)
    else:
        raise ValueError("Método inválido. Escolha 'bilateral' ou 'gaussian'.")

def preprocessar_imagem(
    imagem_bgr: np.ndarray, 
    clip_limit: float = 2.0, 
    metodo_ruido: str = "bilateral"
) -> np.ndarray:
    """
    Pipeline completo de pré-processamento:
    1. Conversão para cinza
    2. CLAHE (Equalização de Contraste)
    3. Remoção de Ruído (Filtro Bilateral)
    """
    cinza = converter_para_cinza(imagem_bgr)
    equalizada = aplicar_clahe(cinza, clip_limit=clip_limit)
    suavizada = remover_ruido(equalizada, metodo=metodo_ruido)
    return suavizada

if __name__ == "__main__":
    from dataset import carregar_caminhos_dataset
    
    # Teste rápido do pré-processamento
    df = carregar_caminhos_dataset()
    if not df.empty:
        caminho_amostra = df.iloc[0]["caminho"]
        img_original = cv2.imread(caminho_amostra)
        img_processada = preprocessar_imagem(img_original)
        
        print("--- Teste do Módulo de Pré-processamento ---")
        print(f"Imagem carregada: {caminho_amostra}")
        print(f"Dimensões original: {img_original.shape}")
        print(f"Dimensões processada (cinza): {img_processada.shape}")
