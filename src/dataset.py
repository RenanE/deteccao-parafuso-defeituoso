from pathlib import Path
import pandas as pd

# Mapeamento das 6 categorias oficiais do conjunto de teste do MVTec AD (parafuso)
CATEGORIAS_TESTE = [
    "good",
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top"
]

def carregar_caminhos_dataset(data_dir="data"):
    """
    Mapeia os caminhos de todas as imagens do dataset e atribui os rótulos correspondentes.
    
    Retorna:
        pd.DataFrame: Tabela contendo as colunas ['caminho', 'classe', 'split']
    """
    data_path = Path(data_dir)
    dados = []
    
    # 1. Processar conjunto de treino (peças normais 'good')
    train_good = data_path / "train" / "good"
    if train_good.exists():
        for img_path in train_good.glob("*.png"):
            dados.append({
                "caminho": str(img_path),
                "classe": "good",
                "split": "train"
            })
            
    # 2. Processar conjunto de teste (6 categorias de peças normais e defeituosas)
    test_path = data_path / "test"
    if test_path.exists():
        for pasta_classe in test_path.iterdir():
            if pasta_classe.is_dir():
                nome_classe = pasta_classe.name
                for img_path in pasta_classe.glob("*.png"):
                    dados.append({
                        "caminho": str(img_path),
                        "classe": nome_classe,
                        "split": "test"
                    })
                    
    df = pd.DataFrame(dados)
    return df

def resumo_dataset(df):
    """
    Retorna a contagem de imagens por categoria e divisão (treino/teste).
    """
    if df.empty:
        return "Aviso: Nenhuma imagem encontrada. Verifique se o dataset está na pasta 'data/'."
    return pd.crosstab(df['classe'], df['split'], margins=True)

if __name__ == "__main__":
    # Teste de execução direta no terminal
    df = carregar_caminhos_dataset()
    print("--- Resumo do Carregamento do Dataset ---")
    print(f"Total de imagens mapeadas: {len(df)}\n")
    print(resumo_dataset(df))