import os
from pathlib import Path
import pandas as pd

def carregar_caminhos_dataset(data_dir="data"):
    """
    Mapeia os caminhos das imagens e seus respectivos rótulos.
    """
    data_path = Path(data_dir)
    dados = []
    
    # Processa conjunto de treino (apenas imagens normais 'good')
    train_good = data_path / "train" / "good"
    if train_good.exists():
        for img in train_good.glob("*.png"):
            dados.append({"caminho": str(img), "classe": "good", "split": "train"})
            
    # Processa conjunto de teste (normais e defeitos)
    test_path = data_path / "test"
    if test_path.exists():
        for pasta_classe in test_path.iterdir():
            if pasta_classe.is_dir():
                classe = pasta_classe.name
                for img in pasta_classe.glob("*.png"):
                    dados.append({"caminho": str(img), "classe": classe, "split": "test"})
                    
    return pd.DataFrame(dados)
