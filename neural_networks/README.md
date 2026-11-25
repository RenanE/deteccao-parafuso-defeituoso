# Redes neurais (experimento separado, fora do escopo final)

O escopo final do projeto é visão computacional clássica (SVM, LOF e Isolation
Forest). Esta pasta guarda, isolado do pipeline principal, o experimento com
redes neurais feito antes dessa definição. Nada aqui é usado por `src/`.

## Conteúdo

| Caminho | O que é |
|---|---|
| `neural.py` | PatchCore (Wide ResNet-50 pré-treinada no ImageNet) e autoencoder convolucional treinado do zero |
| `final_evaluation_all.py` | Avaliação conjunta dos 5 modelos clássicos e das 2 redes no teste completo |
| `tests/test_neural.py` | Testes sintéticos das duas redes |
| `notebooks/04_avaliacao_final_com_redes.ipynb` | Tabelas e figuras da avaliação conjunta |
| `docs/relatorio/`, `docs/poster/` | Relatório e pôster A3 da versão com redes neurais |
| `results/stage7/` | Modelos ajustados e calibrados (ignorado pelo Git) |
| `results/stage8_all_models/` | Avaliação conjunta e verificação de viés de brilho (ignorado pelo Git) |

## Protocolo e resultado (teste completo, 160 imagens)

Mesmo protocolo dos modelos clássicos: ajuste nas 256 boas de desenvolvimento,
limiar no quantil de 95% das 64 boas reservadas, avaliação única no teste.

| Modelo | FN (de 119) | FP (de 41) | AUC | AUC reservadas vs defeitos |
|---|---:|---:|---:|---:|
| PatchCore | 13 | 3 | 0,964 | 0,979 |
| Autoencoder | 117 | 0 | 0,795 | 0,583 |

## Reprodução

Requer `torch` e `torchvision` (não estão no `requirements.txt` principal).
Executar a partir da raiz do repositório:

```bash
python -m neural_networks.neural --study results/stage3/study_v1 --model patchcore \
  --out neural_networks/results/stage7/final_patchcore_v2 --confirm-final-calibration
python -m neural_networks.neural --study results/stage3/study_v1 --model autoencoder \
  --out neural_networks/results/stage7/final_autoencoder_v2 --confirm-final-calibration
python -m unittest neural_networks.tests.test_neural
```

Os modelos já salvos em `results/stage7/*_v1` foram serializados quando o módulo
ainda ficava em `src/neural.py`; `final_evaluation_all.py` faz o mapeamento de
nomes para carregá-los.
