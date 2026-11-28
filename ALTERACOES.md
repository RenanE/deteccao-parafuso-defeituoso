# Histórico técnico do projeto

O [README](README.md) descreve o escopo clássico final, o protocolo, os resultados reproduzidos e a execução do notebook. A exportação em [`docs/results/`](docs/results/README.md) separa os registros históricos das duas execuções locais.

## Etapas 1 a 5 · Representação global e validação

- O pipeline passou a compartilhar o carregador, a segmentação e o alinhamento da ROI. O layout de treino plano passou a exigir declaração explícita dos rótulos. A inspeção do dataset inventaria formatos e duplicatas por SHA-256.
- A validação passou a reservar boas para calibração, impedir sobreposição de IDs e manter imagens relacionadas por grupos ou conteúdo na mesma partição. As transformações de escala e PCA são ajustadas nas amostras de ajuste. Foram congeladas 256 boas para desenvolvimento, 64 para calibração final e 160 imagens para teste.
- A busca global comparou representações de HOG e PCA e implementou One-Class SVM, kNN, Mahalanobis e Isolation Forest. As repetições internas com sementes 42, 43 e 44 usaram 163 boas para ajuste, 41 para limiar e 52 para validação. A regra de busca global minimizou o RMSE em relação à meta de 5% de rejeição de boas.
- O kNN passou a excluir a autovizinhança nos diagnósticos de treino e a calibrar com boas reservadas. Mahalanobis ganhou estimadores Ledoit–Wolf e Elliptic Envelope com verificações explícitas de viabilidade numérica.

[Validação e manifestos](src/validation.py) · [Representações globais](src/representations.py) · [Experimentos globais](src/model_experiments.py)

## Etapas 6 a 9 · Execução histórica

- **Etapa 6.** LOF em modo de novidade entrou na busca global, com escolha de R2 e 5 vizinhos. SVM, LOF e Isolation Forest foram calibrados nas 64 boas reservadas, no quantil de 95%.
- **Etapa 8.** A avaliação global do teste completo registrou AUCs de 0,438 (SVM), 0,622 (LOF) e 0,634 (Isolation Forest). O diagnóstico também constatou menor brilho nas boas de teste. [Métricas históricas globais](docs/results/historico/etapa8_metricas.csv) · [Diagnóstico de brilho](docs/relatorio/img/brightness_robustness.png)
- **Etapa 9.** A ROI alinhada foi dividida em patches sobrepostos. Os detectores passaram a ajustar uma referência por posição com vizinhança espacial, dois ajustes cruzados e padronização dos scores. A semente 2026 congelou a divisão estratificada. A busca usou 58 defeitos de seleção e 64 boas reservadas. O holdout continha 61 defeitos e 21 boas. Foram avaliadas 168 combinações e oito ablações.
- **Resultados históricos.** No holdout, os falsos negativos do LOF passaram de 53 com a representação global para 2 com a local. SVM e Isolation Forest locais tiveram 11 e 19 falsos negativos, respectivamente. Na seleção, a ablação registrou AUC de 0,916 para HOG local e de 0,964 para Gabor local isolado. [Métricas históricas locais](docs/results/historico/etapa9_metricas.csv) · [Ablação](docs/results/historico/ablation.png)

A execução histórica documentada usou Linux e Python 3.14. Os manifestos da época registravam caminhos absolutos e os artefatos assinados usavam hashes dos bytes das fontes. O teste completo havia sido examinado antes de criar o holdout; os ganhos observados dependem da amostra e não estabelecem superioridade estatística.

## Reprodução no Windows e extensão local

- **Compatibilidade.** A telemetria de memória passou a aceitar a ausência de `resource` no Windows. A leitura de seleções e relatórios assinados explicita UTF-8. Os testes passaram a usar caminhos portáveis e fixtures isolados dos CSVs reais.
- **Reprodução.** Foram gerados inventário, manifestos, divisão estratificada e cache de 480 ROIs no Windows com Python 3.12.8. As contagens e as escolhas dos três modelos coincidiram com o histórico. A correspondência exata das amostras históricas permanece sem comprovação por ausência dos manifestos da execução original.
- **kNN e Mahalanobis locais.** Os detectores existentes passaram a ser usados com as mesmas dobras, descritores e posições da etapa 9. A busca adicional avaliou 192 combinações. Foram válidas 144 de kNN e 24 de Ledoit–Wolf. As 24 de Elliptic Envelope foram registradas como inviáveis por deficiência de posto. [Grade adicional](docs/results/reproducao/selection_knn_mahalanobis_v1/runs.csv)
- **Avaliação final.** Em diretórios distintos dos três modelos reproduzidos, o kNN com 10 vizinhos, distância média e raio 2 obteve 6 FN, 1 FP e AUC 0,9578. Mahalanobis com Ledoit–Wolf e raio 1 obteve 11 FN, 2 FP e AUC 0,9141. Ambos usaram patch de 24 px, todos os descritores e `top3`. [Métricas dos cinco modelos](docs/results/avaliacao_final/metricas.csv)
- **Testes.** A integração local ganhou casos sintéticos de pontuação e registro de inviabilidade. A suíte clássica passou com 112 testes na execução Windows.

## Entrega da versão clássica

O [notebook 04](notebooks/04_avaliacao_classica.ipynb) consolidou hiperparâmetros, métricas, ROC, matrizes de confusão, recall por categoria e mapas de anomalia. A exportação leve inclui IDs relativos, SHA-256 do inventário, grades, predições e figuras. A representação portátil mantém a composição das partições e usa hashes estruturais próprios. Os hashes e assinaturas dos artefatos de origem ficam registrados na [proveniência](docs/results/reprodutibilidade/proveniencia.json).

O experimento com redes neurais e os materiais temporários ficaram fora do rastreamento da versão clássica, com cópias locais preservadas. O [relatório](docs/relatorio/index.html) e o [pôster](docs/poster/poster_a3.pdf) ainda correspondem à versão anterior.
