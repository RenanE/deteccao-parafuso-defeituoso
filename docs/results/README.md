# Resultados experimentais

O pacote reúne a Etapa 9 reproduzida no Windows e a avaliação final dos cinco modelos locais no mesmo holdout de 61 defeitos e 21 boas. A seleção usou 64 boas reservadas e 58 defeitos de seleção. O [notebook 04](../../notebooks/04_avaliacao_classica.ipynb) lê este pacote diretamente.

- [Histórico](historico/) reúne as tabelas, contagens e figuras preservadas da execução original, extraídas do notebook. Os manifestos históricos completos estão indisponíveis para confirmar a correspondência imagem a imagem.
- [Reprodução local](reproducao/) reúne as seleções, grades e ablações. Em `selection_knn_mahalanobis_v1/runs.csv`, `skipped` identifica as 24 combinações inviáveis de Elliptic Envelope e `reason` registra a posição e o motivo.
- [Avaliação final](avaliacao_final/) reúne métricas, predições por modelo e imagem, recall por categoria e figuras. `final_v1` agrupa SVM, LOF e Isolation Forest, e `final_knn_mahalanobis_v1` agrupa kNN e Mahalanobis.
- [Análises complementares](analises_complementares/) reúne dois experimentos exploratórios sobre o holdout já consultado. [Pontos de operação](analises_complementares/operating_points_v1/) contém os 1.120 scores, limiares, métricas, recall por categoria, curvas ROC e gráficos de FN/FP. [Custo ponderado](analises_complementares/threshold_cost_v1/) contém os 40 pontos comparados, a fronteira de Pareto e o SHA-256 do CSV de scores usado.
- [Reprodutibilidade](reprodutibilidade/) reúne os manifestos, os IDs relativos à raiz do dataset, os SHA-256 das imagens e o ambiente registrado. `proveniencia.json` vincula a exportação aos hashes e às assinaturas originais. `SHA256SUMS` verifica os arquivos publicados.

A exportação mantém a ordem e a composição das partições e os valores experimentais. Os IDs foram convertidos para caminhos relativos com separadores POSIX. Os hashes estruturais e as referências entre manifestos foram calculados para essa representação portátil. Os originais assinados continuam em `results/`. O pacote contém somente tabelas, metadados e figuras renderizadas.

Confira os checksums a partir desta pasta com `sha256sum -c reprodutibilidade/SHA256SUMS`.

## Análises complementares

A comparação preserva o limiar original, acrescenta dois pontos definidos pelo recall na seleção e cinco pontos por modelo definidos por `J = w × FNR + FPR`. Os limiares ponderados usam a FNR nos 58 defeitos de seleção e a FPR nas 64 boas de calibração. O peso expressa a prioridade relativa de reduzir defeitos aprovados. O custo publicado para o holdout serve à comparação exploratória, já que esse conjunto havia sido consultado anteriormente.

Para refazer apenas a comparação de custo a partir da raiz do clone, use `python -m src.stage9_threshold_cost --scores docs/results/analises_complementares/operating_points_v1/scores.csv --out results/stage9/threshold_cost_reproduction`. O notebook visualiza os dois conjuntos publicados sem ajuste de modelos. Para recuperar os scores com o cache local, siga as [instruções de reprodução no README principal](../../README.md#reproduzir-as-análises-complementares).
