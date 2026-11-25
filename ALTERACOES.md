# Alterações feitas nesta rodada (etapas 6 a 9)

Este arquivo lista tudo o que foi alterado, criado ou movido para concluir o
projeto, e explica por que os resultados dos modelos clássicos melhoraram.
Nenhum commit foi feito; todas as mudanças estão no diretório de trabalho.
Detalhes metodológicos completos estão no [README](README.md), seção
"Etapas 6 a 9".

## 1. Resumo

| Etapa | O que foi feito | Resultado |
|---|---|---|
| 6 | LOF (citado nos slides) no mesmo protocolo das etapas 4/5 | Escolha: R2, k=5 |
| Calibração final | Limiar dos modelos nas 64 boas reservadas (5% de FPR-alvo) | Executada para SVM, LOF e Isolation Forest |
| 8 | Avaliação dos modelos com vetor global no teste completo | AUC entre 0,44 e 0,63: perto do acaso |
| 9 | Diagnóstico e nova representação com descritores locais | LOF local: 59 de 61 defeitos detectados no holdout |
| Organização | Redes neurais movidas para `neural_networks/` | Escopo final só clássico: SVM, LOF, Isolation Forest |

## 2. Por que os resultados melhoraram

### O problema estava na representação, não nos detectores

Os três detectores (SVM, LOF, Isolation Forest) são os mesmos antes e depois.
O que mudou foi o que eles recebem.

Antes, cada parafuso virava um único vetor global de 25.413 atributos, dos quais
25.380 (99,9%) eram HOG da ROI inteira. O diagnóstico na metade de seleção mostrou:

| Bloco de atributos | Dimensão | AUC com LOF | AUC com Isolation Forest |
|---|---:|---:|---:|
| HOG 8x8 global | 25.380 | 0,48 | 0,47 |
| HOG 16x16 global | 3.864 | 0,51 | 0,60 |
| Gabor | 24 | 0,72 | 0,66 |
| Bordas | 3 | 0,61 | 0,65 |
| Geometria | 6 | 0,57 | 0,53 |

AUC 0,5 é o acaso. O HOG global não tinha sinal nenhum e dominava o vetor,
afogando os blocos pequenos que tinham algum sinal.

### Por que o vetor global falha

1. Os defeitos são pequenos e locais: um risco na cabeça, um filete amassado, a
   ponta torta. Ocupam poucos pixels da ROI.
2. Entre parafusos bons há bastante variação normal: a posição exata dos filetes
   da rosca (fase), reflexos de luz, pequenas diferenças de alinhamento.
3. Num vetor que resume a peça inteira, essa variação normal espalhada por toda a
   ROI pesa muito mais que um defeito concentrado num canto. Para o detector, um
   parafuso com defeito parece só "mais um parafuso bom um pouco diferente".
4. Com 25 mil dimensões e 256 exemplos, as distâncias ficam quase iguais entre
   todos os pontos, o que prejudica especialmente SVM e LOF.

A segmentação e o alinhamento (etapas 1 e 2) funcionavam bem; o problema era
resumir a ROI inteira num vetor só.

### O que a representação local muda

1. Aproveita o alinhamento: como todo parafuso sai com a cabeça à esquerda e o eixo
   na horizontal, cada posição de uma grade cai sempre na mesma parte da peça
   (cabeça, pescoço, rosca, ponta).
2. Compara cada região só com a mesma região dos parafusos bons. Um detector por
   posição aprende como aquela parte costuma ser; um risco vira um patch que não
   se parece com nenhum patch bom daquela posição.
3. Decide pelo pior patch (média dos 3 piores), não pela média da peça. O sinal
   de um defeito pequeno não é diluído pelo resto do parafuso.
4. Descritores curtos (cerca de 60 dimensões por patch, em vez de 25 mil) com
   centenas de exemplos por posição: dimensão compatível com a quantidade de dados.
5. Tolerância a vizinhos: o detector de cada posição também vê os patches das
   posições adjacentes nas boas, o que absorve pequenos desalinhamentos e a fase
   da rosca (sem isso, a AUC cai de 0,980 para 0,951).

Evidência de que o ganho vem da localidade: o mesmo HOG que tinha AUC 0,48 como
vetor global chega a 0,916 sozinho quando usado por patch. Gabor por patch chega a
0,964. A combinação de todos os descritores locais dá 0,980.

### Resultado (holdout: 61 defeitos e 21 boas nunca usados em escolhas)

| Modelo | Falsos negativos antes | Falsos negativos depois | Falsos positivos antes | Falsos positivos depois | AUC antes | AUC depois |
|---|---:|---:|---:|---:|---:|---:|
| LOF | 53 | 2 | 0 | 3 | 0,561 | 0,958 |
| One-Class SVM | 57 | 11 | 0 | 3 | 0,440 | 0,903 |
| Isolation Forest | 54 | 19 | 2 | 2 | 0,600 | 0,910 |

O LOF é o que mais se beneficia porque ele mede densidade local: um patch
defeituoso fica longe de todos os vizinhos bons daquela posição. O Isolation
Forest escolhe atributos ao acaso para separar pontos e perde parte do sinal
quando ele está concentrado em poucos atributos.

### Cuidados para o ganho ser real

- O teste completo já tinha sido visto na etapa 8. Antes de qualquer experimento,
  ele foi dividido (estratificado por categoria, semente 2026) em metade de seleção
  e metade de holdout. Todas as escolhas usaram só a seleção; o holdout foi
  avaliado uma única vez no final.
- Na seleção, os negativos foram as 64 boas reservadas (mesma distribuição de
  brilho do treino), nunca as boas de teste, que são mais escuras (viés do dataset).
- Dentro do próprio teste (boas de teste contra defeitos de teste, mesma aquisição),
  o LOF local ainda separa com AUC 0,93 na seleção, então ele detecta defeitos e
  não diferenças de aquisição.

## 3. Arquivos novos

| Arquivo | Para que serve |
|---|---|
| `src/lof.py` | Detector LOF com a mesma calibração externa dos outros modelos |
| `src/lof_experiments.py` | Busca interna do LOF (etapa 6) e calibração final |
| `src/final_evaluation.py` | Avaliação dos modelos globais finais (SVM, LOF, Isolation Forest) no teste |
| `src/robustness_check.py` | Mede o viés de brilho do teste e recalcula AUCs sem ele |
| `src/report_figures.py` | Figuras do pipeline clássico (etapas, ROIs alinhadas) |
| `src/defect_split.py` | Divisão congelada do teste em seleção e holdout (etapa 9) |
| `src/roi_cache.py` | Cache das ROIs alinhadas das 480 imagens com o pipeline original |
| `src/local_features.py` | Descritores locais por patch e detector local (SVM, LOF, Isolation Forest) |
| `src/stage9_experiments.py` | Busca na metade de seleção (retomável) e avaliação única no holdout |
| `src/stage9_figures.py` | Figuras da etapa 9 (antes/depois, ROC, ablação, mapas, categorias) |
| `tests/test_stage6_7.py` | Testes do LOF e das métricas |
| `tests/test_stage9.py` | Testes do detector local e da divisão do teste |
| `notebooks/04_avaliacao_classica.ipynb` | Tabelas e figuras das etapas 8 e 9 (só leitura) |
| `docs/relatorio/index.html` | Relatório final (versão clássica) com figuras |
| `docs/poster/poster.html`, `docs/poster/poster_a3.pdf` | Pôster A3 (versão clássica) |
| `ALTERACOES.md` | Este arquivo |

## 4. Arquivos alterados

| Arquivo | Alteração |
|---|---|
| `README.md` | Nova seção "Etapas 6 a 9" no final (o conteúdo anterior foi mantido) |
| `src/*.py` e `tests/*.py` já existentes | Só os finais de linha: CRLF (checkout Windows) para LF, igual ao Git. Necessário porque as seleções assinadas guardam o SHA-256 desses arquivos. O conteúdo não mudou (`git diff` vazio). |

Nenhum módulo das etapas 1 a 5 teve o código alterado. Manifestos, caches e
seleções assinados também não foram alterados.

## 5. Arquivos movidos para `neural_networks/`

O projeto passou a ser só de visão clássica. O experimento com redes neurais
(PatchCore e autoencoder) foi isolado em [`neural_networks/`](neural_networks/README.md):
código (`neural.py`, `final_evaluation_all.py`), testes, notebook, relatório,
pôster e resultados. Nada em `src/` depende dessa pasta.

## 6. Resultados gerados (pasta `results/`, ignorada pelo Git)

| Pasta | Conteúdo |
|---|---|
| `results/stage5/final_*_v1`, `results/stage6/` | Modelos globais calibrados e busca do LOF |
| `results/stage8/test_evaluation_classical_v1/` | Modelos globais no teste completo |
| `results/stage8/robustness_classical_v1/` | Viés de brilho |
| `results/stage8/report_figures/` | Figuras do pipeline |
| `results/stage9/defect_split_v1.json` | Divisão seleção/holdout |
| `results/stage9/roi_cache_v1.npz` | ROIs alinhadas |
| `results/stage9/selection_v1/` | 176 execuções da busca e a seleção assinada |
| `results/stage9/final_v1/` | Avaliação no holdout, predições e mapas |
| `results/stage9/figures/` | Figuras da etapa 9 |

Como `results/` está no `.gitignore`, esses arquivos não entram num commit sem
mudar essa regra. As figuras usadas no relatório e no pôster estão copiadas em
`docs/relatorio/img/`.

## 7. Ambiente desta execução

- Linux, Python 3.14, scikit-learn 1.9, OpenCV 5.1, scikit-image 0.26.
- O `.venv` do repositório é de Windows; foi usado um ambiente Linux separado.
- Os manifestos guardam o caminho absoluto `/home/gabrieladc/Downloads/data`.
  O dataset foi copiado byte a byte para esse caminho; o SHA-256 de cada imagem
  foi conferido pelo próprio protocolo. Nenhum manifesto foi editado.
- No Windows, converter os `src/*.py` para LF antes de rodar a calibração final
  das etapas 4/5, senão a verificação de procedência recusa a execução.
