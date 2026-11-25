# Detecção de anomalias em parafusos

Extração de características de imagens industriais com HOG, Gabor, bordas e
geometria. Os modelos previstos são One-Class SVM, k-NN de anomalias,
Mahalanobis/Envelope Elíptico e Isolation Forest.
Os quatro detectores são ajustados exclusivamente com imagens normais (`good`),
conforme a formulação One-Class confirmada pelo grupo. Essa escolha de modelagem
não certifica, por si só, os rótulos dos arquivos locais.

**Status da etapa 3:** o grupo confirmou que os 320 PNGs atuais de `train/` são
normais. A declaração `TRAIN_FLAT_IS_GOOD=true` está autorizada para esse dataset.
As pendências de rotulagem descritas nos registros das etapas anteriores foram
resolvidas por essa confirmação humana. O procedimento novo está ao final deste README.

## Ambiente local

Use Python **3.12** em um ambiente virtual. A preparação e os testes sintéticos
desta etapa foram realizados com Python 3.12.7. Na raiz do repositório:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Se usar asdf e o comando Python não tiver uma versão selecionada, use uma versão
já instalada, por exemplo:

```bash
ASDF_PYTHON_VERSION=3.12.7 python -m venv .venv
source .venv/bin/activate
```

No Windows, ative com `.venv\Scripts\Activate.ps1` no PowerShell. As dependências
diretas estão em `requirements.txt`, incluindo NumPy, SciPy, tqdm, joblib e o
ambiente Jupyter. As versões ainda não estão fixadas; a instalação não garante
reprodução numérica de execuções históricas dos notebooks. Os parâmetros e
algoritmos dos descritores não foram alterados nesta etapa.

## Dataset local

O dataset não acompanha o Git. Informe a pasta que **já contém** `train/` e
`test/`, sem mover, renomear ou duplicar imagens. Para o MVTec AD, normalmente é
a pasta da categoria `screw`, não a raiz de todas as categorias de objetos.

```text
<raiz-do-dataset>/
├── train/
│   └── good/
│       └── *.png
└── test/
    ├── good/
    ├── manipulated_front/
    ├── scratch_head/
    ├── scratch_neck/
    ├── thread_side/
    └── thread_top/
```

A resolução do caminho segue a prioridade **`--data-dir` (ou argumento Python)
> variável `DATASET_DIR` > `data/` na raiz deste repositório**. A variável é
consultada a cada chamada; vazia ou ausente mantém o padrão. `~` é expandido.
Prefira caminhos absolutos: valores relativos fornecidos explicitamente ou na
variável são relativos ao diretório de execução. O padrão `data/` é sempre
relativo à raiz do projeto.

Para usar o dataset externo informado, sem copiar imagens:

```bash
source .venv/bin/activate
export DATASET_DIR="$HOME/Downloads/data"
python -m src.dataset
```

O `export` vale para essa sessão do terminal e seus subprocessos. Não há caminho
pessoal fixado no código nem leitura automática de `.env`. Para retornar ao
padrão, execute `unset DATASET_DIR`. Para substituir a variável em uma execução,
use `--data-dir "/outro/caminho"`.

Regras do carregador compartilhado por inspeção, notebooks e extração:

- `train/` e `test/` são obrigatórios. O layout padrão usa `train/good/*.png`.
- Alternativamente, aceita-se `train/*.png` somente mediante declaração explícita
  de que **todas** essas imagens são `good`, conforme a configuração abaixo.
- Treino exige pelo menos uma imagem PNG legível. Outra classe, mesmo vazia,
  interrompe a validação, inclusive quando a declaração está habilitada.
- Não é permitido misturar PNGs diretamente em `train/` com `train/good/`.
  Não há mesclagem, movimentação nem renomeação automática.
- Teste exige pelo menos uma das seis categorias listadas, e toda categoria
  presente precisa conter PNGs legíveis. Subconjuntos são aceitos para inspeção;
  isso não certifica que o conjunto esteja completo para avaliação dos modelos.
- Categorias desconhecidas, imagens soltas em `test/` e subpastas dentro de
  categorias são rejeitadas.
- `.png` e `.PNG` são aceitos. Outros formatos conhecidos de imagem (JPEG, BMP,
  TIFF, WebP e GIF) são inventariados, mas interrompem a validação: a extração
  continua restrita a PNG. Não há conversão automática.
- Arquivos auxiliares como `.txt` são contados por extensão, mas não entram no
  conjunto de imagens. Extensões desconhecidas não são decodificadas.
- A legibilidade é verificada com o mesmo decodificador BGR da extração,
  incluindo caminhos com espaços e acentos. Imagens vazias/corrompidas causam
  erro com o caminho do arquivo; não são descartadas silenciosamente.
- Pastas externas a `train/` e `test/`, como `ground_truth/` e `processed/`, não
  entram no inventário de imagens.

Os rótulos vêm da organização validada. O carregador não consegue certificar
visualmente que uma imagem colocada em `good/` realmente representa peça boa.
No layout plano, o rótulo vem exclusivamente da declaração explícita. Não
reorganize imagens para contornar erros.

## Declaração para treino plano

O layout `train/*.png` é reconhecido estruturalmente, mas o carregamento permanece
bloqueado até que um responsável verifique os rótulos e declare que todos são
normais. Para `train/good/*.png`, o rótulo já está indicado pelo nome da pasta;
essa indicação também não constitui certificação visual ou documental.

**Somente depois de confirmar os rótulos dos arquivos atuais**, configure:

```bash
export DATASET_DIR="$HOME/Downloads/data"
export TRAIN_FLAT_IS_GOOD=true
python -m src.dataset
```

`TRAIN_FLAT_IS_GOOD` aceita apenas `true`, `false` ou vazio (equivalente a false),
sem distinção de maiúsculas/minúsculas. Valores ambíguos como `1`, `yes` ou `good`
são rejeitados. O padrão é **false**. Não há confirmação persistida automaticamente.
Para retirar a declaração, use `unset TRAIN_FLAT_IS_GOOD` ou defina `false`.

Uma declaração por comando tem prioridade sobre a variável:

```bash
# Declara explicitamente que todas as imagens do treino plano são boas:
python -m src.dataset --train-flat-is-good

# Força inspeção sem essa declaração, mesmo que o ambiente contenha true:
python -m src.dataset --no-train-flat-is-good
```

Na API Python, `carregar_caminhos_dataset(data_dir, train_flat_is_good=True)`
habilita o layout plano; `False` desabilita mesmo se o ambiente declarar `true`;
`None` consulta a variável. O argumento deve ser um booleano, não uma string.
A extração usa a mesma variável, argumento Python e opções de linha de comando.
Habilitar a declaração nunca permite classes desconhecidas, treino misto,
categorias vazias ou imagens ilegíveis.

Os rótulos de teste permanecem nas seis categorias originais. A convenção binária
dos notebooks continua `+1` para `good` e `-1` para qualquer uma das cinco
categorias de defeito; a declaração do treino não altera os rótulos de teste.

## Inspeção sem modificar dados

Na raiz do repositório, com o ambiente ativado:

```bash
python -m src.dataset
python -m src.dataset --data-dir "/caminho/para/screw"
```

Também é possível executar `python src/dataset.py --data-dir "..."`.
O comando apresenta:

1. Quantidade de arquivos de imagem encontrados por split e categoria.
2. Quantidade de arquivos por split e extensão, incluindo arquivos auxiliares.
3. Grupos com o mesmo SHA-256, destacando duplicatas entre treino e teste.
4. Validade estrutural, layout do treino, origem da declaração dos rótulos,
   legibilidade e categorias de teste ausentes, separadamente.
5. Contagens dos PNGs efetivamente validados ou o motivo da interrupção.

O inventário inicial inclui arquivos fora da estrutura e não deve ser confundido
com o conjunto validado. O processo retorna **0** quando estrutura, leitura e
requisitos de rotulagem permitem carregar o dataset, e **1** em caso de erro ou
declaração pendente. Portanto, uma estrutura plana pode ser válida e ainda
retornar 1 por falta de confirmação. A inspeção não verifica documentos nem
certifica o conteúdo real das imagens. Duplicatas são avisos para revisão, não são removidas e
não alteram o código de saída. SHA-256 detecta arquivos idênticos byte a byte;
não identifica todas as imagens visualmente iguais com codificações diferentes,
recortes ou aumentações. A inspeção tenta decodificar todas as imagens
inventariadas e informa quantas são legíveis e todos os erros de leitura, mesmo
se a estrutura estiver inválida. A validação do carregador ainda para no primeiro
erro; imagens legíveis não são automaticamente consideradas rotuladas.

Nenhum CSV é criado pela inspeção. Os hashes são calculados em memória, lendo
blocos dos arquivos; a inspeção também decodifica as imagens e pode levar tempo
em datasets grandes. Se o dataset estiver ausente, o comando informa o diretório
obrigatório que falta, sem criar pastas ou apresentar contagens como válidas.

## Situação do dataset externo inspecionado

A inspeção de `~/Downloads/data` encontrou 480 arquivos PNG: 320 diretamente
em `train/` (sem categoria), e 160 em `test/`: `good` (41), `manipulated_front`
(24), `scratch_head` (24), `scratch_neck` (25), `thread_side` (23) e `thread_top`
(23). Todos foram decodificados; não foram encontrados arquivos ilegíveis nem
duplicatas byte a byte por SHA-256 nessa inspeção.

**`train/good/` não existe.** Na etapa 1, a confirmação dos rótulos estava pendente.
Na etapa 3, o grupo confirmou os 320 arquivos como `good`. O layout plano é aceito
com `TRAIN_FLAT_IS_GOOD=true`; sem a declaração explícita, o bloqueio continua
existindo para evitar inferência silenciosa em outros datasets.
Nenhuma imagem foi movida, copiada, renomeada, excluída ou presumida como `good`.

### Evidências documentais locais e limitações

- Não há README, manifesto, metadados externos ou arquivos não PNG dentro de
  `~/Downloads/data` que vinculem cada imagem à origem e ao seu rótulo.
- O código versionado de `src/dataset.py` em `b78eaec` identifica as seis
  categorias como MVTec AD (parafuso) e espera `train/good/`. Já a extração
  versionada usa `train/*.png` e atribui `good`. Isso registra a intenção do
  projeto e a divergência de caminhos; não é verificação independente dos dados.
- O notebook 03 declara treino exclusivamente `good` e contém saídas históricas
  de 320/160 amostras. O README versionado antes desta etapa estava vazio.
- Foi encontrado fora do repositório o documento local
  `~/Downloads/roteiro_apresentacao_parafusos_v2.pdf`, página 3, slide 11:
  ele declara treino com 320 imagens exclusivamente `good` e teste com 160
  imagens nas seis categorias. É evidência documental do projeto a favor dos
  rótulos esperados, mas não um manifesto dos arquivos atuais. O slide 12 ainda
  lista três modelos antigos, reforçando que o documento não descreve todo o
  estado atual do projeto. O PDF não foi copiado para o repositório.
- Não foi localizada documentação oficial do dataset original nos materiais
  locais examinados. Assim, não foi possível certificar a estrutura original
  nem verificar a procedência dos 320 PNGs por um manifesto da fonte.

A confirmação humana de que **estes 320 arquivos são todos normais** foi recebida
na etapa 3. Ela autoriza a declaração documentada acima. A documentação original
e um manifesto da fonte continuam indisponíveis localmente; a base para os rótulos
atuais é a confirmação explícita do grupo.

## Testes pequenos e independentes do dataset

```bash
python -m unittest discover -s tests -v
```

Os testes usam `unittest` da biblioteca padrão e imagens sintéticas em diretórios
temporários. Cobrem estrutura, rótulos, arquivos vazios/corrompidos, extensões,
hashes, caminhos Unicode, CLI, falhas de processamento/escrita e preservação de
CSVs existentes. O lote é simulado nos testes de escrita. Há apenas um teste de
compatibilidade que executa o pipeline real em uma imagem sintética pequena e
confere o vetor de 25.413 atributos; nenhum modelo é treinado.

## Próxima etapa: extração, somente após revisar a inspeção

A extração completa não faz parte da preparação do ambiente. Quando for hora
de executá-la, use uma das opções:

```bash
python -m src.extract_features
python -m src.extract_features --data-dir "/caminho/para/screw"
```

A extração usa `carregar_caminhos_dataset()` e preserva os rótulos validados.
O pipeline permanece com os mesmos parâmetros de pré-processamento, ROI,
HOG, Gabor, bordas e geometria. Produz 25.413 características por imagem.

Os destinos são `<raiz-do-dataset>/processed/features_train.csv` e
`features_test.csv`. O processo recusa a execução se **qualquer um** deles já
existir. Não há opção de sobrescrita. Todos os dados são validados antes do
processamento; todas as imagens precisam produzir vetores finitos com a dimensão
esperada antes da escrita. Falhas tratadas de escrita removem apenas os CSVs
novos daquela execução. A criação exclusiva protege também contra um CSV que
surja durante o processamento; ela não constitui transação resistente a queda
de energia ou encerramento forçado do processo.

Os CSVs mantêm `caminho`, `classe`, `split` e `feat_0` até `feat_25412`.
Caminhos de imagens dentro do repositório são relativos à sua raiz; imagens
externas usam caminhos absolutos. A extração não modifica imagens.

Os notebooks 01 e 02 ainda passam explicitamente `../data`; `DATASET_DIR` não
reconfigura esses argumentos. Na etapa 2, o notebook 03 foi adaptado para usar
`DATASET_DIR`, a confirmação dos rótulos e o protocolo abaixo. Suas saídas antigas
foram preservadas e identificadas como históricas. Para abrir os notebooks:

```bash
cd notebooks
python -m jupyterlab
```

Não execute o notebook 03 nesta etapa: ele treina modelos. Dados, ambientes e
artefatos grandes não devem ser adicionados ao Git; `data/`, `.venv/`, `models/`,
`*.pkl` e `*.joblib` já estão ignorados.

## Etapa 2: ajuste, calibração e teste independentes

Ao concluir a etapa 2, a implementação havia sido testada somente com dados
sintéticos, sem manifesto real nem declaração dos rótulos. A etapa 3 acrescentou
o inventário e as divisões reais descritos abaixo. O notebook 03 não foi executado.

O notebook anterior definia `KNNAnomalyDetector` na célula de treinamento do k-NN.
`kneighbors(X)` consultava o próprio conjunto de referências e incluía a amostra
entre seus vizinhos. O percentil dessas distâncias definia um limiar otimista.
O scaler também era ajustado sobre todo o treino, sem reserva para calibração.

### Detector reutilizável: `src/knn.py`

- `fit(X_fit, sample_ids=...)` indexa somente as referências normais de ajuste.
  Calcula distâncias internas de diagnóstico com `kneighbors(X=None)`, que
  exclui a própria amostra por índice. Exige `n_fit >= k + 1`.
- Zeros entre amostras distintas com vetores iguais são distâncias legítimas;
  não são removidos. Essas distâncias internas **não** calibram o limiar.
- `calibrate(X_calibration, sample_ids=...)` calcula a distância média aos `k`
  vizinhos do ajuste para cada normal reservado. O limiar é o quantil empírico
  `1 - target_fpr`, usando `numpy.quantile(..., method="higher")`.
- `decision_function = limiar - distância média`. `predict` retorna `+1` se
  o resultado for não negativo, e `-1` caso contrário. Empates são normais.
  `score_samples` retorna a distância negativa, compatível com a avaliação
  existente que inverte o score para considerar anomalia como classe positiva.
- Não é possível prever antes da calibração. Reajustar o modelo invalida o
  limiar anterior. Se IDs forem fornecidos no ajuste, a calibração exige IDs
  únicos e disjuntos. Sem IDs, a API de baixo nível não consegue verificar a
  origem dos vetores; use o protocolo com manifesto.
- O parâmetro antigo `contamination` foi substituído por `target_fpr` (padrão
  `0.05`). Essa taxa é um objetivo de rejeição de boas, não a proporção de
  defeitos do teste. O quantil produz uma FPR empírica conservadora, **sem
  garantir** a FPR populacional/futura ou informar recall de defeitos.
- Exige ao menos duas imagens de calibração. Se `n_calibration * target_fpr < 1`,
  emite aviso sobre a resolução insuficiente da amostra para a taxa-alvo.

Exemplo de uso com matrizes **já separadas e transformadas pelo protocolo**:

```python
from src.knn import KNNAnomalyDetector

detector = KNNAnomalyDetector(n_neighbors=5, target_fpr=0.05)
detector.fit(preparados.X_fit, sample_ids=preparados.fit_ids)
detector.calibrate(
    preparados.X_calibration,
    sample_ids=preparados.calibration_ids,
)
# Inspecionar detector.threshold_, target_fpr_, n_calibration_ e calibration_fpr_.
```

### Divisão reproduzível: `src/validation.py`

Defaults iniciais: semente `42`, fração de calibração `0.2`, pelo menos seis
amostras no ajuste e duas na calibração. São configuráveis, não decisões
metodológicas já aprovadas para o experimento final. Para 320 imagens independentes,
20% correspondem a 256 de ajuste e 64 de calibração; essa divisão foi efetivamente
registrada na etapa 3. Com 64 boas, um falso positivo equivale a 1,5625 ponto percentual.

As funções recebem um inventário validado pelo carregador:

| Função | Responsabilidade |
|---|---|
| `adicionar_hashes` | Lê os arquivos e adiciona SHA-256 sem modificá-los. |
| `gerar_divisao` | Ordena IDs canônicos e divide somente `train/good`, com semente e proporção configuráveis. |
| `salvar_divisao` | Cria manifesto JSON exclusivamente; nunca sobrescreve um arquivo existente. |
| `carregar_divisao` / `validar_divisao` | Verificam integridade, inventário, cobertura, grupos e ausência de sobreposição. |
| `obter_ou_criar_divisao` | Reusa a mesma divisão/configuração; alterações exigem um novo destino explícito. |
| `preparar_caracteristicas` | Ajusta seletor opcional, StandardScaler e PCA opcional somente no subconjunto de ajuste. |
| `transformar_teste` | Aplica somente `transform`, na ordem `test_ids`, sem reajustar parâmetros. |

O manifesto registra IDs de ajuste/calibração/teste, parâmetros, fração realizada,
grupos utilizados, impressão digital do inventário e hash de integridade. A
identificação usa caminhos absolutos canônicos; mover o dataset exige uma migração
explícita do manifesto, nunca um remapeamento silencioso. Com SHA-256, alterações
de conteúdo invalidam o inventário; sem hashes, a verificação cobre apenas
caminhos, rótulos e grupos, sem detectar cópias com nomes diferentes.

Colunas `grupo`, `peca_id`, `aquisicao_id`, `group_id`, `piece_id` e
`acquisition_id` são detectadas automaticamente. Outros identificadores devem
ser informados em `group_columns=["nome_da_coluna"]`. Nenhum grupo é inferido
dos números dos arquivos. Identificadores ausentes são rejeitados. Amostras
ligadas por qualquer identificador ou hash idêntico são mantidas juntas,
inclusive por relações transitivas. Compartilhamento com `test` interrompe o
protocolo. Sem identificadores, cada imagem é independente, exceto duplicatas
exatas detectáveis pelos hashes.

Quando há grupos, a proporção se aplica aos grupos/componentes, e pode diferir
da proporção por imagem. A fração realizada fica registrada. Divisões pequenas
ou com menos de dois grupos independentes são rejeitadas; o código não tenta
outras sementes silenciosamente.

O teste é mantido fora da seleção aleatória e do ajuste/calibração. Seus
metadados/hashes são consultados apenas para integridade e prevenção de
sobreposição. Seus vetores não entram em `fit` de transformadores ou detectores.

### Preparação das características sem vazamento

`preparar_caracteristicas` recebe a tabela completa de características com os
mesmos metadados/hashes/grupos do manifesto e uma lista explícita `feat_N`.
Transformadores opcionais são clonados para descartar qualquer ajuste anterior.
A ordem é seletor não supervisionado opcional → StandardScaler → PCA opcional.
Somente `fit_ids` entram no `fit_transform`; calibração recebe apenas `transform`.
Nenhum seletor ou PCA foi ativado no notebook nesta etapa.

```python
from src.validation import preparar_caracteristicas

preparados = preparar_caracteristicas(
    df_features, divisao, cols_features,
    # selector=..., pca=...  # experimentos de etapa futura
)
```

O notebook 03 compara metadados dos CSVs ao inventário atual e reutiliza
`artifacts/splits/one_class_v1.json`. Hashes em CSVs, quando presentes, também são
conferidos. CSVs históricos sem hashes/configuração não têm proveniência completa:
anexar hashes atuais não prova que seus vetores foram extraídos desses mesmos
bytes. Antes de reutilizá-los, verificar sua origem ou gerar um novo cache em
etapa autorizada, sem sobrescrever resultados anteriores.

### Antes da etapa de redução das características

Checklist registrado na etapa 2. Na etapa 3, a confirmação dos rótulos, a semente
42, a reserva de 20% e a geração do manifesto foram concluídas. Os comandos
específicos da etapa 3 abaixo substituem o exemplo de destino desta seção.

1. Confirmar a origem e os rótulos dos 320 PNGs reais. Só então habilitar
   `TRAIN_FLAT_IS_GOOD=true` e validar o carregamento.
2. Confirmar se há IDs de peça/aquisição e adicioná-los ao inventário. A ausência
   de metadados locais não comprova independência entre imagens.
3. Fixar a fração de calibração, semente, tamanho mínimo e FPR-alvo operacional;
   registrar também as versões do ambiente antes dos experimentos.
4. Gerar hashes e salvar um manifesto único. Exemplo futuro, apenas após a
   confirmação dos rótulos (não executado nesta etapa):

   ```python
   from src.dataset import carregar_caminhos_dataset
   from src.validation import adicionar_hashes, obter_ou_criar_divisao

   inventario = adicionar_hashes(carregar_caminhos_dataset())
   # Anexar aqui IDs de peça/aquisição, se disponíveis, antes de gerar a divisão.
   divisao = obter_ou_criar_divisao(
       inventario, "artifacts/splits/one_class_v1.json",
       calibration_fraction=0.2, seed=42,
   )
   ```

5. Validar a procedência do cache de características ou autorizar nova extração,
   preservando o HOG/Gabor/bordas/geometria atuais como referência. Não salvar
   esse novo cache sobre CSVs existentes.
6. Definir como selecionar parâmetros de HOG/PCA/k sem consumir a calibração
   final nem o teste: usar uma validação interna ao ajuste ou um conjunto de
   desenvolvimento separado. Boas reservadas permitem calibrar falsos positivos;
   não permitem escolher configurações por recall de defeitos. Se isso for
   necessário, obter defeitos de desenvolvimento independentes do teste.

Não retreinar o índice k-NN ou as transformações incorporando a calibração após
escolher o limiar: as distâncias mudariam e exigiriam nova reserva independente.
A etapa 2 não implementa Mahalanobis/Isolation Forest nem a calibração dos outros
detectores. Todas as saídas salvas no notebook são anteriores a este protocolo.

Para executar somente os testes sintéticos:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Etapa 3 — comparação reproduzível de representações

O grupo confirmou que as 320 imagens de `~/Downloads/data/train/` são boas.
Essa confirmação autoriza `TRAIN_FLAT_IS_GOOD=true` para este dataset. O novo
comando `src.feature_experiments` reutiliza o carregador e `src.validation`.
As saídas são criadas em `results/`, ignorado pelo Git, e nunca sobrescritas.
Não é necessário executar notebooks nem reutilizar CSVs históricos.

### Inventário, procedência e partições

A preparação executada gerou `results/stage3/study_v1/` com:

- `inventory.json`: caminhos, classes, splits e SHA-256 das 480 imagens;
- `outer_manifest.json`: semente 42, 256 boas para desenvolvimento, 64 boas
  reservadas para calibração final e os 160 arquivos de teste preservados;
- `internal_manifest.json`: três repetições internas, sementes 42, 43 e 44;
  cada repetição usa 163 imagens para ajuste, 41 para calibrar o limiar interno
  e 52 para validar a rejeição de normais, todas entre as 256 de desenvolvimento;
- `audit.json` e `environment.json`: contagens, chaves de metadados PNG,
  arquivos auxiliares, CSVs encontrados, versões e hashes do código;
- `metadata_review.json`: registro adicional da inspeção dos valores dos
  comentários e XMP realizada nesta etapa.

Os nomes do treino são numéricos (`000.png` a `319.png`). Todos os comentários
declaram `MVTec Anomaly Detection Dataset v1.0`. O XMP é igual em todas as
imagens e contém atribuição/licença, sem identificadores de peça ou aquisição.
Não foram encontrados arquivos auxiliares no dataset, nem duplicatas exatas
entre as 480 imagens. Isso não prova independência entre aquisições nem exclui
imagens semelhantes. Os rótulos do treino foram confirmados pelo grupo; esses
metadados não fornecem rótulos individuais.

Se surgirem identificadores reais, criar **outro estudo** com
`prepare --groups-csv caminho/grupos.csv`. O CSV deve conter `caminho` e apenas
colunas de identificadores, cobrindo todas as imagens; caminhos relativos são
resolvidos a partir da raiz do dataset. Os identificadores e hashes mantêm
amostras relacionadas na mesma partição, inclusive nas divisões internas.
Compartilhamento com o teste é rejeitado. Não inferir grupos pelos números dos
arquivos. Com grupos, as contagens por partição podem mudar.

Não foram encontrados CSVs antigos de características na inspeção local.
O comando registra CSVs encontrados dentro do dataset e de `data/`, incluindo
hash, tamanho e quantidade de colunas `feat_`; sua configuração é tratada como
não verificável apenas pelo CSV. Os experimentos usam extrações novas com
procedência registrada. Se houver resultados em outro local, preservá-los e
registrar sua origem antes de considerar seu uso.

O teste (41 boas e 119 defeituosas nas seis categorias) foi apenas inventariado
e verificado quanto à integridade. Nem teste nem as 64 boas de calibração final
entram na extração ou comparação desta etapa.

### Configurações e ordem dos atributos

`HOGConfig` permite variar células/orientações. A chamada original, sem
configuração, continua usando A. ROI 384×128, blocos 2×2 e normalização L1
permanecem iguais, assim como pré-processamento, segmentação e alinhamento.

| Configuração | Célula HOG | Orientações | HOG | Total bruto |
|---|---|---:|---:|---:|
| A — baseline original | 8×8 | 9 | 25.380 | 25.413 |
| B | 16×16 | 9 | 5.796 | 5.829 |
| C | 16×16 | 6 | 3.864 | 3.897 |
| D | 32×32 | 9 | 1.188 | 1.221 |

O vetor bruto mantém a ordem HOG → Gabor (24) → bordas (3) → geometria (6).
Os limites de cada bloco ficam registrados no JSON do cache. Não há seleção
adicional de atributos nesta etapa.

`FeatureRepresentation` aplica StandardScaler e, opcionalmente, PCA, ambos
ajustados **somente nas 163 imagens de ajuste de cada repetição interna**.
Calibração interna e validação recebem apenas `transform`. Há três estratégias:

- `none`: vetor completo padronizado, na ordem original;
- `full`: PCA do vetor completo, com 10, 25, 50 ou 100 atributos de saída;
- `hog`: PCA apenas do HOG padronizado, seguido dos mesmos 33 atributos
  auxiliares padronizados; saídas de 43, 58, 83 ou 133 atributos.

O PCA usa SVD completa, sem whitening e sem nova padronização após a projeção.
Componentes acima de `min(n_ajuste - 1, dimensão)` ou do posto numérico medido
no ajuste são registradas como inviáveis, sem redução silenciosa do pedido.
As dimensões de PCA foram verificadas com dados sintéticos; sua viabilidade
no conjunto real será determinada no experimento completo.

### Procedimento de execução

Na raiz do projeto, com o ambiente instalado conforme as instruções iniciais:

```bash
export DATASET_DIR="$HOME/Downloads/data"
export TRAIN_FLAT_IS_GOOD=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

.venv/bin/python -m unittest discover -s tests -v
```

Preparação e piloto abaixo **já foram executados**. Para repeti-los, escolher
novos destinos, como `study_v2` e `pilot_v2`; um diretório existente é rejeitado.

```bash
.venv/bin/python -m src.feature_experiments prepare \
  --out results/stage3/study_v1

.venv/bin/python -m src.feature_experiments extract \
  --study results/stage3/study_v1 \
  --out results/stage3/pilot_v1 --sample-size 8
```

O piloto escolhe as mesmas 8 imagens do ajuste com semente 42. As quatro
configurações foram extraídas com sucesso, sem ajustar detectores ou PCA reais:

| Configuração | Forma verificada | Extração das 8 imagens | Bytes da matriz float32 |
|---|---|---:|---:|
| A | 8 × 25.413 | 2,549 s | 813.216 |
| B | 8 × 5.829 | 2,486 s | 186.528 |
| C | 8 × 3.897 | 2,492 s | 124.704 |
| D | 8 × 1.221 | 2,547 s | 39.072 |

Esses tempos são observações deste piloto, não uma estimativa de desempenho
final. `pilot_v1/verification.json` registra dimensões, valores finitos, igualdade
dos 33 atributos auxiliares entre configurações e hashes preservados dos 480
arquivos. A suíte passou em 79 testes, incluindo os 64 das etapas anteriores.

**Próximos comandos**, ainda não executados: extração das quatro representações
para todas as 256 imagens de desenvolvimento e comparação interna:

```bash
.venv/bin/python -m src.feature_experiments extract \
  --study results/stage3/study_v1 \
  --out results/stage3/full_v1 --full

.venv/bin/python -m src.feature_experiments compare \
  --study results/stage3/study_v1 \
  --cache results/stage3/full_v1 \
  --out results/stage3/comparison_v1 \
  --components 10 25 50 100 --neighbors 5 --target-fpr 0.05 --threads 1
```

`--full` significa todos os IDs do ajuste externo, não todo o dataset. O cache
piloto é rejeitado pela comparação completa. `--configs A B C D` permite
selecionar configurações explicitamente. Os caches NPZ registram IDs, hashes
das imagens e ordem dos atributos; JSONs registram dimensões, configuração,
semente, tempos por imagem, bytes de arrays/arquivo e versões/hashes do código.
Mudanças no código de extração exigem um novo cache. Falhas podem deixar um
diretório parcial, que deve ser preservado; usar outro destino para nova execução.

### Critérios de comparação e limites da interpretação

A grade padrão contém 36 representações (4 HOG × [sem PCA + 4 PCA completo +
4 PCA HOG]), com três repetições: até 108 execuções válidas. Configurações de
PCA incompatíveis com o posto são registradas como `skipped`, com motivo.

O k-NN é uma referência fixa (`k=5`, FPR-alvo interno 5%), com calibração
exclusiva nas 41 boas internas e avaliação nas 52 boas reservadas. Essa é uma
comparação interna das representações, anterior ao treinamento comparativo
final dos quatro modelos. `comparison.json` e `comparison.csv` registram:

- rejeição de normais na validação e sua dispersão entre repetições;
- dimensões, posto, variância explicada e limiar calibrado;
- tempos de extração, ajuste/aplicação das transformações e k-NN;
- memória aproximada dos arrays e estado dos transformadores.

A contagem de bytes não mede RSS nem pico de memória de SVD/BLAS. As repetições
compartilham imagens, portanto o desvio-padrão não é um intervalo de confiança
de experimentos independentes. Tempos dependem do ambiente e não são idênticos
entre execuções. A SVD completa pode ser custosa, especialmente no baseline.

Não há vencedor automático. Menos atributos, mais variância explicada ou menor
rejeição de boas, isoladamente, não provam melhor detecção de defeitos. Sem
defeitos independentes de desenvolvimento, não é possível escolher pela
sensibilidade a defeitos. O FPR-alvo operacional e eventuais grupos reais ainda
precisam ser definidos pelo grupo antes de congelar o protocolo final.

Após os experimentos internos, registrar a representação escolhida e sua
justificativa. Somente em etapa posterior ajustar transformações/modelos nas
256 boas, usar as 64 boas exclusivamente para calibração final e avaliar uma
vez no teste. Mahalanobis e Isolation Forest ainda não foram implementados.

## Etapa 4 — SVM e k-NN nas representações selecionadas

`src.model_experiments` reutiliza o inventário e as divisões de
`results/stage3/study_v1`, os caches A/C de `full_v1`, `FeatureRepresentation`
e o k-NN corrigido. Não abre imagens das 64 boas de calibração final ou do teste.
Os metadados dessas reservas continuam no manifesto para verificar a separação.
A verificação de bytes nesta etapa abre somente as 256 imagens de desenvolvimento.

As representações são fixas:

| ID | Descrição | Dimensão |
|---|---|---:|
| R1 | HOG A, sem PCA, StandardScaler | 25.413 |
| R2 | HOG C, StandardScaler, PCA HOG 10 + 33 auxiliares | 43 |
| R3 | HOG C, StandardScaler, PCA completo 25 | 25 |

Em cada repetição congelada (42, 43, 44), scaler/PCA são ajustados somente nas
163 boas de ajuste. As mesmas transformações são reutilizadas entre os modelos
naquela repetição. Os limiares usam exclusivamente as 41 boas internas; a rejeição
é medida nas 52 boas de validação. Nenhum descritor ou divisão foi alterado.

### Detectores, grade e critério provisório

- SVM RBF: `nu` em 0,01 / 0,05 / 0,10; `gamma` igual a 0,5 / 1 / 2 vezes
  `gamma="scale"`, calculado apenas no ajuste. A combinação histórica
  `nu=0.05, gamma="scale"` está incluída. `nu` não é uma garantia de FPR.
- k-NN: `k` em 3 / 5 / 10 / 20; agregação `mean` (média, padrão preservado)
  ou `kth` (distância ao k-ésimo vizinho), distância euclidiana. O diagnóstico
  das distâncias de treino exclui a própria amostra em ambas as agregações.
- Grade completa: 17 configurações de modelo × 3 representações × 3 repetições,
  totalizando 153 execuções. Não há busca adaptativa nem uso das reservas.

`src.anomaly.CalibratedAnomalyMixin` compartilha a calibração entre os modelos.
`anomaly_score` cresce com a anomalia; `score_samples` tem sinal oposto.
O limiar é o quantil empírico 95% (`method="higher"`) dos scores das boas
internas. `decision_function = threshold - anomaly_score`; valores >= 0
produzem +1 (normal), os demais -1. Empates são normais.

`CalibratedOneClassSVM` preserva o SVM nativo em `model_` para referência, mas
suas predições operacionais usam o limiar independente. A rejeição pelo limiar
nativo também é registrada, sem ser usada na seleção. O histórico preserva
hiperparâmetros, não os resultados do notebook obtidos sob outro protocolo.

O critério foi definido antes da busca: minimizar a raiz da média dos quadrados
das diferenças entre a rejeição de boas e 5%, nas três repetições (RMSE).
Isso considera proximidade da meta e dispersão, sem premiar automaticamente
rejeição zero. Empates em 12 casas são resolvidos pela menor dimensão e depois
pela ordem lexical da configuração. A regra é exploratória, não uma medida de
sensibilidade a defeitos, nem um teste estatístico de superioridade.

### Testes, piloto e busca completa

Na raiz do repositório:

```bash
export DATASET_DIR="$HOME/Downloads/data"
export TRAIN_FLAT_IS_GOOD=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python -m src.model_experiments \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --out results/stage4/pilot_v1 --pilot

.venv/bin/python -m src.model_experiments \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --out results/stage4/comparison_v1
```

O piloto usa somente R2, a primeira divisão já congelada e dois modelos:
SVM histórico e k-NN k=5/média. Não gera seleção. Os comandos recusam destinos
existentes; para reproduzir, usar `pilot_v2`, `comparison_v2` etc. A comparação
completa usa sempre as três repetições, independentemente do piloto.

Antes de executar, o comando confere a integridade do manifesto, SHA-256 dos
caches, configuração dos descritores, dimensões, ordem dos IDs e hashes das
imagens de desenvolvimento. Confere também margem de 1 GiB de memória disponível
e 256 MiB de disco; são estimativas de planejamento. Execução sequencial com
uma thread BLAS. Falhas após o início ficam em `failure.json`, com resultados
parciais em `runs.jsonl`; não há redução silenciosa da grade ou do protocolo.

Cada diretório registra `protocol.json`, `runs.jsonl`, `comparison.json` e
`summary.csv`; a busca completa também registra `selection.json`, com uma
escolha provisória por modelo. Os registros incluem:

- fontes e versões, hashes dos caches, configurações, sementes e regra de seleção;
- rejeição de boas, dispersão e RMSE em relação à meta;
- tempos de ajuste/aplicação das transformações, ajuste dos modelos,
  calibração dos limiares e cálculo dos scores de validação;
- bytes aproximados de arrays dos modelos, transformadores e matrizes;
  pico RSS do processo inteiro (Linux), não por configuração;
- scores individuais das boas de calibração interna/validação com seus IDs,
  predições e distribuições (média, desvio, quantis, extremos e valores únicos).

Scores brutos de modelos/representações diferentes não compartilham escala.
As distâncias de referência do k-NN excluem a própria amostra; os scores de
referência do SVM são medidos nas amostras de ajuste. Essas estatísticas têm
finalidades diagnósticas, não são métricas comparáveis de generalização.
Tempos de transformação aparecem por configuração para consulta, mas o ajuste
foi realizado uma única vez por representação/repetição; não somar esses tempos
repetidos como se fossem execuções independentes.

### Calibração final: procedimento posterior, separado da etapa 4

`src.final_calibration` implementa o passo posterior, **não executado com a
reserva real nesta etapa**. Verifica a assinatura da seleção, relatório, fontes,
manifesto e caches. Ajusta scaler/PCA e o modelo selecionado nas 256 boas; extrai
somente as 64 boas finais com o descritor selecionado, aplica `transform` e calibra
um limiar a 5%. Não lê o teste, não escolhe hiperparâmetros e não reajusta o modelo
incorporando as boas de calibração. Salva `model.joblib`, características da
calibração e metadados em diretório novo.

Comandos para uma etapa futura, após congelar e aceitar a seleção provisória:

```bash
.venv/bin/python -m src.final_calibration \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --selection results/stage4/comparison_v1/selection.json \
  --model svm --out results/stage4/final_svm_v1 \
  --confirm-final-calibration

.venv/bin/python -m src.final_calibration \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --selection results/stage4/comparison_v1/selection.json \
  --model knn --out results/stage4/final_knn_v1 \
  --confirm-final-calibration
```

A opção explícita evita consumir acidentalmente a reserva durante a busca.
Se o código metodológico mudar após a seleção, o procedimento interrompe e exige
revisão da procedência. Não repetir a calibração com outras configurações para
escolher qual tem melhor comportamento nas 64 boas.

Mahalanobis e Isolation Forest permanecem não implementados: não há comando
executável para eles nesta etapa. Sua implementação futura deverá seguir as
mesmas divisões internas, convenção de scores e calibração independente antes
de integrar a comparação final. A avaliação no teste é outro passo posterior.

### Limitações metodológicas

As representações já foram selecionadas na etapa 3 usando essas divisões, e a
etapa 4 busca hiperparâmetros nas mesmas divisões. As estimativas resultantes são
otimistas para as configurações escolhidas e não substituem avaliação independente.
As três repetições compartilham imagens; o desvio entre elas não é um intervalo
de confiança. Com 52 boas na validação, uma rejeição equivale a cerca de 1,92
ponto percentual; com 41 na calibração, a resolução é cerca de 2,44 pontos.

Resultados exclusivamente normais não permitem concluir recall, precisão,
F1 de defeitos, ROC-AUC ou superioridade de detecção. Não há identificadores de
peça/aquisição disponíveis para comprovar independência. A meta operacional de
5% continua provisória. Para selecionar pela sensibilidade, seria necessário
um conjunto independente de defeitos de desenvolvimento, sem consumir o teste.

### Execução real concluída da etapa 4

A execução em `results/stage4/comparison_v1` concluiu as 153 combinações com
repetição (51 configurações agregadas) em 36,49 s de processo, com pico RSS de
463,09 MiB. O piloto em `pilot_v1` levou 3,25 s. Os 89 testes passaram; os hashes
dos artefatos da etapa 3 permaneceram iguais. Os limiares e as contagens das nove
referências k-NN k=5/média reproduziram exatamente os resultados da etapa 3.

Escolhas provisórias pelo critério previamente definido:

| Modelo | Representação | Parâmetros | Rejeição média de boas ± DP |
|---|---|---|---:|
| SVM | R2, 43 atributos | nu=0,10; gamma=0,5 × scale | 4,49% ± 1,81 pp |
| k-NN | R3, 25 atributos | k=3; distância média | 5,13% ± 0,91 pp |

O k-NN empatou nas taxas com R2/k=3/média e R2/k=5/média; R3 foi escolhido
pelo desempate de menor dimensão. Isso não demonstra superioridade de detecção
nem menor memória total: o estado PCA de R3 é maior que o de R2. A diferença
entre os melhores SVMs de R2/R3 também é pequena. As reservas finais não foram
abertas nem utilizadas nesta busca.

Relatório detalhado: [results/stage4/comparison_v1/report.md](results/stage4/comparison_v1/report.md).
Tabelas: `summary.csv`, `best_by_representation.csv`, `score_distributions.csv`.
Auditoria e logs: `results/stage4/execution_v1/`. A calibração final real não foi
executada; os comandos anteriores para `final_calibration` continuam sendo passos
futuros, após congelar a seleção. Mahalanobis e Isolation Forest permanecem pendentes.

## Etapa 5 — Mahalanobis e Isolation Forest

Esta etapa amplia `src.model_experiments` com `--stage 5`, mantendo a execução
padrão da etapa 4. Reutiliza o mesmo manifesto, caches A/C, representações,
transformações e calibração independente. Somente os 256 IDs de desenvolvimento
podem entrar nos experimentos; as imagens das reservas finais não são abertas.
As referências a modelos pendentes nas seções anteriores descrevem o estado
histórico: nesta etapa os quatro detectores estão implementados.

### Interfaces e cuidados numéricos

`MahalanobisAnomalyDetector` (`src.mahalanobis`) e
`IsolationForestAnomalyDetector` (`src.isolation_forest`) oferecem `fit`,
`calibrate`, `anomaly_score`, `score_samples`, `decision_function` e `predict`.
Ambos reutilizam `CalibratedAnomalyMixin`: `anomaly_score` cresce com a anomalia,
`score_samples` tem sinal contrário (maior = mais normal), `predict` usa +1/-1.
Os limiares são calibrados nas 41 boas internas com `target_fpr=0.05`.
Os métodos de baixo nível recebem `sample_ids` para impedir sobreposição;
o protocolo verifica adicionalmente grupos/hashes e partições.

Mahalanobis usa distância **quadrática** d²; essa escolha preserva a ordenação
por distância. O score de normalidade é -d². O estimador de covariância pode ser:

- `covariance="ledoit_wolf"`: regularização por shrinkage estimada só no ajuste;
- `covariance="elliptic"`: EllipticEnvelope, com `support_fraction=None` ou 0,75.
  Seu limiar nativo é ignorado; o `contamination=0.1` interno não define as
  predições deste detector, que dependem da calibração externa.

Não existe opção de covariância empírica direta. Mahalanobis só participa em
R2/R3; a classe também bloqueia entradas acima de 256 atributos. O cálculo usa
Cholesky/solução triangular, sem inversão direta ou pseudoinversa silenciosa.
Posto zero, covariância não positiva definida, condição >1e12, overflow ou avisos
na estimação são tratados explicitamente. Ledoit-Wolf pode regularizar entradas
com posto reduzido quando ainda há variação. Elliptic exige posto completo e
n>=2p+1, além das verificações na covariância estimada. Essas condições são
critérios numéricos conservadores, não garantia estatística de robustez.
Não removemos colunas nem alteramos R2/R3 para tornar o método viável.

Isolation Forest usa `contamination="auto"`, `n_jobs=1` e `random_state=42`.
O offset nativo é ignorado: o wrapper usa `-model_.score_samples` como score de
anomalia e calibra um limiar separado. `max_samples` inteiro acima do número de
amostras é rejeitado, sem redução silenciosa. Dados sem variabilidade, parâmetros
inválidos e scores não finitos são rejeitados.

### Grade e execução

- Mahalanobis: Ledoit-Wolf, Elliptic com suporte automático, Elliptic com suporte
  0,75, em R2/R3: 18 tentativas nas três repetições.
- Isolation Forest: 100/200 árvores × `max_samples` 64/128 × `max_features` 0,5/1,0,
  em R1/R2/R3: 72 execuções nas três repetições.
- Total previsto: 90 tentativas. Sementes das divisões 42/43/44, semente dos
  detectores sempre 42. StandardScaler/PCA ajustados somente nas 163 boas internas.

Configurações numericamente inviáveis são registradas em `skipped`, com motivo,
sem substituição. Só candidatos viáveis nas três repetições podem ser selecionados.
Falhas de recursos interrompem a execução e são registradas; não há mudança
silenciosa da grade. O critério continua sendo RMSE da rejeição de boas versus
5%; desempates em 12 casas usam dimensão e depois ordem lexical da configuração.

```bash
export DATASET_DIR="$HOME/Downloads/data"
export TRAIN_FLAT_IS_GOOD=true
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python -m src.model_experiments --stage 5 \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --out results/stage5/pilot_v1 --pilot

.venv/bin/python -m src.model_experiments --stage 5 \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --out results/stage5/comparison_v1
```

O piloto usa somente a primeira divisão já congelada: as três variantes de
Mahalanobis em R2/R3 e uma configuração de floresta em R1/R2/R3 (9 tentativas).
Não gera seleção. As saídas são exclusivas; usar nomes v2/v3 se v1 já existir.
Os registros incluem scores individuais e distribuições, rejeição/estabilidade,
RMSE, parâmetros, diagnósticos de covariância/floresta, tempos, arrays retidos
(incluindo nós/valores das árvores) e pico RSS do processo. Não somar tempos de
transformações repetidos nas linhas: elas são reutilizadas entre candidatos.

### Preparar os quatro modelos para a calibração final

A integração altera a orquestração e amplia os hashes de procedência. Para não
ignorar verificações nem modificar a seleção histórica da etapa 4,
`src.selection_bundle` cria uma seleção nova com os quatro modelos. Ele:

1. Confere assinaturas, relatórios, manifesto e caches de ambas as seleções.
2. Exige que o código metodológico antigo (SVM, k-NN, calibração e transformações)
   tenha os mesmos hashes; permite mudanças somente nos dois módulos de
   orquestração existentes (`model_experiments.py` e `final_calibration.py`).
3. Reproduz as seis execuções internas selecionadas de SVM/k-NN, verificando
   scores e limiares com tolerâncias 1e-12 e predições idênticas.
4. Copia as escolhas provisórias sem reselecioná-las e registra fontes antigas,
   atuais, relatórios de origem e resultado da compatibilidade.

Esse comando só usa o desenvolvimento; não consome a calibração final:

```bash
.venv/bin/python -m src.selection_bundle \
  --study results/stage3/study_v1 --cache results/stage3/full_v1 \
  --stage4 results/stage4/comparison_v1/selection.json \
  --stage5 results/stage5/comparison_v1/selection.json \
  --out results/stage5/selection_all_v1
```

Após a integração, usar o novo `selection_all_v1/selection.json` nos comandos
finais, substituindo os exemplos históricos da etapa 4. O procedimento valida
também os hashes das seleções/relatórios de origem e o registro de compatibilidade.

**Etapa posterior, NÃO executar durante a seleção:** os comandos abaixo ajustarão
cada modelo nas 256 boas e usarão as 64 boas somente para calibrar o limiar final.
Não abrem o teste. Os nomes de saída precisam estar livres.

```bash
for model in svm knn mahalanobis isolation_forest; do
  .venv/bin/python -m src.final_calibration \
    --study results/stage3/study_v1 --cache results/stage3/full_v1 \
    --selection results/stage5/selection_all_v1/selection.json \
    --model "$model" --out "results/stage5/final_${model}_v1" \
    --confirm-final-calibration || break
done
```

Os modelos finais ainda não foram calibrados com dados reais nesta etapa.
Os testes do procedimento final usam exclusivamente fixtures sintéticos temporários.

### Limitações da etapa 5

Rejeitar poucas boas ou ficar próximo de 5% não demonstra capacidade de detectar
defeitos. As representações já foram selecionadas usando essas divisões, que
agora também servem à busca dos novos modelos; a seleção é exploratória e suas
estimativas são otimistas. Três repetições correlacionadas não fornecem intervalo
de confiança independente. A variabilidade de Isolation Forest entre sementes
não foi estudada: random_state ficou fixo. EllipticEnvelope pressupõe uma estrutura
aproximadamente elíptica/unimodal e pode ser estatisticamente frágil em poucas
amostras mesmo quando numericamente viável. Não usamos as reservas para contornar
essas limitações ou desempatar configurações.

### Execução real concluída da etapa 5

A busca em `results/stage5/comparison_v1` concluiu **90 execuções válidas**
(30 configurações agregadas), sem inviabilidades nem falhas de recursos.
O piloto levou 7,36 s; a busca completa levou 76,29 s, com pico RSS de 477,68 MiB.
Os 100 testes passaram em 30,28 s. Os artefatos das etapas 3/4 permaneceram intactos.

| Modelo | Escolha provisória | Rejeição média de boas ± DP | RMSE versus 5% |
|---|---|---:|---:|
| Mahalanobis | R3; EllipticEnvelope; suporte automático; semente 42 | 3,85% ± 2,72 pp | 2,954 pp |
| Isolation Forest | R1; 100 árvores; 128 amostras; 50% dos atributos; semente 42 | 5,13% ± 0,91 pp | 0,916 pp |

A regularização Ledoit-Wolf produziu covariâncias mais bem condicionadas que as
robustas, sobretudo em R2. Todas foram numericamente viáveis nesta execução;
isso não valida a hipótese de distribuição elíptica nem comprova detecção de
defeitos. Isolation Forest selecionado em R1 é mais custoso que as alternativas
reduzidas; a escolha seguiu o critério previamente definido, sem mudar a regra
após observar tempos ou métricas.

A seleção conjunta já foi preparada em
`results/stage5/selection_all_v1/selection.json`. As seis execuções selecionadas
de SVM/k-NN reproduziram exatamente scores, limiares e predições anteriores.
Nenhuma imagem reservada foi aberta, e a calibração final real não foi executada.

Relatório: [results/stage5/comparison_v1/report.md](results/stage5/comparison_v1/report.md).
Tabelas: `summary.csv`, `best_by_representation.csv`, `covariance_diagnostics.csv`
e `score_distributions.csv`. Auditoria: `results/stage5/execution_v1/postflight.json`.

## Etapas 6 a 9: LOF, avaliação no teste e melhoria com descritores locais

Escopo final do projeto: visão computacional clássica, com três detectores
One-Class: One-Class SVM, LOF e Isolation Forest. k-NN e Mahalanobis (etapas
4/5) ficam como registro histórico. O experimento com redes neurais foi movido
para a pasta separada [`neural_networks/`](neural_networks/README.md) e não é
usado por `src/`.

A lista completa de arquivos criados, alterados e movidos nesta rodada, e a
explicação de por que os resultados melhoraram, estão em [ALTERACOES.md](ALTERACOES.md).
Relatório final: `docs/relatorio/index.html`. Pôster A3: `docs/poster/poster_a3.pdf`.

### Protocolo comum

- Ajuste somente nas 256 boas de desenvolvimento (`fit_ids` do manifesto externo).
- Limiar no quantil empírico de 95% (`method="higher"`) dos scores das 64 boas
  reservadas (`calibration_ids`), meta de 5% de boas rejeitadas.
- Defeito é a classe positiva: falso negativo = parafuso defeituoso aprovado.
  Critério de escolha (slide 12): menor número de falsos negativos.

### Etapa 6: Local Outlier Factor

`src/lof.py` implementa `LOFAnomalyDetector` (scikit-learn, `novelty=True`,
limiar nativo ignorado). `src/lof_experiments.py compare` repete a busca interna
das etapas 4/5 (sementes 42/43/44, R1/R2/R3, k em 5/10/20/35) com a regra RMSE
versus 5%. Escolha: R2, k=5. Os módulos já assinados não foram alterados.

### Etapa 8: avaliação dos modelos globais no teste completo

`src/final_evaluation.py` aplica os modelos congelados (vetor global de
HOG/Gabor/bordas/geometria) às 160 imagens de teste:

| Modelo | FN (de 119) | FP (de 41) | Recall defeitos | AUC |
|---|---:|---:|---:|---:|
| Isolation Forest (R1) | 100 | 3 | 16,0% | 0,634 |
| LOF (R2, k=5) | 104 | 3 | 12,6% | 0,622 |
| One-Class SVM (R2) | 110 | 1 | 7,6% | 0,438 |

Os três ficam perto do acaso. Uma verificação de robustez (`src/robustness_check.py`)
mostrou ainda um viés do dataset: as 41 boas de teste são mais escuras (brilho
médio 170) que as boas de treino (184) e os defeitos (187); só o brilho separa
boas de defeitos do teste com AUC 1,0. Por isso, nas etapas seguintes as boas
de teste nunca são usadas para escolher nada.

### Etapa 9: diagnóstico e melhoria dos modelos clássicos

#### Diagnóstico

Por bloco de atributos, AUC entre as 64 boas reservadas e os defeitos da metade
de seleção (LOF/Isolation Forest ajustados só no bloco):

| Bloco | Dimensão | LOF | Isolation Forest |
|---|---:|---:|---:|
| HOG 16x16 (config. C) | 3.864 | 0,51 | 0,60 |
| HOG 8x8 (config. A) | 25.380 | 0,48 | 0,47 |
| Gabor | 24 | 0,72 | 0,66 |
| Bordas | 3 | 0,61 | 0,65 |
| Geometria | 6 | 0,57 | 0,53 |

O vetor global é 99,9% HOG e o HOG global não carrega sinal: ele resume a peça
inteira, e a variação normal entre parafusos bons (fase da rosca, reflexos)
domina um defeito de poucos pixels. Os blocos pequenos têm algum sinal, mas são
diluídos. A segmentação e o alinhamento funcionam bem
(`results/stage8/report_figures/aligned_good_rois.png`); o problema é resumir
a ROI inteira num único vetor.

#### Método: descritores locais sobre a ROI alinhada (`src/local_features.py`)

1. Reutiliza a ROI alinhada das etapas 1-2 (cache em `src/roi_cache.py`),
   384x128, cabeça à esquerda. Brilho/contraste normalizados por imagem.
2. Divide a ROI numa grade de patches sobrepostos (lado 24, 32 ou 48, passo de meio patch).
3. Cada patch recebe um descritor clássico curto: HOG local (bloco 2x2), histograma
   LBP uniforme, energia média dos 12 filtros de Gabor do projeto, densidade de
   bordas Canny e silhueta (fração de máscara e bordas superior/inferior).
4. Um detector One-Class (SVM, LOF ou Isolation Forest) por posição da grade,
   ajustado com os patches das boas naquela posição e nas vizinhas (tolerância a
   pequenos desalinhamentos e à fase da rosca).
5. O score do patch é padronizado por média/desvio de boas fora da amostra
   (ajuste cruzado em 2 dobras); o score da imagem é o maior patch ou a média
   dos 3 maiores. Como os scores são por posição, o modelo também indica onde
   está o defeito.

#### Protocolo da etapa 9 (definido antes dos experimentos)

O teste completo já tinha sido consultado na etapa 8. Para não ajustar ao teste:

- `src/defect_split.py` divide o teste, estratificado por categoria e com semente
  2026, em SELEÇÃO (58 defeitos; as 20 boas dessa metade não são usadas) e
  HOLDOUT (61 defeitos e 21 boas), registrado em `results/stage9/defect_split_v1.json`.
- Seleção: maior AUC entre as 64 boas reservadas e os 58 defeitos de seleção;
  desempate pelo recall com 95% de especificidade. As boas de teste (com o viés de
  brilho) não participam.
- Grade: patch 24/32/48 x descritores (todos ou sem HOG) x vizinhança 1/2 x
  agregação máx/top3; LOF k 10/20/40, Isolation Forest 200 árvores com
  `max_samples` auto/512, SVM RBF `nu` 0,05/0,10: 168 combinações.
- Final: limiar nas 64 reservadas e avaliação única no holdout.

Escolhas (`results/stage9/selection_v1/selection.json`):

| Modelo | Patch | Descritores | Parâmetros | AUC seleção |
|---|---:|---|---|---:|
| LOF | 32 | todos | k=10, top3 | 0,984 |
| One-Class SVM | 24 | todos | nu=0,05, top3 | 0,975 |
| Isolation Forest | 24 | sem HOG | 200 árvores, 512 amostras, top3 | 0,948 |

#### Resultado no holdout (61 defeitos, 21 boas)

| Modelo | FN global | FN local | FP global | FP local | AUC global | AUC local | AUC reservadas vs defeitos (local) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOF | 53 | 2 | 0 | 3 | 0,561 | 0,958 | 0,986 |
| One-Class SVM | 57 | 11 | 0 | 3 | 0,440 | 0,903 | 0,935 |
| Isolation Forest | 54 | 19 | 2 | 2 | 0,600 | 0,910 | 0,924 |

Modelo escolhido: LOF local, com 59 de 61 defeitos detectados (96,7%) e 3 de 21
boas rejeitadas. Por categoria: manipulated_front 12/12, scratch_head 12/12,
scratch_neck 12/13, thread_side 11/12, thread_top 12/12. "Global" são os
modelos congelados das etapas 5/6 avaliados nas mesmas 82 imagens.

Ablação (LOF k=20, patch 32; AUC na seleção): HOG local sozinho 0,916; Gabor
sozinho 0,964; LBP 0,581; bordas+silhueta 0,483; sem HOG 0,938; todos 0,980;
sem tolerância a vizinhos 0,951; sem normalizar brilho 0,979. O mesmo HOG que
não tinha sinal no vetor global passa a ter quando usado localmente.

Figuras em `results/stage9/figures/`: `before_after.png`, `ablation.png` e
`local_maps_lof.png` (mapas de anomalia do LOF sobre a ROI: a região marcada cai
sobre a ponta amassada, a cabeça riscada, o pescoço e a rosca danificada).

Limitações: holdout pequeno (uma boa vale 4,8 pontos de especificidade); as boas
do holdout têm o viés de brilho (a normalização por imagem reduz seu efeito, e a
AUC contra as boas reservadas é informada ao lado); a grade foi avaliada na
mesma metade de seleção, então as AUCs de seleção são otimistas, mas o holdout
não participou de nenhuma escolha.

### Reprodução

```bash
export DATASET_DIR="$HOME/Downloads/data"
export TRAIN_FLAT_IS_GOOD=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -m unittest discover -s tests -v

# Etapa 6 (LOF global) e calibração final dos modelos globais
python -m src.lof_experiments compare --study results/stage3/study_v1 \
  --cache results/stage3/full_v1 --out results/stage6/lof_comparison_v1
python -m src.lof_experiments finalize --study results/stage3/study_v1 \
  --cache results/stage3/full_v1 --selection results/stage6/lof_comparison_v1/selection.json \
  --out results/stage6/final_lof_v1 --confirm-final-calibration
for model in svm isolation_forest; do
  python -m src.final_calibration --study results/stage3/study_v1 --cache results/stage3/full_v1 \
    --selection results/stage5/selection_all_v1/selection.json \
    --model "$model" --out "results/stage5/final_${model}_v1" --confirm-final-calibration || break
done

# Etapa 8: modelos globais no teste completo + figuras do pipeline
python -m src.final_evaluation --study results/stage3/study_v1 \
  --out results/stage8/test_evaluation_classical_v1 --confirm-test-evaluation
python -m src.report_figures --study results/stage3/study_v1 --out results/stage8/report_figures

# Etapa 9: divisão do teste, cache de ROIs, seleção, holdout e figuras
python -m src.defect_split --study results/stage3/study_v1 --out results/stage9/defect_split_v1.json
python -m src.roi_cache --study results/stage3/study_v1 --out results/stage9/roi_cache_v1.npz
python -m src.stage9_experiments select --out results/stage9/selection_v1   # --resume retoma
python -m src.stage9_experiments final --selection results/stage9/selection_v1/selection.json \
  --out results/stage9/final_v1
python -m src.stage9_figures --out results/stage9/figures
```

A busca da etapa 9 leva poucos minutos em 16 núcleos. Ela grava cada ajuste em
`runs.jsonl` e pode ser retomada com `--resume` se for interrompida.

Notas de ambiente desta execução (Linux, Python 3.14):

- Os manifestos guardam caminhos absolutos canônicos de `/home/gabrieladc/Downloads/data`.
  Para reutilizá-los em outra máquina, o dataset foi copiado byte a byte para esse
  mesmo caminho; os SHA-256 de todas as imagens foram verificados pelo protocolo.
- As seleções guardam SHA-256 dos arquivos `src/*.py` com finais de linha LF. Num
  checkout Windows com CRLF a verificação de procedência falha; converter para LF
  (conteúdo idêntico ao do Git) antes de rodar a calibração final.
