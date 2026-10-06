# Tipografia Museu Paulista

**Segmentação e classificação de amostras tipográficas provenientes de gavetas do Acervo Tércio Gaudêncio (Museu Paulista - USP)** 

---

## Visão Geral

Esse repositório é dedicado aos resultados da pesquisa "Classificação e agrupamento de fontes tipográficas de acervo do Museu Paulista com uso de técnicas de aprendizado de máquina" (Processo FAPESP 2024/23738-0), um trabalho exploratório que visou **automatizar e agilizar o processo de curadoria museológica** do Acervo Tércio Gaudêncio, uma coleção com quase 200 gavetas tipográficas.

A abordagem utilizada envolveu **classificar impressões sobre papel de um único tipo metálico em uma de quatro grandes famílias tipográficas**: escritural, fantasia, grotesca e serifada. Para isso, houve a necessidade de construir um *pipeline* completo de dados a partir de amostras fotográficas de impressões de cada gaveta, passando por quatro grandes fases:

- **Pré-processamento**: uso de técnicas de processamento de imagens para redução de ruídos e da fragmentação de caracteres por falhas de impressão, auxiliando na segmentação;
- **Segmentação**: recorte individualizado de cada caractere a partir das imagens originais, utilizando abordagem de componentes conectadas por vizinhança-8;
- **Validação Humana**: avaliação dos recortes obtidos na etapa anterior por nove voluntários diferentes, consolidando o *dataset* final, que será utilizado para treinamento dos algoritmos;
- **Treinamento de Algoritmos de Aprendizado de Máquina**: treinamento e avaliação de quatro algoritmos de aprendizado de máquina na tarefa de classificação dos tipos. Foram utilizados Random Forest, XGBoost, ResNet-18 (sem e com *data augmentation*) e EfficientNet-B0 (sem e com *data augmentation*), totalizando, portanto, seis configurações experimentais diferentes;

---
## Dataset

A organização e documentação deste conjunto de dados seguem as diretrizes propostas pelo framework *Datasheets for Datasets* [Gebru et al, 2021](https://arxiv.org/abs/1803.09010), um padrão da comunidade de aprendizado de máquina que visa garantir transparência, facilitar a reprodutibilidade e ajudar os usuários a tomarem decisões informadas sobre o uso dos dados.

O diretório local de dados foi estruturado separando claramente os estágios de processamento:

```text
data/
├── interim/
│   ├── data_1/
│   ├── data_1_2_final/
│   └── data_2/
├── k-fold_split/
│   └── typography_dataset_splits.csv
├── processed/
│   ├── escritural/
│   ├── fantasia/
│   ├── grotesco/
│   └── serifado/
└── raw/
    ├── escritural/
    ├── fantasia/
    ├── grotesco/
    └── serifado/
```

O dataset foi criado com o objetivo principal de automatizar e agilizar o processo de curadoria museológica do Acervo Tércio Gaudêncio. A lacuna que este conjunto preenche é a de fornecer um conjunto de dados estruturados para a tarefa específica de classificação de fontes tipográficas históricas brasileiras.

As instâncias que compõem o dataset representam recortes de imagens de caracteres tipográficos individuais impressos em papel. Cada instância possui um rótulo (label) associado, que a categoriza em uma de quatro famílias tipográficas: escritural, fantasia, grotesco ou serifado. O dataset não contém dados confidenciais ou informações que possam identificar indivíduos.

A adoção dos diretórios `raw`, `interim` e `processed` é inspirada no padrão *Cookiecutter Data Science*, seguindo uma lógica evolutiva das imagens:
- `raw` corresponde aos dados brutos, isto é, as amostras fotográficas originais das impressões tipográficas e dados intocados, exatamente como foram digitalizados e categorizados inicialmente. Não deve ser modificado, sobrescrito ou apagado.
- `interim` é a área de transição, em que localizam-se os dados que sofreram e/ou ainda sofrerão processamento computacional (filtragem, binarização e segmentação), antes da fase da validação humana. 
- `processed` diz respeito aos dados finais e padronizados para alimentar os modelos de aprendizado de máquina na etapa seguinte. Consiste nos recortes individuais de caracteres, devidamente rotulados e validados por humanos como ideais para a tarefa de treinamento dos algoritmos. Um recorte é considerado "ideal" e, portanto, pertence a `processed` se (a) continha um caractere íntegro e legível, sem invasões de outros; (b) apresentava pequenas invasões de outros, mas que não geravam ambiguidade em seu reconhecimento; ou (c) se encaixava em algum dos outros dois casos, mesmo exibindo borrões ou leve desfoque. 

Os dados brutos foram adquiridos diretamente de fotografias de impressões sobre papel de tipos metálicos provenientes de gavetas do Museu Paulista.

Os recortes processados e validados estão distribuídos publicamente na organização de pastas citada acima. O repositório oficial que abriga e documenta o dataset pode ser acessado em Tipografia Museu Paulista. Para garantir a reprodutibilidade dos experimentos de aprendizado de máquina, há divisões de dados previamente recomendadas via validação cruzada estratificada no arquivo `typography_dataset_splits.csv`.

---

## Execução

### 0. Configurando o ambiente virtual no Linux

No projetos, são utilizados o ambiente virtual **Venv** e **Python 3.12.3**. Instale e ative o ambiente:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale todos os pacotes Python utilizados para a implementação:
```bash
python -m pip install -r requirements.txt
```

### 1. Pré-processamento e Segmentação das Imagens

Execute a partir da raiz do repositório. É necessário rodar os dois scripts, pois o dataset `raw` foi manualmente particionado em dois:

```bash
python3 scripts/preprocessing_and_segmentation/crop_data_1.py
python3 scripts/preprocessing_and_segmentation/crop_data_2.py
```

As imagens intermediárias, binarizadas e com fechamento morfológico, são salvas, respectivamente, em `data/interim/data_1/interim_data_1` após a execução do `crop_data_1.py` e `data/interim/data_2/interim_data_2` após `crop_data_2.py`. Os recortes após as segmentações, por sua vez, são salvos em `data/interim/data_1/processed_data_1` e `data/interim/data_2/processed_data_2`.

Para reunir os recortes dos dois conjuntos em uma única pasta (`data/interim/data_1_2_final`):

```bash
python3 scripts/preprocessing_and_segmentation/merge_processed.py
```

### 2. Validação Humana

**Esta etapa foi realizada manualmente e não deve ser reexecutada.**
Os arquivos em `human_evaluation/` são o registro da avaliação feita pelos voluntários e servem de base para o *dataset* final. Reexecutar ou modificar qualquer script ou arquivo desta pasta pode sobrescrever as anotações e comprometer a reprodutibilidade dos resultados.

Os recortes obtidos na segmentação (`data/interim/data_1_2_final`) foram divididos em três *batches* (`batch_1`, `batch_2` e `batch_3`). A avaliação foi feita por nove voluntários, de modo que **cada batch foi analisado por três voluntários diferentes**. As anotações foram registradas por meio de uma aplicação web (`web_annotation_app.py`).

O conteúdo da pasta `human_evaluation/` está organizado assim:

| Arquivo / pasta | Descrição |
|---|---|
| `batch_1/`, `batch_2/`, `batch_3/` | Recortes apresentados aos voluntários em cada batch |
| `annotations_batch_N.csv` | Anotações brutas dos voluntários no batch N |
| `stats_annotations_batch_N.csv` | Estatísticas das anotações do batch N |
| `selected_batch_N.csv` | Recortes selecionados do batch N após a validação |
| `statistics_per_drawer.ipynb` | Análise estatística por gaveta |
| `select_images.py`, `copy_images_final.py` | Seleção dos recortes validados e cópia para o *dataset* final |
| `web_annotation_app.py` | Aplicação web usada na anotação |

Os arquivos `selected_batch_N.csv` e as pastas de batch já estão disponíveis no repositório, de modo que as etapas seguintes (treinamento e avaliação) podem ser executadas diretamente a partir do *dataset* final em `data/processed`.
### 3. Particionamento em folds (validação cruzada)

**O CSV com a alocação das imagens em folds já está disponível em `data/k-fold_split/typography_dataset_splits.csv` e é o utilizado em todos os experimentos. Reexecutar o script o sobrescreve.**

Para avaliar os modelos por validação cruzada, as imagens de `data/processed` são particionadas em 5 folds por meio de `StratifiedGroupKFold` (scikit-learn):

- **Agrupamento por gaveta:** todas as imagens de uma mesma gaveta ficam no mesmo fold. Como caracteres de uma mesma gaveta são muito parecidos entre si, separá-los entre treino e teste causaria vazamento de dados e inflaria as métricas.
- **Estratificação por classe:** o algoritmo busca manter nos folds uma proporção de imagens por classe próxima à do conjunto completo.
- **Mínimo por classe:** o `StratifiedGroupKFold` é uma heurística e não garante, sozinho, que toda classe apareça em todos os folds. Por isso são testadas até 500 sementes (a partir de 42); descartam-se as divisões em que alguma classe fica sem imagens em algum fold, e escolhe-se a mais balanceada entre as válidas.

O script verifica ainda que nenhuma gaveta aparece em mais de um fold.

Para gerar o CSV (a partir da raiz do repositório):

```bash
python3 scripts/k-fold_generator/dataset_split_kfold.py
```

O arquivo gerado tem uma linha por imagem e as colunas `image_path`, `label`
(família tipográfica), `drawer_id` (gaveta, no formato `<classe>_<código>`) e `fold` (0 a 4).

### 4. Treinamento e Avaliação dos Modelos 
São avaliados seis configurações experimentais na tarefa de classificação de caracteres. Todos são treinados, validados e avaliados automaticamente utilizando o script `model_comparison.py`

Para executá-lo:
```bash
python3 scripts/model_comparison/model_comparison.py
```

Todas as métricas quantitativas obtidas (Acurácia, Macro F1-score, tempo de inferência, quantidade de imagens-suporte), obtidas em cada *fold* e para cada modelo estão salvas na forma de CSVs no diretório `results`, além das médias de métricas considerando todos os *folds*. Adicionalmente, há as matrizes de confusão das classes, salvas como PDFs no mesmo diretório.

### 5. Treinamento do ResNet-18 (melhor Macro-F1 score) e Modelo de Predição

Resultados obtidos utilizando as mesmas configurações anteriores devem levar o ResNet-18 (sem augmentation) à melhor acurácia e Macro-F1 score médios dentre todos os modelos testados. Assim, será treinado com *hold-out* simples (particionamento do dataset em treinamento e validação - 85% e teste - 15%). **Uma gaveta está inteiramente contida em um conjunto**, evitando vazamento de dados (estilística) do teste para o treinamento.

Execução:
```bash 
python3 scripts/final_training/resnet18_final_training.py
```

As saídas geradas por esse script são:
- `weights/resnet18_final.pth`: pesos da rede neural após o treinamento.
- `weights/resnet18_final_meta.json`: ordem das classes, pré-processamento, métricas de teste.
- `results/final_test_*`: métricas, matriz de confusão, predições por imagem, split usado.

Por fim, os pesos exportados podem ser utilizados em um modelo de predição. Para utilizá-lo com os pesos padrão (`weights/resnet18_final_meta.json`), basta executar:

```bash 
python3 scripts/inference_models/resnet18_inference.py <nome_da_imagem [.png, .jpg, .jpeg]> 
```

Alternativamente, caso queira mudar o arquivo de pesos:
```bash 
python3 scripts/inference_models/resnet18_inference.py <nome_da_imagem [.png, .jpg, .jpeg]> --weights <caminho_pesos.pth>
```