# Checkpoint 5 — Machine Learning & Modelling / Statistical Computing with R & Python

Trabalho da FIAP (Tecnólogo em Inteligência Artificial — 2º Semestre), usando o **Wine
Dataset for Clustering** (178 vinhos, 13 variáveis físico-químicas, provenientes de 3
cultivares distintas de uva).

- Kaggle: https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering
- Fonte original: https://gist.github.com/tijptjik/9408623

## Integrantes

| Nome completo        | RM     |
|-----------------------|--------|
| Gustavo Lino           | 574157 |
| Lucas Lopes Arias      | 570875 |

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `Checkpoint5_GustavoLino_RM574157.ipynb` | Notebook Jupyter principal, com todo o código, comentários, gráficos e análises já executados. **Este é o entregável oficial do trabalho.** |
| `analise_wine.py` | Versão em script Python do mesmo conteúdo do notebook (gerada automaticamente a partir dele), para quem preferir rodar fora do Jupyter. |
| `wines.csv` | Dataset utilizado na análise (Wine Dataset for Clustering). |
| `README.md` | Este arquivo. |

## Estrutura do trabalho

### Parte 1 — Statistical Computing with R and Python
Análise exploratória das variáveis `Alcohol` e `Malic_Acid`:
- Tabela de distribuição de frequências e gráficos (histograma/boxplot)
- Medidas de tendência central, dispersão e separatrizes
- Análise probabilística (teste de normalidade e cálculos de probabilidade)

### Parte 2 — Machine Learning & Modelling
Clusterização dos vinhos com **K-means**:
- Pré-processamento e padronização dos dados
- Ajuste do modelo K-means
- Escolha do número de clusters (Método Elbow + Silhouette Score)
- Análise e interpretação dos grupos formados

## Como rodar

**Opção 1 — Google Colab (mais fácil, não precisa instalar nada):**
Abra o notebook `.ipynb` neste repositório pelo GitHub e clique em "Open in Colab"
(ou cole a URL do notebook em https://colab.research.google.com).

**Opção 2 — Localmente:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
jupyter notebook Checkpoint5_GustavoLino_RM574157.ipynb
```
