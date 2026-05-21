# Top 10 - Experimentos Controlados Sem Vazamento

Este README documenta os 10 melhores cenários controlados e a baseline estatística.
Os scripts ficam em `src/top10_controlled/` e os resultados isolados ficam em
`data/top10_controlled/`.

## Tabela Resumo

Ranking ordenado por F1-Macro. A baseline fica separada, pois não usa semântica
nem domínio.

| Rank | Arquivo | Estratégia | Acc. | F1-Macro |
|---:|---|---|---:|---:|
| Baseline | `baseline.py` | Estatísticas sem domínio e sem texto | 65.54% | 39.88% |
| 1 | `01_semantic_domain_label_score.py` | Estatísticas + tags semânticas + vocabulário fixo + domínio + score | 72.43% | 46.43% |
| 2 | `02_clusters_domain_interaction.py` | Estatísticas + clusters semânticos + interação domínio/score | 72.43% | 45.79% |
| 3 | `03_fixed_vocab_domain_interaction.py` | Estatísticas + vocabulário fixo + interação domínio/score | 71.56% | 44.59% |
| 4 | `04_clusters_domain_label_score.py` | Estatísticas + clusters semânticos + domínio + score | 69.86% | 44.44% |
| 5 | `05_semantic_domain_score.py` | Estatísticas + tags semânticas + vocabulário fixo + score | 69.86% | 44.15% |
| 6 | `06_tags_domain_label.py` | Estatísticas + tags semânticas + domínio | 69.86% | 44.11% |
| 7 | `07_tags_domain_interaction.py` | Estatísticas + tags semânticas + interação domínio/score | 69.86% | 44.09% |
| 8 | `08_domain_label_score.py` | Estatísticas + domínio + score, sem texto semântico | 69.93% | 44.04% |
| 9 | `09_tags_domain_score.py` | Estatísticas + tags semânticas + score | 70.69% | 44.00% |
| 10 | `10_tags_clusters_domain_score.py` | Estatísticas + tags + clusters semânticos + score | 70.69% | 43.94% |

CSV consolidado: `data/top10_controlled/top10_plus_baseline_summary.csv`.

## Baseline

Arquivo: `src/top10_controlled/baseline.py`

Usa somente as meta-features estatísticas/matemáticas. Remove explicitamente:

- `semantic_text`
- `predicted_domain`
- `domain_score`
- nome do dataset
- colunas de performance dos classificadores

A baseline serve como ponto de comparação limpo: ela mede o quanto o modelo
consegue prever o melhor classificador usando apenas características estruturais
do dataset.

## O Que Cada Família de Teste Usa

### Tags Semânticas

As tags são features interpretáveis criadas a partir do texto limpo. Exemplos:

- `tag_gene_expression`
- `tag_sequence_biology`
- `tag_medical_patient`
- `tag_chemical_qsar`
- `tag_text_documents`
- `tag_image_pixels`
- `tag_financial_business`
- `tag_sparse_high_dimensional`
- `tag_missing_values`
- `tag_categorical_data`
- `tag_numeric_data`

A lógica é transformar o texto em sinais de alto nível, sem deixar o modelo
memorizar palavras específicas de datasets.

### Vocabulário Fixo Seguro

O vocabulário fixo usa uma lista manual e pequena de termos gerais, por exemplo:

`gene`, `expression`, `medical`, `patient`, `image`, `pixel`, `text`,
`document`, `financial`, `sequence`, `molecule`, `sparse`, `tabular`.

Isso evita que o TF-IDF escolha automaticamente tokens suspeitos como nomes de
bases, IDs de atributos, URLs ou probes biológicos.

### Clusters Semânticos

Os clusters usam TF-IDF limpo com vocabulário limitado e depois aplicam
`KMeans(n_clusters=5)`. O cluster vira uma feature categórica por One-Hot.

A ideia é capturar grupos semânticos gerais, sem usar o alvo e sem permitir
tokens identificadores fortes.

### Domínio

Foram testadas três formas principais:

- `label`: `predicted_domain` via One-Hot-Encoding.
- `score`: `domain_score` convertido em features numéricas, incluindo valor bruto,
  `log1p(score)` e indicadores de confiança alta.
- `label_score`: combinação de domínio categórico e score.

Também existem variações com `label_score_interaction`, que criam interações
entre domínio, score e sinais semânticos.

## Como o Vazamento Foi Reduzido

Os scripts usam as seguintes proteções:

1. Remoção de colunas-alvo e de performance:
   `DecisionTree`, `LogisticRegression`, `Perceptron`, `SVM`, `KNN`, `MLP`,
   `best_classifier` e `best_accuracy` nunca entram como features.

2. Remoção do nome do dataset do texto:
   tokens vindos de `name` são bloqueados antes da extração semântica.

3. Remoção de identificadores fortes:
   URLs, tokens com números, tokens com `_`, listas de atributos, probes como
   `1552256_a_at`, IDs, nomes de repositório e termos de fonte são removidos.

4. Corte de trechos de atributos:
   quando aparecem marcadores como `attribute information`, `id_ref`,
   `feature names` ou `column names`, o texto é truncado antes dessa seção.

5. Sem seleção de palavras baseada no alvo:
   os melhores cenários usam tags manuais, vocabulário fixo ou clustering sem
   usar `y`. Não há `SelectKBest` ou escolha supervisionada de palavras.

6. Transformações dentro do pipeline:
   limpeza, TF-IDF, KMeans, One-Hot-Encoding, imputação e normalização rodam
   dentro do `Pipeline`/`ColumnTransformer`, então são ajustados apenas nos folds
   de treino da validação cruzada.

7. `domain_score` isolado:
   no `common.py`, `domain_score` foi removido das meta-features estatísticas.
   Ele só entra quando `domain_mode` explicitamente usa `score`,
   `label_score` ou `label_score_interaction`.

## Observação Sobre `domain_score`

`domain_score` não é o alvo do meta-modelo e não vem da matriz de performance
dos classificadores. Mesmo assim, ele é um sinal derivado do processo de
classificação de domínio. Por isso os scripts deixam claro quando ele é usado.

Se for necessário um cenário ainda mais conservador, priorize resultados sem
`score`, como:

- `06_tags_domain_label.py`
- baseline

## Como Reproduzir

Rodar a baseline:

```powershell
venv\Scripts\python.exe src\top10_controlled\baseline.py
```

Rodar o melhor cenário:

```powershell
venv\Scripts\python.exe src\top10_controlled\01_semantic_domain_label_score.py
```

Rodar todos:

```powershell
$scripts = @(
  'baseline.py',
  '01_semantic_domain_label_score.py',
  '02_clusters_domain_interaction.py',
  '03_fixed_vocab_domain_interaction.py',
  '04_clusters_domain_label_score.py',
  '05_semantic_domain_score.py',
  '06_tags_domain_label.py',
  '07_tags_domain_interaction.py',
  '08_domain_label_score.py',
  '09_tags_domain_score.py',
  '10_tags_clusters_domain_score.py'
)

foreach ($s in $scripts) {
  venv\Scripts\python.exe "src\top10_controlled\$s"
}
```

## Interpretação Principal

A melhor configuração controlada foi:

`Estatísticas + tags semânticas + vocabulário fixo + predicted_domain + domain_score`

Resultado:

- Accuracy: 72.43%
- F1-Macro: 46.43%

Comparada com a baseline:

- Baseline Accuracy: 65.54%
- Baseline F1-Macro: 39.88%

Isso sustenta a hipótese principal do projeto: sinais semânticos e de domínio,
quando tratados de forma controlada, melhoram a escolha do algoritmo em relação
ao uso exclusivo de meta-features estatísticas.
