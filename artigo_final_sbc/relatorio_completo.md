# Relatorio Tecnico Completo do Projeto

## 1. Objetivo do projeto

O objetivo central do projeto foi investigar se informacoes semanticas sobre os
datasets, quando combinadas com meta-features estatisticas tradicionais,
melhoram a recomendacao de algoritmos de classificacao em um cenario de
meta-aprendizagem.

O problema foi formulado como uma tarefa de meta-classificacao:

- cada instancia do meta-dataset representa um dataset do OpenML;
- as features representam caracteristicas do dataset;
- o alvo e o algoritmo que obteve melhor desempenho naquele dataset;
- os algoritmos candidatos considerados na fase principal foram:
  `DecisionTree`, `LogisticRegression` e `Perceptron`.

A hipotese investigada foi:

> Meta-features semanticas e de dominio complementam as meta-features
> estatisticas e podem melhorar a escolha do algoritmo mais adequado.

## 2. Dados utilizados

O projeto trabalhou com 116 datasets selecionados. Para cada dataset, foram
mantidos:

- meta-features estatisticas e estruturais;
- texto semantico originalmente associado ao dataset;
- dominio previsto do dataset;
- score de dominio;
- matriz de desempenho dos algoritmos candidatos.

Os principais arquivos de dados sao:

- `data/metafeatures_selected_datasets.csv`
- `data/performance_matrix.csv`
- `data/final_domain_selection.csv`

O alvo foi recalculado a partir da matriz de desempenho:

```text
best_classifier = argmax(DecisionTree, LogisticRegression, Perceptron)
```

A distribuicao do alvo foi:

- `LogisticRegression`: 75 datasets
- `DecisionTree`: 33 datasets
- `Perceptron`: 8 datasets

Essa distribuicao desbalanceada justifica o uso de F1-Macro como metrica
principal, pois a acuracia tende a favorecer a classe majoritaria.

## 3. Baseline estatistica

O primeiro experimento relevante foi a baseline com apenas meta-features
estatisticas.

Configuracao:

- features: 63 meta-features estatisticas/estruturais;
- sem texto semantico;
- sem dominio;
- sem `domain_score`;
- meta-modelo: `RandomForestClassifier`;
- validacao: `StratifiedKFold`, ate 5 folds.

Resultado controlado:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Baseline estatistica | 65.54% | 39.88% |

Essa baseline define o ponto de comparacao do trabalho.

## 4. Primeira evidencia: texto bruto + dominio

O Teste 1 original combinou:

- meta-features estatisticas;
- TF-IDF do texto semantico bruto;
- dominio previsto via One-Hot-Encoding;
- Random Forest.

Resultado:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Estatistica + texto bruto + dominio | 68.95% | 43.87% |

Esse resultado foi importante porque mostrou ganho claro sobre a baseline.
Entretanto, ao inspecionar os textos semanticos, foi identificado que muitos
continham termos que poderiam funcionar como identificadores fortes de datasets.

Exemplos de identificadores fortes:

- nomes de datasets;
- nomes de repositorios;
- URLs;
- listas de atributos;
- IDs de genes/probes;
- tokens como `AFFX`, `id_ref`, `200001_at`;
- nomes de familias especificas de datasets.

Assim, o ganho inicial poderia estar parcialmente associado a vazamento textual
ou memorizacao de origem/familia, e nao apenas a semantica generalizavel.

## 5. Analise de vazamento textual

Para investigar esse risco, foi criado o Teste 20, que reexecutou a ideia do
Teste 1 com limpeza anti-vazamento.

A limpeza removeu:

- nome do dataset;
- URLs;
- numeros;
- tokens com `_`;
- tokens com digitos;
- listas de atributos;
- secoes iniciadas por marcadores como `attribute information` e `id_ref`;
- termos de fonte/repositirio como `uci`, `openml`, `kaggle`, `gemler`;
- boilerplate como `author`, `source`, `dataset`, `classification`.

Resultados:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Texto limpo sem dominio | 66.41% | 40.35% |
| Texto limpo com dominio | 66.45% | 40.18% |

Esse resultado mostrou que o desempenho caiu quando removemos identificadores
fortes. A conclusao metodologica foi que o texto bruto nao deveria ser usado
diretamente como resultado final do artigo.

## 6. Transicao para semantica controlada

A partir dessa analise, o projeto passou a buscar formas mais controladas de
usar a semantica:

1. texto limpo com TF-IDF limitado;
2. tags semanticas interpretaveis;
3. clusters semanticos;
4. dominio previsto;
5. score de dominio;
6. interacoes entre dominio, score e semantica.

Como decisao final para a escrita do artigo, foram descartados os experimentos
baseados em vocabulario fixo manual. Eles apresentaram bons resultados, mas
dependiam de uma lista curada de palavras, o que poderia gerar questionamentos
sobre generalizacao e intervencao humana.

## 7. Modelo proposto sem vocabulario fixo

O modelo recomendado para o artigo e baseado em clusters semanticos automaticos.

Arquitetura:

```text
meta-features estatisticas
+ texto semantico limpo
+ TF-IDF automatico
+ KMeans
+ cluster semantico
+ predicted_domain
+ domain_score
+ interacao dominio/score
+ Random Forest
```

Etapas:

1. O texto semantico e limpo para remover identificadores fortes.
2. O texto limpo e transformado por TF-IDF automatico.
3. Os vetores TF-IDF sao agrupados por KMeans.
4. O cluster resultante e usado como feature categorica.
5. O dominio previsto e o score de dominio sao adicionados.
6. Interacoes entre dominio, score e semantica sao fornecidas ao meta-modelo.
7. Um Random Forest aprende a prever o melhor algoritmo.

Resultado:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Modelo proposto por clusters | 72.43% | 45.79% |

Esse resultado supera a baseline por:

- +6.89 pontos percentuais em acuracia;
- +5.91 pontos percentuais em F1-Macro.

## 8. Por que o modelo por clusters foi escolhido

O modelo por clusters foi escolhido como principal por tres razoes.

Primeiro, ele nao usa vocabulario manual. O proprio TF-IDF extrai termos do
texto limpo, e o KMeans transforma essa informacao em grupos semanticos.

Segundo, ele nao entrega palavras diretamente ao Random Forest. O texto e usado
para formar clusters, e o modelo final recebe o grupo semantico, reduzindo a
dependencia de tokens especificos.

Terceiro, ele obteve desempenho competitivo com os melhores resultados do
projeto, preservando maior automatizacao e menor intervencao humana.

## 9. Papel do dominio e do score

O Teste 22 investigou explicitamente o impacto de dominio e `domain_score`.

Resultados principais:

| Configuracao | Accuracy | F1-Macro |
|---|---:|---:|
| Estatistica pura | 65.54% | 39.88% |
| Estatistica + dominio | 67.25% | 41.40% |
| Estatistica + score | 69.89% | 43.37% |
| Estatistica + dominio + score | 69.93% | 44.04% |
| Estatistica + semantica controlada | 68.08% | 42.32% |
| Estatistica + semantica + score | 69.86% | 44.15% |

Esses resultados mostram que:

- o dominio categorico ajuda, mas pouco isoladamente;
- o `domain_score` e um sinal forte;
- semantica e score se complementam;
- remover o score em algumas arquiteturas derruba substancialmente o resultado.

Um teste adicional com cluster + dominio sem score obteve:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Cluster + dominio sem score | 64.67% | 38.60% |

Isso reforca que, nessa familia de modelos, o score e mais informativo que o
rotulo de dominio isolado.

## 10. Busca por configuracoes de cluster

O Teste 23 variou:

- numero de clusters;
- numero maximo de features TF-IDF;
- uso de unigramas ou bigramas;
- modo de dominio/score.

Melhor configuracao:

| Clusters | TF-IDF features | N-gramas | Dominio | Accuracy | F1-Macro |
|---:|---:|---|---|---:|---:|
| 4 | 60 | unigramas | label + score + interacao | 72.43% | 45.79% |

Essa busca mostrou que a configuracao por clusters e robusta: uma versao simples
com 4 clusters e 60 termos empatou o melhor desempenho da familia.

## 11. Tentativas que nao foram mantidas como resultado principal

### Texto numerico agregado

Foi testada a conversao do texto para estatisticas numericas, como tamanho,
diversidade lexical, proporcao de tokens com digitos, entropia e densidade.

Resultado maximo:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Texto numerico estendido sem dominio | 64.60% | 40.49% |

Essa abordagem nao capturou semantica suficiente para competir com clusters.

### Score automatico via TF-IDF

Foi tentado substituir o `domain_score` original por um score calculado a partir
de TF-IDF limpo e similaridade com centroides de dominio.

Resultado:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Cluster + score TF-IDF automatico | 66.34% | 41.80% |

O score automatico nao substituiu bem o `domain_score` original.

### Tags adicionadas aos clusters

Foram testadas combinacoes de clusters com tags semanticas.

Resultados:

| Modelo | Accuracy | F1-Macro |
|---|---:|---:|
| Clusters + tags + interacao dominio/score | 68.99% | 40.76% |
| Clusters + tags + dominio + score | 69.86% | 42.83% |
| Clusters + tags + dominio sem score | 65.47% | 40.54% |

As tags nao complementaram bem os clusters nessa familia. A interpretacao e que
elas introduziram redundancia ou ruido.

### LLM

Foi criada uma tentativa com LLM para extrair features estruturadas, mas a
geracao online ficou parcial devido a limite de taxa. Portanto, essa linha foi
separada como trabalho futuro.

## 12. Resultados principais sem vocabulario fixo

| Experimento | Accuracy | F1-Macro | Papel no artigo |
|---|---:|---:|---|
| Baseline estatistica | 65.54% | 39.88% | Controle principal |
| Teste 1 bruto | 68.95% | 43.87% | Evidencia inicial |
| Teste 20 limpo | 66.45% | 40.18% | Analise de vazamento |
| Dominio + score | 69.93% | 44.04% | Ablacao de dominio |
| Modelo proposto por clusters | 72.43% | 45.79% | Resultado principal |

## 13. Conclusao tecnica

O projeto mostrou que a inclusao de informacao semantica pode melhorar a
meta-aprendizagem para selecao de algoritmos, mas tambem demonstrou que o uso
direto de texto bruto traz risco de vazamento.

A abordagem mais defensavel foi usar texto limpo apenas para construir clusters
semanticos automaticos. Esses clusters, combinados com dominio e score,
produziram o melhor equilibrio entre desempenho, automatizacao e controle
metodologico.

Assim, para o artigo, o metodo proposto deve ser:

```text
meta-features estatisticas
+ clusters semanticos automaticos
+ dominio
+ score de dominio
+ interacao dominio/score
```

Os experimentos com vocabulario fixo devem ficar fora da narrativa principal,
pois dependem de curadoria manual. Eles podem ser mantidos como historico do
projeto, mas nao devem sustentar a contribuicao central.

