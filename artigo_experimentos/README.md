# Organização dos Experimentos Para o Artigo

Esta pasta separa os experimentos em quatro grupos:

1. `01_usar_no_artigo`
2. `02_ablações_e_apoio`
3. `03_exploratorio_nao_usar_como_principal`
4. `04_llm_parcial_trabalho_futuro`

Os arquivos originais em `src/` e `data/` nao foram removidos. Esta pasta e uma
curadoria para escrita do artigo.

## 01_usar_no_artigo

Contem os experimentos que sustentam a narrativa principal.

### baseline

Usa apenas meta-features estatisticas.

Resultado controlado:

- Accuracy: 65.54%
- F1-Macro: 39.88%

Uso no artigo: ponto de partida. Mostra o desempenho sem semantica e sem dominio.

### primeira_evidencia_teste1

Teste 1 original: estatisticas + texto semantico + dominio.

Resultado:

- Accuracy: 68.9%
- F1-Macro: 43.8%

Uso no artigo: evidencia inicial de que contexto semantico melhora o resultado.
Nao deve ser o modelo final, porque depois foi identificado risco de vazamento
textual.

### analise_vazamento_teste20

Reexecuta a ideia do Teste 1 com limpeza contra vazamento textual.

Remove:

- nome do dataset;
- URLs;
- numeros;
- tokens com `_`;
- listas de atributos;
- IDs/probes;
- termos de fonte e repositorio.

Uso no artigo: mostra que parte do ganho do texto bruto vinha de identificadores
fortes, fortalecendo a validade metodologica do trabalho.

### modelo_proposto_top2_clusters

Modelo recomendado como principal.

Arquitetura:

```text
meta-features estatisticas
+ texto limpo
+ TF-IDF automatico
+ KMeans
+ cluster semantico
+ dominio
+ domain_score
+ interacao dominio/score
+ Random Forest
```

Resultado:

- Accuracy: 72.43%
- F1-Macro: 45.79%

Por que usar como modelo proposto:

- nao usa vocabulario manual;
- usa texto limpo;
- nao entrega palavras diretamente ao Random Forest;
- usa o texto para gerar grupos semanticos;
- preserva desempenho muito proximo ao melhor resultado absoluto.

### melhor_resultado_absoluto_top1

Melhor resultado numerico.

Arquitetura:

```text
meta-features estatisticas
+ tags semanticas
+ vocabulario fixo
+ dominio
+ domain_score
+ Random Forest
```

Resultado:

- Accuracy: 72.43%
- F1-Macro: 46.43%

Uso no artigo: melhor resultado absoluto. Deve ser apresentado com a observacao
de que usa vocabulario fixo manual, portanto e menos automatico que o Top 2.

### top10_controlado

Contem a tabela Top 10 consolidada e os scripts isolados.

Arquivo principal:

- `top10_plus_baseline_summary.csv`

Uso no artigo: tabela final de resultados.

## 02_ablações_e_apoio

Experimentos que explicam por que o modelo proposto funciona.

### dominio_teste22

Mostra o impacto de `predicted_domain` e `domain_score`.

Uso no artigo: ablation study.

Resultado importante:

```text
Estatistica pura: 65.54% / 39.88%
Estatistica + dominio + score: 69.93% / 44.04%
Estatistica + semantica + dominio + score: 72.43% / 46.43%
```

### busca_clusters_teste23

Busca por configuracoes de cluster.

Uso no artigo: robustez do modelo por clusters.

Conclusao: configuracoes simples, como 4 clusters e 60 features TF-IDF, empatam
o melhor resultado de cluster.

### score_tfidf_teste25

Tenta substituir o `domain_score` original por um score automatico derivado de
TF-IDF.

Uso no artigo: apoio opcional. Mostra que esse score automatico simples nao
substituiu bem o `domain_score` original.

## 03_exploratorio_nao_usar_como_principal

Experimentos que ajudam a entender o problema, mas nao devem ser centrais.

### texto_numerico_teste19

Transforma texto em estatisticas numericas agregadas. Resultado abaixo dos
metodos com clusters/tags.

### cluster_sem_score_teste24

Testa cluster semantico + dominio sem `domain_score`.

Resultado:

- Accuracy: 64.67%
- F1-Macro: 38.60%

Conclusao: nessa arquitetura, dominio categorico sem score nao foi suficiente.

### testes_antigos_invalidos

Reservado para tentativas antigas com maior risco metodologico, como texto
dinamico bruto, excesso de termos e mudancas simultaneas de modelo/meta-modelo.
Caso esses arquivos nao estejam copiados aqui, mantenha-os apenas como registro
historico nos resultados originais.

## 04_llm_parcial_trabalho_futuro

Contem os scripts de features por LLM.

Nao usar como resultado do artigo atual porque:

- a geracao online ficou parcial;
- houve limite de taxa/cota;
- apenas parte dos datasets recebeu features LLM reais.

Uso recomendado: trabalho futuro.

## Recomendacao Final Para a Narrativa

1. Comece com a baseline estatistica.
2. Mostre o Teste 1 como evidencia inicial.
3. Mostre o Teste 20 para discutir vazamento.
4. Apresente a semantica controlada.
5. Defenda o Top 2 como modelo proposto.
6. Mostre o Top 1 como melhor resultado absoluto.
7. Use Teste 22 e Teste 23 como ablacões.
8. Deixe LLM e testes antigos como exploratorios/trabalho futuro.
