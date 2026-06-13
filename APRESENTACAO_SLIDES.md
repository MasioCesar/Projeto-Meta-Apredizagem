# Conteudo Sugerido Para Slides de Apresentacao

## Slide 1 - Titulo

**Informacao de Dominio na Meta-Aprendizagem para Selecao de Algoritmos**

Subtitulo:

**Quando o contexto do dataset ajuda a escolher algoritmos de classificacao?**

Conteudo:

- Projeto de meta-aprendizagem usando datasets do OpenML.
- Objetivo: prever qual algoritmo tende a funcionar melhor em um novo dataset.
- Foco final: verificar se o dominio do dataset adiciona sinal alem das meta-features estatisticas.

---

## Slide 2 - Problema

**Escolher algoritmo ainda e uma decisao dificil**

Conteudo:

- Nao existe um algoritmo universalmente melhor para todos os datasets.
- A meta-aprendizagem tenta aprender padroes entre caracteristicas do dataset e desempenho dos algoritmos.
- Abordagem tradicional: usar meta-features estatisticas e estruturais.
- Pergunta do trabalho: informacoes de dominio tambem ajudam?

Mensagem-chave:

> Se dois datasets possuem caracteristicas estatisticas parecidas, mas pertencem a dominios diferentes, o melhor algoritmo pode mudar.

---

## Slide 3 - Hipotese

**Hipotese central**

Conteudo:

> O dominio de aplicacao de um dataset carrega sinal preditivo sobre quais algoritmos tendem a apresentar melhor desempenho.

Exemplos de intuicao:

- Dados financeiros: frequentemente tabulares e quase-lineares.
- Dados de imagem: podem favorecer modelos capazes de lidar com alta dimensionalidade.
- Dados biologicos: podem ter muitas variaveis e poucos exemplos.

Ponto importante:

- A hipotese nao e que "qualquer texto ajuda".
- A hipotese final e que **dominio bem codificado pode funcionar como um prior sobre o ranking dos algoritmos**.

---

## Slide 4 - Formulacao Como Meta-Aprendizagem

**Cada dataset vira uma instancia do meta-dataset**

Conteudo:

- Entrada: meta-features do dataset.
- Saida tradicional: melhor algoritmo observado.
- Algoritmos candidatos:
  - DecisionTree
  - LogisticRegression
  - Perceptron
  - SVM
  - KNN
  - MLP

Representacao:

```text
Dataset -> meta-features estatisticas + informacao de dominio -> algoritmo recomendado
```

---

## Slide 5 - Evolucao da Base de Dados

**De uma base inicial limitada para uma base V2 mais robusta**

Conteudo:

| Etapa | Datasets |
|---|---:|
| OpenML bruto | 6.408 |
| Filtro tecnico | 1.491 |
| Deduplicacao por familia/tamanho | 796 |
| Base final V2 | 547 |

Filtros principais:

- 200 a 50.000 instancias.
- Pelo menos 2 atributos e 2 classes.
- Menos de 30% de valores ausentes.
- Classe minoritaria com pelo menos 20 exemplos.

Mensagem-chave:

> A base final reduz vies de selecao e evita repetir datasets muito parecidos.

---

## Slide 6 - Dominios Considerados

**Atribuicao de dominio por nome e descricao**

Dominios:

- health
- finance
- biology
- image
- text
- sensor_signal
- education
- social
- other

Observacao:

- A categoria `other` inclui muitos datasets sinteticos, abstratos ou sem dominio tematico claro.
- Isso se tornou importante: dominio ajuda mais quando a categoria representa um contexto real e coerente.

---

## Slide 7 - Primeiros Resultados e Risco de Artefatos

**Os ganhos iniciais eram grandes, mas suspeitos**

Conteudo:

- A base inicial tinha 116 datasets e 3 algoritmos.
- Texto bruto e dominio pareciam melhorar bastante o desempenho.
- Investigacao posterior mostrou riscos:
  - vazamento textual;
  - semente unica;
  - datasets parecidos em treino e teste;
  - `domain_score` confundido com dimensionalidade;
  - troca de modelo em apenas um lado da comparacao.

Mensagem-chave:

> Parte dos ganhos iniciais vinha de artefatos, nao necessariamente de semantica generalizavel.

---

## Slide 8 - Protocolo Rigoroso Final

**Como o resultado final foi validado**

Conteudo:

- Base V2 com 547 datasets.
- 6 algoritmos candidatos.
- Random Forest como meta-modelo.
- 30 sementes.
- Validacao cruzada agrupada por similaridade de colunas.
- Testes pareados.
- Comparacao sempre contra baseline estatistica.
- O dominio e sempre adicionado; as meta-features estatisticas nao sao substituidas.

Mensagem-chave:

> O protocolo foi desenhado para separar sinal real de vazamento ou sorte experimental.

---

## Slide 9 - Problema do Alvo: Melhor Algoritmo e Ruidoso

**O melhor algoritmo nem sempre e claramente melhor**

Conteudo:

- A margem entre o melhor e o segundo melhor algoritmo e pequena.
- Em reavaliacoes repetidas, parte dos rotulos de "melhor algoritmo" muda.
- Isso limita o ganho possivel de qualquer meta-feature.

Mensagem-chave:

> O efeito esperado do dominio nao deveria ser enorme; um ganho pequeno e robusto ja e relevante.

---

## Slide 10 - O Que Nao Funcionou

**Nem toda informacao contextual ajudou**

Abordagens testadas sem ganho robusto:

- Texto bruto.
- Texto limpo com TF-IDF.
- Tags semanticas.
- Vocabulos fixos.
- Embeddings densos.
- PCA de embeddings.
- Clusters semanticos.
- One-hot de dominio.
- `domain_score` bruto.
- Anotacoes por LLM.
- Balanceamento e reagrupamento de classes.

Mensagem-chave:

> O resultado final nao surgiu de escolher o melhor entre muitos testes frageis; ele apareceu depois de descartar varios caminhos que nao resistiram aos controles.

---

## Slide 11 - Ideia Final: Target Encoding de Dominio

**O dominio precisa virar sinal supervisionado**

Conteudo:

Em vez de representar o dataset apenas como:

```text
dominio = finance
```

o metodo representa o dominio como:

```text
em finance, quais algoritmos costumam ranquear melhor?
```

Exemplo conceitual:

| Algoritmo | Rank medio em finance |
|---|---:|
| LogisticRegression | 5.8 |
| SVM | 4.9 |
| MLP | 4.0 |
| DecisionTree | 3.2 |
| KNN | 2.1 |
| Perceptron | 1.0 |

Mensagem-chave:

> O dominio passa a carregar um historico de preferencia algoritimica.

---

## Slide 12 - Como Evitamos Vazamento

**O encoding e calculado apenas no treino**

Conteudo:

Para cada fold:

1. Separa treino e teste.
2. Calcula o rank medio dos algoritmos por dominio usando apenas o treino.
3. Aplica esse encoding nos datasets de treino e teste.
4. Treina o meta-modelo.
5. Avalia no teste.

Mensagem-chave:

> O dataset avaliado nunca participa do calculo do seu proprio encoding.

---

## Slide 13 - Resultado Principal

**Dominio melhora a selecao top-1**

Conteudo:

| Metrica | Baseline | + Dominio rank | Ganho |
|---|---:|---:|---:|
| Acuracia | 40,60% | 41,79% | +1,19 p.p. |
| F1-macro | 29,70% | 30,38% | +0,69 p.p. |
| Top-3 accuracy | 81,22% | 81,26% | ~0 |

Interpretacao:

- O ganho aparece nas metricas top-1.
- Top-3 esta saturado, entao quase nao diferencia os metodos.

Mensagem-chave:

> O efeito e pequeno, mas consistente.

---

## Slide 14 - Resultado em Dominio Real

**O efeito fica maior quando o dominio realmente existe**

Conteudo:

| Grupo | Baseline | + Dominio rank | Ganho |
|---|---:|---:|---:|
| Todos os datasets | 40,60% | 41,79% | +1,19 p.p. |
| Dominio real | 42,74% | 44,66% | +1,92 p.p. |
| Finance | 55,50% | 62,67% | +7,17 p.p. |

Mensagem-chave:

> O dominio ajuda principalmente quando ele representa uma estrutura real, nao uma categoria generica.

---

## Slide 15 - Por Que Finance Melhora Tanto?

**Dominio ajuda quando um algoritmo domina aquele contexto**

Conteudo:

- Em finance, LogisticRegression vence em muitos datasets.
- Isso e coerente com problemas financeiros tabulares, como credito, risco e fraude.
- O encoding por rank captura esse prior historico.
- Quando um dominio tem padrao forte, o prior ajuda.

Contraste:

- Em `other`, os datasets sao muito heterogeneos.
- Nenhum algoritmo domina claramente.
- O prior de dominio pode virar ruido.

Mensagem-chave:

> O dominio e util quando a categoria tem coerencia interna.

---

## Slide 16 - Teste de Permutacao do Dominio

**Controle negativo: e o dominio real ou apenas mais features?**

Comparacao:

- Baseline estatistica.
- Dominio real com rank encoding.
- Dominio embaralhado com o mesmo encoding.

Resultados:

| Metrica | Ganho dominio real | Ganho dominio embaralhado |
|---|---:|---:|
| Acuracia | +1,19 p.p. | +0,26 p.p. |
| F1-macro | +0,54 p.p. | -0,04 p.p. |
| Mean regret | reduz 0,084 p.p. | quase zero |

Mensagem-chave:

> O ganho depende da associacao real entre dataset, dominio e desempenho dos algoritmos.

---

## Slide 17 - Regret

**Nem todo erro tem o mesmo custo**

Definicao:

```text
regret = desempenho do algoritmo otimo - desempenho do algoritmo escolhido
```

Interpretacao:

- Regret baixo significa que, mesmo errando o melhor algoritmo, o modelo escolheu uma alternativa proxima.
- Essa metrica e importante porque os algoritmos frequentemente tem desempenhos muito proximos.

Resultado:

- O meta-learner reduz regret em relacao ao Single Best Algorithm.
- O dominio reduz levemente o regret global.

Mensagem-chave:

> A avaliacao por regret mostra utilidade pratica, nao apenas acerto nominal do melhor algoritmo.

---

## Slide 18 - Regret Por Dominio

**O ganho de regret tambem e concentrado**

Resultados principais:

| Dominio | Regret baseline | Regret + dominio | Reducao |
|---|---:|---:|---:|
| Finance | 2,84 p.p. | 2,04 p.p. | -28,18% |
| Image | 2,45 p.p. | 2,09 p.p. | -14,86% |
| Biology | 2,12 p.p. | 2,05 p.p. | -3,42% |
| Other | 2,89 p.p. | 3,01 p.p. | piora |

Mensagem-chave:

> O dominio melhora regret onde ha padrao; em categorias heterogeneas, pode atrapalhar.

---

## Slide 19 - Novo Teste: Predicao de Ranking Completo

**Mudando a pergunta: nao apenas o melhor, mas a ordem dos algoritmos**

Motivacao:

- O top-1 pode ser ruidoso.
- Ranking completo avalia se o modelo entende melhor a ordem geral dos algoritmos.
- Foi usada uma metrica ponderada de correlacao de ranking.

Formula usada:

```text
r_w(R_pred, R_real) =
1 - 6 * soma_i ((R_pred_i - R_i)^2 * ((n - R_pred_i + 1) + (n - R_i + 1)))
    / (n^4 + n^3 - n^2 - n)
```

Caracteristica:

- Erros nas primeiras posicoes pesam mais do que erros no fim do ranking.

---

## Slide 20 - Resultado do Ranking Completo

**Dominio nao melhorou a ordenacao completa**

Matriz padrao, 387 datasets completos:

| Metrica | Baseline | + Dominio | Delta |
|---|---:|---:|---:|
| Weighted rank corr | 0,4733 | 0,4699 | -0,0034 |
| Spearman | 0,5032 | 0,4999 | -0,0033 |
| Top-1 hit | 0,3188 | 0,3204 | +0,0016 |
| Mean regret | 0,0209 | 0,0209 | ~0 |

Matriz denoised, 492 datasets completos:

| Metrica | Baseline | + Dominio | Delta |
|---|---:|---:|---:|
| Weighted rank corr | 0,4709 | 0,4687 | -0,0022 |

Mensagem-chave:

> O dominio ajuda mais a escolher o topo da lista do que a ordenar corretamente todos os algoritmos.

---

## Slide 21 - Interpretacao Geral Dos Resultados

**O que aprendemos**

Conteudo:

- O dominio contem sinal, mas esse sinal e limitado.
- O sinal aparece melhor quando usado como prior supervisionado.
- O efeito e concentrado em dominios coerentes.
- Ranking completo e mais dificil: o dominio nao parece ter granularidade suficiente para ordenar todos os algoritmos.
- `other` mostra uma limitacao importante: contexto mal definido pode virar ruido.

Mensagem-chave:

> A hipotese se confirma de forma precisa: o contexto importa quando e informativo e bem codificado.

---

## Slide 22 - Contribuicoes

**Principais contribuicoes do trabalho**

Conteudo:

1. Base V2 com 547 datasets, mais robusta que a versao inicial.
2. Protocolo metodologico contra vazamento e artefatos.
3. Evidencia de que codificacoes ingenuas de contexto nao bastam.
4. Resultado positivo com target encoding por rank medio.
5. Teste de permutacao mostrando que o efeito depende do dominio real.
6. Analise de regret por dominio mostrando onde o dominio ajuda.
7. Resultado negativo honesto para ranking completo.

Mensagem-chave:

> O trabalho contribui tanto pelo resultado positivo quanto pelo processo de validacao.

---

## Slide 23 - Ameacas a Validade

**Limites do estudo**

Conteudo:

- O ganho global e pequeno.
- A atribuicao de dominio usa regras por palavras-chave.
- `other` e heterogeneo.
- O metodo exige historico de datasets por dominio.
- O ranking completo usa apenas datasets com performance completa para os 6 algoritmos.
- Os algoritmos avaliados sao um portifolio limitado.

Mensagem-chave:

> A conclusao e valida dentro do escopo testado, mas nao deve ser generalizada como "todo contexto sempre ajuda".

---

## Slide 24 - Trabalho Futuro

**Proximos passos**

Conteudo:

- Melhorar anotacao de dominio com LLM ou curadoria manual controlada.
- Dividir `other` em subcategorias mais informativas.
- Testar outros portifolios de algoritmos.
- Investigar encoding por dominio + familia de algoritmo.
- Usar metricas de ranking parcial, como top-k ou NDCG, alem do ranking completo.
- Aplicar regra de confianca: usar dominio apenas quando o dominio tiver historico suficiente.

Mensagem-chave:

> O caminho mais promissor e tornar o dominio mais confiavel e seletivo.

---

## Slide 25 - Conclusao

**Resposta final**

Conteudo:

> Sim, o dominio ajuda na meta-aprendizagem para selecao de algoritmos, mas nao de qualquer forma.

Conclusoes:

- One-hot, texto cru, embeddings e clusters nao foram suficientes.
- Target encoding por rank medio revelou um sinal pequeno, robusto e interpretavel.
- O ganho e maior em dominios reais e coerentes, especialmente finance.
- O dominio melhora top-1 e regret em contextos especificos.
- O dominio nao melhorou a predicao do ranking completo dos 6 algoritmos.

Frase final:

> O contexto importa, mas apenas quando representa uma regularidade real entre o tipo de problema e o comportamento dos algoritmos.

---

## Slide 26 - Perguntas

**Perguntas?**

Sugestoes de apoio para respostas:

- Se perguntarem por que o ganho e pequeno:
  - porque a margem entre algoritmos e pequena e o alvo e ruidoso.

- Se perguntarem por que usar target encoding:
  - porque o dominio sozinho e apenas uma categoria; o rank encoding transforma dominio em historico de desempenho.

- Se perguntarem sobre vazamento:
  - o encoding e ajustado apenas no treino de cada fold e a CV e agrupada por similaridade de colunas.

- Se perguntarem por que ranking completo nao melhorou:
  - porque o dominio parece carregar sinal para o topo do ranking, mas nao informacao suficiente para ordenar todos os algoritmos.



