"""Auditoria do domain_score: ele e 'dominio' ou proxy de nr_attr / tamanho de texto?"""
import numpy as np
import pandas as pd
from common import INPUT_METAFEATURES

m = pd.read_csv(INPUT_METAFEATURES)
m["text_len"] = m["semantic_text"].astype(str).str.split().apply(len)

print(f"Datasets: {len(m)}")
print(f"domain_score: min={m.domain_score.min():.0f} max={m.domain_score.max():.0f} "
      f"mediana={m.domain_score.median():.0f} media={m.domain_score.mean():.1f}\n")

# correlacao do domain_score com features estruturais / tamanho de texto
cands = ["nr_attr", "nr_inst", "nr_class", "text_len", "attr_to_inst",
         "class_entropy", "minority_class_ratio"]
print("Correlacao de Spearman do domain_score com:")
for c in cands:
    if c in m.columns:
        r = m[["domain_score", c]].corr(method="spearman").iloc[0, 1]
        print(f"  {c:22s} rho = {r:+.3f}")

# correlacao com TODAS as meta-features estatisticas numericas - top 8 absolutas
num = m.select_dtypes(include="number").drop(columns=["did"], errors="ignore")
corr = num.corr(method="spearman")["domain_score"].drop("domain_score").dropna()
top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
print("\nTop 10 meta-features estatisticas mais correlacionadas com domain_score:")
for k, v in top.items():
    print(f"  {k:28s} rho = {v:+.3f}")

# distribuicao do domain_score por dominio (sera que e so 'biologia tem score alto'?)
print("\ndomain_score por predicted_domain (mediana):")
print(m.groupby("predicted_domain")["domain_score"].median().sort_values(ascending=False).to_string())
