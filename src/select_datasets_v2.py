"""
FILTRO V2: aumentar a base de datasets MANTENDO qualidade (sem repetidos).

Melhorias sobre o 01_select_datasets.py:
  1. NAO exige palavra-chave de dominio no nome (removia datasets validos + criava
     vies de selecao). O dominio passa a ser atribuido DEPOIS, a todos.
  2. Deduplicacao robusta: por (familia-do-nome) + assinatura de tamanho
     (nr_inst, nr_attr, nr_class) -> elimina reuploads/versoes/subamostras.
  3. Mantem o filtro tecnico de qualidade.

Este script so faz LISTAGEM + FILTRO + DEDUP (sem avaliar algoritmos) para
reportar QUANTOS datasets de qualidade a base teria. Saida: data/selection_v2_preview.csv
"""
import re
import unicodedata
from pathlib import Path

import openml
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "selection_v2_preview.csv"


def normalize_text(t):
    if pd.isna(t):
        return ""
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s_-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def family_key(name):
    n = normalize_text(name)
    for pat in [r"_seed_?\d+", r"_version_?\d+", r"\bv\d+\b", r"-?dropped",
                r"\d+nrows.*", r"\d+rows.*", r"\d+instances.*",
                r"nclasses_?\d+", r"ncols_?\d+", r"stratify_?\w+", r"_\d+$"]:
        n = re.sub(pat, "", n)
    toks = [t for t in re.split(r"[_\-\s]+", n)
            if t not in {"dataset", "data", "set", "train", "test", "csv", "openml", "the"} and len(t) > 2]
    return "_".join(toks[:2]) if toks else n


def main():
    print("Listando datasets do OpenML...")
    df = openml.datasets.list_datasets(output_format="dataframe")
    print(f"Total bruto no OpenML: {len(df)}")

    req = ["name", "did", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses",
           "NumberOfInstancesWithMissingValues", "MinorityClassSize"]
    df = df.dropna(subset=req).copy()

    # filtro tecnico de qualidade (igual ao original)
    tech = df[
        (df.NumberOfInstances >= 200) & (df.NumberOfInstances <= 50000) &
        (df.NumberOfFeatures >= 2) & (df.NumberOfClasses >= 2) &
        (df.NumberOfInstancesWithMissingValues < df.NumberOfInstances * 0.3) &
        (df.MinorityClassSize >= 20)
    ].copy()
    # opcional: status ativo
    if "status" in tech.columns:
        tech = tech[tech["status"] == "active"].copy()
    print(f"Apos filtro tecnico (sem exigir dominio no nome): {len(tech)}")

    # dedup robusto: familia do nome + assinatura de tamanho
    tech["family"] = tech["name"].apply(family_key)
    tech["sig"] = (tech["family"] + "|" +
                   tech["NumberOfInstances"].round(-1).astype("Int64").astype(str) + "|" +
                   tech["NumberOfFeatures"].astype("Int64").astype(str) + "|" +
                   tech["NumberOfClasses"].astype("Int64").astype(str))
    # mantem 1 por assinatura (o de maior nº de instancias)
    dedup_sig = (tech.sort_values("NumberOfInstances", ascending=False)
                     .drop_duplicates(subset=["sig"], keep="first"))
    print(f"Apos dedup por assinatura (familia+tamanho): {len(dedup_sig)}")
    # dedup adicional: 1 por familia (mais agressivo, remove variacoes de tamanho)
    dedup_fam = (dedup_sig.sort_values("NumberOfInstances", ascending=False)
                          .drop_duplicates(subset=["family"], keep="first"))
    print(f"Apos dedup por familia (1 por familia): {len(dedup_fam)}")

    dedup_fam[["did", "name", "family", "NumberOfInstances", "NumberOfFeatures",
               "NumberOfClasses"]].to_csv(OUT, index=False)
    print(f"\nComparacao:")
    print(f"  Base atual (com exigencia de dominio no nome): 116")
    print(f"  Base V2 (dedup por familia):                   {len(dedup_fam)}")
    print(f"  Base V2 (dedup por assinatura, mais permissivo): {len(dedup_sig)}")
    print(f"\nPreview salvo: {OUT}")
    print("\nObs: a anotacao de dominio (por LLM ou keywords na descricao) seria feita")
    print("DEPOIS, sobre toda a base, removendo o vies de selecao por nome.")


if __name__ == "__main__":
    main()
