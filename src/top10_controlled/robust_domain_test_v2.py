"""
ESTUDO ROBUSTO (base V2, 547 datasets) - OPCAO A: dominio limpo via keywords na
DESCRICAO (normalizado), testado com CV AGRUPADA (por similaridade de colunas) e
CV ALEATORIA, multi-semente. Alvo: 6 algoritmos.

Pergunta: na base ampliada/sem-vies, o dominio supera o baseline estatistico?
Saida: data/top10_controlled/robust_domain_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data"
OUTDIR = DATA / "top10_controlled"
ALL6 = ["DecisionTree", "KNN", "LogisticRegression", "MLP", "Perceptron", "SVM"]
SEEDS = [42, 7, 123]

domain_keywords = {
    "health": {"health":4,"medical":5,"clinical":5,"patient":5,"disease":5,"diagnosis":5,"cancer":7,"tumor":7,"diabetes":7,"heart":7,"blood":5,"breast":6,"thyroid":6,"hepatitis":6,"eeg":7,"ecg":7,"covid":6,"hospital":5,"symptom":4},
    "finance": {"finance":5,"financial":5,"credit":7,"bank":6,"loan":7,"fraud":7,"stock":6,"insurance":6,"bankruptcy":6,"default":6,"payment":5,"investment":6,"transaction":6,"mortgage":7,"trading":7},
    "biology": {"gene":7,"genes":7,"genomic":7,"protein":7,"dna":7,"rna":7,"sequence":6,"yeast":8,"ecoli":8,"microarray":7,"molecular":5,"cell":4,"species":4,"mutation":6,"expression":5,"qsar":8,"chemical":6,"molecule":6},
    "image": {"image":6,"images":6,"pixel":5,"pixels":5,"mnist":8,"digit":6,"face":6,"vision":6,"ocr":6,"letter":5,"texture":5,"rgb":6,"video":5},
    "text": {"text":6,"document":6,"news":5,"review":5,"sentiment":7,"spam":7,"email":6,"language":6,"nlp":7,"tweet":6,"corpus":6,"sarcasm":7},
    "sensor_signal": {"sensor":6,"signal":6,"seismic":7,"robot":6,"fault":6,"vibration":6,"activity":6,"gesture":7,"motion":6,"accelerometer":7,"waveform":6,"radar":7,"sonar":7,"gas":5},
    "education": {"student":7,"students":7,"school":5,"education":7,"exam":5,"grade":5,"academic":5,"university":5,"dropout":7,"mooc":7},
    "social": {"social":6,"network":6,"user":5,"community":5,"rating":6,"movie":5,"recommendation":6,"customer":4,"adult":7,"census":7,"income":5,"churn":7,"marketing":5},
}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t)).strip()


def jaccard_families(did_to_cols):
    dids = list(did_to_cols)
    parent = {d: d for d in dids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(dids)):
        si = did_to_cols[dids[i]]
        if not si: continue
        for j in range(i+1, len(dids)):
            sj = did_to_cols[dids[j]]
            if not sj: continue
            u = len(si | sj)
            if u and len(si & sj)/u >= 0.5:
                parent[find(dids[i])] = find(dids[j])
    fam = {d: find(d) for d in dids}
    uniq = {f: k for k, f in enumerate(sorted(set(fam.values())))}
    return {d: uniq[fam[d]] for d in dids}


def main():
    meta = pd.read_csv(DATA / "metafeatures_v2.csv")
    perf = pd.read_csv(DATA / "performance_matrix_v2.csv")
    txt = {r["did"]: r for r in json.loads((DATA / "v2_descriptions.json").read_text(encoding="utf-8"))}

    df = meta.merge(perf[["did", "best_classifier"] + [a for a in ALL6 if a in perf.columns]], on="did", how="inner")
    df = df.dropna(subset=["best_classifier"]).reset_index(drop=True)

    # dominio limpo (keywords na DESCRICAO, normalizado)
    doms, confs = [], []
    for did in df["did"]:
        d = norm(txt.get(int(did), {}).get("description", ""))
        toks = max(1, len(d.split()))
        scores = {dom: sum(w for kw, w in kws.items() if re.search(rf"\b{kw}\b", d))
                  for dom, kws in domain_keywords.items()}
        best = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
        doms.append(best); confs.append(scores.get(best, 0)/toks)
    df["dom"] = doms
    df["dom_conf"] = confs

    # familias por colunas (CV agrupada)
    d2c = {int(did): set(norm(txt.get(int(did), {}).get("feature_names", "")).split()) - {""} for did in df["did"]}
    fam = jaccard_families(d2c)
    groups = df["did"].astype(int).map(fam).values
    print(f"Datasets: {len(df)} | familias por coluna: {len(set(groups))}")
    print(f"dom_conf vs nr_attr (confound check) rho={df[['dom_conf','nr_attr']].corr('spearman').iloc[0,1]:+.3f}")
    print(f"Distribuicao dominio: {df['dom'].value_counts().to_dict()}\n")

    numeric_cols = [c for c in meta.columns if c not in ("did", "name") and pd.api.types.is_numeric_dtype(meta[c])]
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    # limpa infinitos (pymfe gera inf em alguns datasets) -> NaN (imputado depois)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    yenc = LabelEncoder().fit_transform(df["best_classifier"].astype(str))

    def model(with_dom):
        t = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols)]
        if with_dom:
            t.append(("dom", OneHotEncoder(handle_unknown="ignore"), ["dom"]))
            t.append(("conf", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["dom_conf"]))
        return Pipeline([("pre", ColumnTransformer(t, remainder="drop")),
                         ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])

    rows = []
    for proto in ["random", "grouped"]:
        print(f"--- CV {proto} ---")
        db, ds_ = [], []
        for s in SEEDS:
            if proto == "random":
                cv = StratifiedKFold(5, shuffle=True, random_state=s)
                fb = cross_val_score(model(False), df, yenc, cv=cv, scoring="f1_macro")
                fd = cross_val_score(model(True), df, yenc, cv=cv, scoring="f1_macro")
            else:
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=s)
                fb = cross_val_score(model(False), df, yenc, cv=cv, groups=groups, scoring="f1_macro")
                fd = cross_val_score(model(True), df, yenc, cv=cv, groups=groups, scoring="f1_macro")
            db.append(fb.mean()); ds_.append(fd.mean())
        delta = np.mean(ds_) - np.mean(db)
        _, p = stats.ttest_rel(ds_, db)
        print(f"  F1 baseline={np.mean(db):.4f} | +dominio={np.mean(ds_):.4f} | dF1={delta:+.4f} | p={p:.3f}\n")
        rows.append({"cv": proto, "f1_baseline": np.mean(db), "f1_domain": np.mean(ds_),
                     "delta_f1": float(delta), "p_value": float(p)})

    pd.DataFrame(rows).to_csv(OUTDIR / "robust_domain_v2_summary.csv", index=False)
    print(f"Salvo: {OUTDIR / 'robust_domain_v2_summary.csv'}")


if __name__ == "__main__":
    main()
