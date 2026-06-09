"""
(1) Verifica a config vencedora do test22 (estatistica + semantica controlada +
dominio_label) sob protocolo rigoroso: 3 sementes, CV aleatoria E agrupada,
teste pareado vs baseline. Base V2 (metafeatures_v2_full).
Saida: data/top10_controlled/verify_test22_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import build_model

DATA = Path(__file__).resolve().parents[2] / "data"
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
SEEDS = [42, 7, 123]


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()


def jacc(d2c):
    dids=list(d2c); par={d:d for d in dids}
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in range(len(dids)):
        si=d2c[dids[i]]
        if not si: continue
        for j in range(i+1,len(dids)):
            sj=d2c[dids[j]]
            if sj and len(si|sj) and len(si&sj)/len(si|sj)>=0.5: par[f(dids[i])]=f(dids[j])
    fam={d:f(d) for d in dids}; u={x:k for k,x in enumerate(sorted(set(fam.values())))}
    return {d:u[fam[d]] for d in dids}


def main():
    meta = pd.read_csv(DATA/"metafeatures_v2_full.csv")
    perf = pd.read_csv(DATA/"performance_matrix_v2_full.csv")
    df = meta.merge(perf[["did","best_classifier"]], on="did", how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    txt = {r["did"]: r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}

    numeric_cols = [c for c in meta.columns if c not in ("did","name","semantic_text","predicted_domain","domain_score") and pd.api.types.is_numeric_dtype(meta[c])]
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    df[numeric_cols] = df[numeric_cols].replace([np.inf,-np.inf], np.nan)
    for c in ["semantic_text","name","predicted_domain"]:
        df[c] = df[c].fillna("").astype(str)
    df["domain_score"] = pd.to_numeric(df["domain_score"], errors="coerce").fillna(0)
    yenc = LabelEncoder().fit_transform(df["best_classifier"].astype(str))

    d2c = {int(d): set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]}
    groups = df["did"].astype(int).map(jacc(d2c)).values

    configs = {
        "baseline_stat": build_model(numeric_cols, "none", "none"),
        "+dominio_label": build_model(numeric_cols, "none", "label"),
        "+semantica+dominio_label(test22)": build_model(numeric_cols, "tags_fixed_vocab", "label"),
    }
    rows = []
    for proto in ["random","grouped"]:
        print(f"\n=== CV {proto} (F1-macro, 3 sementes) ===")
        base = []
        for s in SEEDS:
            cv = StratifiedKFold(5, shuffle=True, random_state=s) if proto=="random" else StratifiedGroupKFold(5, shuffle=True, random_state=s)
            kw = {} if proto=="random" else {"groups":groups}
            base.append(cross_val_score(configs["baseline_stat"], df, yenc, cv=cv, scoring="f1_macro", **kw).mean())
        print(f"  baseline = {np.mean(base):.4f}")
        for name in ["+dominio_label","+semantica+dominio_label(test22)"]:
            ds = []
            for s in SEEDS:
                cv = StratifiedKFold(5, shuffle=True, random_state=s) if proto=="random" else StratifiedGroupKFold(5, shuffle=True, random_state=s)
                kw = {} if proto=="random" else {"groups":groups}
                ds.append(cross_val_score(configs[name], df, yenc, cv=cv, scoring="f1_macro", **kw).mean())
            d = np.mean(ds)-np.mean(base); _, p = stats.ttest_rel(ds, base)
            print(f"  {name:38s} F1={np.mean(ds):.4f}  dF1={d:+.4f}  p={p:.3f}")
            rows.append({"cv":proto,"config":name,"f1":np.mean(ds),"delta":float(d),"p":float(p),"baseline":np.mean(base)})

    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"verify_test22_v2_summary.csv", index=False)
    print(f"\nSalvo: verify_test22_v2_summary.csv")


if __name__ == "__main__":
    main()
