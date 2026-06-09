"""
TESTE B FINAL: dominio anotado por LLM (Claude) vs baseline, alvo 3 algoritmos.

Compara baseline (so estatistica) contra:
  - +llm_domain        (categoria 8-vias, one-hot)
  - +llm_confidence    (1-5, numerico)
  - +llm_domain+conf
Tambem checa o CONFUNDIMENTO: correlacao de llm_confidence com nr_attr, e
nr_attr medio por dominio (o dominio-LLM esta entrelacado com dimensionalidade?).

3 sementes, CV 5x20, t pareado + Wilcoxon + mediana. Saida: data/top10_controlled/llm_domain_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data

THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
SEEDS = [42, 7, 123]
ANN = INPUT_METAFEATURES.parent / "llm_domain_annotations.csv"


def build(numeric_cols, num_cols=None, cat_cols=None):
    t = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols)]
    if num_cols:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return Pipeline([("pre", ColumnTransformer(t, remainder="drop")),
                     ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["b6"] = perf[["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["b6"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    yenc = LabelEncoder().fit_transform(y3)

    ann = pd.read_csv(ANN)
    merged = df_exp[["did"]].merge(ann, on="did", how="left")
    assert len(merged) == len(X)
    Xc = X.copy().reset_index(drop=True)
    Xc["llm_domain"] = merged["llm_domain"].fillna("unknown").astype(str).values
    Xc["llm_confidence"] = merged["llm_confidence"].fillna(3).astype(float).values
    Xc["nr_attr_dbg"] = df_exp["nr_attr"].values

    # --- diagnostico de confundimento ---
    print("=== CONFUNDIMENTO do dominio-LLM com dimensionalidade ===")
    r = Xc[["llm_confidence", "nr_attr_dbg"]].corr("spearman").iloc[0, 1]
    print(f"llm_confidence vs nr_attr: rho = {r:+.3f}  (lembrete: domain_score original era +0.40)")
    print("nr_attr medio por llm_domain:")
    print(Xc.groupby("llm_domain")["nr_attr_dbg"].mean().sort_values(ascending=False).round(0).to_string())

    configs = [
        ("baseline_stat_only", None, None),
        ("+llm_domain (categoria)", None, ["llm_domain"]),
        ("+llm_confidence", ["llm_confidence"], None),
        ("+llm_domain+confidence", ["llm_confidence"], ["llm_domain"]),
    ]
    base_by_seed = {s: cross_val_score(build(numeric_cols), Xc, yenc,
                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=s), scoring="f1_macro") for s in SEEDS}
    rows = []
    print("\n=== TESTE (alvo 3 algoritmos, F1-macro) ===")
    for name, ncols, ccols in configs:
        if ncols is None and ccols is None:
            for s in SEEDS:
                rows.append({"config": name, "seed": s, "f1": float(base_by_seed[s].mean()),
                             "delta": 0.0, "median_delta": 0.0, "p_wilcoxon": np.nan, "folds_won": "-"})
            print(f"--- {name} --- F1m medio={np.mean([base_by_seed[s].mean() for s in SEEDS]):.4f}")
            continue
        print(f"--- {name} ---")
        print(f"  {'seed':>5s} {'F1':>8s} {'mean_d':>8s} {'med_d':>8s} {'p_t':>7s} {'p_wil':>7s} {'venc':>8s}")
        for s in SEEDS:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=s)
            sem = cross_val_score(build(numeric_cols, ncols, ccols), Xc, yenc, cv=rskf, scoring="f1_macro")
            base = base_by_seed[s]; d = sem - base
            _, pt = stats.ttest_rel(sem, base)
            try: _, pw = stats.wilcoxon(sem, base)
            except ValueError: pw = np.nan
            print(f"  {s:>5d} {sem.mean():>8.4f} {d.mean():>+8.4f} {np.median(d):>+8.4f} {pt:>7.3f} {pw:>7.3f} {int((d>0).sum())}/{len(d)}")
            rows.append({"config": name, "seed": s, "f1": float(sem.mean()), "delta": float(d.mean()),
                         "median_delta": float(np.median(d)), "p_wilcoxon": float(pw), "folds_won": f"{int((d>0).sum())}/{len(d)}"})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "llm_domain_summary.csv"
    summary.to_csv(out, index=False)
    print("\n=== REPLICACAO (mediana>0 E Wilcoxon<0.05 nas 3 sementes) ===")
    for cfg, g in summary[summary.config != "baseline_stat_only"].groupby("config"):
        rep = (g["median_delta"] > 0).all() and (g["p_wilcoxon"] < 0.05).all()
        pos = (g["delta"] > 0).all()
        print(f"  {cfg:28s} replica? {'SIM' if rep else 'NAO':3s} | sempre+? {'SIM' if pos else 'NAO'} | medianas {[round(v,4) for v in g['median_delta']]}")
    print(f"\nSalvo: {out}")


if __name__ == "__main__":
    main()
