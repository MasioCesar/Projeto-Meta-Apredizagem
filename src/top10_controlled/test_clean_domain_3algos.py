"""
(C) Re-teste do sinal de DOMINIO LIMPO no alvo de 3 algoritmos, multi-semente.

Compara baseline (so estatistica) contra variantes de dominio:
  - domain_score ORIGINAL (confundido, referencia)
  - score_desc        (A: so descricao, sem nome/features)
  - score_desc_norm   (A+B: normalizado por tamanho do texto)
  - hits_desc_norm    (A+B: fracao de keywords distintas)
  - domain_desc label (categoria limpa, one-hot)

Pergunta: o ganho SOBREVIVE quando o sinal e limpo de vazamento e de
verbosidade/dimensionalidade? CV 5x20, 3 sementes, t pareado + Wilcoxon + mediana.
Saida: data/top10_controlled/clean_domain_summary.csv
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
CLEAN = INPUT_METAFEATURES.parent / "domain_score_clean.csv"


def build(numeric_cols, domain_col=None, label_col=None):
    transformers = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), numeric_cols)]
    if domain_col:
        transformers.append(("dom", Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), [domain_col]))
    if label_col:
        transformers.append(("lab", OneHotEncoder(handle_unknown="ignore"), [label_col]))
    return Pipeline([("pre", ColumnTransformer(transformers, remainder="drop")),
                     ("model", RandomForestClassifier(n_estimators=300, random_state=42,
                                                      class_weight="balanced"))])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["b6"] = perf[["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["b6"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    yenc = LabelEncoder().fit_transform(y3)

    clean = pd.read_csv(CLEAN)
    merged = df_exp[["did"]].merge(clean, on="did", how="left")
    assert len(merged) == len(X)
    Xc = X.copy().reset_index(drop=True)
    Xc["domain_score_orig"] = df_exp["domain_score"].values
    for col in ["score_desc", "score_desc_norm", "hits_desc_norm"]:
        Xc[col] = merged[col].values
    Xc["domain_desc"] = merged["domain_desc"].fillna("none").astype(str).values

    configs = [
        ("baseline_stat_only", None, None),
        ("+domain_score_ORIGINAL(confundido)", "domain_score_orig", None),
        ("+score_desc (A)", "score_desc", None),
        ("+score_desc_norm (A+B)", "score_desc_norm", None),
        ("+hits_desc_norm (A+B)", "hits_desc_norm", None),
        ("+domain_desc_label (categoria limpa)", None, "domain_desc"),
    ]
    rskf_by_seed = {s: RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=s) for s in SEEDS}
    base_by_seed = {s: cross_val_score(build(numeric_cols), Xc, yenc, cv=rskf_by_seed[s], scoring="f1_macro") for s in SEEDS}

    rows = []
    for name, dcol, lcol in configs:
        print(f"--- {name} ---")
        if dcol is None and lcol is None:
            for s in SEEDS:
                print(f"  seed {s}: F1m {base_by_seed[s].mean():.4f}")
                rows.append({"config": name, "seed": s, "f1": float(base_by_seed[s].mean()),
                             "delta": 0.0, "median_delta": 0.0, "p_wilcoxon": np.nan, "folds_won": "-"})
            print(); continue
        print(f"  {'seed':>5s} {'F1':>8s} {'mean_d':>8s} {'med_d':>8s} {'p_t':>7s} {'p_wil':>7s} {'venc':>8s}")
        for s in SEEDS:
            sem = cross_val_score(build(numeric_cols, dcol, lcol), Xc, yenc, cv=rskf_by_seed[s], scoring="f1_macro")
            base = base_by_seed[s]; d = sem - base
            _, pt = stats.ttest_rel(sem, base)
            try: _, pw = stats.wilcoxon(sem, base)
            except ValueError: pw = np.nan
            print(f"  {s:>5d} {sem.mean():>8.4f} {d.mean():>+8.4f} {np.median(d):>+8.4f} {pt:>7.3f} {pw:>7.3f} {int((d>0).sum())}/{len(d)}")
            rows.append({"config": name, "seed": s, "f1": float(sem.mean()), "delta": float(d.mean()),
                         "median_delta": float(np.median(d)), "p_wilcoxon": float(pw), "folds_won": f"{int((d>0).sum())}/{len(d)}"})
        print()

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "clean_domain_summary.csv"
    summary.to_csv(out, index=False)
    print("================ REPLICACAO (mediana>0 E Wilcoxon<0.05 nas 3 sementes) ================")
    for cfg, g in summary[summary.config != "baseline_stat_only"].groupby("config"):
        rep = (g["median_delta"] > 0).all() and (g["p_wilcoxon"] < 0.05).all()
        pos = (g["delta"] > 0).all()
        print(f"  {cfg:40s} replica? {'SIM' if rep else 'NAO':3s} | sempre+? {'SIM' if pos else 'NAO'} | medianas {[round(v,4) for v in g['median_delta']]}")
    print(f"\nSalvo: {out}")


if __name__ == "__main__":
    main()
