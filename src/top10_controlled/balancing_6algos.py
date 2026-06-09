"""
TESTE DE BALANCEAMENTO: o desbalanceamento esconde um sinal semantico?

Combina reamostragem (SMOTE / over / under, aplicada SO no treino via imblearn
Pipeline -> sem vazamento de CV) com a comparacao baseline vs +semantica
(interacao modalidade x escala, a melhor ate aqui).

Duas perguntas:
  (1) o balanceamento melhora o F1-macro absoluto?
  (2) COM balanceamento, a semantica passa a VENCER o baseline? (p pareado)

Alvos: 6 classes, 3 vias, "complexo vale" (margem 0.01).
NAO altera meta-features estatisticas. Saida: data/top10_controlled/balancing_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data
from explore_6algos import build

SIMPLE = ["DecisionTree", "LogisticRegression", "Perceptron"]
COMPLEX = ["SVM", "MLP", "KNN"]
GROUP3 = {"DecisionTree": "tree", "LogisticRegression": "linear", "Perceptron": "linear",
          "SVM": "complex", "MLP": "complex", "KNN": "complex"}


def make_model(numeric_cols, semantic_mode, balancer, k_neighbors=5):
    """Reusa o ColumnTransformer de build() e troca a estrategia de balanceamento."""
    base = build(numeric_cols, semantic_mode)
    ct = clone(base.named_steps["preprocess"])
    if balancer == "none":
        # mantem class_weight balanced (estrategia atual do projeto)
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        return ImbPipeline([("pre", ct), ("model", rf)])
    rf = RandomForestClassifier(n_estimators=300, random_state=42)  # sem class_weight
    if balancer == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    elif balancer == "over":
        sampler = RandomOverSampler(random_state=42)
    elif balancer == "under":
        sampler = RandomUnderSampler(random_state=42)
    return ImbPipeline([("pre", ct), ("sampler", sampler), ("model", rf)])


def main():
    X, y, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)

    targets = {
        "6_classes": (y.values, 2),   # k_neighbors menor: Perceptron=5
        "3_way": (y.map(GROUP3).values, 5),
        "complex_worth_m0.01": (np.where(gap > 0.01, "complex_worth", "simple_enough"), 5),
    }
    balancers = ["none", "smote", "over", "under"]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []

    for tname, (yt, kn) in targets.items():
        yenc = LabelEncoder().fit_transform(yt)
        print(f"\n===== {tname} | dist={dict(pd.Series(yt).value_counts())} =====")
        print(f"  {'balanceador':12s} {'baseline':>10s} {'+semantica':>11s} {'delta':>8s} {'p':>7s} {'venc':>6s}")
        for bal in balancers:
            try:
                mb = make_model(numeric_cols, None, bal, kn)
                ms = make_model(numeric_cols, "modality_interaction", bal, kn)
                base = cross_val_score(mb, X, yenc, cv=rskf, scoring="f1_macro")
                sem = cross_val_score(ms, X, yenc, cv=rskf, scoring="f1_macro")
            except Exception as e:
                print(f"  {bal:12s} ERRO: {str(e)[:60]}")
                continue
            d = sem - base
            t, p = stats.ttest_rel(sem, base)
            print(f"  {bal:12s} {base.mean():>10.4f} {sem.mean():>11.4f} {d.mean():>+8.4f} {p:>7.3f} {int((d>0).sum())}/{len(d)}")
            rows.append({"target": tname, "balancer": bal,
                         "f1_baseline": base.mean(), "f1_baseline_std": base.std(),
                         "f1_semantic": sem.mean(), "f1_semantic_std": sem.std(),
                         "delta_semantic_minus_baseline": float(d.mean()),
                         "p_value": float(p), "folds_won": f"{int((d>0).sum())}/{len(d)}"})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "balancing_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
