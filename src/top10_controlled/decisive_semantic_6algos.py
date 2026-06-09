"""
TESTE DECISIVO: alvo 3-vias + oversampling + os configs semanticos MAIS FORTES,
com alto poder estatistico (CV 5x20). Responde de vez: a semantica vence?

Configs semanticos testados (via common.build_model, os historicamente melhores):
  - clusters_only + label_score_interaction  (test02, melhor em 6 classes)
  - tags_fixed_vocab + label_score            (top1 original em 3 algoritmos)
  - tags_only + label_score_interaction
  + modality_interaction (explore_6algos)

Balanceamento: RandomOverSampler (so no treino). Alvos: 3_way e complex_worth.
Cautela: testar varios configs infla risco de falso-positivo; reportamos TODOS.
Saida: data/top10_controlled/decisive_summary.csv
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
from imblearn.over_sampling import RandomOverSampler

from common import (OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data,
                    build_model)
from explore_6algos import build as build_explore

SIMPLE = ["DecisionTree", "LogisticRegression", "Perceptron"]
COMPLEX = ["SVM", "MLP", "KNN"]
GROUP3 = {"DecisionTree": "tree", "LogisticRegression": "linear", "Perceptron": "linear",
          "SVM": "complex", "MLP": "complex", "KNN": "complex"}


def wrap_oversample(sk_pipeline):
    """Extrai o ColumnTransformer e reembrulha com RandomOverSampler (treino-only)."""
    ct = clone(sk_pipeline.named_steps["preprocess"])
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    return ImbPipeline([("pre", ct),
                        ("sampler", RandomOverSampler(random_state=42)),
                        ("model", rf)])


def main():
    X, y, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)

    # modelos candidatos (cada um ja embrulhado com oversampling)
    candidates = {
        "baseline_stat_only": wrap_oversample(build_model(numeric_cols, "none", "none")),
        "clusters+domain_interaction": wrap_oversample(
            build_model(numeric_cols, "clusters_only", "label_score_interaction")),
        "tags_fixed_vocab+label_score": wrap_oversample(
            build_model(numeric_cols, "tags_fixed_vocab", "label_score")),
        "tags+domain_interaction": wrap_oversample(
            build_model(numeric_cols, "tags_only", "label_score_interaction")),
        "modality_interaction": wrap_oversample(build_explore(numeric_cols, "modality_interaction")),
    }

    targets = {
        "3_way": y.map(GROUP3).values,
        "complex_worth_m0.01": np.where(gap > 0.01, "complex_worth", "simple_enough"),
    }
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    rows = []

    for tname, yt in targets.items():
        yenc = LabelEncoder().fit_transform(yt)
        print(f"\n===== {tname} | dist={dict(pd.Series(yt).value_counts())} | CV 5x20 + oversampling =====")
        base_scores = cross_val_score(candidates["baseline_stat_only"], X, yenc, cv=rskf, scoring="f1_macro")
        print(f"  {'config':32s} {'F1m':>8s} {'delta':>8s} {'p':>7s} {'venc':>7s}")
        print(f"  {'baseline_stat_only':32s} {base_scores.mean():>8.4f} {'-':>8s} {'-':>7s} {'-':>7s}")
        rows.append({"target": tname, "config": "baseline_stat_only",
                     "f1_macro_mean": base_scores.mean(), "f1_macro_std": base_scores.std(),
                     "delta": 0.0, "p_value": np.nan, "folds_won": "-"})
        for name, model in candidates.items():
            if name == "baseline_stat_only":
                continue
            s = cross_val_score(model, X, yenc, cv=rskf, scoring="f1_macro")
            d = s - base_scores
            t, p = stats.ttest_rel(s, base_scores)
            flag = " <-- p<0.05" if p < 0.05 else ""
            print(f"  {name:32s} {s.mean():>8.4f} {d.mean():>+8.4f} {p:>7.3f} {int((d>0).sum())}/{len(d)}{flag}")
            rows.append({"target": tname, "config": name,
                         "f1_macro_mean": s.mean(), "f1_macro_std": s.std(),
                         "delta": float(d.mean()), "p_value": float(p),
                         "folds_won": f"{int((d>0).sum())}/{len(d)}"})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "decisive_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
