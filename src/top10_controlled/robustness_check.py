"""
VERIFICACAO DE ROBUSTEZ do unico resultado positivo significativo:
  alvo "complex_worth (margem 0.01)" + oversampling + clusters+domain_interaction
  vs baseline (so estatistica), p=0.004 no teste inicial.

Checa se REPLICA: 3 sementes diferentes de CV, + Wilcoxon (nao-parametrico) +
mediana do delta (robusta a folds outliers). Se replicar nas 3 sementes com
mediana>0 e Wilcoxon significativo, e um achado solido.
Saida: data/top10_controlled/robustness_summary.csv
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

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data, build_model

SIMPLE = ["DecisionTree", "LogisticRegression", "Perceptron"]
COMPLEX = ["SVM", "MLP", "KNN"]


def wrap(sk_pipeline, seed):
    ct = clone(sk_pipeline.named_steps["preprocess"])
    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    return ImbPipeline([("pre", ct),
                        ("sampler", RandomOverSampler(random_state=seed)),
                        ("model", rf)])


def main():
    X, y, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)
    yenc = LabelEncoder().fit_transform(np.where(gap > 0.01, "complex_worth", "simple_enough"))

    rows = []
    print("alvo: complex_worth_m0.01 | oversampling | clusters+domain_interaction vs baseline\n")
    print(f"  {'seed':>5s} {'F1_base':>8s} {'F1_sem':>8s} {'mean_d':>8s} {'med_d':>8s} "
          f"{'p_ttest':>8s} {'p_wilcox':>9s} {'venc':>8s}")
    for seed in [42, 7, 123]:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
        base = cross_val_score(wrap(build_model(numeric_cols, "none", "none"), seed),
                               X, yenc, cv=rskf, scoring="f1_macro")
        sem = cross_val_score(wrap(build_model(numeric_cols, "clusters_only", "label_score_interaction"), seed),
                              X, yenc, cv=rskf, scoring="f1_macro")
        d = sem - base
        _, p_t = stats.ttest_rel(sem, base)
        try:
            _, p_w = stats.wilcoxon(sem, base)
        except ValueError:
            p_w = np.nan
        wins = int((d > 0).sum())
        print(f"  {seed:>5d} {base.mean():>8.4f} {sem.mean():>8.4f} {d.mean():>+8.4f} "
              f"{np.median(d):>+8.4f} {p_t:>8.3f} {p_w:>9.3f} {wins}/{len(d)}")
        rows.append({"seed": seed, "f1_baseline": base.mean(), "f1_semantic": sem.mean(),
                     "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                     "p_ttest": float(p_t), "p_wilcoxon": float(p_w),
                     "folds_won": f"{wins}/{len(d)}"})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "robustness_summary.csv"
    summary.to_csv(out, index=False)
    rep = (summary["median_delta"] > 0).all() and (summary["p_wilcoxon"] < 0.05).all()
    print(f"\nReplica nas 3 sementes (mediana>0 E Wilcoxon<0.05)? {'SIM' if rep else 'NAO'}")
    print(f"Salvo em: {out}")


if __name__ == "__main__":
    main()
