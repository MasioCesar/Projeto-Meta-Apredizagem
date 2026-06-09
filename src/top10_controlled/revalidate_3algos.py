"""
RE-VALIDACAO do resultado original com 3 algoritmos (DT, LR, Perceptron).

Pergunta: o ganho semantico de +6.5pp era REAL/replicavel, ou um falso positivo
de semente unica (como o p=0.004 dos 6 algoritmos)?

Recomputa o alvo entre os 3 originais e roda baseline vs os configs semanticos
originais, com 3 SEMENTES + Wilcoxon + mediana do delta. As features (X) sao as
mesmas; so o alvo muda. NAO altera meta-features estatisticas.
Saida: data/top10_controlled/revalidate_3algos_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data, build_model

THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
SEEDS = [42, 7, 123]


def main():
    X, y6, numeric_cols, _ = load_data()
    # recomputa alvo de 3 algoritmos alinhado a X (mesma ordem do merge do load_data)
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["best6"] = perf[["DecisionTree", "KNN", "LogisticRegression", "MLP",
                          "Perceptron", "SVM"]].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best6"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    assert len(y3) == len(X)
    print(f"Alvo 3 algoritmos | dist={dict(y3.value_counts())} | datasets={len(X)}\n")

    configs = {
        "tags_fixed_vocab+label_score(top1_original)":
            build_model(numeric_cols, "tags_fixed_vocab", "label_score"),
        "clusters+domain_interaction(top2_original)":
            build_model(numeric_cols, "clusters_only", "label_score_interaction"),
    }
    yenc = LabelEncoder().fit_transform(y3)
    rows = []
    for cfgname, cfgpipe in configs.items():
        print(f"--- baseline vs {cfgname} ---")
        print(f"  {'seed':>5s} {'F1_base':>8s} {'F1_sem':>8s} {'mean_d':>8s} {'med_d':>8s} "
              f"{'p_t':>7s} {'p_wil':>7s} {'venc':>8s}")
        for seed in SEEDS:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
            base = cross_val_score(build_model(numeric_cols, "none", "none"),
                                   X, yenc, cv=rskf, scoring="f1_macro")
            sem = cross_val_score(cfgpipe, X, yenc, cv=rskf, scoring="f1_macro")
            d = sem - base
            _, pt = stats.ttest_rel(sem, base)
            try:
                _, pw = stats.wilcoxon(sem, base)
            except ValueError:
                pw = np.nan
            print(f"  {seed:>5d} {base.mean():>8.4f} {sem.mean():>8.4f} {d.mean():>+8.4f} "
                  f"{np.median(d):>+8.4f} {pt:>7.3f} {pw:>7.3f} {int((d>0).sum())}/{len(d)}")
            rows.append({"config": cfgname, "seed": seed,
                         "f1_baseline": base.mean(), "f1_semantic": sem.mean(),
                         "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                         "p_ttest": float(pt), "p_wilcoxon": float(pw),
                         "folds_won": f"{int((d>0).sum())}/{len(d)}"})
        print()

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "revalidate_3algos_summary.csv"
    summary.to_csv(out, index=False)
    print("================ REPLICACAO ================")
    for cfg, g in summary.groupby("config"):
        rep = (g["median_delta"] > 0).all() and (g["p_wilcoxon"] < 0.05).all()
        print(f"  {cfg:46s} replica? {'SIM' if rep else 'NAO'} "
              f"(medianas {[round(v,4) for v in g['median_delta']]}, "
              f"p_wil {[round(v,3) for v in g['p_wilcoxon']]})")
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
