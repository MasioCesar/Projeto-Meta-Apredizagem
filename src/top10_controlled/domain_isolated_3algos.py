"""
TESTE DA HIPOTESE CENTRAL ORIGINAL: o DOMINIO do dataset (saude, financas, etc.)
impacta a selecao de algoritmo?

Isola o dominio (predicted_domain e/ou domain_score) SEM nenhuma outra feature
semantica (sem clusters, tags, texto, embeddings). Alvo limpo de 3 algoritmos,
3 sementes, CV 5x20, testes pareados + mediana. NAO altera meta-features
estatisticas. Saida: data/top10_controlled/domain_isolated_summary.csv
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
# todas com semantic_mode="none" -> dominio isolado
DOMAIN_CONFIGS = [
    ("baseline_stat_only", "none"),
    ("stat+domain_label", "label"),
    ("stat+domain_score", "score"),
    ("stat+domain_label_score", "label_score"),
    ("stat+domain_label_score_interaction", "label_score_interaction"),
]


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["b6"] = perf[["DecisionTree", "KNN", "LogisticRegression", "MLP",
                       "Perceptron", "SVM"]].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["b6"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    assert len(y3) == len(X)
    yenc = LabelEncoder().fit_transform(y3)
    print(f"Alvo 3 algoritmos | dist={dict(y3.value_counts())} | datasets={len(X)}\n")

    # baseline por semente (reusado)
    rows = []
    base_by_seed = {}
    for seed in SEEDS:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
        base_by_seed[seed] = cross_val_score(build_model(numeric_cols, "none", "none"),
                                             X, yenc, cv=rskf, scoring="f1_macro")

    for name, dmode in DOMAIN_CONFIGS:
        print(f"--- {name} ---")
        if dmode == "none":
            for seed in SEEDS:
                b = base_by_seed[seed]
                print(f"  seed {seed}: F1m {b.mean():.4f}")
                rows.append({"config": name, "domain_mode": dmode, "seed": seed,
                             "f1_mean": float(b.mean()), "delta": 0.0,
                             "median_delta": 0.0, "p_ttest": np.nan, "p_wilcoxon": np.nan,
                             "folds_won": "-"})
            print()
            continue
        print(f"  {'seed':>5s} {'F1':>8s} {'mean_d':>8s} {'med_d':>8s} {'p_t':>7s} {'p_wil':>7s} {'venc':>8s}")
        for seed in SEEDS:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
            sem = cross_val_score(build_model(numeric_cols, "none", dmode),
                                  X, yenc, cv=rskf, scoring="f1_macro")
            base = base_by_seed[seed]
            d = sem - base
            _, pt = stats.ttest_rel(sem, base)
            try:
                _, pw = stats.wilcoxon(sem, base)
            except ValueError:
                pw = np.nan
            print(f"  {seed:>5d} {sem.mean():>8.4f} {d.mean():>+8.4f} {np.median(d):>+8.4f} "
                  f"{pt:>7.3f} {pw:>7.3f} {int((d>0).sum())}/{len(d)}")
            rows.append({"config": name, "domain_mode": dmode, "seed": seed,
                         "f1_mean": float(sem.mean()), "delta": float(d.mean()),
                         "median_delta": float(np.median(d)), "p_ttest": float(pt),
                         "p_wilcoxon": float(pw), "folds_won": f"{int((d>0).sum())}/{len(d)}"})
        print()

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "domain_isolated_summary.csv"
    summary.to_csv(out, index=False)
    print("================ REPLICACAO (DOMINIO ISOLADO) ================")
    for cfg, g in summary[summary.domain_mode != "none"].groupby("config"):
        rep = (g["median_delta"] > 0).all() and (g["p_wilcoxon"] < 0.05).all()
        pos = (g["delta"] > 0).all()
        print(f"  {cfg:40s} replica<0.05? {'SIM' if rep else 'NAO':3s} | "
              f"sempre positivo? {'SIM' if pos else 'NAO'} | "
              f"medianas {[round(v,4) for v in g['median_delta']]}")
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
