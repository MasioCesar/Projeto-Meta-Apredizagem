"""
TESTE: manter apenas algoritmos com numero de casos suficiente.

Distribuicao: LR=40, MLP=23, DT=20, SVM=19, KNN=9, Perceptron=5.
Removemos os datasets cujo melhor algoritmo foi KNN ou Perceptron (raros) ->
alvo de 4 classes bem-povoadas (LR, MLP, DT, SVM), min=19, ~2:1.

Compara baseline (so estatistica) vs os 2 melhores configs semanticos, com e sem
oversampling, em 3 SEMENTES (replicacao desde o inicio) + Wilcoxon + mediana.
NAO altera meta-features estatisticas. Saida: data/top10_controlled/highcount_summary.csv
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

from common import OUTPUT_DIR, load_data, build_model
from explore_6algos import build as build_explore

KEEP = {"LogisticRegression", "MLP", "DecisionTree", "SVM"}
SEEDS = [42, 7, 123]


def make(sk_pipeline, balance, seed):
    ct = clone(sk_pipeline.named_steps["preprocess"])
    if balance:
        rf = RandomForestClassifier(n_estimators=300, random_state=seed)
        return ImbPipeline([("pre", ct), ("sampler", RandomOverSampler(random_state=seed)),
                            ("model", rf)])
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    return ImbPipeline([("pre", ct), ("model", rf)])


def main():
    X, y, numeric_cols, _ = load_data()
    mask = y.isin(KEEP).values
    Xf = X.loc[mask].reset_index(drop=True)
    yf = y.loc[mask].reset_index(drop=True)
    print(f"Apos filtro 4 classes: {len(Xf)} datasets (de {len(X)})")
    print(f"Distribuicao: {dict(yf.value_counts())}\n")

    configs = {
        "clusters+domain_interaction": build_model(numeric_cols, "clusters_only", "label_score_interaction"),
        "modality_interaction": build_explore(numeric_cols, "modality_interaction"),
    }
    rows = []
    for balance in [False, True]:
        bname = "oversample" if balance else "class_weight"
        print(f"########## balanceamento: {bname} ##########")
        for cfgname, cfgpipe in configs.items():
            print(f"\n--- baseline vs {cfgname} ---")
            print(f"  {'seed':>5s} {'F1_base':>8s} {'F1_sem':>8s} {'mean_d':>8s} {'med_d':>8s} "
                  f"{'p_t':>7s} {'p_wil':>7s} {'venc':>8s}")
            for seed in SEEDS:
                yenc = LabelEncoder().fit_transform(yf)
                rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
                base = cross_val_score(make(build_model(numeric_cols, "none", "none"), balance, seed),
                                       Xf, yenc, cv=rskf, scoring="f1_macro")
                sem = cross_val_score(make(cfgpipe, balance, seed),
                                      Xf, yenc, cv=rskf, scoring="f1_macro")
                d = sem - base
                _, pt = stats.ttest_rel(sem, base)
                try:
                    _, pw = stats.wilcoxon(sem, base)
                except ValueError:
                    pw = np.nan
                print(f"  {seed:>5d} {base.mean():>8.4f} {sem.mean():>8.4f} {d.mean():>+8.4f} "
                      f"{np.median(d):>+8.4f} {pt:>7.3f} {pw:>7.3f} {int((d>0).sum())}/{len(d)}")
                rows.append({"balance": bname, "config": cfgname, "seed": seed,
                             "f1_baseline": base.mean(), "f1_semantic": sem.mean(),
                             "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
                             "p_ttest": float(pt), "p_wilcoxon": float(pw),
                             "folds_won": f"{int((d>0).sum())}/{len(d)}"})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "highcount_summary.csv"
    summary.to_csv(out, index=False)

    print("\n================ REPLICACAO POR CONFIG ================")
    for (bal, cfg), g in summary.groupby(["balance", "config"]):
        rep = (g["median_delta"] > 0).all() and (g["p_wilcoxon"] < 0.05).all()
        print(f"  {bal:12s} {cfg:30s} replica nas 3 sementes? {'SIM' if rep else 'NAO'} "
              f"(deltas medianos: {[round(v,4) for v in g['median_delta']]})")
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
