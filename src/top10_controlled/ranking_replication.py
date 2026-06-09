"""
Replicacao MULTI-SEMENTE do sinal de dominio na metrica de REGRET (6 algos).
Corrige o bug de semente unica: testa o regret pareado por dataset em CADA semente.
Saida: data/top10_controlled/ranking_replication_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data

ALL6 = ["DecisionTree", "KNN", "LogisticRegression", "MLP", "Perceptron", "SVM"]
SEEDS = [42, 7, 123, 2024, 11]
ANN = INPUT_METAFEATURES.parent / "llm_domain_annotations.csv"


def make_reg(numeric_cols, num_extra=None, cat_extra=None):
    t = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols)]
    if num_extra:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_extra))
    if cat_extra:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_extra))
    return Pipeline([("pre", ColumnTransformer(t, remainder="drop")),
                     ("reg", MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42)))])


def regret_vec(Y, Ypred):
    return np.array([Y[i].max() - Y[i][np.argmax(Ypred[i])] for i in range(len(Y))])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    ann = pd.read_csv(ANN)
    ann_df = df_exp[["did"]].merge(ann, on="did", how="left")
    ann_df["domain_score"] = df_exp["domain_score"].values
    ann_df["llm_domain"] = ann_df["llm_domain"].fillna("unknown").astype(str)
    ann_df["llm_confidence"] = ann_df["llm_confidence"].fillna(3).astype(float)

    Y = df_exp[ALL6].to_numpy(dtype=float)
    keep = ~np.isnan(Y).any(axis=1)
    Y = Y[keep]; Xc = X.loc[keep].reset_index(drop=True); ann_df = ann_df.loc[keep].reset_index(drop=True)
    Xc = Xc.copy()
    Xc["llm_domain"] = ann_df["llm_domain"].values
    Xc["llm_confidence"] = ann_df["llm_confidence"].values
    Xc["domain_score"] = ann_df["domain_score"].values
    print(f"6 algoritmos completos: {len(Y)} datasets | sementes: {SEEDS}\n")

    variants = {
        "+dominio_LLM": (["llm_confidence"], ["llm_domain"]),
        "+domain_score_orig": (["domain_score"], None),
    }
    rows = []
    for vname, (ncols, ccols) in variants.items():
        print(f"--- baseline vs {vname} (regret, menor=melhor) ---")
        print(f"  {'seed':>6s} {'reg_base':>9s} {'reg_var':>9s} {'delta':>9s} {'p_wil':>7s} {'reduz':>7s}")
        for seed in SEEDS:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            base_pred = cross_val_predict(make_reg(numeric_cols), Xc, Y, cv=kf, n_jobs=-1)
            var_pred = cross_val_predict(make_reg(numeric_cols, ncols, ccols), Xc, Y, cv=kf, n_jobs=-1)
            rb, rv = regret_vec(Y, base_pred), regret_vec(Y, var_pred)
            d = rv - rb  # negativo = dominio reduz regret (bom)
            try: _, pw = stats.wilcoxon(rv, rb)
            except ValueError: pw = np.nan
            n_reduz = int((d < 0).sum())
            print(f"  {seed:>6d} {rb.mean():>9.4f} {rv.mean():>9.4f} {d.mean():>+9.4f} {pw:>7.3f} {n_reduz:>3d}/{len(d)}")
            rows.append({"variant": vname, "seed": seed, "regret_base": rb.mean(),
                         "regret_var": rv.mean(), "delta": d.mean(), "p_wilcoxon": pw})
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "ranking_replication_summary.csv", index=False)
    print("=== REPLICACAO (delta<0 reduz regret E p<0.05 em TODAS as sementes) ===")
    for v, g in summary.groupby("variant"):
        rep = (g["delta"] < 0).all() and (g["p_wilcoxon"] < 0.05).all()
        print(f"  {v:22s} reduz regret sempre? {'SIM' if (g['delta']<0).all() else 'NAO'} | "
              f"sig nas {len(g)}? {'SIM' if (g['p_wilcoxon']<0.05).all() else 'NAO'} | "
              f"deltas {[round(x,4) for x in g['delta']]}")
    print(f"\nSalvo: {OUTPUT_DIR / 'ranking_replication_summary.csv'}")


if __name__ == "__main__":
    main()
