"""
TESTE DE RANKING/REGRESSAO: em vez de prever o 'best_classifier' (argmax ruidoso),
preve a PERFORMANCE de cada algoritmo (regressao multi-saida) e avalia por
ranking (Spearman) e REGRET. Mais robusto a empates; da chance justa a semantica.

Compara, em 3 algoritmos (limpo) e 6 algoritmos (subset completo de 107):
  - baseline (so estatistica)
  - + dominio-LLM (categoria + confianca)
  - + domain_score original (referencia confundida)
Baselines triviais: ORACULO (teto), SBA (single best algorithm), ALEATORIO.
Multi-semente, Wilcoxon pareado no regret por dataset.
Saida: data/top10_controlled/ranking_regression_summary.csv
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
THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
SEEDS = [42, 7, 123]
ANN = INPUT_METAFEATURES.parent / "llm_domain_annotations.csv"


def make_reg(numeric_cols, num_extra=None, cat_extra=None):
    t = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols)]
    if num_extra:
        t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_extra))
    if cat_extra:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_extra))
    return Pipeline([("pre", ColumnTransformer(t, remainder="drop")),
                     ("reg", MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42)))])


def eval_ranking(Y_true, Y_pred):
    """Por dataset: Spearman(pred,true), regret (pp), hit top-1."""
    sp, reg, hit = [], [], []
    for t, p in zip(Y_true, Y_pred):
        if np.all(np.isnan(t)):
            continue
        mask = ~np.isnan(t)
        tt, pp = t[mask], p[mask]
        if len(tt) < 2:
            continue
        rho = stats.spearmanr(pp, tt).correlation
        sp.append(0.0 if np.isnan(rho) else rho)
        reg.append(tt.max() - tt[np.argmax(pp)])      # regret em acuracia
        hit.append(int(np.argmax(pp) == np.argmax(tt)))
    return np.array(sp), np.array(reg), np.array(hit)


def run(label_set, cols, X, Y, numeric_cols, ann_df):
    """Retorna dict de metricas medias sobre sementes + regret por dataset (ultima seed) p/ teste."""
    Xc = X.copy().reset_index(drop=True)
    Xc["llm_domain"] = ann_df["llm_domain"].values
    Xc["llm_confidence"] = ann_df["llm_confidence"].values
    Xc["domain_score"] = ann_df["domain_score"].values
    feature_variants = {
        "baseline_stat": (None, None),
        "+dominio_LLM": (["llm_confidence"], ["llm_domain"]),
        "+domain_score_orig": (["domain_score"], None),
    }
    results = {}
    regret_per_ds = {}
    for fname, (ncols, ccols) in feature_variants.items():
        sps, regs, hits = [], [], []
        last_reg = None
        for seed in SEEDS:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            model = make_reg(numeric_cols, ncols, ccols)
            Ypred = cross_val_predict(model, Xc, Y, cv=kf, n_jobs=-1)
            sp, reg, hit = eval_ranking(Y, Ypred)
            sps.append(sp.mean()); regs.append(reg.mean()); hits.append(hit.mean())
            last_reg = reg
        results[fname] = {"spearman": np.mean(sps), "regret": np.mean(regs), "hit": np.mean(hits)}
        regret_per_ds[fname] = last_reg
    return results, regret_per_ds


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    ann = pd.read_csv(ANN)
    ann_df = df_exp[["did"]].merge(ann, on="did", how="left")
    ann_df["domain_score"] = df_exp["domain_score"].values
    ann_df["llm_domain"] = ann_df["llm_domain"].fillna("unknown").astype(str)
    ann_df["llm_confidence"] = ann_df["llm_confidence"].fillna(3).astype(float)

    rows = []
    for setname, cols in [("3_algoritmos", THREE), ("6_algoritmos_completos", ALL6)]:
        Y = df_exp[cols].to_numpy(dtype=float)
        if setname.startswith("6"):
            keep = ~np.isnan(Y).any(axis=1)
            Yc, Xc, annc = Y[keep], X.loc[keep].reset_index(drop=True), ann_df.loc[keep].reset_index(drop=True)
        else:
            Yc, Xc, annc = Y, X, ann_df
        print(f"\n===== {setname} ({len(Yc)} datasets, {len(cols)} algos) =====")

        # baselines triviais (regret)
        oracle = 0.0
        sba_idx = np.nanmean(Yc, axis=0).argmax()
        sba_regret = np.nanmean(Yc.max(axis=1) - Yc[:, sba_idx])
        rng = np.random.default_rng(0)
        rand_regret = np.mean([Yc[i].max() - Yc[i][rng.integers(len(cols))] for i in range(len(Yc))])
        print(f"  ORACULO regret=0.0000 | SBA ({cols[sba_idx]}) regret={sba_regret:.4f} | ALEATORIO regret={rand_regret:.4f}")

        res, regret_ds = run(setname, cols, Xc, Yc, numeric_cols, annc)
        base_reg = regret_ds["baseline_stat"]
        print(f"  {'feature set':22s} {'Spearman':>9s} {'regret':>8s} {'hit@1':>7s} {'vs base p(reg)':>14s}")
        for fname, m in res.items():
            extra = ""
            if fname != "baseline_stat":
                d = regret_ds[fname] - base_reg
                try: _, pw = stats.wilcoxon(regret_ds[fname], base_reg)
                except ValueError: pw = np.nan
                extra = f"  d={d.mean():+.4f} p={pw:.3f}"
            print(f"  {fname:22s} {m['spearman']:>9.3f} {m['regret']:>8.4f} {m['hit']:>7.3f}{extra}")
            rows.append({"set": setname, "feature_set": fname, **m,
                         "sba_regret": sba_regret, "oracle_regret": 0.0})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ranking_regression_summary.csv", index=False)
    print(f"\nSalvo: {OUTPUT_DIR / 'ranking_regression_summary.csv'}")


if __name__ == "__main__":
    main()
