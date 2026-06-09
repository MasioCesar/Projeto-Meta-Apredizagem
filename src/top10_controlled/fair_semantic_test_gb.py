"""
TESTE JUSTO E FINAL: a semantica/dominio ajuda, com MODELO CONSTANTE nos dois lados?

Corrige os 3 artefatos das tabelas antigas:
  1. modelo diferente  -> usa Gradient Boosting (mesmo) em TODOS os bracos
  2. vazamento textual  -> usa so semantica LIMPA (tags/vocab fixo/clusters via SafeTextCleaner)
  3. semente unica      -> 3 sementes, CV 5x10, teste pareado

Alvo: 3 algoritmos (limpo). Compara baseline vs texto-limpo vs dominio vs ambos.
Saida: data/top10_controlled/fair_semantic_gb_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from common import INPUT_MATRIX, INPUT_METAFEATURES, OUTPUT_DIR, load_data, build_model

THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
SEEDS = [42, 7, 123]
GB = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)


def gb_model(numeric_cols, semantic_mode, domain_mode):
    """Reusa o ColumnTransformer do build_model (mesmas features/limpeza) e troca o
    classificador para Gradient Boosting -- igual nos dois lados."""
    base = build_model(numeric_cols, semantic_mode, domain_mode)
    ct = clone(base.named_steps["preprocess"])
    return Pipeline([("pre", ct), ("clf", GradientBoostingClassifier(**GB))])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["b6"] = perf[["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["b6"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    yenc = LabelEncoder().fit_transform(y3)
    print(f"Alvo 3 algoritmos | {len(X)} datasets | modelo: Gradient Boosting (constante)\n")

    configs = [
        ("baseline_stat", "none", "none"),
        ("+texto_limpo_tags_vocab", "tags_fixed_vocab", "none"),
        ("+texto_limpo_clusters", "clusters_only", "none"),
        ("+dominio(label+score)", "none", "label_score"),
        ("+texto_limpo+dominio", "tags_fixed_vocab", "label_score"),
    ]
    base_by_seed = {s: cross_val_score(gb_model(numeric_cols, "none", "none"), X, yenc,
                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=s),
                    scoring="f1_macro") for s in SEEDS}
    acc_by_seed = {s: cross_val_score(gb_model(numeric_cols, "none", "none"), X, yenc,
                   cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=s),
                   scoring="accuracy") for s in SEEDS}

    rows = []
    base_f1 = np.mean([base_by_seed[s].mean() for s in SEEDS])
    base_acc = np.mean([acc_by_seed[s].mean() for s in SEEDS])
    print(f"{'config':28s} {'Acc':>7s} {'F1m':>7s}  {'dF1':>8s} {'p(F1, 3 sementes)':>20s}")
    print(f"{'baseline_stat':28s} {base_acc:>7.3f} {base_f1:>7.3f}  {'-':>8s} {'-':>20s}")
    rows.append({"config": "baseline_stat", "acc": base_acc, "f1": base_f1, "delta_f1": 0, "pvals": ""})

    for name, sm, dm in configs[1:]:
        f1s, accs, deltas, pvals = [], [], [], []
        for s in SEEDS:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=s)
            f1 = cross_val_score(gb_model(numeric_cols, sm, dm), X, yenc, cv=rskf, scoring="f1_macro")
            acc = cross_val_score(gb_model(numeric_cols, sm, dm), X, yenc, cv=rskf, scoring="accuracy")
            f1s.append(f1.mean()); accs.append(acc.mean())
            d = f1 - base_by_seed[s]; deltas.append(d.mean())
            _, pw = stats.wilcoxon(f1, base_by_seed[s])
            pvals.append(pw)
        print(f"{name:28s} {np.mean(accs):>7.3f} {np.mean(f1s):>7.3f}  {np.mean(deltas):>+8.4f} "
              f"  {[round(p,3) for p in pvals]}")
        rows.append({"config": name, "acc": np.mean(accs), "f1": np.mean(f1s),
                     "delta_f1": float(np.mean(deltas)), "pvals": str([round(p,3) for p in pvals])})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fair_semantic_gb_summary.csv", index=False)
    print("\n=== veredito (ganho de F1 sobre baseline, mesmo modelo) ===")
    for r in rows[1:]:
        ps = eval(r["pvals"])
        sig_all = all(p < 0.05 for p in ps)
        pos = r["delta_f1"] > 0
        print(f"  {r['config']:28s} dF1={r['delta_f1']:+.4f} | positivo? {'SIM' if pos else 'NAO'} | "
              f"sig nas 3 sementes? {'SIM' if sig_all else 'NAO'}")
    print(f"\nSalvo: {OUTPUT_DIR / 'fair_semantic_gb_summary.csv'}")


if __name__ == "__main__":
    main()
