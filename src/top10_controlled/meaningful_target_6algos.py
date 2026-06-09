"""
Alvo BEM-POSTO: "vale a pena um modelo COMPLEXO?"

O alvo best_classifier e ruido: 64% dos datasets tem melhor vs 2o-melhor a <1pp.
Aqui redefinimos a pergunta de meta-aprendizagem para algo decidivel e util:

  label = "complex_worth" se  max(complexo) - max(simples) > margem
          "simple_enough" caso contrario

  simples  = {DecisionTree, LogisticRegression, Perceptron}
  complexo = {SVM, MLP, KNN}

Quando nao ha diferenca pratica, a escolha parcimoniosa (modelo simples) e a
correta -> isso REMOVE o ruido de empates em vez de classificar moeda-ao-ar.

Testa varias margens e mede se a semantica (interacao modalidade x escala)
agrega, com CV repetida e teste-t pareado. NAO altera meta-features estatisticas.
Saida: data/top10_controlled/meaningful_target_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data
from explore_6algos import build

SIMPLE = ["DecisionTree", "LogisticRegression", "Perceptron"]
COMPLEX = ["SVM", "MLP", "KNN"]
MARGINS = [0.0, 0.01, 0.02, 0.03]

STRATEGIES = [
    ("baseline_stat_only", None),
    ("stat_plus_interaction", "interaction"),
    ("stat_plus_modality_interaction", "modality_interaction"),
    ("stat_plus_mod_inter_clusters", "modality_interaction_clusters"),
]


def main():
    X, y, numeric_cols, _ = load_data()
    # Reconstroi o gap na MESMA ordem do load_data (df_meta.merge(perf, on='did')).
    df_meta = pd.read_csv(INPUT_METAFEATURES)
    perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)
    assert len(gap) == len(X), f"alinhamento falhou: gap={len(gap)} X={len(X)}"

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []
    for margin in MARGINS:
        label_target = np.where(gap > margin, "complex_worth", "simple_enough")
        dist = dict(pd.Series(label_target).value_counts())
        if len(dist) < 2 or min(dist.values()) < 10:
            print(f"\n[margem {margin}] desbalanceado demais {dist} - pulado")
            continue
        yenc = LabelEncoder().fit_transform(label_target)
        print(f"\n===== margem {margin} | {dist} =====")
        scores = {}
        for lbl, mode in STRATEGIES:
            f1 = cross_val_score(build(numeric_cols, mode), X, yenc, cv=rskf, scoring="f1_macro")
            acc = cross_val_score(build(numeric_cols, mode), X, yenc, cv=rskf, scoring="accuracy")
            scores[lbl] = f1
            rows.append({"margin": margin, "dist": str(dist), "strategy": lbl,
                         "acc_mean": acc.mean(), "f1_macro_mean": f1.mean(),
                         "f1_macro_std": f1.std()})
            print(f"  {lbl:32s} acc {acc.mean():.4f}  F1m {f1.mean():.4f} +/- {f1.std():.4f}")
        base = scores["baseline_stat_only"]
        for lbl in ["stat_plus_interaction", "stat_plus_modality_interaction",
                    "stat_plus_mod_inter_clusters"]:
            d = scores[lbl] - base
            t, p = stats.ttest_rel(scores[lbl], base)
            print(f"  >> {lbl:32s} delta {d.mean():+.4f} | vence {int((d>0).sum())}/{len(d)} | p={p:.4f}")

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "meaningful_target_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
