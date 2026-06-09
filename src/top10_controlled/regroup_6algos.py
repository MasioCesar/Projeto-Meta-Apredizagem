"""
Reagrupamento do alvo para lidar com classes raras (Perceptron=5, KNN=9) que
destroem o F1-macro. Testa varias granularidades e mede, com CV repetida e
teste-t pareado, se a semantica (interacao modalidade x escala) agrega.

NAO altera meta-features estatisticas. Saida: data/top10_controlled/regroup_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from common import OUTPUT_DIR, load_data
from explore_6algos import build


# Diferentes granularidades de alvo (do mais fino ao mais grosso)
GROUPINGS = {
    "6_classes": None,  # identidade
    "5_linear_merge": {  # funde Perceptron em linear (ambos lineares)
        "LogisticRegression": "linear", "Perceptron": "linear",
        "DecisionTree": "tree", "SVM": "svm", "MLP": "mlp", "KNN": "knn",
    },
    "4_families": {  # linear / tree / neural / (kernel+instancia)
        "LogisticRegression": "linear", "Perceptron": "linear",
        "DecisionTree": "tree", "MLP": "neural",
        "SVM": "kernel_instance", "KNN": "kernel_instance",
    },
    "3_way": {
        "DecisionTree": "tree",
        "LogisticRegression": "linear", "Perceptron": "linear",
        "SVM": "complex", "MLP": "complex", "KNN": "complex",
    },
    "2_simple_complex": {
        "DecisionTree": "simple", "LogisticRegression": "simple", "Perceptron": "simple",
        "SVM": "complex", "MLP": "complex", "KNN": "complex",
    },
}

STRATEGIES = [
    ("baseline_stat_only", None),
    ("stat_plus_interaction", "interaction"),
    ("stat_plus_modality_interaction", "modality_interaction"),
]


def main():
    X, y, numeric_cols, _ = load_data()
    print(f"Datasets: {len(X)} | numeric_cols: {len(numeric_cols)}")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    rows = []
    for gname, mapping in GROUPINGS.items():
        yg = y if mapping is None else y.map(mapping)
        yenc = LabelEncoder().fit_transform(yg)
        dist = dict(pd.Series(yg).value_counts())
        print(f"\n===== {gname} | {len(dist)} classes | {dist} =====")

        scores = {}
        for label, mode in STRATEGIES:
            s = cross_val_score(build(numeric_cols, mode), X, yenc, cv=rskf, scoring="f1_macro")
            sa = cross_val_score(build(numeric_cols, mode), X, yenc, cv=rskf, scoring="accuracy")
            scores[label] = s
            rows.append({
                "grouping": gname, "n_classes": len(dist), "strategy": label,
                "acc_mean": sa.mean(), "acc_std": sa.std(),
                "f1_macro_mean": s.mean(), "f1_macro_std": s.std(),
            })
            print(f"  {label:34s} acc {sa.mean():.4f}  F1m {s.mean():.4f} +/- {s.std():.4f}")

        # teste pareado: melhor semantica vs baseline
        base = scores["baseline_stat_only"]
        best_sem = scores["stat_plus_modality_interaction"]
        delta = best_sem - base
        wins = int((delta > 0).sum())
        t, p = stats.ttest_rel(best_sem, base)
        print(f"  >> delta pareado (modality_interaction - baseline): {delta.mean():+.4f} | "
              f"vence {wins}/{len(delta)} folds | p={p:.4f}")
        rows[-1]["paired_delta_vs_baseline"] = float(delta.mean())
        rows[-1]["paired_p_value"] = float(p)
        rows[-1]["folds_won"] = f"{wins}/{len(delta)}"

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "regroup_summary.csv"
    summary.to_csv(out, index=False)

    pd.set_option("display.width", 180); pd.set_option("display.max_columns", None)
    print("\n================ TABELA FINAL ================")
    print(summary[["grouping", "n_classes", "strategy", "acc_mean", "f1_macro_mean",
                   "f1_macro_std"]].to_string(index=False))
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
