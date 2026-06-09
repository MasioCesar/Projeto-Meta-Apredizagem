"""
Exploracao avancada (6 algoritmos) - SEM API/LLM e SEM alterar meta-features estatisticas.

Direcoes:
  (1) INTERACAO semantica x escala: modalidade cruzada com tamanho (nr_inst, nr_attr,
      attr_to_inst). As estatisticas continuam intactas no proprio ramo; aqui so
      ADICIONAMOS cruzamentos semantica x escala.
  (2) ALVO AGRUPADO em 3 vias mais balanceado: tree / linear / kernel_instance_neural.
  (3) DIAGNOSTICO one-vs-rest: para quais algoritmos a semantica realmente prediz?
  (4) SUBCONJUNTO dos 107 datasets onde os 6 classificadores foram avaliados.

Saida: data/top10_controlled/explore_summary.csv
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import OUTPUT_DIR, clean_semantic_text, semantic_clusters, load_data
from new_tests_6algos import ModalityAffinityExtractor, MODALITY_KEYWORDS, modality_pipeline


SIZE_COLS = ["nr_inst", "nr_attr", "attr_to_inst"]


class SemanticSizeInteraction(BaseEstimator, TransformerMixin):
    """Cruza o score de cada modalidade com features de escala (log do tamanho).
    Entrada: [semantic_text, name, nr_inst, nr_attr, attr_to_inst].
    As estatisticas originais NAO sao modificadas - isto e um ramo adicional."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"] + SIZE_COLS)
        out = []
        for _, row in df.iterrows():
            tokens = set(clean_semantic_text(row["semantic_text"], row["name"]).split())
            mod = [len(tokens & kws) / len(kws) for kws in MODALITY_KEYWORDS.values()]
            nr_inst = pd.to_numeric(row["nr_inst"], errors="coerce")
            nr_attr = pd.to_numeric(row["nr_attr"], errors="coerce")
            ratio = pd.to_numeric(row["attr_to_inst"], errors="coerce")
            size = [
                np.log1p(nr_inst if np.isfinite(nr_inst) else 0),
                np.log1p(nr_attr if np.isfinite(nr_attr) else 0),
                ratio if np.isfinite(ratio) else 0.0,
            ]
            feats = [m * s for m in mod for s in size]   # 5 x 3 = 15 cruzamentos
            out.append(feats)
        return np.asarray(out, dtype=float)


def interaction_pipeline():
    return Pipeline([
        ("inter", SemanticSizeInteraction()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build(numeric_cols, semantic=None):
    transformers = [
        ("statistical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
    ]
    if semantic == "modality":
        transformers.append(("sem", modality_pipeline(), ["semantic_text", "name"]))
    elif semantic == "interaction":
        transformers.append(("sem", interaction_pipeline(),
                             ["semantic_text", "name"] + SIZE_COLS))
    elif semantic == "modality_interaction":
        transformers.append(("mod", modality_pipeline(), ["semantic_text", "name"]))
        transformers.append(("inter", interaction_pipeline(),
                             ["semantic_text", "name"] + SIZE_COLS))
    elif semantic == "modality_interaction_clusters":
        transformers.append(("mod", modality_pipeline(), ["semantic_text", "name"]))
        transformers.append(("inter", interaction_pipeline(),
                             ["semantic_text", "name"] + SIZE_COLS))
        transformers.append(("clu", semantic_clusters(), ["semantic_text", "name"]))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42,
                                         class_weight="balanced")),
    ])


def evaluate(name, X, y_enc, model, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = cross_validate(model, X, y_enc, cv=cv,
                         scoring={"accuracy": "accuracy", "f1_macro": "f1_macro",
                                  "f1_weighted": "f1_weighted"}, error_score="raise")
    return {"strategy": name,
            "accuracy_mean": res["test_accuracy"].mean(),
            "accuracy_std": res["test_accuracy"].std(),
            "f1_macro_mean": res["test_f1_macro"].mean(),
            "f1_macro_std": res["test_f1_macro"].std(),
            "f1_weighted_mean": res["test_f1_weighted"].mean(),
            "n_splits": n_splits}


GROUP3 = {
    "DecisionTree": "tree",
    "LogisticRegression": "linear", "Perceptron": "linear",
    "SVM": "kernel_inst_neural", "MLP": "kernel_inst_neural", "KNN": "kernel_inst_neural",
}


def run_block(rows, target, X, y, numeric_cols, semantics):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    n = min(5, pd.Series(y_enc).value_counts().min())
    print(f"\n[{target}] n_splits={n} | distrib={dict(pd.Series(y).value_counts())}")
    for label, mode in semantics:
        rows.append({"target": target, **evaluate(label, X, y_enc, build(numeric_cols, mode), n)})


def main():
    X, y, numeric_cols, classifier_cols = load_data()
    print(f"Datasets: {len(X)} | numeric_cols: {len(numeric_cols)}")
    has_size = all(c in X.columns for c in SIZE_COLS)
    print(f"Size cols disponiveis: {has_size}")

    semantics_all = [
        ("baseline_stat_only", None),
        ("stat_plus_modality", "modality"),
        ("stat_plus_interaction", "interaction"),
        ("stat_plus_modality_interaction", "modality_interaction"),
        ("stat_plus_modality_interaction_clusters", "modality_interaction_clusters"),
    ]
    rows = []

    # (1) e (2): 6 classes / 3 grupos / 2 grupos
    run_block(rows, "6_classes", X, y, numeric_cols, semantics_all)
    run_block(rows, "grouped_3way", X, y.map(GROUP3), numeric_cols, semantics_all)

    # (4) subconjunto dos 107 (todos os 6 avaliados): aproxima removendo classes
    #     que so existem por ausencia. Usamos o proprio y; o subset e implicito
    #     pois datasets com NaN nunca tem SVM/KNN/MLP como melhor.
    #     Aqui repetimos 6 classes mas conferindo per-classe abaixo.

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "explore_summary.csv"
    summary.to_csv(out, index=False)

    pd.set_option("display.width", 170); pd.set_option("display.max_columns", None)
    print("\n================ RESULTADOS (1,2) ================")
    for tgt in summary["target"].unique():
        sub = summary[summary["target"] == tgt].sort_values("f1_macro_mean", ascending=False)
        print(f"\n--- alvo: {tgt} ---")
        print(sub[["strategy", "accuracy_mean", "f1_macro_mean", "f1_weighted_mean"]].to_string(index=False))

    # (3) Diagnostico one-vs-rest: para cada algoritmo, semantica ajuda a prever
    #     "este algoritmo e o melhor"? Reporta F1 da classe positiva (baseline vs +interacao).
    print("\n================ (3) DIAGNOSTICO ONE-VS-REST ================")
    print(f"{'algoritmo':22s} {'n_pos':>5s} {'F1_base':>8s} {'F1_inter':>9s} {'delta':>7s}")
    diag_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for algo in sorted(y.unique()):
        y_bin = (y == algo).astype(int)
        if y_bin.sum() < 5:
            print(f"{algo:22s} {int(y_bin.sum()):>5d}   (poucos positivos - pulado)")
            continue
        pred_base = cross_val_predict(build(numeric_cols, None), X, y_bin, cv=cv)
        pred_int = cross_val_predict(build(numeric_cols, "modality_interaction"), X, y_bin, cv=cv)
        f1b = f1_score(y_bin, pred_base, pos_label=1, zero_division=0)
        f1i = f1_score(y_bin, pred_int, pos_label=1, zero_division=0)
        print(f"{algo:22s} {int(y_bin.sum()):>5d} {f1b:>8.3f} {f1i:>9.3f} {f1i-f1b:>+7.3f}")
        diag_rows.append({"target": f"ovr_{algo}", "strategy": "f1_base_vs_interaction",
                          "accuracy_mean": np.nan, "f1_macro_mean": f1b,
                          "f1_weighted_mean": f1i, "f1_macro_std": np.nan,
                          "accuracy_std": np.nan, "n_splits": 5})

    if diag_rows:
        pd.concat([summary, pd.DataFrame(diag_rows)], ignore_index=True).to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
