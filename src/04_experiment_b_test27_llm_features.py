from pathlib import Path
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import (  # noqa: E402
    OUTPUT_DIR,
    INPUT_METAFEATURES,
    INPUT_MATRIX,
    CLASSIFIER_COLS,
    build_model,
    load_data,
)

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402


INPUT_LLM_FEATURES = OUTPUT_DIR / "llm_features" / "llm_semantic_features.csv"
FALLBACK_LLM_FEATURES = Path(__file__).resolve().parents[1] / "data" / "llm_features" / "llm_semantic_features.csv"
OUTPUT_RESULTS = OUTPUT_DIR / "test27_llm_features.csv"


def load_data_with_llm():
    X, y, numeric_cols, classifier_cols = load_data()

    llm_path = INPUT_LLM_FEATURES if INPUT_LLM_FEATURES.exists() else FALLBACK_LLM_FEATURES
    if not llm_path.exists():
        raise FileNotFoundError(
            f"Features LLM nao encontradas: {llm_path}\n"
            "Execute primeiro: python src/04_experiment_b_test26_generate_llm_features.py"
        )

    df_meta = pd.read_csv(INPUT_METAFEATURES)[["did", "name"]]
    df_llm = pd.read_csv(llm_path)
    name_to_did = dict(zip(df_meta["name"].astype(str), df_meta["did"].astype(int)))

    X = X.copy()
    X["did_for_merge"] = X["name"].map(name_to_did)
    X = X.merge(df_llm.drop(columns=["name"], errors="ignore"), left_on="did_for_merge", right_on="did", how="left")

    X["llm_domain"] = X["llm_domain"].fillna("other").astype(str)
    X["llm_modality"] = X["llm_modality"].fillna("unknown").astype(str)
    X["llm_subtype"] = X["llm_subtype"].fillna("unknown").astype(str)
    X["llm_leakage_risk"] = X["llm_leakage_risk"].fillna("medium").astype(str)
    X["llm_tags"] = X["llm_tags"].fillna("").astype(str)
    X["llm_domain_confidence"] = pd.to_numeric(X["llm_domain_confidence"], errors="coerce").fillna(0)

    tag_values = sorted({
        tag
        for value in X["llm_tags"]
        for tag in str(value).split("|")
        if tag
    })
    for tag in tag_values:
        X[f"llm_tag_{tag}"] = X["llm_tags"].str.split("|").apply(lambda tags: float(tag in tags))

    llm_numeric_cols = ["llm_domain_confidence"] + [f"llm_tag_{tag}" for tag in tag_values]
    llm_categorical_cols = ["llm_domain", "llm_modality", "llm_subtype", "llm_leakage_risk"]

    return X, y, numeric_cols, classifier_cols, llm_numeric_cols, llm_categorical_cols


def build_llm_model(numeric_cols, llm_numeric_cols, llm_categorical_cols, include_top2_cluster=True):
    transformers = [
        (
            "statistical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ),
        (
            "llm_numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            llm_numeric_cols,
        ),
        (
            "llm_categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            llm_categorical_cols,
        ),
    ]

    if include_top2_cluster:
        top2_model = build_model(numeric_cols=[], semantic_mode="clusters_only", domain_mode="label_score_interaction")
        top2_preprocess = top2_model.named_steps["preprocess"]
        for name, transformer, cols in top2_preprocess.transformers:
            if name != "statistical":
                transformers.append((f"top2_{name}", transformer, cols))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def main():
    X, y, numeric_cols, classifier_cols, llm_numeric_cols, llm_categorical_cols = load_data_with_llm()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_splits = min(5, pd.Series(y_encoded).value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    experiments = [
        {
            "strategy": "stat_plus_llm_features",
            "include_top2_cluster": False,
        },
        {
            "strategy": "top2_plus_llm_features",
            "include_top2_cluster": True,
        },
    ]

    print("\nEXPERIMENTO B - TESTE 27: FEATURES LLM")
    print("Usa JSON estruturado gerado por LLM local/Ollama.")
    print("-" * 78)
    print(f"{'estrategia':<28} | {'top2_cluster':<12} | {'acc':>8} | {'f1_macro':>8}")
    print("-" * 78)

    rows = []
    for experiment in experiments:
        model = build_llm_model(
            numeric_cols=numeric_cols,
            llm_numeric_cols=llm_numeric_cols,
            llm_categorical_cols=llm_categorical_cols,
            include_top2_cluster=experiment["include_top2_cluster"],
        )

        results = cross_validate(
            model,
            X,
            y_encoded,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
            },
            return_train_score=False,
            error_score="raise",
        )

        row = {
            "experiment": "B_Test27_LLM_Features",
            "strategy": experiment["strategy"],
            "features": "Stat + LLM Structured Features"
            + (" + Top2 Cluster/Domain Interaction" if experiment["include_top2_cluster"] else ""),
            "base_classifiers_considered": ", ".join(classifier_cols),
            "datasets_used": len(X),
            "n_statistical_features_without_domain_score": len(numeric_cols),
            "llm_numeric_cols": ", ".join(llm_numeric_cols),
            "llm_categorical_cols": ", ".join(llm_categorical_cols),
            "target_classes": ", ".join(encoder.classes_),
            "accuracy_mean": results["test_accuracy"].mean(),
            "accuracy_std": results["test_accuracy"].std(),
            "f1_macro_mean": results["test_f1_macro"].mean(),
            "f1_macro_std": results["test_f1_macro"].std(),
            "f1_weighted_mean": results["test_f1_weighted"].mean(),
            "f1_weighted_std": results["test_f1_weighted"].std(),
            "n_splits": n_splits,
        }
        rows.append(row)

        print(
            f"{row['strategy']:<28} | "
            f"{str(experiment['include_top2_cluster']):<12} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['f1_macro_mean']:.4f}"
        )

    final_df = pd.DataFrame(rows).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_RESULTS, index=False)
    print("-" * 78)
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
