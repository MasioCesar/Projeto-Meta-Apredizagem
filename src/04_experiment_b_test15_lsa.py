from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
INPUT_MATRIX = DATA_DIR / __import__("os").environ.get("MAB_PERF", "performance_matrix.csv")

OUTPUT_RESULTS = DATA_DIR / "experiment_b_test15_lsa.csv"


def main():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_METAFEATURES}")

    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_MATRIX}")

    df_meta = pd.read_csv(INPUT_METAFEATURES)
    performance_matrix = pd.read_csv(INPUT_MATRIX)

    classifier_cols = [
        "DecisionTree",
        "KNN",
        "LogisticRegression",
        "MLP",
        "Perceptron",
        "SVM"
    ]

    existing_classifier_cols = [
        col for col in classifier_cols
        if col in performance_matrix.columns
    ]

    performance_matrix["best_classifier"] = (
        performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
    )
    performance_matrix["best_accuracy"] = (
        performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
    )

    performance_matrix = performance_matrix.dropna(
        subset=["best_classifier", "best_accuracy"]
    ).copy()

    df_exp_b = df_meta.merge(
        performance_matrix,
        on="did",
        how="inner"
    )

    all_possible_classifier_cols = [
        "DecisionTree", "SVM", "KNN", "LogisticRegression", "Perceptron", "MLP"
    ]

    ignore_cols = [
        "did", "name", "domain", "best_classifier", "best_accuracy"
    ] + all_possible_classifier_cols

    X_b = df_exp_b.drop(columns=ignore_cols, errors="ignore").copy()

    if "semantic_text" not in X_b.columns:
        X_b["semantic_text"] = ""
    X_b["semantic_text"] = X_b["semantic_text"].fillna("").astype(str)

    if "predicted_domain" not in X_b.columns:
        X_b["predicted_domain"] = "unknown"
    X_b["predicted_domain"] = X_b["predicted_domain"].fillna("unknown").astype(str)

    numeric_cols = [col for col in X_b.columns if col not in ["semantic_text", "predicted_domain"]]

    for col in numeric_cols:
        X_b[col] = pd.to_numeric(X_b[col], errors="coerce")

    X_b = X_b.dropna(axis=1, how="all")

    numeric_cols = [col for col in X_b.columns if col not in ["semantic_text", "predicted_domain"]]

    X_b[numeric_cols] = X_b[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_b[numeric_cols] = X_b[numeric_cols].mask(X_b[numeric_cols].abs() > 1e12, np.nan)

    nunique = X_b[numeric_cols].nunique(dropna=True)
    valid_numeric_cols = nunique[nunique > 1].index.tolist()

    X_b = X_b[valid_numeric_cols + ["semantic_text", "predicted_domain"]]
    numeric_cols = valid_numeric_cols

    y_b = df_exp_b["best_classifier"].astype(str)

    class_counts = y_b.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    valid_mask = y_b.isin(valid_classes)

    X_b = X_b.loc[valid_mask].reset_index(drop=True)
    y_b = y_b.loc[valid_mask].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_b)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "statistical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]),
                ["predicted_domain"]
            ),
            (
                "semantic",
                Pipeline([
                    # Extrai unigramas e bigramas (sem limite)
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                    # LSA: Reduz todas as milhares de palavras para os 20 principais "Conceitos" (Tópicos Latentes)
                    ("lsa", TruncatedSVD(n_components=20, random_state=42))
                ]),
                "semantic_text"
            )
        ],
        remainder="drop"
    )

    meta_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
    ])

    results = cross_validate(
        meta_model, X_b, y_encoded, cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"},
        return_train_score=False, error_score="raise"
    )

    summary = pd.DataFrame([{
        "experiment": "B_Test15_LSA",
        "features": "Stat(63) + Dom + LSA(20 Conceitos)",
        "accuracy_mean": results["test_accuracy"].mean(),
        "accuracy_std": results["test_accuracy"].std(),
        "f1_macro_mean": results["test_f1_macro"].mean(),
        "f1_macro_std": results["test_f1_macro"].std()
    }])

    summary.to_csv(OUTPUT_RESULTS, index=False)
    print("\nEXPERIMENTO B - TESTE 15: ANÁLISE SEMÂNTICA LATENTE (LSA)")
    print("Extraindo os 20 principais 'conceitos' do texto em vez de palavras isoladas.")
    print("Accuracy:", summary.loc[0, "accuracy_mean"], "+/-", summary.loc[0, "accuracy_std"])
    print("F1 macro:", summary.loc[0, "f1_macro_mean"], "+/-", summary.loc[0, "f1_macro_std"])
    print(f"Salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
