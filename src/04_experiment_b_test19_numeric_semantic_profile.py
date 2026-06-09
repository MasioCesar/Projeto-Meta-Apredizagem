from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
INPUT_MATRIX = DATA_DIR / __import__("os").environ.get("MAB_PERF", "performance_matrix.csv")
OUTPUT_RESULTS = DATA_DIR / "experiment_b_test19_numeric_semantic_profile.csv"


class NumericSemanticProfile(BaseEstimator, TransformerMixin):
    """Converte texto em numeros agregados, sem vocabulário de palavras."""

    def __init__(self, mode="basic"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [
            self._extract_features(text)
            for text in pd.Series(X).fillna("").astype(str)
        ]
        return np.asarray(rows, dtype=float)

    def _extract_features(self, text):
        tokens = re.findall(r"[A-Za-z0-9_]+", text)
        lower_tokens = [token.lower() for token in tokens]

        token_count = len(lower_tokens)
        char_count = len(text)

        if token_count == 0:
            return np.zeros(self._n_features(), dtype=float)

        unique_count = len(set(lower_tokens))
        token_lengths = np.asarray([len(token) for token in lower_tokens], dtype=float)

        digit_tokens = sum(any(char.isdigit() for char in token) for token in lower_tokens)
        alpha_tokens = sum(token.isalpha() for token in lower_tokens)
        mixed_tokens = token_count - digit_tokens - alpha_tokens
        underscore_tokens = sum("_" in token for token in lower_tokens)
        gene_probe_like = sum(
            bool(re.match(r"^\d+_?[a-z]*_?at$", token))
            or token.startswith("affx")
            for token in lower_tokens
        )

        boilerplate = {
            "author", "source", "please", "cite", "dataset", "datasets",
            "repository", "machine", "learning", "classification",
        }
        boilerplate_tokens = sum(token in boilerplate for token in lower_tokens)

        frequencies = pd.Series(lower_tokens).value_counts(normalize=True)
        entropy = -sum(p * math.log2(p) for p in frequencies)

        basic = [
            math.log1p(char_count),
            math.log1p(token_count),
            math.log1p(unique_count),
            unique_count / token_count,
            float(token_lengths.mean()),
            float(token_lengths.std()),
            float(np.median(token_lengths)),
            digit_tokens / token_count,
            alpha_tokens / token_count,
            mixed_tokens / token_count,
            underscore_tokens / token_count,
            gene_probe_like / token_count,
            boilerplate_tokens / token_count,
            entropy,
        ]

        if self.mode == "basic":
            return basic

        punctuation_count = sum(not char.isalnum() and not char.isspace() for char in text)
        whitespace_count = sum(char.isspace() for char in text)
        numeric_only_tokens = sum(token.isdigit() for token in lower_tokens)
        short_tokens = sum(len(token) <= 3 for token in lower_tokens)
        long_tokens = sum(len(token) >= 12 for token in lower_tokens)
        url_markers = len(re.findall(r"https?|www|\.com|\.org|\.edu", text.lower()))
        year_like_tokens = sum(bool(re.match(r"^(19|20)\d\d$", token)) for token in lower_tokens)

        extended = [
            punctuation_count / max(char_count, 1),
            whitespace_count / max(char_count, 1),
            numeric_only_tokens / token_count,
            short_tokens / token_count,
            long_tokens / token_count,
            url_markers / token_count,
            year_like_tokens / token_count,
        ]

        return basic + extended

    def _n_features(self):
        if self.mode == "basic":
            return 14
        return 21


def load_experiment_data():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_METAFEATURES}")

    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_MATRIX}")

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

    if not existing_classifier_cols:
        raise RuntimeError("Nenhum classificador alvo foi encontrado.")

    performance_matrix["best_classifier"] = (
        performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
    )
    performance_matrix["best_accuracy"] = (
        performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
    )

    performance_matrix = performance_matrix.dropna(
        subset=["best_classifier", "best_accuracy"]
    ).copy()

    df_exp = df_meta.merge(performance_matrix, on="did", how="inner")

    all_possible_classifier_cols = [
        "DecisionTree", "SVM", "KNN", "LogisticRegression", "Perceptron", "MLP"
    ]

    ignore_cols = [
        "did",
        "name",
        "domain",
        "best_classifier",
        "best_accuracy",
    ] + all_possible_classifier_cols

    X = df_exp.drop(columns=ignore_cols, errors="ignore").copy()

    if "semantic_text" not in X.columns:
        X["semantic_text"] = ""
    X["semantic_text"] = X["semantic_text"].fillna("").astype(str)

    if "predicted_domain" not in X.columns:
        X["predicted_domain"] = "unknown"
    X["predicted_domain"] = X["predicted_domain"].fillna("unknown").astype(str)

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain"]
    ]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain"]
    ]

    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X[numeric_cols] = X[numeric_cols].mask(X[numeric_cols].abs() > 1e12, np.nan)

    nunique = X[numeric_cols].nunique(dropna=True)
    numeric_cols = nunique[nunique > 1].index.tolist()

    X = X[numeric_cols + ["semantic_text", "predicted_domain"]]
    y = df_exp["best_classifier"].astype(str)

    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    valid_mask = y.isin(valid_classes)

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y, numeric_cols, existing_classifier_cols


def build_model(numeric_cols, include_domain, semantic_mode):
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
            "semantic_numeric",
            Pipeline([
                ("profile", NumericSemanticProfile(mode=semantic_mode)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            "semantic_text",
        ),
    ]

    if include_domain:
        transformers.insert(
            1,
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                ["predicted_domain"],
            ),
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def main():
    X, y, numeric_cols, classifier_cols = load_experiment_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Classes insuficientes para validacao cruzada.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    experiments = [
        {
            "strategy": "stat_plus_domain_plus_text_numeric_basic",
            "include_domain": True,
            "semantic_mode": "basic",
        },
        {
            "strategy": "stat_plus_domain_plus_text_numeric_extended",
            "include_domain": True,
            "semantic_mode": "extended",
        },
        {
            "strategy": "stat_plus_text_numeric_basic_no_domain",
            "include_domain": False,
            "semantic_mode": "basic",
        },
        {
            "strategy": "stat_plus_text_numeric_extended_no_domain",
            "include_domain": False,
            "semantic_mode": "extended",
        },
    ]

    print("\nEXPERIMENTO B - TESTE 19: TEXTO SEMANTICO COMO NUMEROS")
    print("Nao usa palavras cruas, TF-IDF, vocabulario ou selecao baseada no alvo.")
    print("Classificadores alvo:", classifier_cols)
    print("Datasets usados:", len(X))
    print("-" * 92)
    print(f"{'estrategia':<45} | {'acc':>8} | {'f1_macro':>8} | {'f1_weighted':>11}")
    print("-" * 92)

    summaries = []

    for experiment in experiments:
        model = build_model(
            numeric_cols=numeric_cols,
            include_domain=experiment["include_domain"],
            semantic_mode=experiment["semantic_mode"],
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
            "experiment": "B_Test19_Numeric_Semantic_Profile",
            "strategy": experiment["strategy"],
            "features": "Stat + Numeric Semantic Profile"
            + (" + Domain" if experiment["include_domain"] else ""),
            "semantic_mode": experiment["semantic_mode"],
            "uses_raw_words": False,
            "uses_tfidf": False,
            "uses_target_for_text_selection": False,
            "base_classifiers_considered": ", ".join(classifier_cols),
            "datasets_used": len(X),
            "n_statistical_features": len(numeric_cols),
            "target_classes": ", ".join(label_encoder.classes_),
            "accuracy_mean": results["test_accuracy"].mean(),
            "accuracy_std": results["test_accuracy"].std(),
            "f1_macro_mean": results["test_f1_macro"].mean(),
            "f1_macro_std": results["test_f1_macro"].std(),
            "f1_weighted_mean": results["test_f1_weighted"].mean(),
            "f1_weighted_std": results["test_f1_weighted"].std(),
            "n_splits": n_splits,
        }
        summaries.append(row)

        print(
            f"{row['strategy']:<45} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['f1_macro_mean']:.4f} | "
            f"{row['f1_weighted_mean']:.4f}"
        )

    final_df = pd.DataFrame(summaries).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    final_df.to_csv(OUTPUT_RESULTS, index=False)

    print("-" * 92)
    print("\nRanking por F1-Macro:")
    print(final_df[["strategy", "accuracy_mean", "f1_macro_mean"]].to_string(index=False))
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
