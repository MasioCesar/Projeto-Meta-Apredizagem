from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
INPUT_MATRIX = DATA_DIR / __import__("os").environ.get("MAB_PERF", "performance_matrix.csv")
OUTPUT_RESULTS = DATA_DIR / "experiment_b_test18_semantic_strategies.csv"


class SemanticTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_digit_tokens=False, max_tokens=None):
        self.remove_digit_tokens = remove_digit_tokens
        self.max_tokens = max_tokens
        self.noise_words = {
            "author", "source", "unknown", "date", "please", "cite", "https",
            "www", "org", "edu", "com", "dataset", "datasets", "data",
            "repository", "machine", "learning", "classification", "task",
            "attribute", "attributes", "feature", "features", "openml",
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in pd.Series(X).fillna("").astype(str)]

    def _clean(self, text):
        text = text.lower()
        text = re.sub(r"https?\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9_ -]+", " ", text)

        tokens = []
        for token in text.split():
            token = token.strip("_-")
            if len(token) < 3:
                continue
            if token in self.noise_words:
                continue
            if self.remove_digit_tokens and any(char.isdigit() for char in token):
                continue
            tokens.append(token)
            if self.max_tokens is not None and len(tokens) >= self.max_tokens:
                break

        return " ".join(tokens)


class SemanticProfileExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keyword_groups = {
            "biology": {
                "gene", "genes", "protein", "proteins", "molecular", "biology",
                "biological", "bioresponse", "breast", "lung", "ovary", "kidney",
                "cancer", "tumor", "patient", "patients", "clinical", "chemical",
                "molecule", "molecules", "hepatitis", "dna", "rna",
            },
            "text": {
                "text", "document", "documents", "word", "words", "sentence",
                "sentences", "topic", "topics", "news", "review", "reviews",
                "language", "spam", "email",
            },
            "vision": {
                "image", "images", "pixel", "pixels", "object", "objects",
                "digit", "digits", "mnist", "face", "faces", "visual",
            },
            "business": {
                "price", "prices", "cost", "market", "sales", "customer",
                "customers", "income", "credit", "bank", "financial", "loan",
            },
            "generic_ml": {
                "classification", "class", "classes", "attribute", "attributes",
                "feature", "features", "instance", "instances", "missing",
                "nominal", "numeric",
            },
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in pd.Series(X).fillna("").astype(str):
            rows.append(self._profile(text))
        return np.asarray(rows, dtype=float)

    def _profile(self, text):
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", lowered)
        token_count = len(tokens)
        if token_count == 0:
            return [0.0] * (9 + len(self.keyword_groups))

        unique_count = len(set(tokens))
        digit_count = sum(any(char.isdigit() for char in token) for token in tokens)
        alpha_count = sum(token.isalpha() for token in tokens)
        long_count = sum(len(token) >= 12 for token in tokens)
        probe_count = sum(bool(re.match(r"^\d+_[a-z]?_?at$", token)) for token in tokens)
        boilerplate_count = sum(token in {"author", "source", "please", "cite", "dataset"} for token in tokens)
        avg_len = sum(len(token) for token in tokens) / token_count

        features = [
            np.log1p(token_count),
            np.log1p(unique_count),
            unique_count / token_count,
            digit_count / token_count,
            alpha_count / token_count,
            long_count / token_count,
            probe_count / token_count,
            avg_len,
            boilerplate_count / token_count,
        ]

        token_set = set(tokens)
        for keywords in self.keyword_groups.values():
            features.append(len(token_set.intersection(keywords)) / len(keywords))

        return features


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


def semantic_strategies():
    return {
        "raw_tfidf_20_test1_control": TfidfVectorizer(
            max_features=20,
            ngram_range=(1, 1),
            min_df=2,
        ),
        "clean_tfidf_20_no_boilerplate": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=False)),
            ("tfidf", TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2)),
        ]),
        "clean_words_only_tfidf_20": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=True)),
            ("tfidf", TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2)),
        ]),
        "clean_tfidf_50": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=False)),
            ("tfidf", TfidfVectorizer(max_features=50, ngram_range=(1, 1), min_df=2)),
        ]),
        "supervised_chi2_20_from_500": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=False)),
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)),
            ("select", SelectKBest(score_func=chi2, k=20)),
        ]),
        "supervised_chi2_50_from_500": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=False)),
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)),
            ("select", SelectKBest(score_func=chi2, k=50)),
        ]),
        "short_prefix_tfidf_20": Pipeline([
            ("clean", SemanticTextCleaner(remove_digit_tokens=False, max_tokens=250)),
            ("tfidf", TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2)),
        ]),
        "char_wb_ngrams_50": TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50,
            min_df=2,
        ),
        "hashing_unigrams_64": HashingVectorizer(
            n_features=64,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 1),
        ),
        "semantic_profile_features": Pipeline([
            ("profile", SemanticProfileExtractor()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        "raw_tfidf_20_plus_profile": FeatureUnion([
            ("tfidf", TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2)),
            ("profile", Pipeline([
                ("profile", SemanticProfileExtractor()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])),
        ]),
    }


def build_model(numeric_cols, semantic_transformer):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "statistical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                ["predicted_domain"],
            ),
            (
                "semantic",
                semantic_transformer,
                "semantic_text",
            ),
        ],
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

    print("\nEXPERIMENTO B - TESTE 18: ESTRATEGIAS SEMANTICAS")
    print("Base fixa: estatisticas + dominio + RandomForest.")
    print("Classificadores alvo:", classifier_cols)
    print("Datasets usados:", len(X))
    print("-" * 92)
    print(f"{'estrategia':<34} | {'acc':>8} | {'f1_macro':>8} | {'f1_weighted':>11}")
    print("-" * 92)

    summaries = []

    for strategy_name, semantic_transformer in semantic_strategies().items():
        model = build_model(numeric_cols, semantic_transformer)

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
            "experiment": "B_Test18_Semantic_Strategies",
            "strategy": strategy_name,
            "features": "Stat + Semantic Strategy + Domain",
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
            f"{strategy_name:<34} | "
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
