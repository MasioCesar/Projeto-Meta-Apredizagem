from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
INPUT_MATRIX = DATA_DIR / __import__("os").environ.get("MAB_PERF", "performance_matrix.csv")
OUTPUT_RESULTS = DATA_DIR / "experiment_b_test20_no_leakage_test1.csv"


class LeakageSafeSemanticCleaner(BaseEstimator, TransformerMixin):
    """Limpa identificadores fortes antes do TF-IDF.

    O objetivo e manter descricao semantica geral e remover pistas que podem
    identificar diretamente o dataset: nome da base, URLs, autores/fontes,
    numeros, nomes de atributos, probes e tokens com underscore.
    """

    def __init__(self, max_tokens=500):
        self.max_tokens = max_tokens
        self.boilerplate_words = {
            "author", "authors", "source", "sources", "unknown", "date",
            "please", "cite", "citation", "copyright", "donated",
            "available", "download", "repository", "openml", "uci",
            "kaggle", "mlbench", "rdocumentation", "package", "packages",
            "version", "versions", "topic", "topics", "none", "http",
            "https", "www", "com", "org", "edu", "arff", "csv",
            "dataset", "datasets", "data", "database",
            "attribute", "attributes", "feature", "features", "information",
            "class", "classes", "classification", "learning", "machine",
            "algorithm", "algorithms", "benchmark", "benchmarking", "used",
            "use", "using", "different", "number", "set", "collection",
            "problem", "problems", "task", "tasks",
            "archive", "ics", "gemler", "tabarena", "expo", "consortium",
            "project", "international", "public", "availability",
        }
        self.cut_markers = [
            "attribute information",
            "attribute_information",
            "attributes information",
            "feature information",
            "feature names",
            "input attributes",
            "id_ref",
            "variable names",
            "column names",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"])
        cleaned = []

        for _, row in df.iterrows():
            cleaned.append(
                self._clean_text(
                    text="" if pd.isna(row["semantic_text"]) else str(row["semantic_text"]),
                    dataset_name="" if pd.isna(row["name"]) else str(row["name"]),
                )
            )

        return cleaned

    def _clean_text(self, text, dataset_name):
        text = text.lower()
        text = re.sub(r"https?\S+|www\.\S+", " ", text)

        for marker in self.cut_markers:
            marker_pos = text.find(marker)
            if marker_pos != -1:
                text = text[:marker_pos]
                break

        dataset_name_tokens = set(re.findall(r"[a-z]+", dataset_name.lower()))
        family_like_tokens = set(re.findall(r"[a-z]+", dataset_name.lower().replace("_", " ")))
        blocked_name_tokens = dataset_name_tokens.union(family_like_tokens)

        raw_tokens = re.findall(r"[a-z0-9_]+", text)
        cleaned_tokens = []

        for token in raw_tokens:
            if len(token) < 3:
                continue
            if token in self.boilerplate_words:
                continue
            if token in blocked_name_tokens:
                continue
            if "_" in token:
                continue
            if any(char.isdigit() for char in token):
                continue
            if not token.isalpha():
                continue

            cleaned_tokens.append(token)

            if self.max_tokens is not None and len(cleaned_tokens) >= self.max_tokens:
                break

        return " ".join(cleaned_tokens)


def load_data():
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
        "domain",
        "best_classifier",
        "best_accuracy",
    ] + all_possible_classifier_cols

    X = df_exp.drop(columns=ignore_cols, errors="ignore").copy()

    if "semantic_text" not in X.columns:
        X["semantic_text"] = ""
    X["semantic_text"] = X["semantic_text"].fillna("").astype(str)

    if "name" not in X.columns:
        X["name"] = ""
    X["name"] = X["name"].fillna("").astype(str)

    if "predicted_domain" not in X.columns:
        X["predicted_domain"] = "unknown"
    X["predicted_domain"] = X["predicted_domain"].fillna("unknown").astype(str)

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain", "name"]
    ]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain", "name"]
    ]

    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X[numeric_cols] = X[numeric_cols].mask(X[numeric_cols].abs() > 1e12, np.nan)

    nunique = X[numeric_cols].nunique(dropna=True)
    numeric_cols = nunique[nunique > 1].index.tolist()

    X = X[numeric_cols + ["semantic_text", "name", "predicted_domain"]]
    y = df_exp["best_classifier"].astype(str)

    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    valid_mask = y.isin(valid_classes)

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y, numeric_cols, existing_classifier_cols


def build_model(numeric_cols, include_domain=True):
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
            "semantic_safe",
            Pipeline([
                ("cleaner", LeakageSafeSemanticCleaner(max_tokens=500)),
                ("tfidf", TfidfVectorizer(
                    max_features=20,
                    ngram_range=(1, 1),
                    min_df=2,
                    stop_words="english",
                )),
            ]),
            ["semantic_text", "name"],
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


def get_safe_vocabulary_preview(X):
    cleaner = LeakageSafeSemanticCleaner(max_tokens=500)
    cleaned_text = cleaner.fit_transform(X[["semantic_text", "name"]])
    vectorizer = TfidfVectorizer(
        max_features=20,
        ngram_range=(1, 1),
        min_df=2,
        stop_words="english",
    )
    vectorizer.fit(cleaned_text)
    return ", ".join(vectorizer.get_feature_names_out())


def main():
    X, y, numeric_cols, classifier_cols = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Classes insuficientes para validacao cruzada.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    experiments = [
        {
            "strategy": "test1_safe_semantic_with_domain",
            "include_domain": True,
        },
        {
            "strategy": "test1_safe_semantic_no_domain",
            "include_domain": False,
        },
    ]

    safe_vocab_preview = get_safe_vocabulary_preview(X)

    print("\nEXPERIMENTO B - TESTE 20: TESTE 1 SEM VAZAMENTO TEXTUAL")
    print("Base: estatisticas + TF-IDF seguro do texto + dominio opcional + RandomForest.")
    print("Limpeza: remove nome do dataset, URLs, numeros, underscores, atributos/probes e boilerplate.")
    print("Vocabulario seguro global para inspecao:", safe_vocab_preview)
    print("-" * 86)
    print(f"{'estrategia':<36} | {'acc':>8} | {'f1_macro':>8} | {'f1_weighted':>11}")
    print("-" * 86)

    summaries = []

    for experiment in experiments:
        model = build_model(
            numeric_cols=numeric_cols,
            include_domain=experiment["include_domain"],
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
            "experiment": "B_Test20_No_Leakage_Test1",
            "strategy": experiment["strategy"],
            "features": "Stat + Leakage-Safe Semantic TF-IDF"
            + (" + Domain" if experiment["include_domain"] else ""),
            "semantic_cleaning": (
                "remove_dataset_name_urls_numbers_underscores_attribute_lists_"
                "probe_like_tokens_boilerplate_stopwords"
            ),
            "safe_vocab_preview": safe_vocab_preview,
            "base_classifiers_considered": ", ".join(classifier_cols),
            "datasets_used": len(X),
            "n_statistical_features": len(numeric_cols),
            "n_semantic_max_features": 20,
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
            f"{row['strategy']:<36} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['f1_macro_mean']:.4f} | "
            f"{row['f1_weighted_mean']:.4f}"
        )

    final_df = pd.DataFrame(summaries).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    final_df.to_csv(OUTPUT_RESULTS, index=False)

    print("-" * 86)
    print("\nRanking por F1-Macro:")
    print(final_df[["strategy", "accuracy_mean", "f1_macro_mean"]].to_string(index=False))
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
