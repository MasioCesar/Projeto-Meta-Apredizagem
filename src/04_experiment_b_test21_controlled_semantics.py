from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
INPUT_MATRIX = DATA_DIR / __import__("os").environ.get("MAB_PERF", "performance_matrix.csv")
OUTPUT_RESULTS = DATA_DIR / "experiment_b_test21_controlled_semantics.csv"


BOILERPLATE_WORDS = {
    "author", "authors", "source", "sources", "unknown", "date", "please",
    "cite", "citation", "copyright", "donated", "available", "download",
    "repository", "openml", "uci", "kaggle", "mlbench", "rdocumentation",
    "package", "packages", "version", "versions", "topic", "topics",
    "none", "http", "https", "www", "com", "org", "edu", "arff", "csv",
    "dataset", "datasets", "data", "database", "attribute", "attributes",
    "feature", "features", "information", "class", "classes",
    "classification", "learning", "machine", "algorithm", "algorithms",
    "benchmark", "benchmarking", "used", "use", "using", "different",
    "number", "set", "collection", "problem", "problems", "task", "tasks",
    "archive", "ics", "gemler", "tabarena", "expo", "consortium",
    "project", "international", "public", "availability",
}

CUT_MARKERS = [
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

FIXED_SAFE_VOCABULARY = [
    "age", "bank", "binary", "cancer", "categorical", "chemical",
    "clinical", "credit", "customer", "dna", "document", "expression",
    "financial", "gene", "genomics", "health", "image", "medical",
    "microarray", "molecular", "molecule", "numeric", "oncology",
    "patient", "pixel", "protein", "review", "rna", "sample",
    "sequence", "sparse", "survey", "tabular", "text", "time",
    "tissue", "transaction", "tumor", "values",
]

SEMANTIC_TAGS = {
    "tag_gene_expression": {
        "gene", "genes", "expression", "microarray", "genomics", "rna",
        "probe", "probes", "tumor", "tissue", "oncology",
    },
    "tag_sequence_biology": {
        "dna", "sequence", "sequences", "splice", "junction", "molecular",
        "protein", "proteins", "nucleotide", "nucleotides",
    },
    "tag_medical_patient": {
        "patient", "patients", "clinical", "diagnosis", "disease",
        "medical", "health", "hepatitis", "hospital",
    },
    "tag_chemical_qsar": {
        "chemical", "chemistry", "molecule", "molecules", "qsar",
        "biodegradation", "toxicity", "compound", "compounds",
    },
    "tag_text_documents": {
        "text", "document", "documents", "word", "words", "sentence",
        "sentences", "review", "reviews", "email", "spam", "topic",
    },
    "tag_image_pixels": {
        "image", "images", "pixel", "pixels", "digit", "digits", "visual",
        "face", "faces", "object", "objects",
    },
    "tag_financial_business": {
        "financial", "finance", "bank", "credit", "loan", "market",
        "customer", "customers", "sales", "price", "income",
    },
    "tag_time_series_sensor": {
        "time", "series", "sensor", "signal", "signals", "activity",
        "accelerometer", "temporal",
    },
    "tag_tabular_business": {
        "tabular", "record", "records", "survey", "demographic",
        "adult", "census", "income",
    },
    "tag_missing_values": {
        "missing", "unknown", "nan", "null", "incomplete",
    },
    "tag_categorical_data": {
        "categorical", "nominal", "ordinal", "category", "categories",
    },
    "tag_numeric_data": {
        "numeric", "numerical", "continuous", "integer", "real", "values",
    },
    "tag_sparse_high_dimensional": {
        "sparse", "dimensional", "dimension", "dimensions", "large",
        "many", "variables",
    },
    "tag_small_sample": {
        "small", "few", "samples",
    },
    "tag_imbalanced": {
        "imbalanced", "unbalanced", "minority", "rare",
    },
}


def clean_semantic_text(text, dataset_name="", max_tokens=500):
    text = "" if pd.isna(text) else str(text).lower()
    dataset_name = "" if pd.isna(dataset_name) else str(dataset_name).lower()

    text = re.sub(r"https?\S+|www\.\S+", " ", text)

    for marker in CUT_MARKERS:
        marker_pos = text.find(marker)
        if marker_pos != -1:
            text = text[:marker_pos]
            break

    blocked_name_tokens = set(re.findall(r"[a-z]+", dataset_name.replace("_", " ")))
    raw_tokens = re.findall(r"[a-z0-9_]+", text)
    cleaned_tokens = []

    for token in raw_tokens:
        if len(token) < 3:
            continue
        if token in BOILERPLATE_WORDS:
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

        if max_tokens is not None and len(cleaned_tokens) >= max_tokens:
            break

    return " ".join(cleaned_tokens)


class SafeTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, max_tokens=500):
        self.max_tokens = max_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"])
        return [
            clean_semantic_text(row["semantic_text"], row["name"], self.max_tokens)
            for _, row in df.iterrows()
        ]


class SemanticTagExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"])
        rows = []

        for _, row in df.iterrows():
            cleaned = clean_semantic_text(row["semantic_text"], row["name"])
            tokens = set(cleaned.split())
            token_count = max(len(tokens), 1)

            features = []
            for keywords in SEMANTIC_TAGS.values():
                hits = tokens.intersection(keywords)
                features.append(float(len(hits) > 0))
                features.append(len(hits) / len(keywords))

            features.extend([
                np.log1p(len(cleaned.split())),
                np.log1p(len(tokens)),
                len(tokens) / max(len(cleaned.split()), 1),
                sum(token in {"gene", "expression", "microarray"} for token in tokens) / token_count,
                sum(token in {"patient", "clinical", "medical"} for token in tokens) / token_count,
                sum(token in {"text", "document", "image", "pixel"} for token in tokens) / token_count,
            ])
            rows.append(features)

        return np.asarray(rows, dtype=float)


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


def semantic_transformers():
    semantic_tags = Pipeline([
        ("tags", SemanticTagExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    fixed_vocab = Pipeline([
        ("cleaner", SafeTextCleaner(max_tokens=500)),
        ("tfidf", TfidfVectorizer(
            vocabulary=FIXED_SAFE_VOCABULARY,
            ngram_range=(1, 1),
            stop_words="english",
        )),
    ])
    auto_vocab = Pipeline([
        ("cleaner", SafeTextCleaner(max_tokens=500)),
        ("tfidf", TfidfVectorizer(
            max_features=20,
            ngram_range=(1, 1),
            min_df=2,
            stop_words="english",
        )),
    ])
    clusters = Pipeline([
        ("cleaner", SafeTextCleaner(max_tokens=500)),
        ("tfidf", TfidfVectorizer(
            max_features=80,
            ngram_range=(1, 1),
            min_df=2,
            stop_words="english",
        )),
        ("cluster", KMeans(n_clusters=5, random_state=42, n_init=20)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return {
        "semantic_tags_only": semantic_tags,
        "fixed_safe_vocabulary_tfidf": fixed_vocab,
        "safe_tfidf_20_auto_vocab": auto_vocab,
        "safe_semantic_clusters_5": clusters,
        "tags_plus_fixed_vocabulary": FeatureUnion([
            ("tags", semantic_tags),
            ("fixed_vocab", fixed_vocab),
        ]),
        "tags_plus_safe_auto_vocab": FeatureUnion([
            ("tags", semantic_tags),
            ("auto_vocab", auto_vocab),
        ]),
        "tags_plus_clusters": FeatureUnion([
            ("tags", semantic_tags),
            ("clusters", clusters),
        ]),
    }


def build_model(numeric_cols, semantic_transformer, include_domain):
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
            "semantic",
            semantic_transformer,
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


def main():
    X, y, numeric_cols, classifier_cols = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Classes insuficientes para validacao cruzada.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\nEXPERIMENTO B - TESTE 21: SEMANTICA CONTROLADA")
    print("Base fixa: estatisticas + semantica controlada + dominio opcional + RandomForest.")
    print("Semantica limpa remove identificadores fortes antes dos transformadores textuais.")
    print("-" * 98)
    print(f"{'estrategia':<42} | {'dominio':<7} | {'acc':>8} | {'f1_macro':>8} | {'f1_weighted':>11}")
    print("-" * 98)

    summaries = []

    for strategy_name, semantic_transformer in semantic_transformers().items():
        for include_domain in [True, False]:
            model = build_model(
                numeric_cols=numeric_cols,
                semantic_transformer=semantic_transformer,
                include_domain=include_domain,
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
                "experiment": "B_Test21_Controlled_Semantics",
                "strategy": strategy_name,
                "include_domain": include_domain,
                "features": "Stat + Controlled Semantic"
                + (" + Domain" if include_domain else ""),
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
                f"{strategy_name:<42} | "
                f"{str(include_domain):<7} | "
                f"{row['accuracy_mean']:.4f} | "
                f"{row['f1_macro_mean']:.4f} | "
                f"{row['f1_weighted_mean']:.4f}"
            )

    final_df = pd.DataFrame(summaries).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    final_df.to_csv(OUTPUT_RESULTS, index=False)

    print("-" * 98)
    print("\nRanking por F1-Macro:")
    print(
        final_df[
            ["strategy", "include_domain", "accuracy_mean", "f1_macro_mean"]
        ].to_string(index=False)
    )
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
