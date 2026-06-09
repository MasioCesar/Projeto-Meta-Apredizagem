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


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "top10_controlled"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"

CLASSIFIER_COLS = ["DecisionTree", "LogisticRegression", "Perceptron"]

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
    "tag_missing_values": {"missing", "unknown", "nan", "null", "incomplete"},
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
    "tag_small_sample": {"small", "few", "samples"},
    "tag_imbalanced": {"imbalanced", "unbalanced", "minority", "rare"},
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
        if token in BOILERPLATE_WORDS or token in blocked_name_tokens:
            continue
        if "_" in token or any(char.isdigit() for char in token):
            continue
        if not token.isalpha():
            continue
        cleaned_tokens.append(token)
        if max_tokens is not None and len(cleaned_tokens) >= max_tokens:
            break

    return " ".join(cleaned_tokens)


class SafeTextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"])
        return [
            clean_semantic_text(row["semantic_text"], row["name"])
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


class DomainConfidenceFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = pd.to_numeric(pd.Series(np.asarray(X).ravel()), errors="coerce").fillna(0).to_numpy(float)
        return np.column_stack([
            values,
            np.log1p(values),
            (values >= 20).astype(float),
            (values >= 30).astype(float),
        ])


class DomainSemanticInteraction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=["semantic_text", "name", "predicted_domain", "domain_score"])
        self.domains_ = sorted(df["predicted_domain"].fillna("unknown").astype(str).unique())
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name", "predicted_domain", "domain_score"])
        tag_values = SemanticTagExtractor().transform(df[["semantic_text", "name"]])
        domains = df["predicted_domain"].fillna("unknown").astype(str)
        scores = pd.to_numeric(df["domain_score"], errors="coerce").fillna(0).to_numpy(float)
        score_scale = np.log1p(scores).reshape(-1, 1)

        features = []
        for domain in self.domains_:
            mask = (domains == domain).to_numpy(float).reshape(-1, 1)
            features.append(mask * tag_values)
            features.append(mask * score_scale)
        return np.hstack(features)


def load_data():
    df_meta = pd.read_csv(INPUT_METAFEATURES)
    performance_matrix = pd.read_csv(INPUT_MATRIX)

    existing_classifier_cols = [col for col in CLASSIFIER_COLS if col in performance_matrix.columns]
    if not existing_classifier_cols:
        raise RuntimeError("Nenhum classificador alvo foi encontrado.")

    performance_matrix["best_classifier"] = performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
    performance_matrix["best_accuracy"] = performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
    performance_matrix = performance_matrix.dropna(subset=["best_classifier", "best_accuracy"]).copy()

    df_exp = df_meta.merge(performance_matrix, on="did", how="inner")
    classifier_cols_to_drop = ["DecisionTree", "SVM", "KNN", "LogisticRegression", "Perceptron", "MLP"]
    ignore_cols = ["did", "domain", "best_classifier", "best_accuracy"] + classifier_cols_to_drop
    X = df_exp.drop(columns=ignore_cols, errors="ignore").copy()

    X["semantic_text"] = X.get("semantic_text", "").fillna("").astype(str)
    X["name"] = X.get("name", "").fillna("").astype(str)
    X["predicted_domain"] = X.get("predicted_domain", "unknown").fillna("unknown").astype(str)
    X["domain_score"] = pd.to_numeric(X.get("domain_score", 0), errors="coerce").fillna(0)

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain", "name", "domain_score"]
    ]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna(axis=1, how="all")

    numeric_cols = [
        col for col in X.columns
        if col not in ["semantic_text", "predicted_domain", "name", "domain_score"]
    ]
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X[numeric_cols] = X[numeric_cols].mask(X[numeric_cols].abs() > 1e12, np.nan)
    valid_numeric_cols = X[numeric_cols].nunique(dropna=True)
    numeric_cols = valid_numeric_cols[valid_numeric_cols > 1].index.tolist()

    X = X[numeric_cols + ["semantic_text", "name", "predicted_domain", "domain_score"]]
    y = df_exp["best_classifier"].astype(str)
    valid_classes = y.value_counts()[lambda s: s >= 2].index
    valid_mask = y.isin(valid_classes)

    return X.loc[valid_mask].reset_index(drop=True), y.loc[valid_mask].reset_index(drop=True), numeric_cols, existing_classifier_cols


def semantic_tags():
    return Pipeline([
        ("tags", SemanticTagExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def fixed_vocabulary():
    return Pipeline([
        ("cleaner", SafeTextCleaner()),
        ("tfidf", TfidfVectorizer(
            vocabulary=FIXED_SAFE_VOCABULARY,
            ngram_range=(1, 1),
            stop_words="english",
        )),
    ])


def safe_auto_vocabulary():
    return Pipeline([
        ("cleaner", SafeTextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2, stop_words="english")),
    ])


def semantic_clusters():
    return Pipeline([
        ("cleaner", SafeTextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=80, ngram_range=(1, 1), min_df=2, stop_words="english")),
        ("cluster", KMeans(n_clusters=5, random_state=42, n_init=20)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])


def semantic_transformer(mode):
    if mode == "none":
        return None
    if mode == "tags_only":
        return semantic_tags()
    if mode == "fixed_vocab":
        return fixed_vocabulary()
    if mode == "auto_vocab":
        return safe_auto_vocabulary()
    if mode == "clusters_only":
        return semantic_clusters()
    if mode == "tags_fixed_vocab":
        return FeatureUnion([("tags", semantic_tags()), ("fixed_vocab", fixed_vocabulary())])
    if mode == "tags_clusters":
        return FeatureUnion([("tags", semantic_tags()), ("clusters", semantic_clusters())])
    if mode == "tags_auto_vocab":
        return FeatureUnion([("tags", semantic_tags()), ("auto_vocab", safe_auto_vocabulary())])
    raise ValueError(f"Modo semantico desconhecido: {mode}")


def build_model(numeric_cols, semantic_mode, domain_mode):
    transformers = [
        (
            "statistical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        )
    ]

    semantic = semantic_transformer(semantic_mode)
    if semantic is not None:
        transformers.append(("controlled_semantic", semantic, ["semantic_text", "name"]))

    if domain_mode in {"label", "label_score", "label_score_interaction"}:
        transformers.append((
            "domain_label",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            ["predicted_domain"],
        ))

    if domain_mode in {"score", "label_score", "label_score_interaction"}:
        transformers.append((
            "domain_score",
            Pipeline([
                ("confidence", DomainConfidenceFeatures()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ["domain_score"],
        ))

    if domain_mode == "label_score_interaction":
        transformers.append((
            "domain_semantic_interaction",
            Pipeline([
                ("interaction", DomainSemanticInteraction()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ["semantic_text", "name", "predicted_domain", "domain_score"],
        ))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])


def run_experiment(rank, strategy, semantic_mode, domain_mode, output_filename):
    X, y, numeric_cols, classifier_cols = load_data()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_splits = min(5, pd.Series(y_encoded).value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = build_model(numeric_cols, semantic_mode=semantic_mode, domain_mode=domain_mode)
    results = cross_validate(
        model,
        X,
        y_encoded,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"},
        return_train_score=False,
        error_score="raise",
    )

    summary = pd.DataFrame([{
        "rank": rank,
        "strategy": strategy,
        "semantic_mode": semantic_mode,
        "domain_mode": domain_mode,
        "base_classifiers_considered": ", ".join(classifier_cols),
        "datasets_used": len(X),
        "n_statistical_features_without_domain_score": len(numeric_cols),
        "target_classes": ", ".join(encoder.classes_),
        "accuracy_mean": results["test_accuracy"].mean(),
        "accuracy_std": results["test_accuracy"].std(),
        "f1_macro_mean": results["test_f1_macro"].mean(),
        "f1_macro_std": results["test_f1_macro"].std(),
        "f1_weighted_mean": results["test_f1_weighted"].mean(),
        "f1_weighted_std": results["test_f1_weighted"].std(),
        "n_splits": n_splits,
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
    summary.to_csv(output_path, index=False)

    print(f"{strategy}")
    print(f"Accuracy: {summary.loc[0, 'accuracy_mean']:.6f}")
    print(f"F1-Macro: {summary.loc[0, 'f1_macro_mean']:.6f}")
    print(f"Salvo em: {output_path}")
    return summary
