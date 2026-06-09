from pathlib import Path
import sys

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import (  # noqa: E402
    OUTPUT_DIR,
    SafeTextCleaner,
    DomainConfidenceFeatures,
    DomainSemanticInteraction,
    load_data,
)

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402


OUTPUT_RESULTS = OUTPUT_DIR / "test23_cluster_search.csv"


def cluster_transformer(n_clusters, max_features, ngram_range, use_idf=True):
    return Pipeline([
        ("cleaner", SafeTextCleaner()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            stop_words="english",
            use_idf=use_idf,
        )),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=30)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_model(numeric_cols, n_clusters, max_features, ngram_range, domain_mode, use_idf):
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
            "semantic_cluster",
            cluster_transformer(
                n_clusters=n_clusters,
                max_features=max_features,
                ngram_range=ngram_range,
                use_idf=use_idf,
            ),
            ["semantic_text", "name"],
        ),
    ]

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
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def main():
    X, y, numeric_cols, classifier_cols = load_data()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_splits = min(5, pd.Series(y_encoded).value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = []
    for n_clusters in [4, 5, 6, 8]:
        for max_features in [60, 80, 120]:
            for ngram_range in [(1, 1), (1, 2)]:
                for use_idf in [True]:
                    for domain_mode in ["score", "label_score", "label_score_interaction"]:
                        grid.append({
                            "n_clusters": n_clusters,
                            "max_features": max_features,
                            "ngram_range": ngram_range,
                            "use_idf": use_idf,
                            "domain_mode": domain_mode,
                        })

    print("\nEXPERIMENTO B - TESTE 23: BUSCA EM CLUSTERS SEMANTICOS")
    print("Foco: melhorar o Top 2 sem vocabulario fixo manual.")
    print("Texto limpo -> TF-IDF automatico -> KMeans -> dominio/score.")
    print("-" * 105)
    print(f"{'clusters':>8} | {'max_feat':>8} | {'ngram':>7} | {'idf':>5} | {'dominio':<23} | {'acc':>8} | {'f1_macro':>8}")
    print("-" * 105)

    rows = []
    for params in grid:
        model = build_model(
            numeric_cols=numeric_cols,
            n_clusters=params["n_clusters"],
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            domain_mode=params["domain_mode"],
            use_idf=params["use_idf"],
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
            "experiment": "B_Test23_Cluster_Search",
            "features": "Stat + Safe Auto Semantic Clusters + Domain Strategy",
            "base_classifiers_considered": ", ".join(classifier_cols),
            "datasets_used": len(X),
            "n_statistical_features_without_domain_score": len(numeric_cols),
            "target_classes": ", ".join(encoder.classes_),
            "n_clusters": params["n_clusters"],
            "max_features": params["max_features"],
            "ngram_range": str(params["ngram_range"]),
            "use_idf": params["use_idf"],
            "domain_mode": params["domain_mode"],
            "accuracy_mean": results["test_accuracy"].mean(),
            "accuracy_std": results["test_accuracy"].std(),
            "f1_macro_mean": results["test_f1_macro"].mean(),
            "f1_macro_std": results["test_f1_macro"].std(),
            "f1_weighted_mean": results["test_f1_weighted"].mean(),
            "f1_weighted_std": results["test_f1_weighted"].std(),
            "n_splits": n_splits,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(
            by=["f1_macro_mean", "accuracy_mean"],
            ascending=False,
        ).to_csv(OUTPUT_RESULTS, index=False)

        print(
            f"{row['n_clusters']:>8} | "
            f"{row['max_features']:>8} | "
            f"{row['ngram_range']:<7} | "
            f"{str(row['use_idf']):>5} | "
            f"{row['domain_mode']:<23} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['f1_macro_mean']:.4f}"
        )

    final_df = pd.DataFrame(rows).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_RESULTS, index=False)

    print("-" * 105)
    print("\nTop 15 por F1-Macro:")
    print(
        final_df[
            [
                "n_clusters",
                "max_features",
                "ngram_range",
                "use_idf",
                "domain_mode",
                "accuracy_mean",
                "f1_macro_mean",
            ]
        ].head(15).to_string(index=False)
    )
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
