from pathlib import Path
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import (  # noqa: E402
    OUTPUT_DIR,
    SafeTextCleaner,
    load_data,
    semantic_clusters,
)

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, normalize  # noqa: E402


OUTPUT_RESULTS = OUTPUT_DIR / "test25_auto_tfidf_score.csv"


class AutoTfidfDomainScore(BaseEstimator, TransformerMixin):
    """Gera score automatico de dominio a partir do TF-IDF limpo.

    O score nao usa a coluna domain_score original. Em cada fold, o fit aprende
    o vocabulario TF-IDF e os centroides textuais dos dominios usando apenas o
    conjunto de treino.
    """

    def __init__(self, max_features=80, ngram_range=(1, 1), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=["semantic_text", "name", "predicted_domain"])
        self.cleaner_ = SafeTextCleaner()
        cleaned_text = self.cleaner_.fit_transform(df[["semantic_text", "name"]])

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words="english",
        )
        tfidf = self.vectorizer_.fit_transform(cleaned_text)

        domains = df["predicted_domain"].fillna("unknown").astype(str).to_numpy()
        self.domains_ = sorted(pd.unique(domains))
        self.centroids_ = {}

        for domain in self.domains_:
            mask = domains == domain
            if not mask.any():
                continue
            centroid = tfidf[mask].mean(axis=0)
            centroid = np.asarray(centroid)
            centroid = normalize(centroid)
            self.centroids_[domain] = centroid

        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name", "predicted_domain"])
        cleaned_text = self.cleaner_.transform(df[["semantic_text", "name"]])
        tfidf = self.vectorizer_.transform(cleaned_text)
        domains = df["predicted_domain"].fillna("unknown").astype(str).to_numpy()

        rows = []
        for idx, domain in enumerate(domains):
            row = tfidf[idx]
            similarities = []

            for known_domain in self.domains_:
                centroid = self.centroids_.get(known_domain)
                if centroid is None:
                    similarities.append(0.0)
                else:
                    similarities.append(float(cosine_similarity(row, centroid)[0, 0]))

            if similarities:
                max_similarity = max(similarities)
                own_similarity = (
                    similarities[self.domains_.index(domain)]
                    if domain in self.domains_
                    else 0.0
                )
                sorted_sims = sorted(similarities, reverse=True)
                margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]
            else:
                max_similarity = 0.0
                own_similarity = 0.0
                margin = 0.0

            data = row.data
            top1_tfidf = float(data.max()) if data.size else 0.0
            top3_tfidf = float(np.sort(data)[-3:].sum()) if data.size else 0.0
            nonzero_ratio = row.nnz / max(len(self.vectorizer_.get_feature_names_out()), 1)

            rows.append([
                own_similarity,
                max_similarity,
                margin,
                top1_tfidf,
                top3_tfidf,
                nonzero_ratio,
            ])

        return np.asarray(rows, dtype=float)


def build_model(numeric_cols, use_domain_label=True):
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
            semantic_clusters(),
            ["semantic_text", "name"],
        ),
        (
            "auto_tfidf_score",
            Pipeline([
                ("score", AutoTfidfDomainScore(max_features=80, ngram_range=(1, 1), min_df=2)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ["semantic_text", "name", "predicted_domain"],
        ),
    ]

    if use_domain_label:
        transformers.append((
            "domain_label",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            ["predicted_domain"],
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

    experiments = [
        {
            "strategy": "cluster_plus_domain_plus_auto_tfidf_score",
            "use_domain_label": True,
        },
        {
            "strategy": "cluster_plus_auto_tfidf_score_no_domain_label",
            "use_domain_label": False,
        },
    ]

    print("\nEXPERIMENTO B - TESTE 25: SCORE AUTOMATICO VIA TF-IDF")
    print("Nao usa domain_score original. O score e calculado via TF-IDF limpo em cada fold.")
    print("-" * 88)
    print(f"{'estrategia':<48} | {'dominio':<7} | {'acc':>8} | {'f1_macro':>8}")
    print("-" * 88)

    rows = []
    for experiment in experiments:
        model = build_model(
            numeric_cols=numeric_cols,
            use_domain_label=experiment["use_domain_label"],
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
            "experiment": "B_Test25_Auto_Tfidf_Score",
            "strategy": experiment["strategy"],
            "features": "Stat + Clean Text TF-IDF KMeans Cluster + Auto TF-IDF Score"
            + (" + Domain Label" if experiment["use_domain_label"] else ""),
            "uses_original_domain_score": False,
            "auto_score_source": "cleaned semantic_text fitted inside CV folds",
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
        }
        rows.append(row)

        print(
            f"{row['strategy']:<48} | "
            f"{str(experiment['use_domain_label']):<7} | "
            f"{row['accuracy_mean']:.4f} | "
            f"{row['f1_macro_mean']:.4f}"
        )

    final_df = pd.DataFrame(rows).sort_values(
        by=["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_RESULTS, index=False)

    print("-" * 88)
    print("\nRanking por F1-Macro:")
    print(final_df[["strategy", "accuracy_mean", "f1_macro_mean"]].to_string(index=False))
    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
