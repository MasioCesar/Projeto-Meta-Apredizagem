from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from top10_controlled.common import (
    DomainConfidenceFeatures,
    fixed_vocabulary,
    semantic_tags,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "hypothesis_regret"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"

ALL_CLASSIFIERS = [
    "DecisionTree",
    "LogisticRegression",
    "Perceptron",
    "SVM",
    "KNN",
    "MLP",
]

SIMPLE_CLASSIFIERS = ["DecisionTree", "LogisticRegression", "Perceptron"]
HEAVY_CLASSIFIERS = ["SVM", "KNN", "MLP"]

TARGET_SCENARIOS = [
    {
        "name": "simple3_available",
        "target_classifiers": SIMPLE_CLASSIFIERS,
        "required_classifiers": SIMPLE_CLASSIFIERS,
        "description": "Simple classifiers on every dataset with simple-classifier results.",
    },
    {
        "name": "simple3_same107",
        "target_classifiers": SIMPLE_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
        "description": "Simple classifiers restricted to the all-six complete subset.",
    },
    {
        "name": "heavy3_same107",
        "target_classifiers": HEAVY_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
        "description": "SVM/KNN/MLP restricted to the all-six complete subset.",
    },
    {
        "name": "all6_same107",
        "target_classifiers": ALL_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
        "description": "All classifiers restricted to datasets with all six results.",
    },
]

FEATURE_SETS = [
    {
        "name": "stats_only",
        "use_semantic": False,
        "use_domain_label": False,
        "use_domain_score": False,
        "shuffle_context": False,
    },
    {
        "name": "domain_label_score",
        "use_semantic": False,
        "use_domain_label": True,
        "use_domain_score": True,
        "shuffle_context": False,
    },
    {
        "name": "semantic_tags_fixed_vocab",
        "use_semantic": True,
        "use_domain_label": False,
        "use_domain_score": False,
        "shuffle_context": False,
    },
    {
        "name": "context_full",
        "use_semantic": True,
        "use_domain_label": True,
        "use_domain_score": True,
        "shuffle_context": False,
    },
    {
        "name": "domain_label_score_shuffled",
        "use_semantic": False,
        "use_domain_label": True,
        "use_domain_score": True,
        "shuffle_context": True,
    },
    {
        "name": "semantic_tags_fixed_vocab_shuffled",
        "use_semantic": True,
        "use_domain_label": False,
        "use_domain_score": False,
        "shuffle_context": True,
    },
    {
        "name": "context_full_shuffled",
        "use_semantic": True,
        "use_domain_label": True,
        "use_domain_score": True,
        "shuffle_context": True,
    },
]

CONTEXT_COLS = ["semantic_text", "name", "predicted_domain", "domain_score"]


def load_experiment_table():
    df_meta = pd.read_csv(INPUT_METAFEATURES)
    performance_matrix = pd.read_csv(INPUT_MATRIX)

    missing_cols = [col for col in ALL_CLASSIFIERS if col not in performance_matrix.columns]
    if missing_cols:
        raise RuntimeError(
            "Missing classifier columns in performance matrix: "
            + ", ".join(missing_cols)
        )

    keep_cols = ["did"] + ALL_CLASSIFIERS
    performance_matrix = performance_matrix[keep_cols].copy()

    for col in ALL_CLASSIFIERS:
        performance_matrix[col] = pd.to_numeric(performance_matrix[col], errors="coerce")

    df = df_meta.merge(performance_matrix, on="did", how="inner")

    df["semantic_text"] = df.get("semantic_text", "").fillna("").astype(str)
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["predicted_domain"] = df.get("predicted_domain", "unknown").fillna("unknown").astype(str)
    df["domain_score"] = pd.to_numeric(df.get("domain_score", 0), errors="coerce").fillna(0)

    return df


def prepare_target_data(df, target_classifiers, required_classifiers):
    data = df.dropna(subset=required_classifiers).copy()
    data = data.dropna(subset=target_classifiers).copy()

    data["best_classifier"] = data[target_classifiers].idxmax(axis=1, skipna=True)
    data["best_accuracy"] = data[target_classifiers].max(axis=1, skipna=True)
    data = data.dropna(subset=["best_classifier", "best_accuracy"]).copy()

    valid_classes = data["best_classifier"].value_counts()
    valid_classes = valid_classes[valid_classes >= 2].index
    data = data[data["best_classifier"].isin(valid_classes)].copy()

    if data.empty:
        raise RuntimeError("No valid rows left after target preparation.")

    return data.reset_index(drop=True)


def numeric_feature_columns(data):
    ignored = {
        "did",
        "domain",
        "name",
        "semantic_text",
        "predicted_domain",
        "domain_score",
        "best_classifier",
        "best_accuracy",
    }
    ignored.update(ALL_CLASSIFIERS)

    candidates = [col for col in data.columns if col not in ignored]
    X_num = data[candidates].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.mask(X_num.abs() > 1e12, np.nan)

    valid_counts = X_num.nunique(dropna=True)
    return valid_counts[valid_counts > 1].index.tolist()


def feature_frame(data, numeric_cols):
    X = data[numeric_cols + CONTEXT_COLS].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X[numeric_cols] = X[numeric_cols].mask(X[numeric_cols].abs() > 1e12, np.nan)
    X["semantic_text"] = X["semantic_text"].fillna("").astype(str)
    X["name"] = X["name"].fillna("").astype(str)
    X["predicted_domain"] = X["predicted_domain"].fillna("unknown").astype(str)
    X["domain_score"] = pd.to_numeric(X["domain_score"], errors="coerce").fillna(0)
    return X


def shuffled_context_frame(X, random_state=42):
    rng = np.random.default_rng(random_state)
    shuffled_idx = rng.permutation(len(X))
    X_shuffled = X.copy()
    shuffled_values = X_shuffled[CONTEXT_COLS].iloc[shuffled_idx].reset_index(drop=True)
    X_shuffled[CONTEXT_COLS] = shuffled_values
    return X_shuffled


def semantic_transformer():
    return FeatureUnion([
        ("tags", semantic_tags()),
        ("fixed_vocab", fixed_vocabulary()),
    ])


def build_model(numeric_cols, feature_set):
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

    if feature_set["use_semantic"]:
        transformers.append((
            "semantic",
            semantic_transformer(),
            ["semantic_text", "name"],
        ))

    if feature_set["use_domain_label"]:
        transformers.append((
            "domain_label",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            ["predicted_domain"],
        ))

    if feature_set["use_domain_score"]:
        transformers.append((
            "domain_score",
            Pipeline([
                ("confidence", DomainConfidenceFeatures()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ["domain_score"],
        ))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def evaluate_feature_set(data, target_classifiers, feature_set, scenario_name):
    numeric_cols = numeric_feature_columns(data)
    X = feature_frame(data, numeric_cols)

    if feature_set["shuffle_context"]:
        X = shuffled_context_frame(X)

    y = data["best_classifier"].astype(str).reset_index(drop=True)
    perf_values = data[target_classifiers].reset_index(drop=True)
    oracle_values = perf_values.max(axis=1).to_numpy(float)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_counts = pd.Series(y_encoded).value_counts()
    n_splits = min(5, int(class_counts.min()))

    if n_splits < 2:
        raise RuntimeError(f"Not enough samples per class for {scenario_name}.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_rows = []
    all_true = []
    all_pred = []
    all_regret = []
    all_normalized_regret = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
        model = build_model(numeric_cols, feature_set)
        model.fit(X.iloc[train_idx], y_encoded[train_idx])
        pred_encoded = model.predict(X.iloc[test_idx])
        pred_labels = encoder.inverse_transform(pred_encoded)

        chosen_values = np.array([
            perf_values.iloc[row_idx][pred_label]
            for row_idx, pred_label in zip(test_idx, pred_labels)
        ], dtype=float)

        regret = oracle_values[test_idx] - chosen_values
        normalized_regret = regret / np.maximum(oracle_values[test_idx], 1e-12)

        y_true_fold = y_encoded[test_idx]
        acc = accuracy_score(y_true_fold, pred_encoded)
        f1_macro = f1_score(
            y_true_fold,
            pred_encoded,
            labels=np.arange(len(encoder.classes_)),
            average="macro",
            zero_division=0,
        )
        f1_weighted = f1_score(
            y_true_fold,
            pred_encoded,
            labels=np.arange(len(encoder.classes_)),
            average="weighted",
            zero_division=0,
        )

        fold_rows.append({
            "target_scenario": scenario_name,
            "feature_set": feature_set["name"],
            "fold": fold,
            "n_test": len(test_idx),
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mean_regret": regret.mean(),
            "median_regret": np.median(regret),
            "p90_regret": np.quantile(regret, 0.9),
            "mean_normalized_regret": normalized_regret.mean(),
            "zero_regret_rate": np.mean(regret <= 1e-12),
            "within_1pp_rate": np.mean(regret <= 0.01),
            "within_5pp_rate": np.mean(regret <= 0.05),
        })

        all_true.extend(y_true_fold.tolist())
        all_pred.extend(pred_encoded.tolist())
        all_regret.extend(regret.tolist())
        all_normalized_regret.extend(normalized_regret.tolist())

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    all_regret = np.asarray(all_regret)
    all_normalized_regret = np.asarray(all_normalized_regret)

    summary = {
        "target_scenario": scenario_name,
        "feature_set": feature_set["name"],
        "target_classifiers": ", ".join(target_classifiers),
        "datasets_used": len(data),
        "n_statistical_features": len(numeric_cols),
        "target_classes": ", ".join(encoder.classes_),
        "n_classes": len(encoder.classes_),
        "n_splits": n_splits,
        "accuracy": accuracy_score(all_true, all_pred),
        "f1_macro": f1_score(
            all_true,
            all_pred,
            labels=np.arange(len(encoder.classes_)),
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            all_true,
            all_pred,
            labels=np.arange(len(encoder.classes_)),
            average="weighted",
            zero_division=0,
        ),
        "mean_regret": all_regret.mean(),
        "median_regret": np.median(all_regret),
        "p90_regret": np.quantile(all_regret, 0.9),
        "mean_normalized_regret": all_normalized_regret.mean(),
        "zero_regret_rate": np.mean(all_regret <= 1e-12),
        "within_1pp_rate": np.mean(all_regret <= 0.01),
        "within_5pp_rate": np.mean(all_regret <= 0.05),
    }

    return summary, fold_rows


def add_baseline_deltas(summary_df):
    rows = []
    for _, group in summary_df.groupby("target_scenario", sort=False):
        baseline = group[group["feature_set"] == "stats_only"]

        if baseline.empty:
            rows.append(group)
            continue

        baseline = baseline.iloc[0]
        group = group.copy()
        group["f1_macro_delta_vs_stats"] = group["f1_macro"] - baseline["f1_macro"]
        group["accuracy_delta_vs_stats"] = group["accuracy"] - baseline["accuracy"]
        group["regret_reduction_vs_stats"] = baseline["mean_regret"] - group["mean_regret"]
        group["normalized_regret_reduction_vs_stats"] = (
            baseline["mean_normalized_regret"] - group["mean_normalized_regret"]
        )
        rows.append(group)

    return pd.concat(rows, ignore_index=True)


def build_contrast(summary_df):
    contrast_rows = []
    for scenario_name, group in summary_df.groupby("target_scenario", sort=False):
        by_feature = {row["feature_set"]: row for _, row in group.iterrows()}
        stats = by_feature.get("stats_only")
        context = by_feature.get("context_full")
        shuffled = by_feature.get("context_full_shuffled")
        domain = by_feature.get("domain_label_score")
        semantic = by_feature.get("semantic_tags_fixed_vocab")

        if stats is None or context is None:
            continue

        row = {
            "target_scenario": scenario_name,
            "datasets_used": stats["datasets_used"],
            "n_classes": stats["n_classes"],
            "stats_f1_macro": stats["f1_macro"],
            "context_f1_macro": context["f1_macro"],
            "context_f1_delta": context["f1_macro"] - stats["f1_macro"],
            "stats_mean_regret": stats["mean_regret"],
            "context_mean_regret": context["mean_regret"],
            "context_regret_reduction": stats["mean_regret"] - context["mean_regret"],
            "stats_zero_regret_rate": stats["zero_regret_rate"],
            "context_zero_regret_rate": context["zero_regret_rate"],
            "context_zero_regret_delta": (
                context["zero_regret_rate"] - stats["zero_regret_rate"]
            ),
        }

        if shuffled is not None:
            row["shuffled_context_f1_macro"] = shuffled["f1_macro"]
            row["shuffled_context_f1_delta"] = shuffled["f1_macro"] - stats["f1_macro"]
            row["shuffled_context_regret_reduction"] = (
                stats["mean_regret"] - shuffled["mean_regret"]
            )

        if domain is not None:
            row["domain_f1_delta"] = domain["f1_macro"] - stats["f1_macro"]
            row["domain_regret_reduction"] = stats["mean_regret"] - domain["mean_regret"]

        if semantic is not None:
            row["semantic_f1_delta"] = semantic["f1_macro"] - stats["f1_macro"]
            row["semantic_regret_reduction"] = (
                stats["mean_regret"] - semantic["mean_regret"]
            )

        contrast_rows.append(row)

    return pd.DataFrame(contrast_rows)


def target_distribution(data, scenario_name):
    rows = []
    counts = data["best_classifier"].value_counts()
    for classifier, count in counts.items():
        rows.append({
            "target_scenario": scenario_name,
            "classifier": classifier,
            "count": count,
            "share": count / len(data),
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_experiment_table()

    summary_rows = []
    fold_rows = []
    distribution_rows = []

    for scenario in TARGET_SCENARIOS:
        data = prepare_target_data(
            df,
            target_classifiers=scenario["target_classifiers"],
            required_classifiers=scenario["required_classifiers"],
        )
        distribution_rows.extend(target_distribution(data, scenario["name"]))

        print(f"\nTarget scenario: {scenario['name']}")
        print(f"Datasets: {len(data)}")
        print(data["best_classifier"].value_counts())

        for feature_set in FEATURE_SETS:
            print(f"  Feature set: {feature_set['name']} ...", end=" ")
            summary, folds = evaluate_feature_set(
                data=data,
                target_classifiers=scenario["target_classifiers"],
                feature_set=feature_set,
                scenario_name=scenario["name"],
            )
            summary_rows.append(summary)
            fold_rows.extend(folds)
            print(
                f"F1={summary['f1_macro']:.4f}, "
                f"regret={summary['mean_regret']:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_baseline_deltas(summary_df)
    fold_df = pd.DataFrame(fold_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    contrast_df = build_contrast(summary_df)

    summary_df.to_csv(OUTPUT_DIR / "context_hypothesis_summary.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "context_hypothesis_folds.csv", index=False)
    distribution_df.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)
    contrast_df.to_csv(OUTPUT_DIR / "context_hypothesis_contrast.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "context_hypothesis_summary.csv")
    print(OUTPUT_DIR / "context_hypothesis_folds.csv")
    print(OUTPUT_DIR / "target_distribution.csv")
    print(OUTPUT_DIR / "context_hypothesis_contrast.csv")


if __name__ == "__main__":
    main()
