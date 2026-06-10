from pathlib import Path
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from top10_controlled.common import (
    DomainConfidenceFeatures,
    fixed_vocabulary,
    semantic_tags,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "hypothesis_regret_repeated"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"

N_SPLITS = int(os.getenv("REGRET_N_SPLITS", "5"))
N_REPEATS = int(os.getenv("REGRET_N_REPEATS", "3"))
N_SHUFFLES = int(os.getenv("REGRET_N_SHUFFLES", "10"))
MODEL_N_ESTIMATORS = int(os.getenv("REGRET_N_ESTIMATORS", "200"))
RANDOM_STATE = int(os.getenv("REGRET_RANDOM_STATE", "42"))

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
    },
    {
        "name": "simple3_same107",
        "target_classifiers": SIMPLE_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
    },
    {
        "name": "heavy3_same107",
        "target_classifiers": HEAVY_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
    },
    {
        "name": "all6_same107",
        "target_classifiers": ALL_CLASSIFIERS,
        "required_classifiers": ALL_CLASSIFIERS,
    },
]

REAL_FEATURE_SETS = [
    {
        "name": "stats_only",
        "context_group": "none",
        "use_semantic": False,
        "use_domain_label": False,
        "use_domain_score": False,
    },
    {
        "name": "domain_label_score",
        "context_group": "domain",
        "use_semantic": False,
        "use_domain_label": True,
        "use_domain_score": True,
    },
    {
        "name": "semantic_tags_fixed_vocab",
        "context_group": "semantic",
        "use_semantic": True,
        "use_domain_label": False,
        "use_domain_score": False,
    },
    {
        "name": "context_full",
        "context_group": "full",
        "use_semantic": True,
        "use_domain_label": True,
        "use_domain_score": True,
    },
]

SHUFFLED_FEATURE_SETS = [
    {
        "name": "domain_label_score_shuffled",
        "context_group": "domain",
        "use_semantic": False,
        "use_domain_label": True,
        "use_domain_score": True,
    },
    {
        "name": "semantic_tags_fixed_vocab_shuffled",
        "context_group": "semantic",
        "use_semantic": True,
        "use_domain_label": False,
        "use_domain_score": False,
    },
    {
        "name": "context_full_shuffled",
        "context_group": "full",
        "use_semantic": True,
        "use_domain_label": True,
        "use_domain_score": True,
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

    performance_matrix = performance_matrix[["did"] + ALL_CLASSIFIERS].copy()

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


def shuffled_context_frame(X, seed):
    rng = np.random.default_rng(seed)
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


def build_model(numeric_cols, feature_set, seed):
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
            n_estimators=MODEL_N_ESTIMATORS,
            random_state=seed,
            class_weight="balanced",
        )),
    ])


def split_iterator(y_encoded):
    class_counts = pd.Series(y_encoded).value_counts()
    n_splits = min(N_SPLITS, int(class_counts.min()))

    if n_splits < 2:
        raise RuntimeError("Not enough samples per class for repeated CV.")

    if N_REPEATS <= 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )

    return cv, n_splits


def evaluate_once(
    X,
    y_encoded,
    perf_values,
    oracle_values,
    encoder,
    target_classifiers,
    feature_set,
    numeric_cols,
    eval_seed,
):
    cv, n_splits = split_iterator(y_encoded)
    true_all = []
    pred_all = []
    regret_all = []
    normalized_regret_all = []
    fold_rows = []

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
        model_seed = eval_seed + split_idx
        model = build_model(numeric_cols, feature_set, seed=model_seed)
        model.fit(X.iloc[train_idx], y_encoded[train_idx])
        pred_encoded = model.predict(X.iloc[test_idx])
        pred_labels = encoder.inverse_transform(pred_encoded)

        chosen_values = np.array([
            perf_values.iloc[row_idx][pred_label]
            for row_idx, pred_label in zip(test_idx, pred_labels)
        ], dtype=float)

        regret = oracle_values[test_idx] - chosen_values
        normalized_regret = regret / np.maximum(oracle_values[test_idx], 1e-12)

        true_fold = y_encoded[test_idx]
        true_all.extend(true_fold.tolist())
        pred_all.extend(pred_encoded.tolist())
        regret_all.extend(regret.tolist())
        normalized_regret_all.extend(normalized_regret.tolist())

        fold_rows.append({
            "split_idx": split_idx,
            "n_splits": n_splits,
            "n_test": len(test_idx),
            "accuracy": accuracy_score(true_fold, pred_encoded),
            "f1_macro": f1_score(
                true_fold,
                pred_encoded,
                labels=np.arange(len(encoder.classes_)),
                average="macro",
                zero_division=0,
            ),
            "mean_regret": regret.mean(),
            "zero_regret_rate": np.mean(regret <= 1e-12),
            "within_1pp_rate": np.mean(regret <= 0.01),
            "within_5pp_rate": np.mean(regret <= 0.05),
        })

    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)
    regret_all = np.asarray(regret_all)
    normalized_regret_all = np.asarray(normalized_regret_all)

    metrics = {
        "accuracy": accuracy_score(true_all, pred_all),
        "f1_macro": f1_score(
            true_all,
            pred_all,
            labels=np.arange(len(encoder.classes_)),
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            true_all,
            pred_all,
            labels=np.arange(len(encoder.classes_)),
            average="weighted",
            zero_division=0,
        ),
        "mean_regret": regret_all.mean(),
        "median_regret": np.median(regret_all),
        "p90_regret": np.quantile(regret_all, 0.9),
        "mean_normalized_regret": normalized_regret_all.mean(),
        "zero_regret_rate": np.mean(regret_all <= 1e-12),
        "within_1pp_rate": np.mean(regret_all <= 0.01),
        "within_5pp_rate": np.mean(regret_all <= 0.05),
    }

    return metrics, fold_rows


def evaluate_real_feature_set(data, target_classifiers, feature_set, scenario_name):
    numeric_cols = numeric_feature_columns(data)
    X = feature_frame(data, numeric_cols)
    y = data["best_classifier"].astype(str).reset_index(drop=True)
    perf_values = data[target_classifiers].reset_index(drop=True)
    oracle_values = perf_values.max(axis=1).to_numpy(float)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    metrics, fold_rows = evaluate_once(
        X=X,
        y_encoded=y_encoded,
        perf_values=perf_values,
        oracle_values=oracle_values,
        encoder=encoder,
        target_classifiers=target_classifiers,
        feature_set=feature_set,
        numeric_cols=numeric_cols,
        eval_seed=RANDOM_STATE,
    )

    row = {
        "target_scenario": scenario_name,
        "feature_set": feature_set["name"],
        "context_group": feature_set["context_group"],
        "variant": "real",
        "shuffle_id": np.nan,
        "target_classifiers": ", ".join(target_classifiers),
        "datasets_used": len(data),
        "n_statistical_features": len(numeric_cols),
        "target_classes": ", ".join(encoder.classes_),
        "n_classes": len(encoder.classes_),
        "n_repeats": N_REPEATS,
        "n_splits": min(N_SPLITS, pd.Series(y_encoded).value_counts().min()),
    }
    row.update(metrics)

    for fold in fold_rows:
        fold.update({
            "target_scenario": scenario_name,
            "feature_set": feature_set["name"],
            "variant": "real",
            "shuffle_id": np.nan,
        })

    return row, fold_rows


def evaluate_shuffled_feature_set(data, target_classifiers, feature_set, scenario_name):
    numeric_cols = numeric_feature_columns(data)
    X_base = feature_frame(data, numeric_cols)
    y = data["best_classifier"].astype(str).reset_index(drop=True)
    perf_values = data[target_classifiers].reset_index(drop=True)
    oracle_values = perf_values.max(axis=1).to_numpy(float)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    rows = []
    fold_rows = []

    for shuffle_id in range(N_SHUFFLES):
        shuffle_seed = RANDOM_STATE + 10_000 + shuffle_id
        X = shuffled_context_frame(X_base, seed=shuffle_seed)
        metrics, folds = evaluate_once(
            X=X,
            y_encoded=y_encoded,
            perf_values=perf_values,
            oracle_values=oracle_values,
            encoder=encoder,
            target_classifiers=target_classifiers,
            feature_set=feature_set,
            numeric_cols=numeric_cols,
            eval_seed=shuffle_seed,
        )

        row = {
            "target_scenario": scenario_name,
            "feature_set": feature_set["name"],
            "context_group": feature_set["context_group"],
            "variant": "shuffled",
            "shuffle_id": shuffle_id,
            "target_classifiers": ", ".join(target_classifiers),
            "datasets_used": len(data),
            "n_statistical_features": len(numeric_cols),
            "target_classes": ", ".join(encoder.classes_),
            "n_classes": len(encoder.classes_),
            "n_repeats": N_REPEATS,
            "n_splits": min(N_SPLITS, pd.Series(y_encoded).value_counts().min()),
        }
        row.update(metrics)
        rows.append(row)

        for fold in folds:
            fold.update({
                "target_scenario": scenario_name,
                "feature_set": feature_set["name"],
                "variant": "shuffled",
                "shuffle_id": shuffle_id,
            })
        fold_rows.extend(folds)

    return rows, fold_rows


def summarize_shuffles(result_df):
    shuffled = result_df[result_df["variant"] == "shuffled"].copy()
    if shuffled.empty:
        return pd.DataFrame()

    rows = []
    metric_cols = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "mean_regret",
        "mean_normalized_regret",
        "zero_regret_rate",
        "within_1pp_rate",
        "within_5pp_rate",
    ]

    for keys, group in shuffled.groupby(["target_scenario", "context_group", "feature_set"]):
        target_scenario, context_group, feature_set = keys
        row = {
            "target_scenario": target_scenario,
            "context_group": context_group,
            "feature_set": feature_set,
            "n_shuffles": len(group),
        }

        for metric in metric_cols:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = np.quantile(values, 0.025)
            row[f"{metric}_ci95_high"] = np.quantile(values, 0.975)

        rows.append(row)

    return pd.DataFrame(rows)


def compare_real_to_stats_and_shuffles(result_df, shuffle_summary_df):
    rows = []
    real = result_df[result_df["variant"] == "real"].copy()
    real_by_key = {
        (row["target_scenario"], row["feature_set"]): row
        for _, row in real.iterrows()
    }
    stats_by_scenario = {
        row["target_scenario"]: row
        for _, row in real[real["feature_set"] == "stats_only"].iterrows()
    }
    shuffle_by_key = {
        (row["target_scenario"], row["context_group"]): row
        for _, row in shuffle_summary_df.iterrows()
    }

    for _, row in real.iterrows():
        scenario = row["target_scenario"]
        feature_set = row["feature_set"]
        context_group = row["context_group"]
        stats = stats_by_scenario.get(scenario)

        out = {
            "target_scenario": scenario,
            "feature_set": feature_set,
            "context_group": context_group,
            "datasets_used": row["datasets_used"],
            "n_classes": row["n_classes"],
            "real_f1_macro": row["f1_macro"],
            "real_mean_regret": row["mean_regret"],
            "real_zero_regret_rate": row["zero_regret_rate"],
        }

        if stats is not None:
            out["f1_delta_vs_stats"] = row["f1_macro"] - stats["f1_macro"]
            out["regret_reduction_vs_stats"] = stats["mean_regret"] - row["mean_regret"]
            out["zero_regret_delta_vs_stats"] = (
                row["zero_regret_rate"] - stats["zero_regret_rate"]
            )

        shuffle = shuffle_by_key.get((scenario, context_group))
        if shuffle is not None and context_group != "none":
            out["shuffle_f1_macro_mean"] = shuffle["f1_macro_mean"]
            out["shuffle_f1_macro_ci95_low"] = shuffle["f1_macro_ci95_low"]
            out["shuffle_f1_macro_ci95_high"] = shuffle["f1_macro_ci95_high"]
            out["real_minus_shuffle_f1_mean"] = row["f1_macro"] - shuffle["f1_macro_mean"]
            out["real_f1_above_shuffle_ci95"] = (
                row["f1_macro"] > shuffle["f1_macro_ci95_high"]
            )
            out["shuffle_mean_regret_mean"] = shuffle["mean_regret_mean"]
            out["shuffle_mean_regret_ci95_low"] = shuffle["mean_regret_ci95_low"]
            out["shuffle_mean_regret_ci95_high"] = shuffle["mean_regret_ci95_high"]
            out["shuffle_minus_real_regret_mean"] = (
                shuffle["mean_regret_mean"] - row["mean_regret"]
            )
            out["real_regret_below_shuffle_ci95"] = (
                row["mean_regret"] < shuffle["mean_regret_ci95_low"]
            )

        rows.append(out)

    return pd.DataFrame(rows)


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

    result_rows = []
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

        for feature_set in REAL_FEATURE_SETS:
            print(f"  Real: {feature_set['name']} ...", end=" ")
            row, folds = evaluate_real_feature_set(
                data=data,
                target_classifiers=scenario["target_classifiers"],
                feature_set=feature_set,
                scenario_name=scenario["name"],
            )
            result_rows.append(row)
            fold_rows.extend(folds)
            print(f"F1={row['f1_macro']:.4f}, regret={row['mean_regret']:.4f}")

        for feature_set in SHUFFLED_FEATURE_SETS:
            print(f"  Shuffled x{N_SHUFFLES}: {feature_set['name']} ...")
            rows, folds = evaluate_shuffled_feature_set(
                data=data,
                target_classifiers=scenario["target_classifiers"],
                feature_set=feature_set,
                scenario_name=scenario["name"],
            )
            result_rows.extend(rows)
            fold_rows.extend(folds)
            mean_f1 = np.mean([row["f1_macro"] for row in rows])
            mean_regret = np.mean([row["mean_regret"] for row in rows])
            print(f"    mean F1={mean_f1:.4f}, mean regret={mean_regret:.4f}")

    result_df = pd.DataFrame(result_rows)
    fold_df = pd.DataFrame(fold_rows)
    distribution_df = pd.DataFrame(distribution_rows)
    shuffle_summary_df = summarize_shuffles(result_df)
    comparison_df = compare_real_to_stats_and_shuffles(result_df, shuffle_summary_df)

    result_df.to_csv(OUTPUT_DIR / "repeated_results_by_run.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "repeated_fold_metrics.csv", index=False)
    distribution_df.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)
    shuffle_summary_df.to_csv(OUTPUT_DIR / "shuffle_summary.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "real_vs_stats_and_shuffle.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "repeated_results_by_run.csv")
    print(OUTPUT_DIR / "repeated_fold_metrics.csv")
    print(OUTPUT_DIR / "target_distribution.csv")
    print(OUTPUT_DIR / "shuffle_summary.csv")
    print(OUTPUT_DIR / "real_vs_stats_and_shuffle.csv")


if __name__ == "__main__":
    main()
