"""Domain permutation test for the V2 domain rank encoding.

Question:
    Does the gain from + dominio(rank) come from the real domain assignment,
    or from adding extra target-encoded features with the same dimensionality?

Protocol:
    - Same V2 data and grouped CV used by the final domain test.
    - Compare baseline stats, real domain rank encoding, and shuffled-domain
      rank encoding.
    - Encoding is always fitted only on the training fold.
    - The shuffled control preserves the domain frequency distribution, but
      breaks the dataset-domain relationship.

Outputs:
    data/top10_controlled/{OUTPUT_PREFIX}_by_run.csv
    data/top10_controlled/{OUTPUT_PREFIX}_fold_metrics.csv
    data/top10_controlled/{OUTPUT_PREFIX}_null_by_permutation.csv
    data/top10_controlled/{OUTPUT_PREFIX}_summary.csv
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, str(Path(__file__).parent))
from target_encode_advanced_v2 import (  # noqa: E402
    ALL6,
    DOMKW,
    SEEDS as DEFAULT_SEEDS,
    apply_enc,
    enc_perf,
    jacc,
    kw_assign,
    norm,
)

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "top10_controlled"

N_SEEDS = int(os.getenv("DOMAIN_PERM_N_SEEDS", str(len(DEFAULT_SEEDS))))
N_PERMUTATIONS = int(os.getenv("DOMAIN_PERM_N_PERMUTATIONS", "30"))
N_ESTIMATORS = int(os.getenv("DOMAIN_PERM_N_ESTIMATORS", "300"))
PERM_RANDOM_STATE = int(os.getenv("DOMAIN_PERM_RANDOM_STATE", "42"))
MODEL_RANDOM_STATE = int(os.getenv("DOMAIN_PERM_MODEL_RANDOM_STATE", "42"))
N_SPLITS = int(os.getenv("DOMAIN_PERM_N_SPLITS", "5"))
OUTPUT_PREFIX = os.getenv("DOMAIN_PERM_OUTPUT_PREFIX", "domain_permutation")
N_JOBS = int(os.getenv("DOMAIN_PERM_N_JOBS", "1"))

SEEDS = list(DEFAULT_SEEDS[:N_SEEDS])

METRICS = [
    ("accuracy", "higher"),
    ("f1_macro", "higher"),
    ("top3_accuracy", "higher"),
    ("mean_regret", "lower"),
    ("zero_regret_rate", "higher"),
    ("within_1pp_rate", "higher"),
]


def load_data():
    meta = pd.read_csv(DATA / "metafeatures_v2.csv")
    perf = pd.read_csv(DATA / "performance_matrix_v2.csv")
    txt = {
        int(r["did"]): r
        for r in json.loads((DATA / "v2_descriptions.json").read_text(encoding="utf-8"))
    }

    cols = ["did", "best_classifier"] + [a for a in ALL6 if a in perf.columns]
    df = (
        meta.merge(perf[cols], on="did", how="inner")
        .dropna(subset=["best_classifier"])
        .reset_index(drop=True)
    )
    df["dom"] = [
        kw_assign(nm, txt.get(int(d), {}).get("description", ""), DOMKW)
        for d, nm in zip(df["did"], df["name"])
    ]

    did_to_features = {
        int(d): set(norm(txt.get(int(d), {}).get("feature_names", "")).split()) - {""}
        for d in df["did"]
    }
    groups = df["did"].astype(int).map(jacc(did_to_features)).to_numpy()

    num_cols = [
        c
        for c in meta.columns
        if c not in ("did", "name") and pd.api.types.is_numeric_dtype(meta[c])
    ]
    num_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1]
    x_stats = df[num_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["best_classifier"].astype(str))
    classes = np.arange(len(label_encoder.classes_))

    perf_values = df[ALL6].to_numpy(float)
    algo_to_perf_idx = {name: i for i, name in enumerate(ALL6)}

    return {
        "df": df,
        "x_stats": x_stats,
        "y": y,
        "classes": classes,
        "label_encoder": label_encoder,
        "groups": groups,
        "perf_values": perf_values,
        "algo_to_perf_idx": algo_to_perf_idx,
        "domain": df["dom"].to_numpy(),
    }


def build_features(x_stats, perf_values, domain_values, train_idx, test_idx):
    x_train = x_stats[train_idx]
    x_test = x_stats[test_idx]

    if domain_values is None:
        return x_train, x_test

    tab, prior = enc_perf(domain_values[train_idx], perf_values[train_idx], "rank")
    dom_train = apply_enc(domain_values[train_idx], tab, prior)
    dom_test = apply_enc(domain_values[test_idx], tab, prior)
    return np.hstack([x_train, dom_train]), np.hstack([x_test, dom_test])


def top3_flags(y_true, proba, proba_classes):
    flags = []
    k = min(3, len(proba_classes))
    for i in range(len(y_true)):
        top = proba_classes[np.argsort(proba[i])[-k:]]
        flags.append(int(y_true[i] in set(top)))
    return np.asarray(flags, dtype=int)


def regret_values(test_idx, pred_encoded, label_encoder, perf_values, algo_to_perf_idx):
    pred_names = label_encoder.inverse_transform(pred_encoded)
    chosen_cols = np.asarray([algo_to_perf_idx[name] for name in pred_names])
    chosen = perf_values[test_idx, chosen_cols]
    oracle = np.nanmax(perf_values[test_idx], axis=1)
    return oracle - chosen


def regret_summary(regret):
    known = regret[~np.isnan(regret)]
    if len(known) == 0:
        return {
            "mean_regret": np.nan,
            "median_regret": np.nan,
            "p90_regret": np.nan,
            "zero_regret_rate": np.nan,
            "within_1pp_rate": np.nan,
            "regret_coverage": 0.0,
        }
    return {
        "mean_regret": float(np.mean(known)),
        "median_regret": float(np.median(known)),
        "p90_regret": float(np.quantile(known, 0.90)),
        "zero_regret_rate": float(np.mean(known <= 1e-12)),
        "within_1pp_rate": float(np.mean(known <= 0.01)),
        "regret_coverage": float(len(known) / len(regret)),
    }


def evaluate_config(data, seed, config, domain_values=None, permutation_id=None):
    x_stats = data["x_stats"]
    y = data["y"]
    groups = data["groups"]
    perf_values = data["perf_values"]

    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []
    all_true = []
    all_pred = []
    all_top3 = []
    all_regret = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_stats, y, groups=groups), start=1):
        x_train, x_test = build_features(
            x_stats, perf_values, domain_values, train_idx, test_idx
        )
        imputer = SimpleImputer(strategy="median").fit(x_train)
        x_train = imputer.transform(x_train)
        x_test = imputer.transform(x_test)

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=MODEL_RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        )
        clf.fit(x_train, y[train_idx])

        y_true = y[test_idx]
        pred = clf.predict(x_test)
        proba = clf.predict_proba(x_test)
        top3 = top3_flags(y_true, proba, clf.classes_)
        regret = regret_values(
            test_idx,
            pred,
            data["label_encoder"],
            perf_values,
            data["algo_to_perf_idx"],
        )
        reg = regret_summary(regret)

        fold_rows.append(
            {
                "config": config,
                "seed": seed,
                "permutation_id": permutation_id,
                "fold": fold,
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_true, pred),
                "f1_macro": f1_score(
                    y_true,
                    pred,
                    labels=data["classes"],
                    average="macro",
                    zero_division=0,
                ),
                "top3_accuracy": top3.mean(),
                "mean_regret": reg["mean_regret"],
                "zero_regret_rate": reg["zero_regret_rate"],
                "within_1pp_rate": reg["within_1pp_rate"],
                "regret_coverage": reg["regret_coverage"],
            }
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(pred.tolist())
        all_top3.extend(top3.tolist())
        all_regret.extend(regret.tolist())

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    all_top3 = np.asarray(all_top3)
    all_regret = np.asarray(all_regret)
    reg = regret_summary(all_regret)

    run_row = {
        "config": config,
        "seed": seed,
        "permutation_id": permutation_id,
        "accuracy": accuracy_score(all_true, all_pred),
        "f1_macro": f1_score(
            all_true,
            all_pred,
            labels=data["classes"],
            average="macro",
            zero_division=0,
        ),
        "top3_accuracy": all_top3.mean(),
        "mean_regret": reg["mean_regret"],
        "median_regret": reg["median_regret"],
        "p90_regret": reg["p90_regret"],
        "zero_regret_rate": reg["zero_regret_rate"],
        "within_1pp_rate": reg["within_1pp_rate"],
        "regret_coverage": reg["regret_coverage"],
    }
    return run_row, fold_rows


def add_effects(rows):
    baseline = (
        rows[rows["config"] == "baseline_stats"]
        .set_index("seed")
        [[metric for metric, _ in METRICS]]
    )
    out = rows.copy()
    for metric, direction in METRICS:
        base_values = out["seed"].map(baseline[metric])
        raw_delta = out[metric] - base_values
        effect = raw_delta if direction == "higher" else -raw_delta
        out[f"{metric}_delta_vs_baseline"] = raw_delta
        out[f"{metric}_effect_vs_baseline"] = effect
    return out


def safe_ttest(a, b):
    try:
        _, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        return np.nan


def safe_wilcoxon(effect):
    try:
        _, p = stats.wilcoxon(effect)
        return float(p)
    except Exception:
        return np.nan


def summarize(rows):
    rows = add_effects(rows)
    baseline = rows[rows["config"] == "baseline_stats"].set_index("seed")
    real = rows[rows["config"] == "real_domain_rank"].set_index("seed")
    shuffled = rows[rows["config"] == "shuffled_domain_rank"].copy()

    null_rows = []
    for perm_id, grp in shuffled.groupby("permutation_id"):
        item = {"permutation_id": int(perm_id)}
        for metric, _ in METRICS:
            item[metric] = grp[metric].mean()
            item[f"{metric}_effect_vs_baseline"] = grp[
                f"{metric}_effect_vs_baseline"
            ].mean()
        null_rows.append(item)
    null = pd.DataFrame(null_rows)

    summary_rows = []
    for metric, direction in METRICS:
        base_values = baseline[metric]
        real_values = real[metric]
        real_effect = real[f"{metric}_effect_vs_baseline"]

        if len(null):
            null_effect = null[f"{metric}_effect_vs_baseline"].to_numpy(float)
            empirical_p = (1.0 + np.sum(null_effect >= real_effect.mean())) / (
                len(null_effect) + 1.0
            )
            shuffle_effect_mean = float(np.mean(null_effect))
            shuffle_effect_low = float(np.quantile(null_effect, 0.025))
            shuffle_effect_high = float(np.quantile(null_effect, 0.975))
            real_above_shuffle_95 = bool(real_effect.mean() > shuffle_effect_high)
            shuffle_metric_mean = float(null[metric].mean())
        else:
            empirical_p = np.nan
            shuffle_effect_mean = np.nan
            shuffle_effect_low = np.nan
            shuffle_effect_high = np.nan
            real_above_shuffle_95 = False
            shuffle_metric_mean = np.nan

        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline_mean": float(base_values.mean()),
                "real_domain_mean": float(real_values.mean()),
                "real_delta_vs_baseline": float((real_values - base_values).mean()),
                "real_effect_vs_baseline": float(real_effect.mean()),
                "real_ttest_p": safe_ttest(real_values, base_values),
                "real_wilcoxon_p": safe_wilcoxon(real_effect),
                "real_positive_seeds": int((real_effect > 0).sum()),
                "n_seeds": int(len(real_effect)),
                "shuffle_metric_mean": shuffle_metric_mean,
                "shuffle_effect_mean": shuffle_effect_mean,
                "shuffle_effect_ci95_low": shuffle_effect_low,
                "shuffle_effect_ci95_high": shuffle_effect_high,
                "empirical_p_real_vs_shuffle": float(empirical_p)
                if not np.isnan(empirical_p)
                else np.nan,
                "real_effect_above_shuffle_ci95": real_above_shuffle_95,
            }
        )

    return rows, null, pd.DataFrame(summary_rows)


def print_summary(summary):
    print("\nResumo do teste de permutacao de dominio")
    print(
        f"{'metric':18s} {'base':>8s} {'real':>8s} {'effect':>9s} "
        f"{'shuffle':>9s} {'p_emp':>7s} {'real>95%':>9s}"
    )
    for _, row in summary.iterrows():
        print(
            f"{row['metric']:18s} "
            f"{row['baseline_mean']:>8.4f} "
            f"{row['real_domain_mean']:>8.4f} "
            f"{row['real_effect_vs_baseline']:>+9.4f} "
            f"{row['shuffle_effect_mean']:>+9.4f} "
            f"{row['empirical_p_real_vs_shuffle']:>7.3f} "
            f"{str(row['real_effect_above_shuffle_ci95']):>9s}"
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    domain = data["domain"]

    print(
        "Domain permutation V2 | "
        f"datasets={len(domain)} | seeds={len(SEEDS)} | "
        f"permutations={N_PERMUTATIONS} | estimators={N_ESTIMATORS} | "
        f"n_jobs={N_JOBS}"
    )

    run_rows = []
    fold_rows = []

    for seed_idx, seed in enumerate(SEEDS, start=1):
        print(f"[{seed_idx:02d}/{len(SEEDS):02d}] seed={seed}")

        row, folds = evaluate_config(data, seed, "baseline_stats", None, None)
        run_rows.append(row)
        fold_rows.extend(folds)

        row, folds = evaluate_config(data, seed, "real_domain_rank", domain, None)
        run_rows.append(row)
        fold_rows.extend(folds)

        for perm_id in range(N_PERMUTATIONS):
            rng = np.random.default_rng(PERM_RANDOM_STATE + seed * 1009 + perm_id)
            shuffled_domain = rng.permutation(domain)
            row, folds = evaluate_config(
                data,
                seed,
                "shuffled_domain_rank",
                shuffled_domain,
                perm_id,
            )
            run_rows.append(row)
            fold_rows.extend(folds)

    runs = pd.DataFrame(run_rows)
    folds = pd.DataFrame(fold_rows)
    runs_with_effects, null, summary = summarize(runs)

    runs_with_effects.to_csv(OUT / f"{OUTPUT_PREFIX}_by_run.csv", index=False)
    folds.to_csv(OUT / f"{OUTPUT_PREFIX}_fold_metrics.csv", index=False)
    null.to_csv(OUT / f"{OUTPUT_PREFIX}_null_by_permutation.csv", index=False)
    summary.to_csv(OUT / f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print_summary(summary)
    print("\nArquivos salvos em data/top10_controlled/")
    print(f" - {OUTPUT_PREFIX}_by_run.csv")
    print(f" - {OUTPUT_PREFIX}_fold_metrics.csv")
    print(f" - {OUTPUT_PREFIX}_null_by_permutation.csv")
    print(f" - {OUTPUT_PREFIX}_summary.csv")


if __name__ == "__main__":
    main()
