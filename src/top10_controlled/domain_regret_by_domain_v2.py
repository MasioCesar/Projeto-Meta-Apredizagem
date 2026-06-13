"""Regret by domain for the V2 domain rank encoding.

Question:
    In which domains does + dominio(rank) reduce regret compared with the
    statistical meta-feature baseline?

Protocol:
    - Same V2 data and grouped CV used by the final domain tests.
    - Compare baseline stats vs real domain rank encoding.
    - Encoding is fitted only on the training fold.
    - Save per-dataset predictions so regret can be aggregated by domain.

Outputs:
    data/top10_controlled/{OUTPUT_PREFIX}_observations.csv
    data/top10_controlled/{OUTPUT_PREFIX}_by_seed.csv
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
from sklearn.metrics import accuracy_score
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

N_SEEDS = int(os.getenv("DOMAIN_REGRET_N_SEEDS", str(len(DEFAULT_SEEDS))))
N_ESTIMATORS = int(os.getenv("DOMAIN_REGRET_N_ESTIMATORS", "300"))
N_JOBS = int(os.getenv("DOMAIN_REGRET_N_JOBS", "1"))
N_SPLITS = int(os.getenv("DOMAIN_REGRET_N_SPLITS", "5"))
MODEL_RANDOM_STATE = int(os.getenv("DOMAIN_REGRET_MODEL_RANDOM_STATE", "42"))
OUTPUT_PREFIX = os.getenv("DOMAIN_REGRET_OUTPUT_PREFIX", "domain_regret_by_domain")

SEEDS = list(DEFAULT_SEEDS[:N_SEEDS])
CONFIGS = {
    "baseline_stats": False,
    "real_domain_rank": True,
}


def safe_paired_ttest(a, b):
    try:
        _, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        return np.nan


def safe_wilcoxon(delta):
    try:
        _, p = stats.wilcoxon(delta)
        return float(p)
    except Exception:
        return np.nan


def load_data():
    meta = pd.read_csv(DATA / "metafeatures_v2.csv")
    perf = pd.read_csv(DATA / "performance_matrix_v2.csv")
    txt = {
        int(r["did"]): r
        for r in json.loads((DATA / "v2_descriptions.json").read_text(encoding="utf-8"))
    }

    cols = ["did", "best_classifier", "best_accuracy"] + [
        a for a in ALL6 if a in perf.columns
    ]
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
    perf_values = df[ALL6].to_numpy(float)

    return {
        "df": df,
        "x_stats": x_stats,
        "y": y,
        "groups": groups,
        "label_encoder": label_encoder,
        "perf_values": perf_values,
        "domain": df["dom"].to_numpy(),
        "algo_to_perf_idx": {name: i for i, name in enumerate(ALL6)},
    }


def build_features(data, train_idx, test_idx, use_domain):
    x_stats = data["x_stats"]
    x_train = x_stats[train_idx]
    x_test = x_stats[test_idx]

    if not use_domain:
        return x_train, x_test

    domain = data["domain"]
    perf_values = data["perf_values"]
    tab, prior = enc_perf(domain[train_idx], perf_values[train_idx], "rank")
    domain_train = apply_enc(domain[train_idx], tab, prior)
    domain_test = apply_enc(domain[test_idx], tab, prior)
    return np.hstack([x_train, domain_train]), np.hstack([x_test, domain_test])


def chosen_accuracy(pred_names, test_idx, perf_values, algo_to_perf_idx):
    chosen_cols = np.asarray([algo_to_perf_idx[name] for name in pred_names])
    return perf_values[test_idx, chosen_cols]


def evaluate_seed_config(data, seed, config, use_domain):
    x_stats = data["x_stats"]
    y = data["y"]
    groups = data["groups"]
    df = data["df"]
    perf_values = data["perf_values"]
    label_encoder = data["label_encoder"]

    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_stats, y, groups=groups), start=1):
        x_train, x_test = build_features(data, train_idx, test_idx, use_domain)
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

        pred = clf.predict(x_test)
        pred_names = label_encoder.inverse_transform(pred)
        true_names = label_encoder.inverse_transform(y[test_idx])
        chosen = chosen_accuracy(
            pred_names, test_idx, perf_values, data["algo_to_perf_idx"]
        )
        oracle = np.nanmax(perf_values[test_idx], axis=1)
        regret = oracle - chosen

        for local_i, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "seed": seed,
                    "config": config,
                    "fold": fold,
                    "did": int(df.loc[row_idx, "did"]),
                    "name": df.loc[row_idx, "name"],
                    "domain": df.loc[row_idx, "dom"],
                    "true_best": true_names[local_i],
                    "predicted": pred_names[local_i],
                    "hit_best": int(pred_names[local_i] == true_names[local_i]),
                    "oracle_accuracy": oracle[local_i],
                    "chosen_accuracy": chosen[local_i],
                    "regret": regret[local_i],
                    "zero_regret": int(regret[local_i] <= 1e-12)
                    if not np.isnan(regret[local_i])
                    else np.nan,
                    "within_1pp": int(regret[local_i] <= 0.01)
                    if not np.isnan(regret[local_i])
                    else np.nan,
                }
            )
    return rows


def aggregate_group(group):
    known = group.dropna(subset=["regret"])
    out = {
        "n_observations": int(len(group)),
        "n_unique_datasets": int(group["did"].nunique()),
        "regret_coverage": float(len(known) / len(group)) if len(group) else np.nan,
        "hit_best_rate": float(group["hit_best"].mean()) if len(group) else np.nan,
    }
    if len(known):
        out.update(
            {
                "mean_regret": float(known["regret"].mean()),
                "median_regret": float(known["regret"].median()),
                "p90_regret": float(known["regret"].quantile(0.90)),
                "zero_regret_rate": float(known["zero_regret"].mean()),
                "within_1pp_rate": float(known["within_1pp"].mean()),
                "chosen_accuracy_mean": float(known["chosen_accuracy"].mean()),
                "oracle_accuracy_mean": float(known["oracle_accuracy"].mean()),
            }
        )
    else:
        out.update(
            {
                "mean_regret": np.nan,
                "median_regret": np.nan,
                "p90_regret": np.nan,
                "zero_regret_rate": np.nan,
                "within_1pp_rate": np.nan,
                "chosen_accuracy_mean": np.nan,
                "oracle_accuracy_mean": np.nan,
            }
        )
    return pd.Series(out)


def build_seed_summary(obs):
    by_domain = (
        obs.groupby(["seed", "config", "domain"], dropna=False)
        .apply(aggregate_group)
        .reset_index()
    )
    all_domains = (
        obs.groupby(["seed", "config"], dropna=False)
        .apply(aggregate_group)
        .reset_index()
    )
    all_domains["domain"] = "ALL"
    return pd.concat([all_domains, by_domain], ignore_index=True, sort=False)


def summarize(seed_summary):
    base = seed_summary[seed_summary["config"] == "baseline_stats"]
    dom = seed_summary[seed_summary["config"] == "real_domain_rank"]
    joined = base.merge(
        dom,
        on=["seed", "domain"],
        suffixes=("_baseline", "_domain"),
        how="inner",
    )

    rows = []
    for domain, g in joined.groupby("domain", dropna=False):
        regret_reduction = g["mean_regret_baseline"] - g["mean_regret_domain"]
        hit_delta = g["hit_best_rate_domain"] - g["hit_best_rate_baseline"]
        zero_delta = g["zero_regret_rate_domain"] - g["zero_regret_rate_baseline"]
        within_delta = g["within_1pp_rate_domain"] - g["within_1pp_rate_baseline"]
        chosen_acc_delta = (
            g["chosen_accuracy_mean_domain"] - g["chosen_accuracy_mean_baseline"]
        )

        baseline_regret = g["mean_regret_baseline"].mean()
        domain_regret = g["mean_regret_domain"].mean()
        rel = (
            100 * regret_reduction.mean() / baseline_regret
            if baseline_regret and not np.isnan(baseline_regret)
            else np.nan
        )

        rows.append(
            {
                "domain": domain,
                "n_datasets": int(g["n_unique_datasets_baseline"].iloc[0]),
                "n_seeds": int(len(g)),
                "baseline_mean_regret": float(baseline_regret),
                "domain_mean_regret": float(domain_regret),
                "regret_reduction": float(regret_reduction.mean()),
                "regret_reduction_pct": float(rel),
                "regret_positive_seeds": int((regret_reduction > 0).sum()),
                "regret_ttest_p": safe_paired_ttest(
                    g["mean_regret_baseline"], g["mean_regret_domain"]
                ),
                "regret_wilcoxon_p": safe_wilcoxon(regret_reduction),
                "baseline_zero_regret_rate": float(g["zero_regret_rate_baseline"].mean()),
                "domain_zero_regret_rate": float(g["zero_regret_rate_domain"].mean()),
                "zero_regret_delta": float(zero_delta.mean()),
                "baseline_within_1pp_rate": float(g["within_1pp_rate_baseline"].mean()),
                "domain_within_1pp_rate": float(g["within_1pp_rate_domain"].mean()),
                "within_1pp_delta": float(within_delta.mean()),
                "baseline_hit_best_rate": float(g["hit_best_rate_baseline"].mean()),
                "domain_hit_best_rate": float(g["hit_best_rate_domain"].mean()),
                "hit_best_delta": float(hit_delta.mean()),
                "chosen_accuracy_delta": float(chosen_acc_delta.mean()),
                "regret_coverage_baseline": float(g["regret_coverage_baseline"].mean()),
                "regret_coverage_domain": float(g["regret_coverage_domain"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["sort_key"] = np.where(summary["domain"] == "ALL", 1, 0)
    summary = summary.sort_values(
        ["sort_key", "regret_reduction"], ascending=[False, False]
    ).drop(columns=["sort_key"])
    return summary


def print_summary(summary):
    cols = [
        "domain",
        "n_datasets",
        "baseline_mean_regret",
        "domain_mean_regret",
        "regret_reduction",
        "regret_reduction_pct",
        "regret_positive_seeds",
        "regret_wilcoxon_p",
        "zero_regret_delta",
        "hit_best_delta",
    ]
    view = summary[cols].copy()
    print("\nRegret por dominio")
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    domain_counts = data["df"]["dom"].value_counts().to_dict()
    print(
        "Regret by domain V2 | "
        f"datasets={len(data['df'])} | seeds={len(SEEDS)} | "
        f"estimators={N_ESTIMATORS} | n_jobs={N_JOBS}"
    )
    print(f"Dominios: {domain_counts}")

    rows = []
    for i, seed in enumerate(SEEDS, start=1):
        print(f"[{i:02d}/{len(SEEDS):02d}] seed={seed}")
        for config, use_domain in CONFIGS.items():
            rows.extend(evaluate_seed_config(data, seed, config, use_domain))

    observations = pd.DataFrame(rows)
    seed_summary = build_seed_summary(observations)
    summary = summarize(seed_summary)

    observations.to_csv(OUT / f"{OUTPUT_PREFIX}_observations.csv", index=False)
    seed_summary.to_csv(OUT / f"{OUTPUT_PREFIX}_by_seed.csv", index=False)
    summary.to_csv(OUT / f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print_summary(summary)
    print("\nArquivos salvos em data/top10_controlled/")
    print(f" - {OUTPUT_PREFIX}_observations.csv")
    print(f" - {OUTPUT_PREFIX}_by_seed.csv")
    print(f" - {OUTPUT_PREFIX}_summary.csv")


if __name__ == "__main__":
    main()
