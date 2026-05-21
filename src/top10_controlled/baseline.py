from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank="baseline",
        strategy="baseline_statistical_only_no_domain_score",
        semantic_mode="none",
        domain_mode="none",
        output_filename="baseline.csv",
    )
