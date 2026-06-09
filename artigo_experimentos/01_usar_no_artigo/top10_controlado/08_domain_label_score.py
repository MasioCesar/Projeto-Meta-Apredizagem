from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=8,
        strategy="stat_plus_domain_label_score",
        semantic_mode="none",
        domain_mode="label_score",
        output_filename="08_domain_label_score.csv",
    )
