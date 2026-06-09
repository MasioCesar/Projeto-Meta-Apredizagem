from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=6,
        strategy="stat_plus_tags_plus_domain_label",
        semantic_mode="tags_only",
        domain_mode="label",
        output_filename="06_tags_domain_label.csv",
    )
