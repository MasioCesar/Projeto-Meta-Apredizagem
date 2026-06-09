from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=9,
        strategy="stat_plus_tags_plus_domain_score",
        semantic_mode="tags_only",
        domain_mode="score",
        output_filename="09_tags_domain_score.csv",
    )
