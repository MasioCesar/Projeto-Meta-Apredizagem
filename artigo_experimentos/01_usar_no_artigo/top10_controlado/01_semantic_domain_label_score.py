from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=1,
        strategy="stat_plus_controlled_semantic_plus_domain_label_score",
        semantic_mode="tags_fixed_vocab",
        domain_mode="label_score",
        output_filename="01_semantic_domain_label_score.csv",
    )
