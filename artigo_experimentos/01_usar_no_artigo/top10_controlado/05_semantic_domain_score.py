from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=5,
        strategy="stat_plus_tags_fixed_vocab_plus_domain_score",
        semantic_mode="tags_fixed_vocab",
        domain_mode="score",
        output_filename="05_semantic_domain_score.csv",
    )
