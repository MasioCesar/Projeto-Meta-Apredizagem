from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=3,
        strategy="stat_plus_fixed_vocab_plus_domain_label_score_interaction",
        semantic_mode="fixed_vocab",
        domain_mode="label_score_interaction",
        output_filename="03_fixed_vocab_domain_interaction.csv",
    )
