from common import run_lodo_experiment

if __name__ == "__main__":
    run_lodo_experiment(
        rank=1,
        strategy="stat_plus_tags_fixed_vocab_plus_domain_interaction",
        semantic_mode="tags_fixed_vocab",
        domain_mode="label_score_interaction",
        output_filename="1_tags_vocab_domain_score_LODO.csv",
    )