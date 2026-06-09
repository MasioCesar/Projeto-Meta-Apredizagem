from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=7,
        strategy="stat_plus_tags_plus_domain_label_score_interaction",
        semantic_mode="tags_only",
        domain_mode="label_score_interaction",
        output_filename="07_tags_domain_interaction.csv",
    )
