from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=2,
        strategy="stat_plus_clusters_plus_domain_label_score_interaction",
        semantic_mode="clusters_only",
        domain_mode="label_score_interaction",
        output_filename="02_clusters_domain_interaction.csv",
    )
