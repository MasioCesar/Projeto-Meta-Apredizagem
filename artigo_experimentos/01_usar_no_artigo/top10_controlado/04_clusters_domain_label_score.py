from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=4,
        strategy="stat_plus_clusters_plus_domain_label_score",
        semantic_mode="clusters_only",
        domain_mode="label_score",
        output_filename="04_clusters_domain_label_score.csv",
    )
