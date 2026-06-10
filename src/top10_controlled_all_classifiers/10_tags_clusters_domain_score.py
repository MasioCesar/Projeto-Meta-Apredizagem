from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        rank=10,
        strategy="stat_plus_tags_clusters_plus_domain_score",
        semantic_mode="tags_clusters",
        domain_mode="score",
        output_filename="10_tags_clusters_domain_score.csv",
    )
