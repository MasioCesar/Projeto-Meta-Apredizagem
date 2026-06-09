from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import run_experiment  # noqa: E402


if __name__ == "__main__":
    run_experiment(
        rank="test24",
        strategy="stat_plus_clean_text_tfidf_kmeans_cluster_plus_domain_label",
        semantic_mode="clusters_only",
        domain_mode="label",
        output_filename="test24_cluster_domain_no_score.csv",
    )
