from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import run_experiment  # noqa: E402


if __name__ == "__main__":
    run_experiment(
        rank="test29",
        strategy="stat_plus_clusters_plus_domain_score_interaction_plus_tags",
        semantic_mode="tags_clusters",
        domain_mode="label_score_interaction",
        output_filename="test29_clusters_interaction_tags.csv",
    )
