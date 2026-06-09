from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
TOP10_DIR = CURRENT_DIR / "top10_controlled"
sys.path.insert(0, str(TOP10_DIR))

from common import run_experiment  # noqa: E402


if __name__ == "__main__":
    run_experiment(
        rank="test31",
        strategy="stat_plus_clusters_plus_tags_plus_domain_no_score",
        semantic_mode="tags_clusters",
        domain_mode="label",
        output_filename="test31_clusters_tags_domain_no_score.csv",
    )
