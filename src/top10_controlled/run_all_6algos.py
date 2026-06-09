"""Executa todos os experimentos top10 com 6 classificadores e imprime o ranking final."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "baseline.py",
    "01_semantic_domain_label_score.py",
    "02_clusters_domain_interaction.py",
    "03_fixed_vocab_domain_interaction.py",
    "04_clusters_domain_label_score.py",
    "05_semantic_domain_score.py",
    "06_tags_domain_label.py",
    "07_tags_domain_interaction.py",
    "08_domain_label_score.py",
    "09_tags_domain_score.py",
    "10_tags_clusters_domain_score.py",
]

HERE = Path(__file__).parent

for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Rodando: {script}")
    print('='*60)
    result = subprocess.run(
        [sys.executable, script],
        cwd=HERE,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"ERRO em {script} (returncode={result.returncode})")
