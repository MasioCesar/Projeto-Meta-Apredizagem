"""Torna os caminhos de input sobrescrevíveis por env (MAB_META/MAB_PERF),
de forma NAO-destrutiva (sem env -> arquivos originais). Aplica em common.py e
em todos os scripts que leem os CSVs originais."""
from pathlib import Path

SRC = Path(__file__).parent
REPL = [
    ('"metafeatures_selected_datasets.csv"',
     '__import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")'),
    ('"performance_matrix.csv"',
     '__import__("os").environ.get("MAB_PERF", "performance_matrix.csv")'),
]

targets = list(SRC.glob("04_experiment_b_test*.py")) + [SRC / "top10_controlled" / "common.py"]
patched = []
for f in targets:
    txt = f.read_text(encoding="utf-8")
    orig = txt
    for a, b in REPL:
        if a in txt and b not in txt:  # evita dupla aplicacao
            txt = txt.replace(a, b)
    if txt != orig:
        f.write_text(txt, encoding="utf-8")
        patched.append(f.name)

print(f"Patched {len(patched)} arquivos:")
for n in patched:
    print(" ", n)
