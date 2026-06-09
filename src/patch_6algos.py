"""Patch: atualiza classifier_cols (3 -> 6 algoritmos) em todos os test scripts."""
import re
from pathlib import Path

SRC = Path(__file__).parent
SIX = (
    'classifier_cols = [\n'
    '        "DecisionTree",\n'
    '        "KNN",\n'
    '        "LogisticRegression",\n'
    '        "MLP",\n'
    '        "Perceptron",\n'
    '        "SVM"\n'
    '    ]'
)

# Match `classifier_cols = [...]` (assignment only), nao precedido por _/letra,
# evitando existing_classifier_cols / all_possible_classifier_cols.
pattern = re.compile(r'(?<![\w])classifier_cols = \[[^\]]*\]')

patched = []
for f in sorted(SRC.glob("04_experiment_b_test*.py")):
    text = f.read_text(encoding="utf-8")
    new_text, n = pattern.subn(SIX, text)
    if n > 0 and new_text != text:
        f.write_text(new_text, encoding="utf-8")
        patched.append((f.name, n))

for name, n in patched:
    print(f"  patched {name} ({n} ocorrencia)")
print(f"\nTotal: {len(patched)} arquivos atualizados.")
