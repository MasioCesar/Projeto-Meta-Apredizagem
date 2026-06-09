"""Roda todos os test scripts (6 algoritmos) e reporta sucesso/falha + metricas."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent

# Geradores que exigem API key -> nao rodar aqui
SKIP = {
    "04_experiment_b_test26_generate_llm_features.py",
    "04_experiment_b_test28_generate_gemini_features.py",
}

scripts = sorted(SRC.glob("04_experiment_b_test*.py"))

results = []
for script in scripts:
    if script.name in SKIP:
        results.append((script.name, "PULADO (gerador/API)", ""))
        continue
    print(f"\n{'='*70}\n>>> {script.name}\n{'='*70}", flush=True)
    proc = subprocess.run(
        [sys.executable, script.name],
        cwd=SRC,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:], flush=True)
    status = "OK" if proc.returncode == 0 else f"FALHOU (rc={proc.returncode})"
    # extrai ultima linha de erro se falhou
    err_tail = ""
    if proc.returncode != 0:
        err_lines = [l for l in out.strip().splitlines() if l.strip()]
        err_tail = err_lines[-1] if err_lines else ""
    results.append((script.name, status, err_tail))

print(f"\n\n{'#'*70}\nRESUMO\n{'#'*70}")
for name, status, err in results:
    line = f"{status:24s} {name}"
    if err:
        line += f"  -> {err[:90]}"
    print(line)
