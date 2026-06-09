"""Gera as figuras novas do artigo: diagnostico de ruido e historia da escala."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[1] / "figuras"
FIG.mkdir(exist_ok=True)

# ---- Figura 1: diagnostico de margem/ruido (3 vs 6 algoritmos) ----
labels = ["margem < 1 p.p.", "margem < 2 p.p.", "margem mediana"]
tres = [31, 41, 3.1]      # % , % , p.p.
seis = [64, 81, 0.64]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
x = np.arange(2)
w = 0.35
ax[0].bar(x - w/2, [31, 41], w, label="3 algoritmos", color="#4C72B0")
ax[0].bar(x + w/2, [64, 81], w, label="6 algoritmos", color="#C44E52")
ax[0].set_xticks(x); ax[0].set_xticklabels(["< 1 p.p.", "< 2 p.p."])
ax[0].set_ylabel("% de datasets")
ax[0].set_title("Datasets com rotulo em quase-empate")
ax[0].legend(); ax[0].set_ylim(0, 100)
for i, v in enumerate([31, 41]): ax[0].text(i - w/2, v + 2, f"{v}%", ha="center", fontsize=9)
for i, v in enumerate([64, 81]): ax[0].text(i + w/2, v + 2, f"{v}%", ha="center", fontsize=9)

ax[1].bar(["3 algoritmos", "6 algoritmos"], [3.1, 0.64], color=["#4C72B0", "#C44E52"], width=0.5)
ax[1].set_ylabel("margem mediana (p.p.)")
ax[1].set_title("Margem melhor vs. 2o melhor")
for i, v in enumerate([3.1, 0.64]): ax[1].text(i, v + 0.06, f"{v} p.p.", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_margem_ruido.png", dpi=160)
print("salvo:", FIG / "fig_margem_ruido.png")

# ---- Figura 2: historia da escala (F1-macro baseline vs semantica) ----
fig, ax = plt.subplots(figsize=(6.2, 3.8))
cenarios = ["3 algoritmos\n(alvo separavel)", "6 algoritmos\n(alvo ruidoso)"]
base = [0.409, 0.194]
sem = [0.424, 0.20]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, base, w, label="Baseline (estatistica)", color="#999999")
b2 = ax.bar(x + w/2, sem, w, label="+ semantica controlada", color="#4C72B0")
ax.set_xticks(x); ax.set_xticklabels(cenarios)
ax.set_ylabel("F1-Macro")
ax.set_title("Ganho semantico depende da separabilidade do alvo")
ax.legend(loc="upper right")
ax.set_ylim(0, 0.5)
ax.annotate("+1 a +2 p.p.\n(real, replicavel)", xy=(0.18, 0.43), fontsize=8.5, color="#1f5fa6")
ax.annotate("~0 p.p.\n(nao replica)", xy=(1.05, 0.24), fontsize=8.5, color="#a33")
for i, v in enumerate(base): ax.text(i - w/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(sem): ax.text(i + w/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_escala_semantica.png", dpi=160)
print("salvo:", FIG / "fig_escala_semantica.png")
