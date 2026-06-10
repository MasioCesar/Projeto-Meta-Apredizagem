"""Figura: ganho do dominio por dominio (revela a concentracao do efeito)."""
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = Path(__file__).resolve().parents[1] / "figuras"; FIG.mkdir(parents=True, exist_ok=True)

# (dominio, n, delta_pp, %algoritmo_dominante)
dados = [
    ("finance", 60, 7.67, "LR 58%"),
    ("text", 15, 3.33, "LR 40%"),
    ("image", 35, 2.57, "LR 31%"),
    ("biology", 143, 1.68, "LR 38%"),
    ("education", 40, 1.00, "LR 48%"),
    ("health", 48, 0.42, "LR 38%"),
    ("sensor_signal", 16, 0.00, "MLP 31%"),
    ("other (sintéticos)", 164, -1.10, "DT 32%"),
    ("social", 26, -3.46, "LR 42%"),
]
dados = sorted(dados, key=lambda x: x[2])
labels = [f"{d}  (n={n})" for d,n,_,_ in dados]
vals = [d for _,_,d,_ in dados]
conc = [c for *_,c in dados]
colors = ["#2e7d32" if v>0.3 else ("#c62828" if v<-0.3 else "#9aa0a6") for v in vals]

fig, ax = plt.subplots(figsize=(8.0, 4.4))
y = np.arange(len(dados))
ax.barh(y, vals, color=colors)
for i,(v,c) in enumerate(zip(vals,conc)):
    ax.text(v + (0.15 if v>=0 else -0.15), i, f"{v:+.2f} p.p.  [{c}]",
            va="center", ha="left" if v>=0 else "right", fontsize=8.5)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(1.19, color="#1f77b4", ls="--", lw=1.3)
ax.text(1.19, len(dados)-0.4, "média global +1,19", color="#1f77b4", fontsize=8, ha="center")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Ganho de acurácia do domínio (p.p.)"); ax.set_xlim(-5.2, 9.5)
ax.set_title("O efeito do domínio é concentrado: grande onde um algoritmo domina (finanças → LR 58%)")
fig.tight_layout(); fig.savefig(FIG/"fig_por_dominio.png", dpi=160); plt.close(fig)
print("ok fig_por_dominio.png")
