"""Figuras extras para o relatorio: crescimento da base e ruido do alvo."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
FIG = BASE / "artigo_final_sbc" / "figuras"; FIG.mkdir(parents=True, exist_ok=True)
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]

# ===== Crescimento/qualidade da base =====
fig, ax = plt.subplots(figsize=(6.6, 3.6))
etapas = ["Bruto\nOpenML","Filtro\ntécnico","Dedup\n(assinatura)","Base V2\n(final)"]
vals = [6408, 1491, 796, 547]
ax.bar(range(4), vals, color=["#cfd8dc","#90a4ae","#4db6ac","#1f77b4"], width=0.6)
for i,v in enumerate(vals): ax.text(i, v+120, f"{v}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(116, color="#c62828", ls="--", lw=1.5)
ax.text(3.45, 116+120, "base antiga: 116", color="#c62828", ha="right", fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels(etapas, fontsize=9)
ax.set_ylabel("Nº de datasets"); ax.set_ylim(0, 7000)
ax.set_title("Construção da base: 547 datasets de qualidade, sem viés (4,7× a base inicial)")
fig.tight_layout(); fig.savefig(FIG/"fig_base_crescimento.png", dpi=160); plt.close(fig)
print("ok fig_base_crescimento.png")

# ===== Ruido do alvo: distribuicao de margens =====
perf = pd.read_csv(BASE/"data"/"performance_matrix_v2.csv")
Y = perf[[a for a in ALL6 if a in perf.columns]].to_numpy(float)
def margin(r):
    v = np.sort(r[~np.isnan(r)])[::-1]
    return v[0]-v[1] if len(v)>=2 else np.nan
m = np.array([margin(r) for r in Y]); m = m[~np.isnan(m)]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.hist(np.clip(m,0,0.10)*100, bins=30, color="#1f77b4", alpha=0.85)
ax.axvline(np.median(m)*100, color="#c62828", lw=2, label=f"mediana = {np.median(m)*100:.2f} p.p.")
ax.axvline(1.0, color="#2e7d32", ls="--", lw=1.5, label="1 p.p. (ruído da CV)")
pct = (m<0.01).mean()*100
ax.text(0.55, ax.get_ylim()[1]*0.8, f"{pct:.0f}% dos datasets\ncom margem < 1 p.p.", fontsize=9, color="#333")
ax.set_xlabel("Margem entre melhor e 2º melhor algoritmo (p.p.)"); ax.set_ylabel("Nº de datasets")
ax.set_title("O alvo é ruidoso: algoritmos quase empatados na maioria dos datasets")
ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG/"fig_ruido_alvo.png", dpi=160); plt.close(fig)
print("ok fig_ruido_alvo.png")
