"""Gera as 3 figuras finais do artigo/relatorio."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
FIG = BASE / "artigo_final_sbc" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
C_BASE, C_DOM, C_OK, C_NO = "#9aa0a6", "#1f77b4", "#2e7d32", "#c62828"

# ===== Figura 1: resultado principal (baseline vs +dominio) =====
fig, ax = plt.subplots(figsize=(6.2, 3.8))
metrics = ["Acurácia", "F1-macro"]
base = [0.406, 0.297]; dom = [0.418, 0.304]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w/2, base, w, label="Baseline (estatística)", color=C_BASE)
b2 = ax.bar(x + w/2, dom, w, label="+ domínio (target encoding por rank)", color=C_DOM)
for i,(bv,dv) in enumerate(zip(base,dom)):
    ax.text(i-w/2, bv+0.004, f"{bv:.3f}", ha="center", fontsize=9)
    ax.text(i+w/2, dv+0.004, f"{dv:.3f}", ha="center", fontsize=9, fontweight="bold")
    delta = (dv-bv)*100
    ax.annotate(f"+{delta:.2f} p.p.\n(p<0,001)", xy=(i+w/2, dv), xytext=(i+w/2, dv+0.045),
                ha="center", fontsize=8.5, color=C_OK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Desempenho")
ax.set_ylim(0, 0.50); ax.set_title("Domínio melhora a seleção de algoritmos (547 datasets, CV agrupada)")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG/"fig_resultado_principal.png", dpi=160); plt.close(fig)
print("ok fig_resultado_principal.png")

# ===== Figura 2: comparacao de codificacoes (delta F1) =====
fig, ax = plt.subplots(figsize=(7.0, 3.8))
labels = ["Rank médio\n(supervisionada)", "P(é o melhor)\n(supervisionada)",
          "Clusters\n(embeddings)", "One-hot / TF-IDF /\nembeddings"]
deltas = [0.69, 0.44, -0.26, 0.0]; ps = ["p<0,001", "p=0,06", "n.s.", "n.s."]
colors = [C_OK, "#90caf9", C_NO, C_NO]
bars = ax.bar(range(4), deltas, color=colors, width=0.6)
for i,(d,p) in enumerate(zip(deltas,ps)):
    va = "bottom" if d>=0 else "top"; off = 0.04 if d>=0 else -0.04
    ax.text(i, d+off, f"{d:+.2f} p.p.\n{p}", ha="center", va=va, fontsize=8.5,
            fontweight=("bold" if i==0 else "normal"))
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Ganho de F1-macro (p.p.)"); ax.set_ylim(-0.7, 1.1)
ax.set_title("Só a codificação supervisionada por rank revela o sinal de domínio")
fig.tight_layout(); fig.savefig(FIG/"fig_codificacoes.png", dpi=160); plt.close(fig)
print("ok fig_codificacoes.png")

# ===== Figura 3: regret (sistema) =====
perf = pd.read_csv(BASE/"data"/"performance_matrix_v2.csv")
Y = perf[[a for a in ALL6 if a in perf.columns]].to_numpy(float)
Y = Y[~np.isnan(Y).any(axis=1)]
oracle = Y.max(axis=1)
sba = oracle - Y[:, np.nanmean(Y,axis=0).argmax()]
rng = np.random.default_rng(0)
rand = np.mean([oracle[i]-Y[i][rng.integers(Y.shape[1])] for i in range(len(Y))])
regrets = {"Aleatório": float(rand), "Single Best\nAlgorithm": float(sba.mean()),
           "Meta-learner\n(estatística)": 0.0226, "Oráculo": 0.0}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
names = list(regrets); vals = [regrets[n] for n in names]
cols = ["#bdbdbd", "#bdbdbd", C_DOM, C_OK]
bars = ax.bar(range(len(names)), vals, color=cols, width=0.6)
for i,v in enumerate(vals):
    ax.text(i, v+0.0012, f"{v:.4f}", ha="center", fontsize=9, fontweight=("bold" if i==2 else "normal"))
red = (1 - regrets["Meta-learner\n(estatística)"]/regrets["Single Best\nAlgorithm"])*100
ax.annotate(f"−{red:.0f}% vs SBA", xy=(2,0.0226), xytext=(2,0.0226+0.012), ha="center",
            fontsize=9, color=C_OK, fontweight="bold")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("Regret de seleção (menor = melhor)")
ax.set_title("O meta-learner estatístico reduz o regret em ~33% vs SBA")
fig.tight_layout(); fig.savefig(FIG/"fig_regret.png", dpi=160); plt.close(fig)
print("ok fig_regret.png")
