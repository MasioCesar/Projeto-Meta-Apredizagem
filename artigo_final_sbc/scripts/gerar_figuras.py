from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / "figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def percent(value):
    return value * 100


def save_bar_comparison():
    data = pd.DataFrame([
        {
            "Experimento": "Baseline\nestatistica",
            "Acuracia": 0.655435,
            "F1-Macro": 0.398791,
        },
        {
            "Experimento": "Teste 1\ntexto bruto",
            "Acuracia": 0.689493,
            "F1-Macro": 0.438740,
        },
        {
            "Experimento": "Teste 20\ntexto limpo",
            "Acuracia": 0.664493,
            "F1-Macro": 0.401785,
        },
        {
            "Experimento": "Proposto\nclusters+dominio",
            "Acuracia": 0.724275,
            "F1-Macro": 0.457884,
        },
    ])

    x = range(len(data))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar([i - width / 2 for i in x], data["Acuracia"].map(percent), width, label="Acuracia", color="#2f6f73")
    ax.bar([i + width / 2 for i in x], data["F1-Macro"].map(percent), width, label="F1-Macro", color="#b65f35")
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["Experimento"])
    ax.set_ylim(0, 80)
    ax.set_ylabel("Resultado medio (%)")
    ax.set_title("Evolucao dos principais experimentos")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_principais_resultados.png", dpi=220)
    plt.close(fig)


def save_domain_ablation():
    df = pd.read_csv(BASE_DIR / "data" / "experiment_b_test22_domain_strategies.csv")
    selected = [
        "stat_only_no_domain_score",
        "stat_plus_domain_label",
        "stat_plus_domain_score",
        "stat_plus_domain_label_score",
        "stat_plus_semantic_no_domain_score",
        "stat_plus_semantic_domain_score",
    ]
    labels = {
        "stat_only_no_domain_score": "Estatistica",
        "stat_plus_domain_label": "+ dominio",
        "stat_plus_domain_score": "+ score",
        "stat_plus_domain_label_score": "+ dominio\n+ score",
        "stat_plus_semantic_no_domain_score": "+ cluster\nsemantico",
        "stat_plus_semantic_domain_score": "+ cluster\n+ score",
    }
    plot_df = df[df["strategy"].isin(selected)].copy()
    plot_df["label"] = plot_df["strategy"].map(labels)
    plot_df["order"] = plot_df["strategy"].map({name: idx for idx, name in enumerate(selected)})
    plot_df = plot_df.sort_values("order")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(plot_df["label"], plot_df["accuracy_mean"].map(percent), marker="o", label="Acuracia", color="#2f6f73", linewidth=2)
    ax.plot(plot_df["label"], plot_df["f1_macro_mean"].map(percent), marker="s", label="F1-Macro", color="#b65f35", linewidth=2)
    ax.set_ylim(35, 75)
    ax.set_ylabel("Resultado medio (%)")
    ax.set_title("Ablacao: efeito de dominio, score e semantica")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_ablacao_dominio_score.png", dpi=220)
    plt.close(fig)


def save_cluster_search():
    df = pd.read_csv(BASE_DIR / "data" / "top10_controlled" / "test23_cluster_search.csv")
    top = df.head(10).copy()
    top["config"] = (
        top["n_clusters"].astype(str)
        + " cl. / "
        + top["max_features"].astype(str)
        + " termos / "
        + top["ngram_range"].astype(str)
        + " / "
        + top["domain_mode"].astype(str).str.replace("_", " ")
    )
    top = top.sort_values("f1_macro_mean")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(top["config"], top["f1_macro_mean"].map(percent), color="#456990")
    ax.set_xlim(38, 48)
    ax.set_xlabel("F1-Macro medio (%)")
    ax.set_title("Busca de hiperparametros dos clusters semanticos")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_busca_clusters.png", dpi=220)
    plt.close(fig)


def save_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off")

    boxes = [
        (0.04, 0.58, "Meta-features\nestatisticas"),
        (0.04, 0.18, "Texto semantico\noriginal"),
        (0.28, 0.18, "Limpeza\nanti-vazamento"),
        (0.48, 0.18, "TF-IDF\nautomatico"),
        (0.66, 0.18, "KMeans\ncluster semantico"),
        (0.66, 0.58, "Dominio + score\n+ interacao"),
        (0.84, 0.38, "Random Forest\nmeta-modelo"),
    ]

    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.14, 0.22, fill=True, facecolor="#f3f1e7", edgecolor="#333333", linewidth=1.2))
        ax.text(x + 0.07, y + 0.11, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((0.18, 0.29), (0.28, 0.29)),
        ((0.42, 0.29), (0.48, 0.29)),
        ((0.62, 0.29), (0.66, 0.29)),
        ((0.80, 0.29), (0.84, 0.44)),
        ((0.80, 0.69), (0.84, 0.49)),
        ((0.18, 0.69), (0.84, 0.49)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#333333"))

    ax.set_title("Arquitetura do modelo proposto sem vocabulario fixo", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_pipeline_modelo_proposto.png", dpi=220)
    plt.close(fig)


def save_tables():
    top10 = pd.read_csv(BASE_DIR / "data" / "top10_controlled" / "top10_plus_baseline_summary.csv")
    no_fixed = top10[~top10["semantic_mode"].astype(str).str.contains("fixed", case=False, na=False)].copy()
    no_fixed.to_csv(OUT_DIR / "tabela_top_sem_vocabulario_fixo.csv", index=False)


def main():
    save_bar_comparison()
    save_domain_ablation()
    save_cluster_search()
    save_pipeline_diagram()
    save_tables()
    print(f"Figuras salvas em: {OUT_DIR}")


if __name__ == "__main__":
    main()
