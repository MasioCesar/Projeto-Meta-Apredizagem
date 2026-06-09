"""
Bateria com metrica TOP-3 ACCURACY: % de datasets em que o verdadeiro melhor
algoritmo esta entre os 3 mais provaveis previstos pelo meta-modelo.
Cobre as configuracoes dos testes do projeto (baseline + dominio/semantica/clusters).
Base V2 (metafeatures_v2_full), 3 sementes, CV aleatoria. Reporta top1 e top3.
Saida: data/top10_controlled/top3_accuracy_summary.csv
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import build_model

DATA = Path(__file__).resolve().parents[2] / "data"
SEEDS = [42, 7, 123]

# (rotulo, semantic_mode, domain_mode) cobrindo os testes do projeto
CONFIGS = [
    ("baseline (so estatistica)",          "none",             "none"),
    ("+ dominio label (test1/test8)",      "none",             "label"),
    ("+ dominio score (test22)",           "none",             "score"),
    ("+ dominio label+score",              "none",             "label_score"),
    ("+ tags (test21)",                    "tags_only",        "none"),
    ("+ vocabulario fixo (test21)",        "fixed_vocab",      "none"),
    ("+ tags+vocab (test1 controlado)",    "tags_fixed_vocab", "none"),
    ("+ clusters (test23/24)",             "clusters_only",    "none"),
    ("+ tags + dominio (test22)",          "tags_only",        "label"),
    ("+ tags+vocab + dominio (test22 top)","tags_fixed_vocab", "label"),
    ("+ clusters + interacao (test29)",    "clusters_only",    "label_score_interaction"),
    ("+ tudo (semantica+dominio+inter)",   "tags_fixed_vocab", "label_score_interaction"),
]


def topk_acc(proba, y_true, classes, k):
    """proba: (n, C); classes: ordem das colunas; y_true: rotulos codificados."""
    # indice da classe verdadeira em 'classes'
    topk = np.argsort(proba, axis=1)[:, -k:]  # indices das k maiores colunas
    hits = [int(y_true[i] in set(classes[topk[i]])) for i in range(len(y_true))]
    return np.mean(hits)


def main():
    meta = pd.read_csv(DATA/"metafeatures_v2_full.csv")
    perf = pd.read_csv(DATA/"performance_matrix_v2_full.csv")
    df = meta.merge(perf[["did","best_classifier"]], on="did", how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    numeric_cols = [c for c in meta.columns if c not in ("did","name","semantic_text","predicted_domain","domain_score") and pd.api.types.is_numeric_dtype(meta[c])]
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    df[numeric_cols] = df[numeric_cols].replace([np.inf,-np.inf], np.nan)
    for c in ["semantic_text","name","predicted_domain"]:
        df[c] = df[c].fillna("").astype(str)
    df["domain_score"] = pd.to_numeric(df["domain_score"], errors="coerce").fillna(0)
    le = LabelEncoder(); y = le.fit_transform(df["best_classifier"].astype(str))
    print(f"Base V2: {len(df)} datasets | {len(le.classes_)} algoritmos | metrica = TOP-3 accuracy\n")

    rows = []
    base_t3 = None
    print(f"{'configuracao':40s} {'top1':>7s} {'top3':>7s} {'vs base(top3)':>14s}")
    for label, sm, dm in CONFIGS:
        t1s, t3s = [], []
        for s in SEEDS:
            cv = StratifiedKFold(5, shuffle=True, random_state=s)
            model = build_model(numeric_cols, sm, dm)
            proba = cross_val_predict(model, df, y, cv=cv, method="predict_proba")
            # classes do modelo = ordenadas (RF usa np.unique(y))
            classes = np.unique(y)
            t1s.append(topk_acc(proba, y, classes, 1))
            t3s.append(topk_acc(proba, y, classes, 3))
        t1, t3 = np.mean(t1s), np.mean(t3s)
        if base_t3 is None:
            base_t3 = t3
            delta = "-"
        else:
            delta = f"{t3-base_t3:+.4f}"
        print(f"{label:40s} {t1:>7.3f} {t3:>7.3f} {delta:>14s}")
        rows.append({"config": label, "semantic_mode": sm, "domain_mode": dm,
                     "top1_acc": t1, "top3_acc": t3, "delta_top3_vs_baseline": (t3-base_t3)})

    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"top3_accuracy_summary.csv", index=False)
    print(f"\nSalvo: top3_accuracy_summary.csv")


if __name__ == "__main__":
    main()
