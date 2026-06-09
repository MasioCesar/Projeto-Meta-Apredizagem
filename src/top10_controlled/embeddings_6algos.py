"""
TESTE A: embeddings densos locais (sentence-transformers, SEM API) como features
semanticas. Ultimo lever para a hipotese "semantica importa".

Duas variantes:
  - CLEAN: embed do texto LIMPO (clean_semantic_text remove nome/URL/numeros/
           atributos) -> controlado, baixo vazamento.
  - RAW:   embed do texto CRU -> teto OTIMISTA (alto risco de vazamento).

Modelo: all-MiniLM-L6-v2 (384 dims, ~80MB, baixado uma vez). Os embeddings sao
pre-computados 1x (sao deterministicos por dataset, sem risco de leakage de CV)
e entram como colunas numericas adicionais ao lado das estatisticas.

Avalia em 3 alvos (6 classes, 3 vias, complexo-vale margem 0.01) com CV repetida
e teste-t pareado. NAO altera meta-features estatisticas.
Saida: data/top10_controlled/embeddings_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sentence_transformers import SentenceTransformer

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, clean_semantic_text, load_data

MODEL_NAME = "all-MiniLM-L6-v2"
SIMPLE = ["DecisionTree", "LogisticRegression", "Perceptron"]
COMPLEX = ["SVM", "MLP", "KNN"]
GROUP3 = {"DecisionTree": "tree", "LogisticRegression": "linear", "Perceptron": "linear",
          "SVM": "complex", "MLP": "complex", "KNN": "complex"}


def embed(texts, model):
    emb = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=float)


def build_with_emb(numeric_cols, emb_cols):
    """ColumnTransformer: estatisticas (intactas) + embeddings (colunas extra)."""
    transformers = [
        ("statistical", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", StandardScaler())]), numeric_cols),
    ]
    if emb_cols is not None:
        transformers.append(("embeddings", Pipeline([("sc", StandardScaler())]), emb_cols))
    return Pipeline([
        ("pre", ColumnTransformer(transformers, remainder="drop")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42,
                                         class_weight="balanced")),
    ])


def targets(X):
    df_meta = pd.read_csv(INPUT_METAFEATURES)
    perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)
    return gap


def main():
    X, y, numeric_cols, _ = load_data()
    print(f"Datasets: {len(X)} | numeric_cols: {len(numeric_cols)}")
    print(f"Carregando modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # textos
    raw_text = X["semantic_text"].astype(str).tolist()
    clean_text = [clean_semantic_text(t, n) for t, n in zip(X["semantic_text"], X["name"])]

    # embeddings -> colunas no X
    emb_clean = embed(clean_text, model)
    emb_raw = embed(raw_text, model)
    clean_cols = [f"emb_clean_{i}" for i in range(emb_clean.shape[1])]
    raw_cols = [f"emb_raw_{i}" for i in range(emb_raw.shape[1])]
    Xe = X.copy().reset_index(drop=True)
    Xe[clean_cols] = emb_clean
    Xe[raw_cols] = emb_raw

    gap = targets(X)
    assert len(gap) == len(Xe)

    target_defs = {
        "6_classes": y.values,
        "3_way": y.map(GROUP3).values,
        "complex_worth_m0.01": np.where(gap > 0.01, "complex_worth", "simple_enough"),
    }
    strategies = [
        ("baseline_stat_only", None),
        ("stat_plus_emb_CLEAN", clean_cols),
        ("stat_plus_emb_RAW_leaky", raw_cols),
    ]

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []
    for tname, yt in target_defs.items():
        yenc = LabelEncoder().fit_transform(yt)
        dist = dict(pd.Series(yt).value_counts())
        print(f"\n===== {tname} | {dist} =====")
        scores = {}
        for lbl, cols in strategies:
            f1 = cross_val_score(build_with_emb(numeric_cols, cols), Xe, yenc, cv=rskf, scoring="f1_macro")
            acc = cross_val_score(build_with_emb(numeric_cols, cols), Xe, yenc, cv=rskf, scoring="accuracy")
            scores[lbl] = f1
            rows.append({"target": tname, "strategy": lbl, "acc_mean": acc.mean(),
                         "f1_macro_mean": f1.mean(), "f1_macro_std": f1.std()})
            print(f"  {lbl:26s} acc {acc.mean():.4f}  F1m {f1.mean():.4f} +/- {f1.std():.4f}")
        base = scores["baseline_stat_only"]
        for lbl in ["stat_plus_emb_CLEAN", "stat_plus_emb_RAW_leaky"]:
            d = scores[lbl] - base
            t, p = stats.ttest_rel(scores[lbl], base)
            print(f"  >> {lbl:26s} delta {d.mean():+.4f} | vence {int((d>0).sum())}/{len(d)} | p={p:.4f}")

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "embeddings_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
