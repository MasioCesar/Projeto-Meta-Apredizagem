"""
TESTE A (versao JUSTA): embeddings densos REDUZIDOS via PCA.

Jogar 384 dims ao lado de 63 estatisticas afoga o sinal (maldicao da
dimensionalidade). Aqui reduzimos os embeddings a poucos componentes (PCA
DENTRO do pipeline -> ajustado so no treino, sem vazamento de CV) antes de
concatenar com as estatisticas. Se NEM assim ajudar, a conclusao e definitiva.

Testa n_components in {10, 20, 30} no texto LIMPO, em 3 alvos, CV repetida + t pareado.
Saida: data/top10_controlled/embeddings_pca_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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
COMPONENTS = [10, 20, 30]


def build(numeric_cols, emb_cols, n_comp):
    transformers = [
        ("statistical", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", StandardScaler())]), numeric_cols),
    ]
    if emb_cols is not None:
        transformers.append(("embeddings", Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=42)),
        ]), emb_cols))
    return Pipeline([
        ("pre", ColumnTransformer(transformers, remainder="drop")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42,
                                         class_weight="balanced")),
    ])


def main():
    X, y, numeric_cols, _ = load_data()
    print(f"Datasets: {len(X)} | numeric_cols: {len(numeric_cols)}")
    model = SentenceTransformer(MODEL_NAME)
    clean_text = [clean_semantic_text(t, n) for t, n in zip(X["semantic_text"], X["name"])]
    emb = np.asarray(model.encode(clean_text, normalize_embeddings=True), dtype=float)
    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    Xe = X.copy().reset_index(drop=True)
    Xe[emb_cols] = emb

    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    perf["best_classifier"] = perf[SIMPLE + COMPLEX].idxmax(axis=1, skipna=True)
    perf = perf.dropna(subset=["best_classifier"])
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    gap = (df_exp[COMPLEX].max(axis=1) - df_exp[SIMPLE].max(axis=1)).reset_index(drop=True)

    target_defs = {
        "6_classes": y.values,
        "3_way": y.map(GROUP3).values,
        "complex_worth_m0.01": np.where(gap > 0.01, "complex_worth", "simple_enough"),
    }
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []
    for tname, yt in target_defs.items():
        yenc = LabelEncoder().fit_transform(yt)
        print(f"\n===== {tname} =====")
        base = cross_val_score(build(numeric_cols, None, 0), Xe, yenc, cv=rskf, scoring="f1_macro")
        print(f"  baseline_stat_only            F1m {base.mean():.4f} +/- {base.std():.4f}")
        rows.append({"target": tname, "strategy": "baseline", "n_comp": 0,
                     "f1_macro_mean": base.mean(), "f1_macro_std": base.std(), "p_value": np.nan})
        for nc in COMPONENTS:
            f1 = cross_val_score(build(numeric_cols, emb_cols, nc), Xe, yenc, cv=rskf, scoring="f1_macro")
            d = f1 - base
            t, p = stats.ttest_rel(f1, base)
            print(f"  stat_plus_emb_pca{nc:<3d}        F1m {f1.mean():.4f} +/- {f1.std():.4f} | "
                  f"delta {d.mean():+.4f} | vence {int((d>0).sum())}/{len(d)} | p={p:.4f}")
            rows.append({"target": tname, "strategy": f"emb_pca{nc}", "n_comp": nc,
                         "f1_macro_mean": f1.mean(), "f1_macro_std": f1.std(),
                         "delta": float(d.mean()), "p_value": float(p)})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "embeddings_pca_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
