"""
IDEIA DO USUARIO: restringir a datasets de MARGEM ALTA (vencedor claro) e testar
se o CONTEXTO (dominio-LLM + clusters de texto limpo) ajuda ali.

Alvo: 3 algoritmos (maiores margens). Subconjuntos por limiar de margem.
Para cada subconjunto reporta: N, distribuicao de vencedores (expoe trivialidade),
e baseline vs +contexto (mesmo modelo RF, 3 sementes, F1-macro, Wilcoxon).
Saida: data/top10_controlled/high_margin_context_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from common import OUTPUT_DIR, INPUT_MATRIX, INPUT_METAFEATURES, load_data, build_model

THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
SEEDS = [42, 7, 123]
ANN = INPUT_METAFEATURES.parent / "llm_domain_annotations.csv"
THRESHOLDS = [0.0, 0.02, 0.03, 0.05]


def context_model(numeric_cols):
    """estatistica + clusters de texto limpo + dominio-LLM (onehot+conf). Mesmo RF."""
    base = build_model(numeric_cols, "clusters_only", "none")
    ct_base = clone(base.named_steps["preprocess"])  # stat + clusters
    full = ColumnTransformer([
        ("base", ct_base, list(range(0))),  # placeholder, substituido abaixo
    ])
    # Em vez de aninhar, montamos um ColumnTransformer novo com tudo:
    return None  # (nao usado; ver build_full)


def build_full(numeric_cols, with_context):
    transformers = [("stat", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), numeric_cols)]
    if with_context:
        # clusters do texto limpo (reusa transformer do build_model)
        clu = build_model(numeric_cols, "clusters_only", "none")
        # extrai so o transformer de clusters do ColumnTransformer
        ct = clu.named_steps["preprocess"]
        for name, trans, cols in ct.transformers:
            if name != "statistical" and name != "stat":
                transformers.append((f"ctx_{name}", clone(trans), cols))
        transformers.append(("llm_dom", OneHotEncoder(handle_unknown="ignore"), ["llm_domain"]))
        transformers.append(("llm_conf", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                    ("sc", StandardScaler())]), ["llm_confidence"]))
    return Pipeline([("pre", ColumnTransformer(transformers, remainder="drop")),
                     ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    ann = pd.read_csv(ANN)
    annm = df_exp[["did"]].merge(ann, on="did", how="left")
    Xc = X.copy().reset_index(drop=True)
    Xc["llm_domain"] = annm["llm_domain"].fillna("unknown").astype(str).values
    Xc["llm_confidence"] = annm["llm_confidence"].fillna(3).astype(float).values

    Y3 = df_exp[THREE].to_numpy(dtype=float)
    y3 = pd.Series(THREE)[Y3.argmax(axis=1)].reset_index(drop=True)
    margin = np.sort(Y3, axis=1)[:, -1] - np.sort(Y3, axis=1)[:, -2]

    rows = []
    for thr in THRESHOLDS:
        mask = margin >= thr
        Xs = Xc.loc[mask].reset_index(drop=True)
        ys = y3.loc[mask].reset_index(drop=True)
        dist = dict(ys.value_counts())
        n = len(ys)
        # precisa de >=2 classes e amostras suficientes
        if len(dist) < 2 or min(dist.values()) < 5:
            print(f"\n=== margem >= {thr:.2f} | N={n} | {dist} -> classes insuficientes, pulado")
            continue
        yenc = LabelEncoder().fit_transform(ys)
        print(f"\n=== margem >= {thr:.2f} | N={n} | vencedores={dist} ===")
        for seed in SEEDS:
            pass
        # baseline vs contexto, 3 sementes
        deltas, pvals, f1b_list, f1c_list = [], [], [], []
        for seed in SEEDS:
            rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
            f1b = cross_val_score(build_full(numeric_cols, False), Xs, yenc, cv=rskf, scoring="f1_macro")
            f1c = cross_val_score(build_full(numeric_cols, True), Xs, yenc, cv=rskf, scoring="f1_macro")
            d = f1c - f1b
            try: _, pw = stats.wilcoxon(f1c, f1b)
            except ValueError: pw = np.nan
            deltas.append(d.mean()); pvals.append(pw); f1b_list.append(f1b.mean()); f1c_list.append(f1c.mean())
        print(f"  F1 baseline={np.mean(f1b_list):.4f} | F1 +contexto={np.mean(f1c_list):.4f} | "
              f"dF1={np.mean(deltas):+.4f} | p={[round(p,3) for p in pvals]}")
        sig = all(p < 0.05 for p in pvals) and np.mean(deltas) > 0
        print(f"  contexto ajuda de forma replicavel? {'SIM' if sig else 'NAO'}")
        rows.append({"threshold": thr, "N": n, "winners": str(dist),
                     "f1_baseline": np.mean(f1b_list), "f1_context": np.mean(f1c_list),
                     "delta_f1": float(np.mean(deltas)), "pvals": str([round(p,3) for p in pvals])})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "high_margin_context_summary.csv", index=False)
    print(f"\nSalvo: {OUTPUT_DIR / 'high_margin_context_summary.csv'}")


if __name__ == "__main__":
    main()
