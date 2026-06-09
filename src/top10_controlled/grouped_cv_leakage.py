"""
DEMONSTRACAO DE VAZAMENTO: TF-IDF cru (nome+descricao+colunas) parece ajudar sob
CV aleatoria, mas DESABA sob CV agrupada por familia (leave-one-family-out),
provando que o ganho era memorizacao de identidade, nao conhecimento generalizavel.

Familia = componentes conexas por similaridade de Jaccard dos nomes de colunas
(datasets quase-duplicados, ex.: AP_*/OVA_* de microarray, compartilham sondas).

Alvo: 3 algoritmos. Compara stats-only vs stats+TFIDF-cru, sob KFold vs GroupKFold.
Saida: data/top10_controlled/grouped_cv_leakage_summary.csv
"""
import warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import INPUT_MATRIX, INPUT_METAFEATURES, OUTPUT_DIR, load_data

THREE = ["DecisionTree", "LogisticRegression", "Perceptron"]
DESC = INPUT_METAFEATURES.parent / "dataset_descriptions.json"


def build_families(dids):
    """União-find por Jaccard>0.5 dos conjuntos de nomes de colunas."""
    data = {r["did"]: set(str(r.get("feature_names_sample", "")).lower().split(", "))
            for r in json.loads(DESC.read_text(encoding="utf-8"))}
    parent = {d: d for d in dids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    dl = list(dids)
    for i in range(len(dl)):
        si = data.get(dl[i], set()) - {""}
        if not si:
            continue
        for j in range(i + 1, len(dl)):
            sj = data.get(dl[j], set()) - {""}
            if not sj:
                continue
            inter = len(si & sj); uni = len(si | sj)
            if uni and inter / uni >= 0.5:
                union(dl[i], dl[j])
    fam = {d: find(d) for d in dids}
    # remapeia para ids 0..k
    uniq = {f: i for i, f in enumerate(sorted(set(fam.values())))}
    return {d: uniq[fam[d]] for d in dids}


def raw_text_model(numeric_cols, max_features=20):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
            ("rawtext", TfidfVectorizer(max_features=max_features), "semantic_text"),
        ], remainder="drop")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])


def stat_model(numeric_cols):
    return Pipeline([
        ("pre", ColumnTransformer([
            ("stat", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
        ], remainder="drop")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])


def main():
    X, _, numeric_cols, _ = load_data()
    df_meta = pd.read_csv(INPUT_METAFEATURES); perf = pd.read_csv(INPUT_MATRIX)
    df_exp = df_meta.merge(perf, on="did", how="inner").reset_index(drop=True)
    y3 = df_exp[THREE].idxmax(axis=1).reset_index(drop=True)
    yenc = LabelEncoder().fit_transform(y3)
    dids = df_exp["did"].astype(int).tolist()

    fam = build_families(dids)
    groups = np.array([fam[d] for d in dids])
    n_fam = len(set(groups))
    sizes = pd.Series(groups).value_counts()
    print(f"{len(dids)} datasets agrupados em {n_fam} familias (por Jaccard de colunas).")
    print(f"Maiores familias (quase-duplicados): {sizes.head(6).tolist()}\n")

    Xc = X.copy().reset_index(drop=True)
    n_splits = 5
    rows = []
    SEEDS = [42, 7, 123]

    def mean_over_seeds(model_fn, grouped):
        accs = []
        for s in SEEDS:
            if grouped:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
                sc = cross_val_score(model_fn(), Xc, yenc, cv=cv, groups=groups, scoring="accuracy")
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
                sc = cross_val_score(model_fn(), Xc, yenc, cv=cv, scoring="accuracy")
            accs.append(sc.mean())
        return float(np.mean(accs))

    # baselines (stats) por protocolo
    base_rand = mean_over_seeds(lambda: stat_model(numeric_cols), grouped=False)
    base_grp = mean_over_seeds(lambda: stat_model(numeric_cols), grouped=True)
    print(f"Baseline (estatistica)  |  CV aleatoria acc={base_rand:.4f}  |  CV agrupada acc={base_grp:.4f}\n")
    print(f"{'TF-IDF cru':>10s} {'ganho random':>13s} {'ganho agrupado':>15s} {'veredito'}")

    for mf in [20, 50, 300]:
        t_rand = mean_over_seeds(lambda: raw_text_model(numeric_cols, mf), grouped=False)
        t_grp = mean_over_seeds(lambda: raw_text_model(numeric_cols, mf), grouped=True)
        g_rand = t_rand - base_rand
        g_grp = t_grp - base_grp
        verdict = "vazamento" if (g_rand > 0.005 and g_grp < g_rand - 0.005) else "sem ganho claro"
        print(f"{mf:>10d} {g_rand:>+13.4f} {g_grp:>+15.4f}   {verdict}")
        rows.append({"max_features": mf, "acc_text_random": t_rand, "acc_text_grouped": t_grp,
                     "gain_random": g_rand, "gain_grouped": g_grp,
                     "base_random": base_rand, "base_grouped": base_grp})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "grouped_cv_leakage_summary.csv", index=False)
    print("\n=== INTERPRETACAO ===")
    print("Vazamento => o ganho do texto e POSITIVO sob CV aleatoria e CAI sob CV agrupada")
    print("(o modelo so 'acertava' reconhecendo familias de datasets quase-iguais).")
    print(f"\nSalvo: {OUTPUT_DIR / 'grouped_cv_leakage_summary.csv'}")


if __name__ == "__main__":
    main()
