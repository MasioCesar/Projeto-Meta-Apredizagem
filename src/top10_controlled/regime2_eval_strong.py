"""
CAMINHO 1 - FASE 1: reavaliar com um portfolio de modelos FORTES e DIVERSOS,
para ver se os gaps de performance ficam maiores (alvo menos ruidoso) -> dando
chance a semantica/dominio.

Portfolio (8 familias distintas): HistGradientBoosting, RandomForest, ExtraTrees,
SVM-RBF, LogisticRegression, KNN, MLP, GaussianNB.
Subamostragem: max 2000 linhas (estratificado) p/ viabilizar modelos pesados.
CV 5-fold estratificada, acuracia. Reaproveita os 116 datasets (did do metafeatures).

Saida: data/performance_matrix_strong.csv + analise de margem no fim.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import openml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC

from common import INPUT_METAFEATURES

OUT = INPUT_METAFEATURES.parent / "performance_matrix_strong.csv"
MAX_ROWS = 2000

# modelos com n_jobs=1 (a paralelizacao fica no cross_val_score, evita oversubscription)
MODELS = {
    "HistGB": HistGradientBoostingClassifier(max_iter=200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=1),
    "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "LogReg": LogisticRegression(max_iter=1000, C=1.0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "GaussianNB": GaussianNB(),
}


def preprocessor(X):
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
    ], remainder="drop")


def subsample(X, y, max_rows=MAX_ROWS, seed=42):
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        take = max(1, int(round(len(ci) * max_rows / len(X))))
        idx.extend(rng.choice(ci, size=min(take, len(ci)), replace=False))
    idx = np.array(sorted(idx))
    return X.iloc[idx].reset_index(drop=True), y[idx]


def main():
    meta = pd.read_csv(INPUT_METAFEATURES)
    dids = meta["did"].dropna().astype(int).tolist()
    print(f"Avaliando portfolio forte em {len(dids)} datasets (max {MAX_ROWS} linhas)...\n")

    rows = []
    for i, did in enumerate(dids, 1):
        try:
            ds = openml.datasets.get_dataset(int(did), download_data=True,
                                             download_qualities=False, download_features_meta_data=False)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
        except Exception as e:
            print(f"  [{i}/{len(dids)}] did={did} FALHA download: {str(e)[:40]}")
            continue
        m = y.notna().to_numpy()
        X = X.loc[m].reset_index(drop=True)
        y = LabelEncoder().fit_transform(y[m].astype(str))
        if len(np.unique(y)) < 2:
            continue
        X, y = subsample(X, y)
        minc = pd.Series(y).value_counts().min()
        nsp = min(5, minc)
        if nsp < 2:
            continue
        cv = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=42)
        pre = preprocessor(X)
        row = {"did": did, "n_rows": len(X), "n_feat": X.shape[1]}
        for name, clf in MODELS.items():
            try:
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                sc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1, error_score=np.nan)
                row[name] = float(np.nanmean(sc))
            except Exception:
                row[name] = np.nan
        rows.append(row)
        # grava incremental + progresso (flush) para monitorar
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  [{i}/{len(dids)}] did={did} ({len(X)}x{X.shape[1]}) ok", flush=True)

    df = pd.DataFrame(rows)
    cols = list(MODELS.keys())
    df["best"] = df[cols].idxmax(axis=1)
    df.to_csv(OUT, index=False)
    print(f"\nSalvo: {OUT} ({len(df)} datasets avaliados)")

    # === analise de margem (porta de decisao) ===
    def margin(r):
        v = r[cols].dropna().sort_values(ascending=False).values
        return v[0] - v[1] if len(v) >= 2 else np.nan
    mg = df.apply(margin, axis=1)
    print("\n=== PORTA DE DECISAO: distribuicao de margens (portfolio forte) ===")
    print(f"  margem mediana = {mg.median():.4f}  (regime antigo 6 algos: 0.0064)")
    for thr in [0.01, 0.02, 0.05]:
        print(f"  margem < {thr:.2f}: {(mg<thr).mean()*100:.0f}%  (antigo: {'64%' if thr==0.01 else '81%' if thr==0.02 else '90%'})")
    print(f"\n  Distribuicao do melhor modelo:")
    print(df["best"].value_counts().to_string())
    print("\n  -> Se a margem mediana for >> 0.0064 e a distribuicao for diversa,")
    print("     o regime mudou e a semantica tem chance (seguir p/ Fase 2).")


if __name__ == "__main__":
    main()
