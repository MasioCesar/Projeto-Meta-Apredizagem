"""
ESTUDO ROBUSTO (base V2 ~580 datasets): gera matriz de performance + meta-features.
Modelado no 02_evaluate_and_extract.py, mas:
  - le data/selection_v2_preview.csv (base V2 deduplicada)
  - GRAVACAO INCREMENTAL + RESUMIVEL (pula dids ja processados)
  - tratamento de erro por dataset (download/pymfe/classificador)

Saidas: data/performance_matrix_v2.csv, data/metafeatures_v2.csv, data/errors_v2.csv
Dominio sera atribuido DEPOIS (keywords na descricao / LLM), sobre toda a base.
"""
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT = DATA / "selection_v2_preview.csv"
OUT_PERF = DATA / "performance_matrix_v2.csv"
OUT_META = DATA / "metafeatures_v2.csv"
OUT_ERR = DATA / "errors_v2.csv"

classifiers = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(random_state=42, kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=3000),
    "Perceptron": Perceptron(random_state=42, max_iter=1000),
    "MLP": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100,)),
}
HEAVY = {"SVM", "KNN", "MLP"}
MAX_ROWS_HEAVY, MAX_FEAT_HEAVY, MAX_FEAT_PYMFE = 10000, 1000, 1000


def load_ds(did):
    ds = openml.datasets.get_dataset(int(did), download_data=True,
                                     download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    return X, y


def build_pre(X):
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
    ], remainder="drop")


def valid_cv(y, max_splits=5):
    c = pd.Series(y).value_counts()
    if len(c) < 2:
        return None
    n = min(max_splits, c.min())
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=42) if n >= 2 else None


def manual_mf(Xp, y):
    n_i, n_f = Xp.shape
    d = pd.Series(y).value_counts(normalize=True)
    return {"nr_inst": n_i, "nr_attr": n_f, "nr_class": len(np.unique(y)),
            "attr_to_inst": n_f / n_i if n_i else 0, "inst_to_attr": n_i / n_f if n_f else 0,
            "class_entropy": -(d * np.log2(d + 1e-12)).sum(),
            "minority_class_ratio": d.min(), "majority_class_ratio": d.max()}


def extract_mf(X, y):
    pre = build_pre(X)
    Xp = np.nan_to_num(np.asarray(pre.fit_transform(X), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    feats = manual_mf(Xp, y)
    feats["pymfe_failed"] = 0
    try:
        if Xp.shape[1] > MAX_FEAT_PYMFE:
            cols = np.random.default_rng(42).choice(Xp.shape[1], MAX_FEAT_PYMFE, replace=False)
            Xm = Xp[:, cols]
        else:
            Xm = Xp
        mfe = MFE(groups=["general", "statistical"], summary=["mean", "sd"])
        mfe.fit(Xm, y)
        names, vals = mfe.extract()
        feats.update(dict(zip(names, vals)))
        return feats, None
    except Exception as e:
        feats["pymfe_failed"] = 1
        return feats, str(e)


def skip_heavy(name, X):
    return name in HEAVY and (X.shape[0] > MAX_ROWS_HEAVY or X.shape[1] > MAX_FEAT_HEAVY)


def load_done():
    done = set()
    if OUT_PERF.exists():
        done = set(pd.read_csv(OUT_PERF)["did"].astype(int))
    return done


def main():
    sel = pd.read_csv(INPUT)
    dids = sel["did"].dropna().astype(int).tolist()
    done = load_done()
    pending = [d for d in dids if d not in done]
    print(f"Base V2: {len(dids)} | ja feitos: {len(done)} | pendentes: {len(pending)}", flush=True)

    perf_rows = pd.read_csv(OUT_PERF).to_dict("records") if OUT_PERF.exists() else []
    meta_rows = pd.read_csv(OUT_META).to_dict("records") if OUT_META.exists() else []
    err_rows = pd.read_csv(OUT_ERR).to_dict("records") if OUT_ERR.exists() else []
    name_by_did = dict(zip(sel["did"].astype(int), sel["name"]))

    for i, did in enumerate(pending, 1):
        nm = name_by_did.get(did, "")
        try:
            X, y = load_ds(did)
            m = pd.Series(y).notna().to_numpy()
            X = X.loc[m].reset_index(drop=True)
            y = LabelEncoder().fit_transform(pd.Series(y)[m].astype(str))
            cv = valid_cv(y)
            if cv is None:
                err_rows.append({"did": did, "stage": "cv", "error": "classes insuficientes"})
                print(f"[{i}/{len(pending)}] did={did} pulado (cv)", flush=True)
                continue
            mf, mferr = extract_mf(X, y)
            mf.update({"did": did, "name": nm})
            meta_rows.append(mf)
            if mferr:
                err_rows.append({"did": did, "stage": "pymfe", "error": mferr[:100]})
            pre = build_pre(X)
            prow = {"did": did}
            for cn, clf in classifiers.items():
                if skip_heavy(cn, X):
                    prow[cn] = np.nan
                    continue
                try:
                    pipe = Pipeline([("pre", pre), ("sc", StandardScaler(with_mean=False)), ("clf", clf)])
                    sc = cross_validate(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1, error_score=np.nan)
                    prow[cn] = float(np.nanmean(sc["test_score"]))
                except Exception as e:
                    prow[cn] = np.nan
                    err_rows.append({"did": did, "stage": f"clf_{cn}", "error": str(e)[:100]})
            existing = [c for c in classifiers if c in prow and not (isinstance(prow[c], float) and np.isnan(prow[c]))]
            if existing:
                prow["best_classifier"] = max(existing, key=lambda c: prow[c])
                prow["best_accuracy"] = max(prow[c] for c in existing)
            perf_rows.append(prow)
            # grava incremental
            pd.DataFrame(perf_rows).to_csv(OUT_PERF, index=False)
            pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)
            if err_rows:
                pd.DataFrame(err_rows).to_csv(OUT_ERR, index=False)
            print(f"[{i}/{len(pending)}] did={did} ({X.shape[0]}x{X.shape[1]}) ok -> {prow.get('best_classifier','?')}", flush=True)
        except Exception as e:
            err_rows.append({"did": did, "stage": "dataset", "error": str(e)[:120]})
            if err_rows:
                pd.DataFrame(err_rows).to_csv(OUT_ERR, index=False)
            print(f"[{i}/{len(pending)}] did={did} FALHA: {str(e)[:50]}", flush=True)

    print(f"\nConcluido. Datasets com performance: {len(perf_rows)} | meta-features: {len(meta_rows)}", flush=True)


if __name__ == "__main__":
    main()
