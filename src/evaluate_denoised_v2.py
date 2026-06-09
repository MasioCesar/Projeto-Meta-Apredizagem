"""
(2) Re-avalia o alvo com CV REPETIDA (5 folds x 3 sementes = 15 estimativas) para
um best_classifier ESTAVEL (menos ruidoso), na base V2. Subamostra 3000 linhas.
Resumivel. Saida: data/performance_matrix_v2_denoised.csv
"""
import warnings
from pathlib import Path
import numpy as np, openml, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT = DATA / "metafeatures_v2.csv"
OUT = DATA / "performance_matrix_v2_denoised.csv"
MAXROWS = 3000

CLS = {
 "DecisionTree": DecisionTreeClassifier(random_state=42),
 "SVM": SVC(kernel="rbf", random_state=42),
 "KNN": KNeighborsClassifier(n_neighbors=5),
 "LogisticRegression": LogisticRegression(max_iter=3000),
 "Perceptron": Perceptron(max_iter=1000, random_state=42),
 "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}
HEAVY = {"SVM","KNN","MLP"}


def pre(X):
    num = X.select_dtypes(include=["number","bool"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number","bool"]).columns.tolist()
    return ColumnTransformer([
        ("n", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler(with_mean=False))]), num),
        ("c", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("e",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]), cat),
    ], remainder="drop")


def subsample(X, y, n=MAXROWS, seed=42):
    if len(X) <= n: return X, y
    rng = np.random.default_rng(seed); idx=[]
    for c in np.unique(y):
        ci = np.where(y==c)[0]; idx.extend(rng.choice(ci, min(len(ci), max(1,int(len(ci)*n/len(X)))), replace=False))
    idx = np.array(sorted(idx)); return X.iloc[idx].reset_index(drop=True), y[idx]


def main():
    dids = pd.read_csv(INPUT)["did"].dropna().astype(int).tolist()
    done = set(pd.read_csv(OUT)["did"].astype(int)) if OUT.exists() else set()
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    pending = [d for d in dids if d not in done]
    print(f"V2 denoised: {len(dids)} | feitos: {len(done)} | pendentes: {len(pending)}", flush=True)

    for i, did in enumerate(pending, 1):
        try:
            ds = openml.datasets.get_dataset(int(did), download_data=True, download_qualities=False, download_features_meta_data=False)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
            m = pd.Series(y).notna().to_numpy(); X = X.loc[m].reset_index(drop=True)
            y = LabelEncoder().fit_transform(pd.Series(y)[m].astype(str))
            X, y = subsample(X, y)
            minc = pd.Series(y).value_counts().min()
            if minc < 2: continue
            cv = RepeatedStratifiedKFold(n_splits=min(5, minc), n_repeats=3, random_state=42)
            p = pre(X); row = {"did": did}
            for cn, clf in CLS.items():
                if cn in HEAVY and X.shape[1] > 1000:
                    row[cn] = np.nan; continue
                try:
                    sc = cross_val_score(Pipeline([("p",p),("c",clf)]), X, y, cv=cv, scoring="accuracy", n_jobs=-1, error_score=np.nan)
                    row[cn] = float(np.nanmean(sc))
                except Exception:
                    row[cn] = np.nan
            ex = [c for c in CLS if c in row and not (isinstance(row[c],float) and np.isnan(row[c]))]
            if ex:
                row["best_classifier"] = max(ex, key=lambda c: row[c]); row["best_accuracy"] = max(row[c] for c in ex)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[{i}/{len(pending)}] did={did} ({X.shape[0]}x{X.shape[1]}) -> {row.get('best_classifier','?')}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(pending)}] did={did} FALHA: {str(e)[:40]}", flush=True)

    print(f"\nConcluido: {len(rows)} datasets", flush=True)


if __name__ == "__main__":
    main()
