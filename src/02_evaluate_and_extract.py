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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_SELECTION = DATA_DIR / "final_domain_selection.csv"
OUTPUT_PERFORMANCES = DATA_DIR / "performances_selected_datasets.csv"
OUTPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
OUTPUT_MATRIX = DATA_DIR / "performance_matrix.csv"
OUTPUT_ERRORS = DATA_DIR / "error_logs.csv"


classifiers = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=3000),
    "Perceptron": Perceptron(random_state=42, max_iter=1000),
    "SVM": SVC(
        random_state=42,
        kernel="rbf",
        max_iter=5000,  # Added to prevent infinite hangs on massive datasets
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "MLP": MLPClassifier(
        random_state=42,
        max_iter=1000,
        hidden_layer_sizes=(100,)
    )
}


# No classifier is skipped by dataset size anymore. These constants are kept
# only for compatibility with older references in this script.
HEAVY_CLASSIFIERS = set()
MAX_ROWS_FOR_HEAVY_MODELS = float('inf')
MAX_FEATURES_FOR_HEAVY_MODELS = float('inf')
MAX_FEATURES_FOR_PYMFE = float('inf')


def load_openml_dataset(did):
    dataset = openml.datasets.get_dataset(
        int(did),
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False
    )

    target = dataset.default_target_attribute

    if target is None:
        raise ValueError(f"Dataset {did} não possui target padrão.")

    X, y, _, _ = dataset.get_data(
        target=target,
        dataset_format="dataframe"
    )

    return X, y


def encode_target(y):
    y = pd.Series(y).reset_index(drop=True)

    valid_mask = y.notna().to_numpy()

    y_valid = y.loc[valid_mask].astype(str).to_numpy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_valid)

    y_encoded = np.asarray(y_encoded, dtype=int).ravel()

    return y_encoded, valid_mask

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )


def get_valid_cv(y_encoded, max_splits=5):
    counts = pd.Series(y_encoded).value_counts()

    if len(counts) < 2:
        return None

    min_class_count = counts.min()
    n_splits = min(max_splits, min_class_count)

    if n_splits < 2:
        return None

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )


def compute_manual_metafeatures(X_processed, y_encoded):
    X_processed = np.asarray(X_processed, dtype=float)

    X_processed = np.nan_to_num(
        X_processed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    n_instances, n_features = X_processed.shape
    n_classes = len(np.unique(y_encoded))

    class_dist = pd.Series(y_encoded).value_counts(normalize=True)

    return {
        "nr_inst": n_instances,
        "nr_attr": n_features,
        "nr_class": n_classes,
        "attr_to_inst": n_features / n_instances if n_instances > 0 else 0,
        "inst_to_attr": n_instances / n_features if n_features > 0 else 0,
        "class_entropy": -(class_dist * np.log2(class_dist + 1e-12)).sum(),
        "minority_class_ratio": class_dist.min(),
        "majority_class_ratio": class_dist.max(),
    }


def extract_statistical_metafeatures(X, y_encoded, max_features_for_pymfe=MAX_FEATURES_FOR_PYMFE):
    y_encoded = np.asarray(y_encoded, dtype=int).ravel()
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_processed = np.asarray(X_processed, dtype=float)

    X_processed = np.nan_to_num(
        X_processed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    n_instances, n_features = X_processed.shape

    manual_features = compute_manual_metafeatures(X_processed, y_encoded)

    manual_features["pymfe_failed"] = 0
    manual_features["pymfe_feature_sampling"] = 0
    manual_features["pymfe_original_nr_attr"] = n_features
    manual_features["pymfe_used_nr_attr"] = n_features

    try:
        if n_features > max_features_for_pymfe:
            rng = np.random.default_rng(42)

            selected_cols = rng.choice(
                n_features,
                size=max_features_for_pymfe,
                replace=False
            )

            X_for_mfe = X_processed[:, selected_cols]

            manual_features["pymfe_feature_sampling"] = 1
            manual_features["pymfe_used_nr_attr"] = max_features_for_pymfe
        else:
            X_for_mfe = X_processed

        mfe = MFE(
            groups=["general", "statistical"],
            summary=["mean", "sd"]
        )

        mfe.fit(X_for_mfe, y_encoded)
        names, values = mfe.extract()

        pymfe_features = dict(zip(names, values))

        result = {}
        result.update(manual_features)
        result.update(pymfe_features)

        return result, None

    except Exception as e:
        manual_features["pymfe_failed"] = 1
        return manual_features, str(e)


def should_skip_heavy_classifier(clf_name, X):
    return False


def get_completed_performance_keys(performances_df):
    if performances_df.empty:
        return set()

    required_cols = {"did", "Classifier", "acc_mean"}
    if not required_cols.issubset(performances_df.columns):
        return set()

    df = performances_df.copy()
    df["did"] = pd.to_numeric(df["did"], errors="coerce")
    df["acc_mean"] = pd.to_numeric(df["acc_mean"], errors="coerce")
    df = df.dropna(subset=["did", "Classifier", "acc_mean"])
    df = df[df["acc_mean"].between(0, 1, inclusive="both")]

    return {
        (int(row["did"]), str(row["Classifier"]))
        for _, row in df.iterrows()
    }


def prepare_output_frames(performance_results, metafeature_results):
    performances_df = pd.DataFrame(performance_results)

    if not performances_df.empty:
        performances_df = performances_df.drop_duplicates(
            subset=["did", "Classifier"],
            keep="last"
        )

    df_meta = pd.DataFrame(metafeature_results)

    if not df_meta.empty and "did" in df_meta.columns:
        df_meta = df_meta.drop_duplicates(
            subset=["did"],
            keep="last"
        )

    return performances_df, df_meta


def filter_resolved_error_logs(error_logs, performances_df):
    errors_df = pd.DataFrame(error_logs)

    if errors_df.empty or performances_df.empty:
        return errors_df

    required_cols = {"did", "stage", "classifier"}
    if not required_cols.issubset(errors_df.columns):
        return errors_df

    completed_keys = get_completed_performance_keys(performances_df)

    def is_resolved_classifier_error(row):
        stage = str(row.get("stage", ""))
        if stage not in {"classifier_skipped", "classifier_evaluation"}:
            return False

        did = pd.to_numeric(row.get("did"), errors="coerce")
        classifier = row.get("classifier")

        if pd.isna(did) or pd.isna(classifier):
            return False

        return (int(did), str(classifier)) in completed_keys

    resolved_mask = errors_df.apply(is_resolved_classifier_error, axis=1)
    return errors_df.loc[~resolved_mask].copy()


def save_partial_outputs(performance_results, metafeature_results, error_logs):
    performances_df, df_meta = prepare_output_frames(
        performance_results,
        metafeature_results,
    )
    errors_df = filter_resolved_error_logs(error_logs, performances_df)

    if not performances_df.empty:
        performances_df.to_csv(OUTPUT_PERFORMANCES, index=False)

        performance_matrix = performances_df.pivot_table(
            index="did",
            columns="Classifier",
            values="acc_mean"
        ).reset_index()

        classifier_cols = list(classifiers.keys())

        existing_classifier_cols = [
            col for col in classifier_cols
            if col in performance_matrix.columns
        ]

        if existing_classifier_cols:
            # Create a mask to only select rows that have at least one non-NaN value
            valid_mask = performance_matrix[existing_classifier_cols].notna().any(axis=1)

            performance_matrix.loc[valid_mask, "best_classifier"] = (
                performance_matrix.loc[valid_mask, existing_classifier_cols].idxmax(axis=1)
            )

            performance_matrix.loc[valid_mask, "best_accuracy"] = (
                performance_matrix.loc[valid_mask, existing_classifier_cols].max(axis=1)
            )

            performance_matrix.to_csv(OUTPUT_MATRIX, index=False)

    if not df_meta.empty:
        df_meta.to_csv(OUTPUT_METAFEATURES, index=False)

    if not errors_df.empty:
        errors_df.to_csv(OUTPUT_ERRORS, index=False)


def main():
    if not INPUT_SELECTION.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_SELECTION}\n"
            "Execute primeiro: python src/01_select_datasets.py"
        )

    final_domain_selection = pd.read_csv(INPUT_SELECTION)

    # ============================================================
    # 1. CARREGAR RESULTADOS JÁ EXISTENTES
    # ============================================================

    if OUTPUT_PERFORMANCES.exists():
        old_performances_df = pd.read_csv(OUTPUT_PERFORMANCES)
        performance_results = old_performances_df.to_dict("records")
    else:
        old_performances_df = pd.DataFrame()
        performance_results = []

    if OUTPUT_METAFEATURES.exists():
        old_metafeatures_df = pd.read_csv(OUTPUT_METAFEATURES)
        metafeature_results = old_metafeatures_df.to_dict("records")
    else:
        old_metafeatures_df = pd.DataFrame()
        metafeature_results = []

    if OUTPUT_ERRORS.exists():
        old_errors_df = pd.read_csv(OUTPUT_ERRORS)
        error_logs = old_errors_df.to_dict("records")
    else:
        old_errors_df = pd.DataFrame()
        error_logs = []

    # ============================================================
    # 2. IDENTIFICAR DATASETS JÁ PROCESSADOS COM SUCESSO
    # ============================================================

    processed_meta_dids = set()

    if not old_metafeatures_df.empty and "did" in old_metafeatures_df.columns:
        processed_meta_dids = set(old_metafeatures_df["did"].dropna().astype(int))

    expected_classifiers = list(classifiers.keys())
    completed_performance_keys = get_completed_performance_keys(old_performances_df)
    processed_perf_dids = {did for did, _ in completed_performance_keys}

    all_selected_dids = set(final_domain_selection["did"].dropna().astype(int))

    missing_classifiers_by_did = {}
    for did in all_selected_dids:
        missing_classifiers = [
            clf_name
            for clf_name in expected_classifiers
            if (did, clf_name) not in completed_performance_keys
        ]

        if missing_classifiers:
            missing_classifiers_by_did[did] = missing_classifiers

    processed_complete_dids = {
        did
        for did in all_selected_dids
        if did in processed_meta_dids and did not in missing_classifiers_by_did
    }

    pending_dids = {
        did
        for did in all_selected_dids
        if did not in processed_meta_dids or did in missing_classifiers_by_did
    }

    print("\nResumo inicial:")
    print(f"Datasets selecionados: {len(all_selected_dids)}")
    print(f"Algoritmos esperados por dataset: {', '.join(expected_classifiers)}")
    print(f"Datasets com meta-features: {len(processed_meta_dids)}")
    print(f"Datasets com alguma performance valida: {len(processed_perf_dids)}")
    print(f"Datasets com classificadores faltantes: {len(missing_classifiers_by_did)}")
    print(f"Datasets completos: {len(processed_complete_dids)}")
    print(f"Datasets pendentes: {len(pending_dids)}")

    if len(pending_dids) == 0:
        print("\nNenhum dataset pendente. Nada para processar.")
        return

    final_domain_selection = final_domain_selection[
        final_domain_selection["did"].astype(int).isin(pending_dids)
    ].copy()

    def pending_reason(did):
        reasons = []

        if did not in processed_meta_dids:
            reasons.append("meta-features")

        reasons.extend(missing_classifiers_by_did.get(did, []))
        return ", ".join(reasons)

    final_domain_selection["pendente"] = (
        final_domain_selection["did"].astype(int).apply(pending_reason)
    )

    print("\nRodando apenas datasets pendentes:")
    print(final_domain_selection[["did", "name", "predicted_domain", "pendente"]])

    # ============================================================
    # 3. PROCESSAR APENAS DATASETS PENDENTES
    # ============================================================

    for idx, row in final_domain_selection.reset_index(drop=True).iterrows():
        did = int(row["did"])
        dataset_name = row["name"]
        needs_metafeatures = did not in processed_meta_dids
        missing_classifiers = missing_classifiers_by_did.get(did, [])

        print(f"\n[{idx + 1}/{len(final_domain_selection)}] Dataset pendente: {dataset_name} | DID={did}")

        try:
            X, y = load_openml_dataset(did)

            if X is None or y is None:
                print("  Pulando: X ou y vazio.")

                error_logs.append({
                    "did": did,
                    "dataset": dataset_name,
                    "stage": "load_dataset",
                    "classifier": None,
                    "error": "X ou y vazio"
                })

                continue

            y_encoded, valid_mask = encode_target(y)

            X = X.iloc[np.where(valid_mask)[0]].reset_index(drop=True)
            y_encoded = np.asarray(y_encoded, dtype=int).ravel()

            if len(np.unique(y_encoded)) < 2:
                print("  Pulando: menos de 2 classes.")

                error_logs.append({
                    "did": did,
                    "dataset": dataset_name,
                    "stage": "target_validation",
                    "classifier": None,
                    "error": "menos de 2 classes"
                })

                continue

            cv = get_valid_cv(y_encoded)

            if cv is None:
                print("  Pulando: classes insuficientes para validação cruzada.")

                error_logs.append({
                    "did": did,
                    "dataset": dataset_name,
                    "stage": "cross_validation",
                    "classifier": None,
                    "error": "classes insuficientes para validação cruzada"
                })

                continue

            if needs_metafeatures:
                print("  Extraindo meta-features estatisticas...")

                meta_feats, mfe_error = extract_statistical_metafeatures(X, y_encoded)

                if meta_feats is not None:
                    meta_feats["did"] = did
                    meta_feats["name"] = dataset_name
                    meta_feats["predicted_domain"] = row["predicted_domain"]
                    meta_feats["domain_score"] = row["domain_score"]
                    meta_feats["semantic_text"] = row.get("semantic_text", "")

                    metafeature_results.append(meta_feats)

                if mfe_error is not None:
                    print(f"    PyMFE falhou, usando fallback manual. Erro: {mfe_error}")

                    error_logs.append({
                        "did": did,
                        "dataset": dataset_name,
                        "stage": "metafeature_extraction",
                        "classifier": None,
                        "error": mfe_error
                    })
            else:
                print("  Meta-features ja existem; reaproveitando.")

            if not missing_classifiers:
                print("  Nenhum classificador faltante para este dataset.")

            preprocessor = build_preprocessor(X)

            for clf_name, clf in classifiers.items():
                if clf_name not in missing_classifiers:
                    continue

                print(f"  Avaliando {clf_name}...", end=" ")

                model = Pipeline([
                    ("preprocess", preprocessor),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("classifier", clf)
                ])

                try:
                    cv_results = cross_validate(
                        model,
                        X,
                        y_encoded,
                        cv=cv,
                        scoring="accuracy",
                        return_train_score=False,
                        n_jobs=-1,
                        error_score=np.nan
                    )

                    fold_accs = cv_results["test_score"]

                    if np.isnan(fold_accs).all():
                        raise ValueError("Todos os folds retornaram NaN.")

                    result_row = {
                        "did": did,
                        "Dataset": dataset_name,
                        "domain": row["predicted_domain"],
                        "Classifier": clf_name,
                        "acc_mean": np.nanmean(fold_accs),
                        "acc_stddev": np.nanstd(fold_accs),
                        "train_time": np.nansum(cv_results["fit_time"]),
                        "test_time": np.nansum(cv_results["score_time"])
                    }

                    for i, acc in enumerate(fold_accs, start=1):
                        result_row[f"acc_fold{i}"] = acc

                    performance_results.append(result_row)

                    print("OK")

                except Exception as e:
                    print(f"Erro: {e}")

                    error_logs.append({
                        "did": did,
                        "dataset": dataset_name,
                        "stage": "classifier_evaluation",
                        "classifier": clf_name,
                        "error": str(e)
                    })

            # Salva parcial a cada dataset pendente processado
            save_partial_outputs(performance_results, metafeature_results, error_logs)

        except Exception as e:
            print(f"Erro ao processar dataset {dataset_name}: {e}")

            error_logs.append({
                "did": did,
                "dataset": dataset_name,
                "stage": "dataset_processing",
                "classifier": None,
                "error": str(e)
            })

            save_partial_outputs(performance_results, metafeature_results, error_logs)

    # ============================================================
    # 4. SALVAR RESULTADOS FINAIS UNIFICADOS
    # ============================================================

    performances_df, df_meta = prepare_output_frames(
        performance_results,
        metafeature_results,
    )

    if performances_df.empty:
        raise RuntimeError("Nenhum resultado de performance foi gerado.")

    performance_matrix = performances_df.pivot_table(
        index="did",
        columns="Classifier",
        values="acc_mean"
    ).reset_index()

    classifier_cols = list(classifiers.keys())

    existing_classifier_cols = [
        col for col in classifier_cols
        if col in performance_matrix.columns
    ]

    valid_mask = performance_matrix[existing_classifier_cols].notna().any(axis=1)

    performance_matrix.loc[valid_mask, "best_classifier"] = (
        performance_matrix.loc[valid_mask, existing_classifier_cols].idxmax(axis=1)
    )

    performance_matrix.loc[valid_mask, "best_accuracy"] = (
        performance_matrix.loc[valid_mask, existing_classifier_cols].max(axis=1)
    )

    performances_df.to_csv(OUTPUT_PERFORMANCES, index=False)
    df_meta.to_csv(OUTPUT_METAFEATURES, index=False)
    performance_matrix.to_csv(OUTPUT_MATRIX, index=False)

    errors_df = filter_resolved_error_logs(error_logs, performances_df)

    if not errors_df.empty:
        errors_df = errors_df.drop_duplicates(
            subset=["did", "stage", "classifier", "error"],
            keep="last"
        )

        errors_df.to_csv(OUTPUT_ERRORS, index=False)

    print("\nFinalizado!")
    print("Datasets avaliados:", performance_matrix.shape[0])
    print("Meta-features extraídas:", df_meta.shape[0])

    print("\nDistribuição do melhor classificador:")
    print(performance_matrix["best_classifier"].value_counts())

    print("\nArquivos salvos em:")
    print(f"- {OUTPUT_PERFORMANCES}")
    print(f"- {OUTPUT_METAFEATURES}")
    print(f"- {OUTPUT_MATRIX}")

    if not errors_df.empty:
        print(f"- {OUTPUT_ERRORS}")

if __name__ == "__main__":
    main()
