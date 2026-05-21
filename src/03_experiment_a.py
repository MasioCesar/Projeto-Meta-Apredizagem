# from pathlib import Path

# import numpy as np
# import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import StratifiedKFold, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder, StandardScaler


# BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = BASE_DIR / "data"

# INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
# INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"
# OUTPUT_RESULTS = DATA_DIR / "experiment_a_results.csv"


# def main():
#     if not INPUT_METAFEATURES.exists():
#         raise FileNotFoundError(
#             f"Arquivo não encontrado: {INPUT_METAFEATURES}\n"
#             "Execute primeiro: python src/02_evaluate_and_extract.py"
#         )

#     if not INPUT_MATRIX.exists():
#         raise FileNotFoundError(
#             f"Arquivo não encontrado: {INPUT_MATRIX}\n"
#             "Execute primeiro: python src/02_evaluate_and_extract.py"
#         )

#     df_meta = pd.read_csv(INPUT_METAFEATURES)
#     performance_matrix = pd.read_csv(INPUT_MATRIX)

#     print("Meta-features carregadas:", df_meta.shape)
#     print("Performance matrix carregada:", performance_matrix.shape)

#     classifier_cols = [
#         "DecisionTree",
#         "SVM",
#         "KNN",
#         "LogisticRegression",
#         "Perceptron",
#         "MLP"
#     ]

#     existing_classifier_cols = [
#         col for col in classifier_cols
#         if col in performance_matrix.columns
#     ]

#     if not existing_classifier_cols:
#         raise RuntimeError(
#             "Nenhuma coluna de classificador foi encontrada no performance_matrix.csv"
#         )

#     # Recalcula best_classifier de forma segura, ignorando NaN
#     performance_matrix["best_classifier"] = (
#         performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
#     )

#     performance_matrix["best_accuracy"] = (
#         performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
#     )

#     # Remove datasets sem nenhum classificador válido
#     performance_matrix = performance_matrix.dropna(
#         subset=["best_classifier", "best_accuracy"]
#     ).copy()

#     df_exp_a = df_meta.merge(
#         performance_matrix,
#         on="did",
#         how="inner"
#     )

#     print("Datasets após merge:", df_exp_a.shape[0])

#     ignore_cols = [
#         "did",
#         "name",
#         "predicted_domain",
#         "domain",
#         "domain_score",
#         "semantic_text",
#         "best_classifier",
#         "best_accuracy"
#     ] + classifier_cols

#     X_a = df_exp_a.drop(columns=ignore_cols, errors="ignore")
#     X_a = X_a.apply(pd.to_numeric, errors="coerce")

#     # Remove colunas completamente vazias
#     X_a = X_a.dropna(axis=1, how="all")

#     # Remove colunas constantes, pois não ajudam o modelo
#     nunique = X_a.nunique(dropna=True)
#     X_a = X_a.loc[:, nunique > 1]

#     # Trata infinito
#     X_a = X_a.replace([np.inf, -np.inf], np.nan)

#     y_a = df_exp_a["best_classifier"].astype(str)

#     # Remove classes com menos de 2 exemplos, senão StratifiedKFold quebra
#     class_counts = y_a.value_counts()
#     valid_classes = class_counts[class_counts >= 2].index

#     valid_mask = y_a.isin(valid_classes)

#     X_a = X_a.loc[valid_mask].reset_index(drop=True)
#     y_a = y_a.loc[valid_mask].reset_index(drop=True)

#     print("\nDistribuição do alvo após limpeza:")
#     print(y_a.value_counts())

#     if len(y_a.unique()) < 2:
#         raise RuntimeError(
#             "Depois da limpeza, restou menos de 2 classes de melhor classificador."
#         )

#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_a)

#     min_class_count = pd.Series(y_encoded).value_counts().min()
#     n_splits = min(5, min_class_count)

#     if n_splits < 2:
#         raise RuntimeError("Classes insuficientes para validação cruzada.")

#     cv = StratifiedKFold(
#         n_splits=n_splits,
#         shuffle=True,
#         random_state=42
#     )

#     meta_model = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#         ("model", RandomForestClassifier(
#             n_estimators=300,
#             random_state=42,
#             class_weight="balanced"
#         ))
#     ])

#     results = cross_validate(
#         meta_model,
#         X_a,
#         y_encoded,
#         cv=cv,
#         scoring={
#             "accuracy": "accuracy",
#             "f1_macro": "f1_macro",
#             "f1_weighted": "f1_weighted"
#         },
#         return_train_score=False,
#         error_score="raise"
#     )

#     summary = pd.DataFrame([{
#         "experiment": "A",
#         "features": "Statistical meta-features",
#         "datasets_used": len(X_a),
#         "n_features": X_a.shape[1],
#         "target_classes": ", ".join(label_encoder.classes_),
#         "accuracy_mean": results["test_accuracy"].mean(),
#         "accuracy_std": results["test_accuracy"].std(),
#         "f1_macro_mean": results["test_f1_macro"].mean(),
#         "f1_macro_std": results["test_f1_macro"].std(),
#         "f1_weighted_mean": results["test_f1_weighted"].mean(),
#         "f1_weighted_std": results["test_f1_weighted"].std(),
#         "n_splits": n_splits
#     }])

#     summary.to_csv(OUTPUT_RESULTS, index=False)

#     print("\nEXPERIMENTO A — SOMENTE META-FEATURES ESTATÍSTICAS")
#     print("Datasets usados:", len(X_a))
#     print("Número de meta-features:", X_a.shape[1])
#     print("Classes previstas:", list(label_encoder.classes_))
#     print()
#     print("Accuracy:", summary.loc[0, "accuracy_mean"], "+/-", summary.loc[0, "accuracy_std"])
#     print("F1 macro:", summary.loc[0, "f1_macro_mean"], "+/-", summary.loc[0, "f1_macro_std"])
#     print("F1 weighted:", summary.loc[0, "f1_weighted_mean"], "+/-", summary.loc[0, "f1_weighted_std"])

#     print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


# if __name__ == "__main__":
#     main()

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"
OUTPUT_RESULTS = DATA_DIR / "experiment_a_results.csv"


def main():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_METAFEATURES}\n"
            "Execute primeiro: python src/02_evaluate_and_extract.py"
        )

    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_MATRIX}\n"
            "Execute primeiro: python src/02_evaluate_and_extract.py"
        )

    df_meta = pd.read_csv(INPUT_METAFEATURES)
    performance_matrix = pd.read_csv(INPUT_MATRIX)

    print("Meta-features carregadas:", df_meta.shape)
    print("Performance matrix carregada:", performance_matrix.shape)

    # Apenas classificadores considerados neste teste inicial
    classifier_cols = [
        "DecisionTree",
        "LogisticRegression",
        "Perceptron"
    ]

    existing_classifier_cols = [
        col for col in classifier_cols
        if col in performance_matrix.columns
    ]

    if not existing_classifier_cols:
        raise RuntimeError(
            "Nenhuma coluna dos classificadores selecionados foi encontrada no performance_matrix.csv"
        )

    print("Classificadores considerados:", existing_classifier_cols)

    # Recalcula o melhor classificador considerando APENAS os 3 escolhidos
    performance_matrix["best_classifier"] = (
        performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
    )

    performance_matrix["best_accuracy"] = (
        performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
    )

    # Remove datasets sem nenhum dos 3 classificadores avaliados
    performance_matrix = performance_matrix.dropna(
        subset=["best_classifier", "best_accuracy"]
    ).copy()

    df_exp_a = df_meta.merge(
        performance_matrix,
        on="did",
        how="inner"
    )

    print("Datasets após merge:", df_exp_a.shape[0])

    # Aqui mantemos também os classificadores não usados no ignore,
    # caso eles existam no CSV, para não virarem feature por acidente.
    all_possible_classifier_cols = [
        "DecisionTree",
        "SVM",
        "KNN",
        "LogisticRegression",
        "Perceptron",
        "MLP"
    ]

    ignore_cols = [
        "did",
        "name",
        "predicted_domain",
        "domain",
        "domain_score",
        "semantic_text",
        "best_classifier",
        "best_accuracy"
    ] + all_possible_classifier_cols

    X_a = df_exp_a.drop(columns=ignore_cols, errors="ignore")
    X_a = X_a.apply(pd.to_numeric, errors="coerce")

    # Remove colunas completamente vazias
    X_a = X_a.dropna(axis=1, how="all")

    # Remove colunas constantes
    nunique = X_a.nunique(dropna=True)
    X_a = X_a.loc[:, nunique > 1]

    # Trata infinito
    X_a = X_a.replace([np.inf, -np.inf], np.nan)

    y_a = df_exp_a["best_classifier"].astype(str)

    # Remove classes com menos de 2 exemplos
    class_counts = y_a.value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    valid_mask = y_a.isin(valid_classes)

    X_a = X_a.loc[valid_mask].reset_index(drop=True)
    y_a = y_a.loc[valid_mask].reset_index(drop=True)

    print("\nDistribuição do alvo após limpeza:")
    print(y_a.value_counts())

    if len(y_a.unique()) < 2:
        raise RuntimeError(
            "Depois da limpeza, restou menos de 2 classes de melhor classificador."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_a)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Classes insuficientes para validação cruzada.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    meta_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    results = cross_validate(
        meta_model,
        X_a,
        y_encoded,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted"
        },
        return_train_score=False,
        error_score="raise"
    )

    summary = pd.DataFrame([{
        "experiment": "A",
        "features": "Statistical meta-features",
        "base_classifiers_considered": ", ".join(existing_classifier_cols),
        "datasets_used": len(X_a),
        "n_features": X_a.shape[1],
        "target_classes": ", ".join(label_encoder.classes_),
        "accuracy_mean": results["test_accuracy"].mean(),
        "accuracy_std": results["test_accuracy"].std(),
        "f1_macro_mean": results["test_f1_macro"].mean(),
        "f1_macro_std": results["test_f1_macro"].std(),
        "f1_weighted_mean": results["test_f1_weighted"].mean(),
        "f1_weighted_std": results["test_f1_weighted"].std(),
        "n_splits": n_splits
    }])

    summary.to_csv(OUTPUT_RESULTS, index=False)

    print("\nEXPERIMENTO A — SOMENTE META-FEATURES ESTATÍSTICAS")
    print("Classificadores considerados:", existing_classifier_cols)
    print("Datasets usados:", len(X_a))
    print("Número de meta-features:", X_a.shape[1])
    print("Classes previstas:", list(label_encoder.classes_))
    print()
    print("Accuracy:", summary.loc[0, "accuracy_mean"], "+/-", summary.loc[0, "accuracy_std"])
    print("F1 macro:", summary.loc[0, "f1_macro_mean"], "+/-", summary.loc[0, "f1_macro_std"])
    print("F1 weighted:", summary.loc[0, "f1_weighted_mean"], "+/-", summary.loc[0, "f1_weighted_std"])

    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()