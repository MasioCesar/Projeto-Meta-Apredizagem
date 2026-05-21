# from pathlib import Path

# import numpy as np
# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import StratifiedKFold, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder, StandardScaler


# BASE_DIR = Path(__file__).resolve().parents[1]
# DATA_DIR = BASE_DIR / "data"

# INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
# INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"
# INPUT_EXP_A = DATA_DIR / "experiment_a_results.csv"

# OUTPUT_RESULTS = DATA_DIR / "experiment_b_results.csv"
# OUTPUT_COMPARISON = DATA_DIR / "experiments_comparison.csv"


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

#     # Recalcula melhor classificador de forma segura
#     performance_matrix["best_classifier"] = (
#         performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
#     )

#     performance_matrix["best_accuracy"] = (
#         performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
#     )

#     performance_matrix = performance_matrix.dropna(
#         subset=["best_classifier", "best_accuracy"]
#     ).copy()

#     df_exp_b = df_meta.merge(
#         performance_matrix,
#         on="did",
#         how="inner"
#     )

#     print("Datasets após merge:", df_exp_b.shape[0])

#     ignore_cols = [
#         "did",
#         "name",
#         "predicted_domain",
#         "domain",
#         "domain_score",
#         "best_classifier",
#         "best_accuracy"
#     ] + classifier_cols

#     X_b = df_exp_b.drop(columns=ignore_cols, errors="ignore").copy()

#     if "semantic_text" not in X_b.columns:
#         X_b["semantic_text"] = ""

#     X_b["semantic_text"] = X_b["semantic_text"].fillna("").astype(str)

#     numeric_cols = [
#         col for col in X_b.columns
#         if col != "semantic_text"
#     ]

#     for col in numeric_cols:
#         X_b[col] = pd.to_numeric(X_b[col], errors="coerce")

#     # Remove colunas totalmente vazias
#     X_b = X_b.dropna(axis=1, how="all")

#     # Recalcula as colunas numéricas depois do drop
#     numeric_cols = [
#         col for col in X_b.columns
#         if col != "semantic_text"
#     ]

#     # Trata inf, -inf e valores absurdamente grandes
#     X_b[numeric_cols] = X_b[numeric_cols].replace([np.inf, -np.inf], np.nan)

#     # Alguns valores podem ser grandes demais para float64/scaler
#     X_b[numeric_cols] = X_b[numeric_cols].mask(
#         X_b[numeric_cols].abs() > 1e12,
#         np.nan
#     )

#     # Remove colunas constantes, pois não ajudam
#     nunique = X_b[numeric_cols].nunique(dropna=True)
#     valid_numeric_cols = nunique[nunique > 1].index.tolist()

#     X_b = X_b[valid_numeric_cols + ["semantic_text"]]
#     numeric_cols = valid_numeric_cols

#     y_b = df_exp_b["best_classifier"].astype(str)

#     # Remove classes com menos de 2 exemplos
#     class_counts = y_b.value_counts()
#     valid_classes = class_counts[class_counts >= 2].index

#     valid_mask = y_b.isin(valid_classes)

#     X_b = X_b.loc[valid_mask].reset_index(drop=True)
#     y_b = y_b.loc[valid_mask].reset_index(drop=True)

#     print("\nDistribuição do alvo após limpeza:")
#     print(y_b.value_counts())

#     if len(y_b.unique()) < 2:
#         raise RuntimeError(
#             "Depois da limpeza, restou menos de 2 classes de melhor classificador."
#         )

#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y_b)

#     min_class_count = pd.Series(y_encoded).value_counts().min()
#     n_splits = min(5, min_class_count)

#     if n_splits < 2:
#         raise RuntimeError("Classes insuficientes para validação cruzada.")

#     cv = StratifiedKFold(
#         n_splits=n_splits,
#         shuffle=True,
#         random_state=42
#     )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 "statistical",
#                 Pipeline([
#                     ("imputer", SimpleImputer(strategy="median")),
#                     ("scaler", StandardScaler())
#                 ]),
#                 numeric_cols
#             ),
#             (
#                 "semantic",
#                 TfidfVectorizer(
#                     max_features=300,
#                     ngram_range=(1, 2),
#                     min_df=2
#                 ),
#                 "semantic_text"
#             )
#         ],
#         remainder="drop"
#     )

#     meta_model = Pipeline([
#         ("preprocess", preprocessor),
#         ("model", RandomForestClassifier(
#             n_estimators=300,
#             random_state=42,
#             class_weight="balanced"
#         ))
#     ])

#     results = cross_validate(
#         meta_model,
#         X_b,
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
#         "experiment": "B",
#         "features": "Statistical + semantic meta-features",
#         "datasets_used": len(X_b),
#         "n_statistical_features": len(numeric_cols),
#         "n_semantic_max_features": 300,
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

#     print("\nEXPERIMENTO B — META-FEATURES ESTATÍSTICAS + SEMÂNTICAS")
#     print("Datasets usados:", len(X_b))
#     print("Meta-features estatísticas:", len(numeric_cols))
#     print("Features semânticas máximas TF-IDF:", 300)
#     print("Classes previstas:", list(label_encoder.classes_))
#     print()
#     print("Accuracy:", summary.loc[0, "accuracy_mean"], "+/-", summary.loc[0, "accuracy_std"])
#     print("F1 macro:", summary.loc[0, "f1_macro_mean"], "+/-", summary.loc[0, "f1_macro_std"])
#     print("F1 weighted:", summary.loc[0, "f1_weighted_mean"], "+/-", summary.loc[0, "f1_weighted_std"])

#     if INPUT_EXP_A.exists():
#         exp_a = pd.read_csv(INPUT_EXP_A)
#         comparison = pd.concat([exp_a, summary], ignore_index=True)
#         comparison.to_csv(OUTPUT_COMPARISON, index=False)

#         print("\nCOMPARAÇÃO FINAL:")
#         print(comparison)
#         print(f"\nComparação salva em: {OUTPUT_COMPARISON}")

#     print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


# if __name__ == "__main__":
#     main()

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"
INPUT_EXP_A = DATA_DIR / "experiment_a_results.csv"

OUTPUT_RESULTS = DATA_DIR / "experiment_b_results.csv"
OUTPUT_COMPARISON = DATA_DIR / "experiments_comparison.csv"


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

    df_exp_b = df_meta.merge(
        performance_matrix,
        on="did",
        how="inner"
    )

    print("Datasets após merge:", df_exp_b.shape[0])

    # Remove todos os classificadores do conjunto de entrada,
    # inclusive os que não serão considerados como alvo,
    # para evitar vazamento de informação.
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
        "best_classifier",
        "best_accuracy"
    ] + all_possible_classifier_cols

    X_b = df_exp_b.drop(columns=ignore_cols, errors="ignore").copy()

    if "semantic_text" not in X_b.columns:
        X_b["semantic_text"] = ""

    X_b["semantic_text"] = X_b["semantic_text"].fillna("").astype(str)

    numeric_cols = [
        col for col in X_b.columns
        if col != "semantic_text"
    ]

    for col in numeric_cols:
        X_b[col] = pd.to_numeric(X_b[col], errors="coerce")

    # Remove colunas totalmente vazias
    X_b = X_b.dropna(axis=1, how="all")

    # Recalcula as colunas numéricas depois do drop
    numeric_cols = [
        col for col in X_b.columns
        if col != "semantic_text"
    ]

    # Trata inf, -inf e valores absurdamente grandes
    X_b[numeric_cols] = X_b[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Alguns valores podem ser grandes demais para float64/scaler
    X_b[numeric_cols] = X_b[numeric_cols].mask(
        X_b[numeric_cols].abs() > 1e12,
        np.nan
    )

    # Remove colunas constantes, pois não ajudam
    nunique = X_b[numeric_cols].nunique(dropna=True)
    valid_numeric_cols = nunique[nunique > 1].index.tolist()

    X_b = X_b[valid_numeric_cols + ["semantic_text"]]
    numeric_cols = valid_numeric_cols

    y_b = df_exp_b["best_classifier"].astype(str)

    # Remove classes com menos de 2 exemplos
    class_counts = y_b.value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    valid_mask = y_b.isin(valid_classes)

    X_b = X_b.loc[valid_mask].reset_index(drop=True)
    y_b = y_b.loc[valid_mask].reset_index(drop=True)

    print("\nDistribuição do alvo após limpeza:")
    print(y_b.value_counts())

    if len(y_b.unique()) < 2:
        raise RuntimeError(
            "Depois da limpeza, restou menos de 2 classes de melhor classificador."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_b)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise RuntimeError("Classes insuficientes para validação cruzada.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "statistical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            ),
            (
                "semantic",
                TfidfVectorizer(
                    max_features=20,
                    ngram_range=(1, 1),
                    min_df=2
                ),
                "semantic_text"
            )
        ],
        remainder="drop"
    )

    meta_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    results = cross_validate(
        meta_model,
        X_b,
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
        "experiment": "B",
        "features": "Statistical + semantic meta-features",
        "base_classifiers_considered": ", ".join(existing_classifier_cols),
        "datasets_used": len(X_b),
        "n_statistical_features": len(numeric_cols),
        "n_semantic_max_features": 300,
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

    print("\nEXPERIMENTO B — META-FEATURES ESTATÍSTICAS + SEMÂNTICAS")
    print("Classificadores considerados:", existing_classifier_cols)
    print("Datasets usados:", len(X_b))
    print("Meta-features estatísticas:", len(numeric_cols))
    print("Features semânticas máximas TF-IDF:", 20)
    print("Classes previstas:", list(label_encoder.classes_))
    print()
    print("Accuracy:", summary.loc[0, "accuracy_mean"], "+/-", summary.loc[0, "accuracy_std"])
    print("F1 macro:", summary.loc[0, "f1_macro_mean"], "+/-", summary.loc[0, "f1_macro_std"])
    print("F1 weighted:", summary.loc[0, "f1_weighted_mean"], "+/-", summary.loc[0, "f1_weighted_std"])

    if INPUT_EXP_A.exists():
        exp_a = pd.read_csv(INPUT_EXP_A)
        comparison = pd.concat([exp_a, summary], ignore_index=True)
        comparison.to_csv(OUTPUT_COMPARISON, index=False)

        print("\nCOMPARAÇÃO FINAL:")
        print(comparison)
        print(f"\nComparação salva em: {OUTPUT_COMPARISON}")

    print(f"\nResultado salvo em: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()