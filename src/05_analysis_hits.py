from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_METAFEATURES = DATA_DIR / "metafeatures_selected_datasets.csv"
INPUT_MATRIX = DATA_DIR / "performance_matrix.csv"


def main():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_METAFEATURES}")

    if not INPUT_MATRIX.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_MATRIX}")

    df_meta = pd.read_csv(INPUT_METAFEATURES)
    performance_matrix = pd.read_csv(INPUT_MATRIX)

    classifier_cols = [
        "DecisionTree",
        "LogisticRegression",
        "Perceptron"
    ]

    existing_classifier_cols = [
        col for col in classifier_cols
        if col in performance_matrix.columns
    ]

    performance_matrix["best_classifier"] = (
        performance_matrix[existing_classifier_cols].idxmax(axis=1, skipna=True)
    )
    performance_matrix["best_accuracy"] = (
        performance_matrix[existing_classifier_cols].max(axis=1, skipna=True)
    )

    performance_matrix = performance_matrix.dropna(
        subset=["best_classifier", "best_accuracy"]
    ).copy()

    df_exp_b = df_meta.merge(
        performance_matrix,
        on="did",
        how="inner"
    )

    all_possible_classifier_cols = [
        "DecisionTree", "SVM", "KNN", "LogisticRegression", "Perceptron", "MLP"
    ]

    ignore_cols = [
        "did", "name", "domain", "best_classifier", "best_accuracy"
    ] + all_possible_classifier_cols

    X_b = df_exp_b.drop(columns=ignore_cols, errors="ignore").copy()

    if "semantic_text" not in X_b.columns:
        X_b["semantic_text"] = ""
    X_b["semantic_text"] = X_b["semantic_text"].fillna("").astype(str)

    if "predicted_domain" not in X_b.columns:
        X_b["predicted_domain"] = "unknown"
    X_b["predicted_domain"] = X_b["predicted_domain"].fillna("unknown").astype(str)

    numeric_cols = [col for col in X_b.columns if col not in ["semantic_text", "predicted_domain"]]

    for col in numeric_cols:
        X_b[col] = pd.to_numeric(X_b[col], errors="coerce")

    X_b = X_b.dropna(axis=1, how="all")

    numeric_cols = [col for col in X_b.columns if col not in ["semantic_text", "predicted_domain"]]

    X_b[numeric_cols] = X_b[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_b[numeric_cols] = X_b[numeric_cols].mask(X_b[numeric_cols].abs() > 1e12, np.nan)

    nunique = X_b[numeric_cols].nunique(dropna=True)
    valid_numeric_cols = nunique[nunique > 1].index.tolist()

    X_b = X_b[valid_numeric_cols + ["semantic_text", "predicted_domain"]]
    numeric_cols = valid_numeric_cols

    y_b = df_exp_b["best_classifier"].astype(str)

    class_counts = y_b.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    valid_mask = y_b.isin(valid_classes)

    X_b = X_b.loc[valid_mask].reset_index(drop=True)
    y_b = y_b.loc[valid_mask].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_b)

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, min_class_count)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # MODELO 1: EXPERIMENTO A (BASELINE)
    # Apenas as 63 numéricas
    # ---------------------------------------------------------
    preprocessor_a = ColumnTransformer(
        transformers=[
            (
                "statistical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            )
        ],
        remainder="drop"
    )

    meta_model_a = Pipeline([
        ("preprocess", preprocessor_a),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # ---------------------------------------------------------
    # MODELO 2: TESTE 1 (CAMPEÃO)
    # 63 Numéricas + Domínio + 20 Textos
    # ---------------------------------------------------------
    preprocessor_test1 = ColumnTransformer(
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
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]),
                ["predicted_domain"]
            ),
            (
                "semantic",
                TfidfVectorizer(max_features=20, ngram_range=(1, 1), min_df=2),
                "semantic_text"
            )
        ],
        remainder="drop"
    )

    meta_model_test1 = Pipeline([
        ("preprocess", preprocessor_test1),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # ---------------------------------------------------------
    # GERANDO AS PREVISÕES (CROSS_VAL_PREDICT)
    # ---------------------------------------------------------
    print("\nExecutando predições para o Experimento A (Cego)...")
    y_pred_a = cross_val_predict(meta_model_a, X_b, y_encoded, cv=cv)

    print("Executando predições para o Teste 1 (Com Semântica)...")
    y_pred_test1 = cross_val_predict(meta_model_test1, X_b, y_encoded, cv=cv)

    # ---------------------------------------------------------
    # ANÁLISE DE ACERTOS POR CLASSE
    # ---------------------------------------------------------
    y_true_labels = label_encoder.inverse_transform(y_encoded)
    y_pred_a_labels = label_encoder.inverse_transform(y_pred_a)
    y_pred_test1_labels = label_encoder.inverse_transform(y_pred_test1)

    classes_encontradas = np.unique(y_true_labels)

    analysis_data = []

    for cls in classes_encontradas:
        # Quantos datasets realmente tem esse classificador como o melhor
        total_real = np.sum(y_true_labels == cls)
        
        # Quantos o modelo A acertou
        acertos_a = np.sum((y_pred_a_labels == cls) & (y_true_labels == cls))
        
        # Quantos o Teste 1 acertou
        acertos_test1 = np.sum((y_pred_test1_labels == cls) & (y_true_labels == cls))
        
        # Quantos o modelo A "chutou" e errou (Falsos Positivos)
        erros_a = np.sum((y_pred_a_labels == cls) & (y_true_labels != cls))

        # Quantos o Teste 1 "chutou" e errou (Falsos Positivos)
        erros_test1 = np.sum((y_pred_test1_labels == cls) & (y_true_labels != cls))

        analysis_data.append({
            "Algoritmo": cls,
            "Total Real (Gabarito)": total_real,
            "Acertos Exp A (S/ Texto)": acertos_a,
            "Acertos Teste 1 (C/ Texto)": acertos_test1,
            "Ganho de Acertos": acertos_test1 - acertos_a
        })

    # Adicionar linha de total
    total_datasets = len(y_true_labels)
    total_acertos_a = np.sum(y_pred_a_labels == y_true_labels)
    total_acertos_test1 = np.sum(y_pred_test1_labels == y_true_labels)

    analysis_data.append({
        "Algoritmo": "TOTAL GERAL",
        "Total Real (Gabarito)": total_datasets,
        "Acertos Exp A (S/ Texto)": total_acertos_a,
        "Acertos Teste 1 (C/ Texto)": total_acertos_test1,
        "Ganho de Acertos": total_acertos_test1 - total_acertos_a
    })

    df_analysis = pd.DataFrame(analysis_data)
    
    print("\n" + "="*80)
    print("ANÁLISE DE ACERTOS: EXPERIMENTO A vs TESTE 1")
    print("="*80)
    print(df_analysis.to_string(index=False))
    print("="*80)
    
    output_path = DATA_DIR / "hit_analysis_a_vs_test1.csv"
    df_analysis.to_csv(output_path, index=False)
    print(f"\nSalvo em: {output_path}")


if __name__ == "__main__":
    main()
