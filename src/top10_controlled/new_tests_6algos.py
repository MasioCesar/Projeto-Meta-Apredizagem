"""
Novos testes para tentar melhorar a selecao de algoritmo com 6 classificadores.

NAO altera nenhuma meta-feature estatistica. Apenas:
  (A) cria features SEMANTICAS novas focadas em MODALIDADE/AFINIDADE-DE-ALGORITMO
      (em vez de dominio de aplicacao), porque o que prediz o melhor algoritmo
      e a *estrutura* dos dados (alta dimensao, perceptual, tabular...), nao o tema.
  (B) testa um ALVO AGRUPADO (modelo simples vs complexo), que e uma pergunta
      de meta-aprendizagem mais robusta sob 6 classes muito desbalanceadas.

Saida: data/top10_controlled/new_tests_summary.csv
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from common import (
    OUTPUT_DIR,
    clean_semantic_text,
    semantic_tags,
    semantic_clusters,
    DomainConfidenceFeatures,
    load_data,
)


# ----------------------------------------------------------------------------
# (A) Features semanticas novas: MODALIDADE / AFINIDADE DE ALGORITMO
# ----------------------------------------------------------------------------
# Hipotese: a *modalidade* dos dados (nao o dominio) prediz a familia de
# algoritmo vencedora.
#   - alta dimensao / esparso (gene, texto)  -> SVM / lineares
#   - perceptual / denso (imagem, sinal)      -> MLP / SVM
#   - tabular / categorico misto              -> arvore / LR
MODALITY_KEYWORDS = {
    "mod_highdim_sparse": {
        "gene", "genes", "expression", "microarray", "genomics", "rna",
        "probe", "probes", "sequence", "dna", "protein", "sparse",
        "dimensional", "dimension", "text", "document", "word", "words",
    },
    "mod_perceptual_dense": {
        "image", "images", "pixel", "pixels", "digit", "digits", "face",
        "faces", "visual", "object", "signal", "signals", "sensor",
        "accelerometer", "activity", "audio", "speech", "wave",
    },
    "mod_tabular_mixed": {
        "tabular", "survey", "demographic", "census", "adult", "record",
        "records", "categorical", "nominal", "ordinal", "customer",
        "credit", "bank", "income", "transaction", "financial",
    },
    "mod_biomedical": {
        "patient", "patients", "clinical", "medical", "diagnosis",
        "disease", "tumor", "cancer", "tissue", "oncology", "hospital",
    },
    "mod_chemical": {
        "chemical", "chemistry", "molecule", "molecules", "qsar",
        "compound", "compounds", "toxicity", "biodegradation",
    },
}


class ModalityAffinityExtractor(BaseEstimator, TransformerMixin):
    """Para cada dataset, conta evidencia de cada modalidade (score continuo +
    binario) e ainda marca a modalidade dominante (argmax). Tudo derivado so do
    texto semantico limpo -- nenhuma meta-feature estatistica e tocada."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=["semantic_text", "name"])
        rows = []
        for _, row in df.iterrows():
            tokens = set(clean_semantic_text(row["semantic_text"], row["name"]).split())
            scores = []
            feats = []
            for kws in MODALITY_KEYWORDS.values():
                hits = len(tokens & kws)
                ratio = hits / len(kws)
                feats.append(float(hits > 0))      # binario: modalidade presente
                feats.append(ratio)                # intensidade
                scores.append(ratio)
            # modalidade dominante (one-hot via argmax; -1 se nada)
            dominant = int(np.argmax(scores)) if max(scores) > 0 else -1
            for i in range(len(MODALITY_KEYWORDS)):
                feats.append(float(i == dominant))
            rows.append(feats)
        return np.asarray(rows, dtype=float)


def modality_pipeline():
    return Pipeline([
        ("modality", ModalityAffinityExtractor()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ----------------------------------------------------------------------------
# Montagem de modelos
# ----------------------------------------------------------------------------
def build(numeric_cols, semantic=None, domain_score=False):
    transformers = [
        ("statistical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
    ]
    if semantic == "modality":
        transformers.append(("semantic", modality_pipeline(), ["semantic_text", "name"]))
    elif semantic == "modality_tags":
        transformers.append(("semantic", FeatureUnion([
            ("modality", modality_pipeline()),
            ("tags", semantic_tags()),
        ]), ["semantic_text", "name"]))
    elif semantic == "modality_clusters":
        transformers.append(("modality", modality_pipeline(), ["semantic_text", "name"]))
        transformers.append(("clusters", semantic_clusters(), ["semantic_text", "name"]))

    if domain_score:
        transformers.append(("domain_score", Pipeline([
            ("confidence", DomainConfidenceFeatures()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["domain_score"]))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])


def evaluate(name, X, y_encoded, model, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = cross_validate(model, X, y_encoded, cv=cv,
                         scoring={"accuracy": "accuracy", "f1_macro": "f1_macro",
                                  "f1_weighted": "f1_weighted"},
                         error_score="raise")
    return {
        "strategy": name,
        "accuracy_mean": res["test_accuracy"].mean(),
        "accuracy_std": res["test_accuracy"].std(),
        "f1_macro_mean": res["test_f1_macro"].mean(),
        "f1_macro_std": res["test_f1_macro"].std(),
        "f1_weighted_mean": res["test_f1_weighted"].mean(),
        "n_splits": n_splits,
    }


# Agrupamento do alvo: modelo SIMPLES/rapido vs COMPLEXO/pesado
TARGET_GROUPS = {
    "DecisionTree": "simple",
    "LogisticRegression": "simple",
    "Perceptron": "simple",
    "SVM": "complex",
    "MLP": "complex",
    "KNN": "complex",
}


def main():
    X, y, numeric_cols, classifier_cols = load_data()
    print(f"Datasets: {len(X)} | Classificadores: {classifier_cols}")
    rows = []

    # ===== BLOCO 1: alvo original de 6 classes =====
    enc6 = LabelEncoder()
    y6 = enc6.fit_transform(y)
    n6 = min(5, pd.Series(y6).value_counts().min())
    print(f"\n[6 classes] n_splits={n6} | distrib: {dict(y.value_counts())}")

    rows.append({"target": "6_classes", **evaluate(
        "baseline_stat_only", X, y6, build(numeric_cols), n6)})
    rows.append({"target": "6_classes", **evaluate(
        "stat_plus_modality", X, y6, build(numeric_cols, "modality"), n6)})
    rows.append({"target": "6_classes", **evaluate(
        "stat_plus_modality_tags", X, y6, build(numeric_cols, "modality_tags"), n6)})
    rows.append({"target": "6_classes", **evaluate(
        "stat_plus_modality_clusters", X, y6, build(numeric_cols, "modality_clusters"), n6)})
    rows.append({"target": "6_classes", **evaluate(
        "stat_plus_modality_domainscore", X, y6,
        build(numeric_cols, "modality", domain_score=True), n6)})

    # ===== BLOCO 2: alvo agrupado (simples vs complexo) =====
    y_grp = y.map(TARGET_GROUPS)
    encg = LabelEncoder()
    yg = encg.fit_transform(y_grp)
    ng = min(5, pd.Series(yg).value_counts().min())
    print(f"\n[agrupado simples/complexo] n_splits={ng} | distrib: {dict(y_grp.value_counts())}")

    rows.append({"target": "grouped_simple_complex", **evaluate(
        "baseline_stat_only", X, yg, build(numeric_cols), ng)})
    rows.append({"target": "grouped_simple_complex", **evaluate(
        "stat_plus_modality", X, yg, build(numeric_cols, "modality"), ng)})
    rows.append({"target": "grouped_simple_complex", **evaluate(
        "stat_plus_modality_tags", X, yg, build(numeric_cols, "modality_tags"), ng)})
    rows.append({"target": "grouped_simple_complex", **evaluate(
        "stat_plus_modality_domainscore", X, yg,
        build(numeric_cols, "modality", domain_score=True), ng)})

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "new_tests_summary.csv"
    summary.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n================ RESULTADOS ================")
    for tgt in summary["target"].unique():
        sub = summary[summary["target"] == tgt].sort_values("f1_macro_mean", ascending=False)
        print(f"\n--- alvo: {tgt} ---")
        print(sub[["strategy", "accuracy_mean", "f1_macro_mean", "f1_weighted_mean"]].to_string(index=False))
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
