import re
import unicodedata
from pathlib import Path

import openml
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "final_domain_selection.csv"


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


domain_queries = {
    "health": [
        "health", "medical", "clinical", "patient", "disease", "diagnosis",
        "cancer", "tumor", "breast", "thyroid", "diabetes", "heart",
        "cardio", "blood", "liver", "kidney", "lung", "hepatitis",
        "stroke", "parkinsons", "dermatology", "mammographic",
        "maternal", "transfusion", "eeg", "ecg", "arrhythmia",
        "covid", "hospital", "symptom", "syndrome"
    ],

    "finance": [
        "credit", "bank", "loan", "fraud", "finance", "financial",
        "stock", "insurance", "bankruptcy", "approval", "default",
        "payment", "investment", "valuation", "transaction",
        "pricing", "mortgage", "economy", "trading"
    ],

    "biology": [
        "gene", "genes", "genomic", "genomics", "protein", "dna", "rna",
        "microarray", "bio", "bioinformatics", "yeast", "ecoli",
        "molecular", "cell", "splice", "promoter", "plant",
        "soybean", "species", "ecology", "toxic", "toxicity",
        "mutation", "peptide", "expression", "microbiome", "fungi"
    ],

    "image": [
        "image", "images", "pixel", "pixels", "mnist", "digit", "digits",
        "face", "faces", "vision", "letter", "letters", "ocr",
        "optdigits", "pendigits", "satimage", "segment", "texture",
        "led", "display", "shuttle", "fashion", "cifar",
        "rgb", "grayscale", "video", "facial"
    ],

    "text": [
        "text", "document", "documents", "news", "review", "reviews",
        "sentiment", "spam", "email", "emails", "language", "nlp",
        "topic", "tweets", "twitter", "comment", "comments",
        "product", "imdb", "amazon", "yelp", "reuters",
        "corpus", "article", "sarcasm", "fake", "embedding",
        "translation"
    ],

    "sensor_signal": [
        "sensor", "sensors", "signal", "signals", "seismic", "robot",
        "navigation", "fault", "faults", "vibration", "accelerometer",
        "gyro", "motion", "activity", "har", "gesture", "phase",
        "electricity", "power", "energy", "steel", "plate", "plates",
        "waveform", "gas", "iot", "telemetry", "radar", "sonar"
    ],

    "education": [
        "student", "students", "school", "education", "exam", "grade",
        "grades", "academic", "university", "college", "performance",
        "knowledge", "learning", "course", "math", "score", "scores",
        "dropout", "mooc", "tutor", "teaching"
    ],

    "social": [
        "user", "users", "social", "network", "networks", "rating",
        "ratings", "community", "communities", "profile", "profiles",
        "movie", "movies", "recommendation", "recommender",
        "customer", "customers", "marketing", "adult", "census",
        "income", "demographic", "click", "ads", "advertising",
        "purchase", "behavior", "shopping", "ecommerce",
        "churn", "retention"
    ]
}


domain_keywords = {
    "health": {
        "health": 4, "medical": 5, "clinical": 5, "patient": 5,
        "disease": 5, "diagnosis": 5, "cancer": 7, "tumor": 7,
        "diabetes": 7, "heart": 7, "blood": 5, "breast": 6,
        "thyroid": 6, "hepatitis": 6, "maternal": 5,
        "transfusion": 6, "eeg": 7, "ecg": 7,
        "arrhythmia": 7, "covid": 6, "hospital": 5,
        "symptom": 4, "syndrome": 5
    },

    "finance": {
        "finance": 5, "financial": 5, "credit": 7, "bank": 6,
        "loan": 7, "fraud": 7, "stock": 6, "insurance": 6,
        "bankruptcy": 6, "approval": 4, "default": 6,
        "payment": 5, "valuation": 5, "investment": 6,
        "transaction": 6, "pricing": 5,
        "mortgage": 7, "economy": 5, "trading": 7
    },

    "biology": {
        "gene": 7, "genes": 7, "genomic": 7, "genomics": 7,
        "protein": 7, "dna": 7, "rna": 7,
        "sequence": 6, "splice": 8,
        "yeast": 8, "ecoli": 8,
        "promoter": 7, "microarray": 7,
        "bioinformatics": 7, "molecular": 5,
        "cell": 4, "species": 4,
        "mutation": 6, "peptide": 7,
        "expression": 5, "microbiome": 7, "fungi": 5,
        "qsar": 8, "biodeg": 7, "biodegradation": 7,
        "chemical": 6, "compound": 6, "molecule": 6
    },

    "image": {
        "image": 6, "images": 6,
        "pixel": 5, "pixels": 5,
        "mnist": 8, "digit": 6, "digits": 6,
        "face": 6, "faces": 6,
        "vision": 6, "ocr": 6,
        "letter": 5, "letters": 5,
        "optdigits": 8, "pendigits": 8,
        "satimage": 8,
        "texture": 5, "segment": 5,
        "rgb": 6, "grayscale": 6,
        "video": 5, "facial": 6
    },

    "text": {
        "text": 6, "document": 6, "documents": 6,
        "news": 5, "review": 5, "reviews": 5,
        "sentiment": 7, "spam": 7,
        "email": 6, "emails": 6,
        "language": 6, "nlp": 7,
        "imdb": 6, "tweet": 6, "twitter": 6,
        "comment": 5, "comments": 5,
        "reuters": 6,
        "corpus": 6, "article": 5,
        "sarcasm": 7, "fake": 6,
        "embedding": 7, "translation": 6
    },

    "sensor_signal": {
        "sensor": 6, "sensors": 6,
        "signal": 6, "signals": 6,
        "seismic": 7,
        "robot": 6, "navigation": 6,
        "fault": 6, "faults": 6,
        "vibration": 6,
        "activity": 6,
        "gesture": 7,
        "phase": 5,
        "motion": 6,
        "har": 7,
        "accelerometer": 7,
        "gyro": 7,
        "waveform": 6,
        "electricity": 5,
        "power": 5,
        "energy": 5,
        "steel": 5,
        "plates": 5,
        "telemetry": 6,
        "radar": 7,
        "sonar": 7
    },

    "education": {
        "student": 7, "students": 7,
        "school": 5,
        "education": 7,
        "exam": 5,
        "grade": 5, "grades": 5,
        "academic": 5,
        "university": 5,
        "college": 5,
        "performance": 2,
        "knowledge": 5,
        "learning": 2,
        "course": 4,
        "score": 3, "scores": 3,
        "dropout": 7,
        "mooc": 7,
        "tutor": 6,
        "teaching": 5
    },

    "social": {
        "social": 6,
        "network": 6, "networks": 6,
        "user": 5, "users": 5,
        "community": 5,
        "rating": 6, "ratings": 6,
        "movie": 5, "movies": 5,
        "recommendation": 6,
        "recommender": 6,
        "customer": 4, "customers": 4,
        "adult": 7,
        "census": 7,
        "income": 5,
        "demographic": 5,
        "behavior": 5,
        "shopping": 5,
        "ecommerce": 5,
        "churn": 7,
        "retention": 6
    }
}


def get_openml_semantic_text(did):
    try:
        dataset = openml.datasets.get_dataset(int(did), download_data=False)

        name = dataset.name or ""
        description = dataset.description or ""

        try:
            features = dataset.features
            feature_names = " ".join([f.name for f in features.values()])
        except Exception:
            feature_names = ""

        return normalize_text(f"{name} {description} {feature_names}")

    except Exception:
        return ""


def compute_score(text, keywords):
    score = 0

    for word, weight in keywords.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            score += weight

    return score


def extract_family(name):
    name = normalize_text(name)

    patterns = [
        r"_seed_\d+",
        r"_version_\d+",
        r"\bversion\s*\d+\b",
        r"\bv\d+\b",
        r"-dropped",
        r"_dropped",
        r"\d+nrows.*",
        r"\d+rows.*",
        r"\d+instances.*",
        r"nclasses_\d+",
        r"ncols_\d+",
        r"stratify_\w+"
    ]

    for pattern in patterns:
        name = re.sub(pattern, "", name)

    tokens = re.split(r"[_\-\s]+", name)
    stopwords = {"dataset", "data", "set", "train", "test", "csv", "openml"}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

    return "_".join(tokens[:2]) if tokens else name


def main():
    print("Carregando datasets do OpenML...")
    all_datasets = openml.datasets.list_datasets(output_format="dataframe")

    required_cols = [
        "name",
        "did",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfClasses",
        "NumberOfInstancesWithMissingValues",
        "MinorityClassSize",
        "NumberOfNumericFeatures"
    ]

    all_datasets = all_datasets.dropna(subset=required_cols).copy()

    technical_filtered = all_datasets[
        (all_datasets["NumberOfInstances"] >= 200) &
        (all_datasets["NumberOfInstances"] <= 50000) &
        (all_datasets["NumberOfFeatures"] >= 2) &
        (all_datasets["NumberOfClasses"] >= 2) &
        (
            all_datasets["NumberOfInstancesWithMissingValues"]
            < all_datasets["NumberOfInstances"] * 0.3
        ) &
        (all_datasets["MinorityClassSize"] >= 20)
    ].copy()

    technical_filtered["name_norm"] = technical_filtered["name"].apply(normalize_text)

    candidate_rows = []

    for domain, keywords in domain_queries.items():
        pattern = "|".join([re.escape(k) for k in keywords])

        subset = technical_filtered[
            technical_filtered["name_norm"].str.contains(pattern, case=False, na=False)
        ].copy()

        subset["candidate_domain"] = domain
        candidate_rows.append(subset)

    domain_candidates = pd.concat(candidate_rows, ignore_index=True)
    domain_candidates = domain_candidates.drop_duplicates(subset=["did", "candidate_domain"])

    print("\nCandidatos por domínio:")
    print(domain_candidates["candidate_domain"].value_counts())
    print("Total de candidatos:", len(domain_candidates))

    print("\nBuscando textos semânticos no OpenML...")
    domain_candidates["semantic_text"] = domain_candidates["did"].apply(get_openml_semantic_text)

    for domain, keywords in domain_keywords.items():
        domain_candidates[f"{domain}_score"] = domain_candidates["semantic_text"].apply(
            lambda text: compute_score(text, keywords)
        )

    score_cols = [f"{d}_score" for d in domain_keywords]

    # Apenas 1 domínio por dataset: escolhe o domínio de maior score
    domain_candidates["predicted_domain"] = (
        domain_candidates[score_cols]
        .idxmax(axis=1)
        .str.replace("_score", "", regex=False)
    )

    domain_candidates["domain_score"] = domain_candidates.apply(
        lambda row: row[f"{row['predicted_domain']}_score"],
        axis=1
    )

    domain_candidates = domain_candidates[
        domain_candidates["domain_score"] > 7
    ].copy()

    print("\nClassificação por domínio antes de remover famílias:")
    print(domain_candidates["predicted_domain"].value_counts())

    domain_candidates["family"] = domain_candidates["name"].apply(extract_family)

    final_domain_selection = (
        domain_candidates
        .sort_values(
            by=["predicted_domain", "domain_score", "NumberOfInstances"],
            ascending=[True, False, False]
        )
        .groupby(["predicted_domain", "family"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    final_columns = [
        "name",
        "did",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfClasses",
        "family",
        "predicted_domain",
        "domain_score",
        "semantic_text"
    ]

    final_domain_selection = final_domain_selection[final_columns]

    print("\nSeleção final por domínio:")
    print(final_domain_selection["predicted_domain"].value_counts())

    final_domain_selection.to_csv(OUTPUT_PATH, index=False)

    print(f"\nArquivo salvo em: {OUTPUT_PATH}")
    print(f"Total final de datasets: {len(final_domain_selection)}")


if __name__ == "__main__":
    main()





