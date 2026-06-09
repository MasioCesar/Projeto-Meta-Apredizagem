from pathlib import Path
import json
import re
import urllib.error
import urllib.request

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "llm_features"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
OUTPUT_FEATURES = OUTPUT_DIR / "llm_semantic_features.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


BOILERPLATE_WORDS = {
    "author", "authors", "source", "sources", "unknown", "date", "please",
    "cite", "citation", "copyright", "donated", "available", "download",
    "repository", "openml", "uci", "kaggle", "mlbench", "rdocumentation",
    "package", "packages", "version", "versions", "topic", "topics",
    "none", "http", "https", "www", "com", "org", "edu", "arff", "csv",
    "dataset", "datasets", "data", "database", "attribute", "attributes",
    "feature", "features", "information", "class", "classes",
    "classification", "learning", "machine", "algorithm", "algorithms",
    "benchmark", "benchmarking", "used", "use", "using", "different",
    "number", "set", "collection", "problem", "problems", "task", "tasks",
    "archive", "ics", "gemler", "tabarena", "expo", "consortium",
    "project", "international", "public", "availability",
}

CUT_MARKERS = [
    "attribute information",
    "attribute_information",
    "attributes information",
    "feature information",
    "feature names",
    "input attributes",
    "id_ref",
    "variable names",
    "column names",
]

ALLOWED_DOMAINS = [
    "biology",
    "health",
    "finance",
    "image",
    "text",
    "education",
    "social",
    "sensor_signal",
    "other",
]

ALLOWED_MODALITIES = [
    "tabular",
    "text",
    "image",
    "sequence",
    "time_series",
    "sensor",
    "graph",
    "mixed",
    "unknown",
]

ALLOWED_SUBTYPES = [
    "gene_expression",
    "dna_sequence",
    "medical_patient",
    "chemical_qsar",
    "financial_credit",
    "image_pixels",
    "text_documents",
    "survey_social",
    "sensor_activity",
    "education_records",
    "generic_tabular",
    "unknown",
]

ALLOWED_TAGS = [
    "high_dimensional",
    "small_sample",
    "many_samples",
    "sparse",
    "mostly_numeric",
    "mostly_categorical",
    "binary_features",
    "multiclass",
    "imbalanced",
    "missing_values",
    "clinical",
    "biological",
    "financial",
    "visual",
    "textual",
    "temporal",
    "chemical",
]


def clean_semantic_text(text, dataset_name="", max_tokens=350):
    text = "" if pd.isna(text) else str(text).lower()
    dataset_name = "" if pd.isna(dataset_name) else str(dataset_name).lower()
    text = re.sub(r"https?\S+|www\.\S+", " ", text)

    for marker in CUT_MARKERS:
        marker_pos = text.find(marker)
        if marker_pos != -1:
            text = text[:marker_pos]
            break

    blocked_name_tokens = set(re.findall(r"[a-z]+", dataset_name.replace("_", " ")))
    raw_tokens = re.findall(r"[a-z0-9_]+", text)
    cleaned_tokens = []

    for token in raw_tokens:
        if len(token) < 3:
            continue
        if token in BOILERPLATE_WORDS or token in blocked_name_tokens:
            continue
        if "_" in token or any(char.isdigit() for char in token):
            continue
        if not token.isalpha():
            continue
        cleaned_tokens.append(token)
        if max_tokens is not None and len(cleaned_tokens) >= max_tokens:
            break

    return " ".join(cleaned_tokens)


def build_prompt(clean_text):
    return f"""
You are extracting controlled semantic metadata from a cleaned dataset description.
Return ONLY valid JSON. Do not explain.

Allowed domain values: {ALLOWED_DOMAINS}
Allowed modality values: {ALLOWED_MODALITIES}
Allowed subtype values: {ALLOWED_SUBTYPES}
Allowed tags: {ALLOWED_TAGS}

Rules:
- Do not infer a specific dataset name.
- Do not recommend a machine learning algorithm.
- Use only the cleaned description below.
- Confidence must be a number from 0.0 to 1.0.
- If uncertain, use "unknown" or "other" and lower confidence.

JSON schema:
{{
  "llm_domain": "one allowed domain",
  "llm_domain_confidence": 0.0,
  "llm_modality": "one allowed modality",
  "llm_subtype": "one allowed subtype",
  "llm_tags": ["allowed tags only"],
  "llm_leakage_risk": "low|medium|high"
}}

Cleaned description:
\"\"\"{clean_text[:2500]}\"\"\"
""".strip()


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["response"]


def check_ollama_available():
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Return only JSON: {\"ok\": true}",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama nao esta acessivel em http://localhost:11434.\n"
            "Instale e rode uma LLM local gratuita com:\n"
            "  winget install Ollama.Ollama\n"
            "  ollama pull llama3.2:3b\n"
            "  ollama serve\n"
            f"Erro original: {exc}"
        ) from exc


def parse_llm_json(raw_response):
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    tags = parsed.get("llm_tags", [])
    if not isinstance(tags, list):
        tags = []

    clean_tags = [tag for tag in tags if tag in ALLOWED_TAGS]

    return {
        "llm_domain": parsed.get("llm_domain", "other")
        if parsed.get("llm_domain", "other") in ALLOWED_DOMAINS
        else "other",
        "llm_domain_confidence": float(parsed.get("llm_domain_confidence", 0.0)),
        "llm_modality": parsed.get("llm_modality", "unknown")
        if parsed.get("llm_modality", "unknown") in ALLOWED_MODALITIES
        else "unknown",
        "llm_subtype": parsed.get("llm_subtype", "unknown")
        if parsed.get("llm_subtype", "unknown") in ALLOWED_SUBTYPES
        else "unknown",
        "llm_tags": "|".join(clean_tags),
        "llm_leakage_risk": parsed.get("llm_leakage_risk", "medium")
        if parsed.get("llm_leakage_risk", "medium") in {"low", "medium", "high"}
        else "medium",
    }


def main():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_METAFEATURES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(INPUT_METAFEATURES)
    existing = pd.read_csv(OUTPUT_FEATURES) if OUTPUT_FEATURES.exists() else pd.DataFrame()
    done_dids = set(existing["did"].astype(int)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    print("\nGERANDO FEATURES LLM VIA OLLAMA")
    print(f"Modelo: {OLLAMA_MODEL}")
    print(f"Saida: {OUTPUT_FEATURES}")
    print("Se falhar, verifique: ollama serve / ollama pull llama3.2:3b")

    check_ollama_available()

    for idx, row in df_meta.iterrows():
        did = int(row["did"])
        if did in done_dids:
            continue

        clean_text = clean_semantic_text(
            row.get("semantic_text", ""),
            row.get("name", ""),
        )

        prompt = build_prompt(clean_text)

        try:
            raw_response = call_ollama(prompt)
            features = parse_llm_json(raw_response)
            error = ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as exc:
            features = {
                "llm_domain": "other",
                "llm_domain_confidence": 0.0,
                "llm_modality": "unknown",
                "llm_subtype": "unknown",
                "llm_tags": "",
                "llm_leakage_risk": "medium",
            }
            error = str(exc)

        output_row = {
            "did": did,
            "name": row.get("name", ""),
            "clean_text_tokens": len(clean_text.split()),
            **features,
            "llm_model": OLLAMA_MODEL,
            "error": error,
        }
        rows.append(output_row)
        pd.DataFrame(rows).to_csv(OUTPUT_FEATURES, index=False)

        status = "OK" if not error else "ERRO"
        print(f"[{idx + 1}/{len(df_meta)}] DID={did} {status} {features['llm_domain']} {features['llm_subtype']}")

    print(f"\nFeatures LLM salvas em: {OUTPUT_FEATURES}")


if __name__ == "__main__":
    main()
