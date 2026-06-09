from pathlib import Path
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "llm_features"

INPUT_METAFEATURES = DATA_DIR / __import__("os").environ.get("MAB_META", "metafeatures_selected_datasets.csv")
OUTPUT_FEATURES = OUTPUT_DIR / "llm_semantic_features.csv"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def load_shared_llm_helpers():
    helper_path = Path(__file__).resolve().parent / "04_experiment_b_test26_generate_llm_features.py"
    spec = importlib.util.spec_from_file_location("llm_feature_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_shared_llm_helpers()


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY nao configurada.\n"
            "Crie uma chave no Google AI Studio e rode:\n"
            "  $env:GEMINI_API_KEY='SUA_CHAVE_AQUI'\n"
            "Depois execute este script novamente."
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 503} and attempt < 3:
                wait_seconds = 30 * (attempt + 1)
                print(f"Rate limit/servico indisponivel ({exc.code}). Aguardando {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue
            raise
    else:
        raise last_error

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Resposta inesperada da Gemini API: {result}") from exc


def check_gemini_available():
    call_gemini('Return only JSON: {"ok": true}')


def main():
    if not INPUT_METAFEATURES.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_METAFEATURES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(INPUT_METAFEATURES)
    existing = pd.read_csv(OUTPUT_FEATURES) if OUTPUT_FEATURES.exists() else pd.DataFrame()
    if not existing.empty and "error" in existing.columns:
        existing = existing[existing["error"].fillna("") == ""].copy()
    done_dids = set(existing["did"].astype(int)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []

    print("\nGERANDO FEATURES LLM VIA GEMINI API ONLINE")
    print(f"Modelo: {GEMINI_MODEL}")
    print(f"Saida: {OUTPUT_FEATURES}")
    print("Variavel esperada: GEMINI_API_KEY ou GOOGLE_API_KEY")

    check_gemini_available()

    for idx, row in df_meta.iterrows():
        did = int(row["did"])
        if did in done_dids:
            continue

        clean_text = helpers.clean_semantic_text(
            row.get("semantic_text", ""),
            row.get("name", ""),
        )
        prompt = helpers.build_prompt(clean_text)

        raw_response = call_gemini(prompt)
        features = helpers.parse_llm_json(raw_response)
        error = ""

        output_row = {
            "did": did,
            "name": row.get("name", ""),
            "clean_text_tokens": len(clean_text.split()),
            **features,
            "llm_model": GEMINI_MODEL,
            "llm_provider": "google_gemini_api",
            "error": error,
        }
        rows.append(output_row)
        pd.DataFrame(rows).to_csv(OUTPUT_FEATURES, index=False)

        status = "OK" if not error else "ERRO"
        print(f"[{idx + 1}/{len(df_meta)}] DID={did} {status} {features['llm_domain']} {features['llm_subtype']}")
        time.sleep(7)

    print(f"\nFeatures LLM salvas em: {OUTPUT_FEATURES}")
    print("Agora rode: venv\\Scripts\\python.exe src\\04_experiment_b_test27_llm_features.py")


if __name__ == "__main__":
    main()
