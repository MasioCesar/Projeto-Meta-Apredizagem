"""
(A)+(B): recalcula domain_score LIMPO a partir da DESCRICAO isolada do OpenML
(sem nome do dataset, sem nomes das features), e tambem normalizado por tamanho.

Gera variantes por dataset:
  - score_desc        : soma ponderada de keywords de dominio SO na descricao
  - score_desc_norm   : score_desc / num_tokens_descricao  (remove verbosidade)
  - hits_desc_norm    : (n keywords distintas encontradas) / num_tokens
  - domain_desc       : dominio argmax pela descricao isolada (categoria limpa)
Cache: data/domain_score_clean.csv
"""
import re
import time
import unicodedata
from pathlib import Path

import openml
import pandas as pd

from common import INPUT_METAFEATURES

OUT = Path(INPUT_METAFEATURES).parent / "domain_score_clean.csv"

# ---- mesmos pesos do 01_select_datasets.py (autoritativos) ----
domain_keywords = {
    "health": {"health":4,"medical":5,"clinical":5,"patient":5,"disease":5,"diagnosis":5,"cancer":7,"tumor":7,"diabetes":7,"heart":7,"blood":5,"breast":6,"thyroid":6,"hepatitis":6,"maternal":5,"transfusion":6,"eeg":7,"ecg":7,"arrhythmia":7,"covid":6,"hospital":5,"symptom":4,"syndrome":5},
    "finance": {"finance":5,"financial":5,"credit":7,"bank":6,"loan":7,"fraud":7,"stock":6,"insurance":6,"bankruptcy":6,"approval":4,"default":6,"payment":5,"valuation":5,"investment":6,"transaction":6,"pricing":5,"mortgage":7,"economy":5,"trading":7},
    "biology": {"gene":7,"genes":7,"genomic":7,"genomics":7,"protein":7,"dna":7,"rna":7,"sequence":6,"splice":8,"yeast":8,"ecoli":8,"promoter":7,"microarray":7,"bioinformatics":7,"molecular":5,"cell":4,"species":4,"mutation":6,"peptide":7,"expression":5,"microbiome":7,"fungi":5,"qsar":8,"biodeg":7,"biodegradation":7,"chemical":6,"compound":6,"molecule":6},
    "image": {"image":6,"images":6,"pixel":5,"pixels":5,"mnist":8,"digit":6,"digits":6,"face":6,"faces":6,"vision":6,"ocr":6,"letter":5,"letters":5,"optdigits":8,"pendigits":8,"satimage":8,"texture":5,"segment":5,"rgb":6,"grayscale":6,"video":5,"facial":6},
    "text": {"text":6,"document":6,"documents":6,"news":5,"review":5,"reviews":5,"sentiment":7,"spam":7,"email":6,"emails":6,"language":6,"nlp":7,"imdb":6,"tweet":6,"twitter":6,"comment":5,"comments":5,"reuters":6,"corpus":6,"article":5,"sarcasm":7,"fake":6,"embedding":7,"translation":6},
    "sensor_signal": {"sensor":6,"sensors":6,"signal":6,"signals":6,"seismic":7,"robot":6,"navigation":6,"fault":6,"faults":6,"vibration":6,"activity":6,"gesture":7,"phase":5,"motion":6,"har":7,"accelerometer":7,"gyro":7,"waveform":6,"electricity":5,"power":5,"energy":5,"steel":5,"plates":5,"telemetry":6,"radar":7,"sonar":7},
    "education": {"student":7,"students":7,"school":5,"education":7,"exam":5,"grade":5,"grades":5,"academic":5,"university":5,"college":5,"performance":2,"knowledge":5,"learning":2,"course":4,"score":3,"scores":3,"dropout":7,"mooc":7,"tutor":6,"teaching":5},
    "social": {"social":6,"network":6,"networks":6,"user":5,"users":5,"community":5,"rating":6,"ratings":6,"movie":5,"movies":5,"recommendation":6,"recommender":6,"customer":4,"customers":4,"adult":7,"census":7,"income":5,"demographic":5,"behavior":5,"shopping":5,"ecommerce":5,"churn":7,"retention":6},
}


def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_and_hits(text, keywords):
    score, hits = 0, 0
    for word, weight in keywords.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            score += weight
            hits += 1
    return score, hits


def main():
    meta = pd.read_csv(INPUT_METAFEATURES)
    dids = meta["did"].dropna().astype(int).tolist()
    print(f"Buscando descricao isolada de {len(dids)} datasets no OpenML...")

    rows = []
    for i, did in enumerate(dids, 1):
        desc = ""
        try:
            ds = openml.datasets.get_dataset(did, download_data=False,
                                             download_qualities=False,
                                             download_features_meta_data=False)
            desc = ds.description or ""
        except Exception as e:
            print(f"  [{i}/{len(dids)}] did={did} FALHA: {str(e)[:50]}")
        desc_norm = normalize_text(desc)
        n_tokens = max(1, len(desc_norm.split()))
        per_domain = {d: score_and_hits(desc_norm, kw) for d, kw in domain_keywords.items()}
        scores = {d: s for d, (s, h) in per_domain.items()}
        best = max(scores, key=scores.get) if max(scores.values()) > 0 else "none"
        best_score = scores[best] if best != "none" else 0
        best_hits = per_domain[best][1] if best != "none" else 0
        rows.append({
            "did": did,
            "desc_tokens": n_tokens,
            "score_desc": best_score,
            "score_desc_norm": best_score / n_tokens,
            "hits_desc_norm": best_hits / n_tokens,
            "domain_desc": best,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(dids)}] ok")
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSalvo: {OUT}")
    # cobertura e diagnostico de confundimento
    got = (df["desc_tokens"] > 1).sum()
    print(f"Descricoes nao-vazias: {got}/{len(df)}")
    m = meta.merge(df, on="did")
    m["text_len_full"] = m["semantic_text"].astype(str).str.split().apply(len)
    print("\nCorrelacao Spearman (score LIMPO vs confundidores):")
    for col in ["score_desc", "score_desc_norm", "hits_desc_norm"]:
        r_attr = m[[col, "nr_attr"]].corr("spearman").iloc[0,1]
        r_len = m[[col, "text_len_full"]].corr("spearman").iloc[0,1]
        print(f"  {col:16s} vs nr_attr={r_attr:+.3f} | vs text_len={r_len:+.3f}")
    print("\nComparacao com domain_score original:")
    r_orig_attr = m[["domain_score","nr_attr"]].corr("spearman").iloc[0,1]
    print(f"  domain_score (original) vs nr_attr = {r_orig_attr:+.3f}")


if __name__ == "__main__":
    main()
