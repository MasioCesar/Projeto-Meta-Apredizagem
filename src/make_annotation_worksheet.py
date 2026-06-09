"""Gera planilha compacta para anotacao de dominio por LLM dos 547 datasets V2.
Inclui palpite por keyword (kw_domain) como prior, p/ acelerar a anotacao.
Saida: data/v2_annotation_worksheet.csv"""
import json, re, unicodedata
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

domain_keywords = {
    "health": {"health","medical","clinical","patient","disease","diagnosis","cancer","tumor","diabetes","heart","blood","breast","thyroid","hepatitis","eeg","ecg","covid","hospital","symptom"},
    "finance": {"finance","financial","credit","bank","loan","fraud","stock","insurance","bankruptcy","default","payment","investment","transaction","mortgage","trading"},
    "biology": {"gene","genes","genomic","protein","dna","rna","sequence","yeast","ecoli","microarray","molecular","cell","species","mutation","expression","qsar","chemical","molecule"},
    "image": {"image","images","pixel","pixels","mnist","digit","face","vision","ocr","letter","texture","rgb","video"},
    "text": {"text","document","news","review","sentiment","spam","email","language","nlp","tweet","corpus","sarcasm"},
    "sensor_signal": {"sensor","signal","seismic","robot","fault","vibration","activity","gesture","motion","accelerometer","waveform","radar","sonar","gas"},
    "education": {"student","students","school","education","exam","grade","academic","university","dropout","mooc"},
    "social": {"social","network","user","community","rating","movie","recommendation","customer","adult","census","income","churn","marketing"},
}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t)).strip()


def kw_domain(desc):
    d = norm(desc)
    sc = {dom: sum(1 for kw in kws if re.search(rf"\b{kw}\b", d)) for dom, kws in domain_keywords.items()}
    return max(sc, key=sc.get) if max(sc.values()) > 0 else "unknown"


def main():
    txt = {r["did"]: r for r in json.loads((DATA / "v2_descriptions.json").read_text(encoding="utf-8"))}
    meta = pd.read_csv(DATA / "metafeatures_v2.csv")[["did", "name", "nr_inst", "nr_attr", "nr_class"]]
    rows = []
    for _, r in meta.iterrows():
        did = int(r["did"]); t = txt.get(did, {})
        desc = re.sub(r"\s+", " ", str(t.get("description", "")))[:180]
        feats = str(t.get("feature_names", ""))[:120]
        rows.append({"did": did, "name": r["name"], "kw": kw_domain(t.get("description", "")),
                     "nr_attr": int(r["nr_attr"]), "desc": desc, "feats": feats})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "v2_annotation_worksheet.csv", index=False)
    print(f"Salvo: {DATA/'v2_annotation_worksheet.csv'} ({len(df)} linhas)")
    print(f"kw_domain unknown: {(df['kw']=='unknown').sum()}")


if __name__ == "__main__":
    main()
