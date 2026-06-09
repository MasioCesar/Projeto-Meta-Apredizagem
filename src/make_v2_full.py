"""Gera metafeatures_v2_full.csv e performance_matrix_v2_full.csv no MESMO schema
da base original (com semantic_text, predicted_domain, domain_score, best_accuracy),
para rodar todos os scripts de teste originais sobre a base V2."""
import re, unicodedata, json
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]

domain_keywords = {
 "health":{"health":4,"medical":5,"clinical":5,"patient":5,"disease":5,"diagnosis":5,"cancer":7,"tumor":7,"diabetes":7,"heart":7,"blood":5,"breast":6,"thyroid":6,"hepatitis":6,"eeg":7,"ecg":7,"covid":6,"hospital":5,"cleveland":6,"cholesterol":6,"dermatology":6,"obesity":6,"myocardial":7,"ilpd":6},
 "finance":{"finance":5,"financial":5,"credit":7,"bank":6,"loan":7,"fraud":7,"stock":6,"insurance":6,"bankruptcy":6,"default":6,"payment":5,"investment":6,"transaction":6,"mortgage":7,"trading":7,"forex":7,"currency":6},
 "biology":{"gene":7,"genes":7,"genomic":7,"protein":7,"dna":7,"rna":7,"sequence":6,"yeast":8,"ecoli":8,"microarray":7,"molecular":5,"cell":4,"species":4,"mutation":6,"expression":5,"qsar":8,"chemical":6,"molecule":6,"mushroom":6,"soybean":6},
 "image":{"image":6,"images":6,"pixel":5,"pixels":5,"mnist":8,"digit":6,"face":6,"vision":6,"ocr":6,"letter":5,"texture":5,"rgb":6,"svhn":7,"mfeat":6,"optdigits":7},
 "text":{"text":6,"document":6,"news":5,"review":5,"sentiment":7,"spam":7,"email":6,"language":6,"nlp":7,"tweet":6,"corpus":6,"authorship":6,"lyrics":6},
 "sensor_signal":{"sensor":6,"signal":6,"seismic":7,"robot":6,"fault":6,"vibration":6,"activity":6,"gesture":7,"accelerometer":7,"electricity":5,"electric":5,"grid":5,"vowel":5,"water":4,"ozone":5,"wind":4},
 "education":{"student":7,"students":7,"school":5,"education":7,"exam":5,"grade":5,"academic":5,"university":5,"dropout":7,"scores":3},
 "social":{"social":6,"network":6,"user":5,"community":5,"rating":6,"movie":5,"customer":4,"adult":7,"census":7,"income":5,"churn":7,"marketing":5,"compas":6,"survey":4,"employee":4,"baseball":5},
}


def normtext(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s_-]"," ",t)).strip()


def score(text):
    s = {dom: sum(w for kw,w in kws.items() if re.search(rf"\b{re.escape(kw)}\b", text)) for dom,kws in domain_keywords.items()}
    best = max(s, key=s.get)
    return (best, s[best]) if s[best] > 0 else ("unknown", 0)


def main():
    meta = pd.read_csv(DATA/"metafeatures_v2.csv")
    perf = pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt = {r["did"]: r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}

    sem, doms, scrs = [], [], []
    for _, r in meta.iterrows():
        did = int(r["did"]); t = txt.get(did, {})
        s = normtext(str(r["name"]) + " " + t.get("description","") + " " + t.get("feature_names",""))
        d, sc = score(s)
        sem.append(s); doms.append(d); scrs.append(sc)
    meta["semantic_text"] = sem
    meta["predicted_domain"] = doms
    meta["domain_score"] = scrs
    meta.to_csv(DATA/"metafeatures_v2_full.csv", index=False)

    existing = [c for c in ALL6 if c in perf.columns]
    perf["best_classifier"] = perf[existing].idxmax(axis=1, skipna=True)
    perf["best_accuracy"] = perf[existing].max(axis=1, skipna=True)
    perf.to_csv(DATA/"performance_matrix_v2_full.csv", index=False)

    print(f"metafeatures_v2_full: {meta.shape} | dominio: {pd.Series(doms).value_counts().to_dict()}")
    print(f"domain_score>7: {(np.array(scrs)>7).sum()} de {len(scrs)}")
    print(f"performance_matrix_v2_full: {perf.shape}")


if __name__ == "__main__":
    main()
