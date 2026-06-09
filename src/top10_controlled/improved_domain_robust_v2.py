"""
OPCAO B (eficiente): dominio melhorado para a base V2 (547) - nome+descricao,
keywords expandidas + categoria explicita 'other' (sinteticos/jogos/software/misc),
reduzindo drasticamente os 'unknown'. Mesmo teste robusto: baseline vs +dominio,
CV aleatoria e AGRUPADA (por colunas), multi-semente, alvo 6 algoritmos.
Saida: data/top10_controlled/improved_domain_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data"
OUTDIR = DATA / "top10_controlled"
ALL6 = ["DecisionTree", "KNN", "LogisticRegression", "MLP", "Perceptron", "SVM"]
SEEDS = [42, 7, 123]

# marcadores de "sem dominio real" (sintetico/jogo/software/teste) -> other
OTHER_MARK = ["binarized version", "artificial", "gametes", "-pmlb", "pmlb", "twonorm",
    "ringnorm", "banana", "madelon", "hill-valley", "hill valley", "parity", "pie chart",
    "piechart", "pizzacutter", "pizza cutter", "castmetal", "megawatt", "meanwhile",
    "costamadre", "fri_c", "calendardow", "tic-tac", "chess", "kr-vs-k", "kropt", "jungle",
    "jm1", "mozilla", "jedit", "defect", "software", " test ", "dummy", "mofn", "threeof9",
    "xd6", "monks", "led7", "led24", "waveform", "analcat", "chscase", "delve", "fri c"]

DOM = {
    "health": ["health","medical","clinical","patient","disease","diagnos","cancer","tumor",
        "diabet","heart","cardio","thyroid","hepatit","liver","ilpd","myocard","cleveland",
        "cholesterol","apnea","obesity","dermatolog","mammograph","breast","dmft","blood","sick","allrep"],
    "finance": ["financ","credit","bank","loan","fraud","stock","insurance","bankrupt","forex",
        "currency","exchange","betting","bwin","valuation","creditab","euro","usd","gbp"],
    "biology": ["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli",
        "qsar","chemical","mushroom","soybean","eucalyptus","species","bio","cell"],
    "image": ["image","pixel","mnist","svhn","digit","face","vision","ocr","letter","mfeat",
        "cifar","texture","optdigit","pendigit","photo","picture","indian_pines","binary_alpha"],
    "text": ["text","document","news","review","sentiment","spam","email","nlp","tweet",
        "corpus","authorship","lyrics","song","language","codexglue","code"],
    "sensor_signal": ["sensor","signal","seismic","robot","fault","vibration","accelerom",
        "electric","grid","vowel","speech","audio","eating","water","pm10","no2","el_nino",
        "el nino","ozone","weather","wind","touch","gesture","activity"],
    "education": ["student","school","education","exam","grade","academic","univers","dropout","scores","entrance"],
    "social": ["social","census","survey","employee","churn","customer","marketing","compas",
        "ipums","tourism","income","adult","demograph","baseball","football","sport","poll",
        "phishing","strikes","mobility","socmob"],
}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", "".join(c for c in t if not unicodedata.combining(c)))).strip()


def assign_domain(name, desc):
    blob = " " + norm(name) + " | " + norm(desc) + " "
    if any(mk in blob for mk in OTHER_MARK):
        # ainda assim, se tiver dominio tematico forte, prioriza-o (ex.: cleveland binarized=health)
        pass
    best, bestn = "other", 0
    for dom, kws in DOM.items():
        n = sum(1 for kw in kws if kw in blob)
        if n > bestn:
            best, bestn = dom, n
    if bestn == 0:
        return "other"
    return best


def jaccard_families(d2c):
    dids = list(d2c); parent = {d: d for d in dids}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(dids)):
        si = d2c[dids[i]]
        if not si: continue
        for j in range(i+1, len(dids)):
            sj = d2c[dids[j]]
            if not sj: continue
            u = len(si | sj)
            if u and len(si & sj)/u >= 0.5: parent[find(dids[i])] = find(dids[j])
    fam = {d: find(d) for d in dids}; uniq = {f: k for k, f in enumerate(sorted(set(fam.values())))}
    return {d: uniq[fam[d]] for d in dids}


def main():
    meta = pd.read_csv(DATA/"metafeatures_v2.csv"); perf = pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt = {r["did"]: r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df = meta.merge(perf[["did","best_classifier"]], on="did", how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)

    df["dom"] = [assign_domain(txt.get(int(d),{}).get("description",""), m)  # name from meta
                 for d, m in zip(df["did"], df["name"])]
    # corrige: usa name + desc
    df["dom"] = [assign_domain(nm, txt.get(int(d),{}).get("description","")) for d, nm in zip(df["did"], df["name"])]
    print("Distribuicao dominio (melhorado):", df["dom"].value_counts().to_dict())
    print(f"'other': {(df['dom']=='other').sum()} de {len(df)}\n")

    d2c = {int(d): set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]}
    fam = jaccard_families(d2c); groups = df["did"].astype(int).map(fam).values

    numeric_cols = [c for c in meta.columns if c not in ("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    df[numeric_cols] = df[numeric_cols].replace([np.inf,-np.inf], np.nan)
    yenc = LabelEncoder().fit_transform(df["best_classifier"].astype(str))

    def model(with_dom):
        t = [("stat", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), numeric_cols)]
        if with_dom: t.append(("dom", OneHotEncoder(handle_unknown="ignore"), ["dom"]))
        return Pipeline([("pre", ColumnTransformer(t, remainder="drop")),
                         ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))])

    rows = []
    for proto in ["random","grouped"]:
        db, dd = [], []
        for s in SEEDS:
            if proto == "random":
                cv = StratifiedKFold(5, shuffle=True, random_state=s)
                fb = cross_val_score(model(False), df, yenc, cv=cv, scoring="f1_macro")
                fd = cross_val_score(model(True), df, yenc, cv=cv, scoring="f1_macro")
            else:
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=s)
                fb = cross_val_score(model(False), df, yenc, cv=cv, groups=groups, scoring="f1_macro")
                fd = cross_val_score(model(True), df, yenc, cv=cv, groups=groups, scoring="f1_macro")
            db.append(fb.mean()); dd.append(fd.mean())
        delta = np.mean(dd)-np.mean(db); _, p = stats.ttest_rel(dd, db)
        print(f"CV {proto:8s}: baseline={np.mean(db):.4f} | +dominio={np.mean(dd):.4f} | dF1={delta:+.4f} | p={p:.3f}")
        rows.append({"cv":proto,"f1_baseline":np.mean(db),"f1_domain":np.mean(dd),"delta_f1":float(delta),"p_value":float(p)})

    pd.DataFrame(rows).to_csv(OUTDIR/"improved_domain_v2_summary.csv", index=False)
    print(f"\nSalvo: {OUTDIR/'improved_domain_v2_summary.csv'}")


if __name__ == "__main__":
    main()
