"""
BATERIA COMPLETA na base robusta V2 (547 datasets, 6 algoritmos).
Modelo constante (RF), 3 sementes, CV ALEATORIA e AGRUPADA (por colunas).

Arms (F1-macro vs baseline):
  baseline (stats) | +dominio | +texto_limpo(tfidf desc) | +texto_cru(tfidf nome+desc+cols, vazavel) | +dominio+texto
Bloco extra: REGRET (regressao multi-saida) meta-learner vs SBA vs aleatorio.
Saida: data/top10_controlled/full_battery_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data"
OUTDIR = DATA / "top10_controlled"
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
SEEDS = [42, 7, 123]

OTHER_MARK = ["binarized version","artificial","gametes","-pmlb","pmlb","twonorm","ringnorm","banana","madelon","hill-valley","parity","pie chart","piechart","pizzacutter","castmetal","megawatt","meanwhile","costamadre","fri_c","calendardow","tic-tac","chess","kr-vs-k","kropt","jungle","jm1","mozilla","jedit","defect","software","dummy","mofn","threeof9","xd6","monks","led7","led24","analcat","chscase"]
DOM = {
 "health":["health","medical","clinical","patient","disease","diagnos","cancer","tumor","diabet","heart","thyroid","hepatit","liver","ilpd","myocard","cleveland","cholesterol","apnea","obesity","dermatolog","mammograph","breast","dmft","blood","sick"],
 "finance":["financ","credit","bank","loan","fraud","stock","insurance","bankrupt","forex","currency","exchange","betting","bwin","valuation","creditab"],
 "biology":["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli","qsar","chemical","mushroom","soybean","eucalyptus","species"],
 "image":["image","pixel","mnist","svhn","digit","face","vision","ocr","letter","mfeat","cifar","texture","optdigit","pendigit","indian_pines"],
 "text":["text","document","news","review","sentiment","spam","email","nlp","tweet","corpus","authorship","lyrics","song","codexglue","code"],
 "sensor_signal":["sensor","signal","seismic","robot","fault","vibration","accelerom","electric","grid","vowel","speech","audio","water","pm10","no2","ozone","wind","gesture"],
 "education":["student","school","education","exam","grade","academic","univers","dropout","scores"],
 "social":["social","census","survey","employee","churn","customer","marketing","compas","ipums","tourism","income","adult","baseball","football","phishing","strikes"],
}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()


def assign_domain(name, desc):
    blob = " "+norm(name)+" | "+norm(desc)+" "
    best, bn = "other", 0
    for dom, kws in DOM.items():
        n = sum(1 for kw in kws if kw in blob)
        if n > bn: best, bn = dom, n
    return best if bn else "other"


def jacc(d2c):
    dids=list(d2c); par={d:d for d in dids}
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in range(len(dids)):
        si=d2c[dids[i]]
        if not si: continue
        for j in range(i+1,len(dids)):
            sj=d2c[dids[j]]
            if sj and len(si|sj) and len(si&sj)/len(si|sj)>=0.5: par[f(dids[i])]=f(dids[j])
    fam={d:f(d) for d in dids}; u={x:k for k,x in enumerate(sorted(set(fam.values())))}
    return {d:u[fam[d]] for d in dids}


def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv"); perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]+[a for a in ALL6 if a in perf.columns]], on="did", how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)

    df["dom"]=[assign_domain(nm, txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    df["text_clean"]=[norm(txt.get(int(d),{}).get("description","")) for d in df["did"]]
    df["text_raw"]=[norm(str(nm)+" "+txt.get(int(d),{}).get("description","")+" "+txt.get(int(d),{}).get("feature_names","")) for d,nm in zip(df["did"],df["name"])]
    d2c={int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]}
    groups=df["did"].astype(int).map(jacc(d2c)).values

    num=[c for c in meta.columns if c not in ("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1]
    df[num]=df[num].replace([np.inf,-np.inf],np.nan)
    yenc=LabelEncoder().fit_transform(df["best_classifier"].astype(str))

    stat=("stat",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num)
    def mdl(arms):
        t=[stat]
        if "dom" in arms: t.append(("dom",OneHotEncoder(handle_unknown="ignore"),["dom"]))
        if "clean" in arms: t.append(("tc",TfidfVectorizer(max_features=50),"text_clean"))
        if "raw" in arms: t.append(("tr",TfidfVectorizer(max_features=50),"text_raw"))
        return Pipeline([("pre",ColumnTransformer(t,remainder="drop")),("clf",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])

    ARMS={"+dominio":["dom"],"+texto_limpo":["clean"],"+texto_cru(vazavel)":["raw"],"+dominio+texto":["dom","clean"]}
    rows=[]
    for proto in ["random","grouped"]:
        print(f"\n=== CV {proto} (F1-macro) ===")
        base=[]
        for s in SEEDS:
            cv=StratifiedKFold(5,shuffle=True,random_state=s) if proto=="random" else StratifiedGroupKFold(5,shuffle=True,random_state=s)
            kw={} if proto=="random" else {"groups":groups}
            base.append(cross_val_score(mdl([]),df,yenc,cv=cv,scoring="f1_macro",**kw).mean())
        print(f"  baseline = {np.mean(base):.4f}")
        for name,arms in ARMS.items():
            ds=[]
            for s in SEEDS:
                cv=StratifiedKFold(5,shuffle=True,random_state=s) if proto=="random" else StratifiedGroupKFold(5,shuffle=True,random_state=s)
                kw={} if proto=="random" else {"groups":groups}
                ds.append(cross_val_score(mdl(arms),df,yenc,cv=cv,scoring="f1_macro",**kw).mean())
            d=np.mean(ds)-np.mean(base); _,p=stats.ttest_rel(ds,base)
            print(f"  {name:22s} F1={np.mean(ds):.4f}  dF1={d:+.4f}  p={p:.3f}")
            rows.append({"cv":proto,"arm":name,"f1":np.mean(ds),"delta":float(d),"p":float(p),"baseline":np.mean(base)})

    # REGRET (positivo do sistema)
    print("\n=== REGRET: meta-learner (stats) vs SBA ===")
    Y=df[[a for a in ALL6 if a in df.columns]].to_numpy(float)
    keep=~np.isnan(Y).any(axis=1)
    Yk=Y[keep]; dfk=df.loc[keep].reset_index(drop=True)
    sba=np.nanmean(Yk,axis=0).argmax()
    sba_reg=np.nanmean(Yk.max(axis=1)-Yk[:,sba])
    reg=MultiOutputRegressor(RandomForestRegressor(n_estimators=200,random_state=42))
    pre=ColumnTransformer([stat],remainder="drop")
    pipe=Pipeline([("pre",pre),("reg",reg)])
    Yp=cross_val_predict(pipe,dfk,Yk,cv=KFold(5,shuffle=True,random_state=42))
    ml_reg=np.mean([Yk[i].max()-Yk[i][np.argmax(Yp[i])] for i in range(len(Yk))])
    print(f"  SBA regret={sba_reg:.4f} | meta-learner regret={ml_reg:.4f} | reducao={100*(1-ml_reg/sba_reg):.0f}%")
    rows.append({"cv":"regret","arm":"meta_vs_sba","f1":ml_reg,"delta":sba_reg,"p":np.nan,"baseline":0.0})

    pd.DataFrame(rows).to_csv(OUTDIR/"full_battery_v2_summary.csv",index=False)
    print(f"\nSalvo: {OUTDIR/'full_battery_v2_summary.csv'}")


if __name__=="__main__":
    main()
