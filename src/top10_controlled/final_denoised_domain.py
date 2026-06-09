"""Teste final: dominio vs baseline sobre o ALVO LIMPO (CV repetida) na base V2.
CV aleatoria + agrupada, 3 sementes. Saida: data/top10_controlled/final_denoised_summary.csv"""
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

DATA=Path(__file__).resolve().parents[2]/"data"; SEEDS=[42,7,123]
DOM={"health":["health","medical","clinical","patient","disease","cancer","heart","thyroid","cleveland","cholesterol","dermatolog","obesity","ilpd","diabet"],"finance":["financ","credit","bank","loan","fraud","stock","insurance","forex","currency","betting"],"biology":["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli","qsar","mushroom"],"image":["image","pixel","mnist","svhn","digit","face","ocr","letter","mfeat","texture"],"text":["text","document","news","review","sentiment","spam","nlp","authorship","lyrics"],"sensor_signal":["sensor","signal","seismic","robot","fault","electric","grid","vowel","water","wind"],"education":["student","school","education","exam","grade","univers","dropout"],"social":["social","census","survey","employee","churn","customer","compas","income","adult","baseball"]}

def norm(t):
    t=unicodedata.normalize("NFKD",str(t).lower());return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()
def assign(nm,ds):
    b=" "+norm(nm)+" | "+norm(ds)+" ";best,bn="other",0
    for d,ks in DOM.items():
        n=sum(1 for k in ks if k in b)
        if n>bn:best,bn=d,n
    return best if bn else "other"
def jacc(d2c):
    dd=list(d2c);par={d:d for d in dd}
    def f(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    for i in range(len(dd)):
        si=d2c[dd[i]]
        if not si:continue
        for j in range(i+1,len(dd)):
            sj=d2c[dd[j]]
            if sj and len(si|sj) and len(si&sj)/len(si|sj)>=0.5:par[f(dd[i])]=f(dd[j])
    fam={d:f(d) for d in dd};u={x:k for k,x in enumerate(sorted(set(fam.values())))};return{d:u[fam[d]] for d in dd}

def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2_denoised.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[assign(nm,txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1];df[num]=df[num].replace([np.inf,-np.inf],np.nan)
    y=LabelEncoder().fit_transform(df["best_classifier"].astype(str))
    print(f"Alvo LIMPO: {len(df)} datasets | dist={pd.Series(df['best_classifier']).value_counts().to_dict()}\n")
    def mdl(dom):
        t=[("s",Pipeline([("i",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num)]
        if dom:t.append(("d",OneHotEncoder(handle_unknown="ignore"),["dom"]))
        return Pipeline([("p",ColumnTransformer(t,remainder="drop")),("c",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
    rows=[]
    for proto in["random","grouped"]:
        b=[]
        for s in SEEDS:
            cv=StratifiedKFold(5,shuffle=True,random_state=s) if proto=="random" else StratifiedGroupKFold(5,shuffle=True,random_state=s)
            kw={} if proto=="random" else{"groups":g}
            b.append(cross_val_score(mdl(False),df,y,cv=cv,scoring="f1_macro",**kw).mean())
        dd=[]
        for s in SEEDS:
            cv=StratifiedKFold(5,shuffle=True,random_state=s) if proto=="random" else StratifiedGroupKFold(5,shuffle=True,random_state=s)
            kw={} if proto=="random" else{"groups":g}
            dd.append(cross_val_score(mdl(True),df,y,cv=cv,scoring="f1_macro",**kw).mean())
        delta=np.mean(dd)-np.mean(b);_,p=stats.ttest_rel(dd,b)
        print(f"CV {proto:8s}: baseline={np.mean(b):.4f} | +dominio={np.mean(dd):.4f} | dF1={delta:+.4f} | p={p:.3f}")
        rows.append({"cv":proto,"f1_baseline":np.mean(b),"f1_domain":np.mean(dd),"delta":float(delta),"p":float(p)})
    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"final_denoised_summary.csv",index=False)
    print("\nSalvo: final_denoised_summary.csv")

if __name__=="__main__":main()
