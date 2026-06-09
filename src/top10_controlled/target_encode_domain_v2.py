"""
TESTE DEFINITIVO de "dominio como numero": TARGET ENCODING supervisionado.
Cada dominio vira 6 numeros = P(cada algoritmo ser o melhor | dominio), ajustado
SO no treino (sem vazamento). E a forma mais forte de transformar dominio em numero.
Compara baseline vs +dominio_target_encoded, F1-macro e TOP-3 acc, CV aleatoria e
agrupada, 3 sementes. Base V2. Saida: data/top10_controlled/target_encode_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


class DomainTargetEncoder(BaseEstimator, TransformerMixin):
    """dominio -> vetor P(classe | dominio), ajustado no treino (com suavizacao)."""
    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        doms = np.asarray(X).ravel()
        d = pd.DataFrame({"d": doms, "y": y})
        self.prior_ = pd.Series(y).value_counts(normalize=True).reindex(self.classes_, fill_value=0).values
        tab = {}
        for dom, grp in d.groupby("d"):
            vc = grp["y"].value_counts(normalize=True).reindex(self.classes_, fill_value=0).values
            n = len(grp); m = 5.0  # suavizacao bayesiana p/ dominios pequenos
            tab[dom] = (n*vc + m*self.prior_)/(n+m)
        self.tab_ = tab
        return self
    def transform(self, X):
        doms = np.asarray(X).ravel()
        return np.array([self.tab_.get(dm, self.prior_) for dm in doms])


def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[assign(nm,txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1];df[num]=df[num].replace([np.inf,-np.inf],np.nan)
    from sklearn.preprocessing import LabelEncoder
    y=LabelEncoder().fit_transform(df["best_classifier"].astype(str))
    classes=np.unique(y)
    print(f"Base V2: {len(df)} datasets | dominio target-encoded (6 numeros/dominio)\n")

    stat=("s",Pipeline([("i",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num)
    def mdl(te):
        t=[stat]
        if te:t.append(("te",DomainTargetEncoder(),["dom"]))
        return Pipeline([("p",ColumnTransformer(t,remainder="drop")),("c",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])

    def top3(proba): return np.mean([int(y[i] in set(classes[np.argsort(proba[i])[-3:]])) for i in range(len(y))])

    rows=[]
    for proto in["random","grouped"]:
        print(f"=== CV {proto} ===")
        bf,bt,df1,dt=[],[],[],[]
        for s in SEEDS:
            cv=StratifiedKFold(5,shuffle=True,random_state=s) if proto=="random" else StratifiedGroupKFold(5,shuffle=True,random_state=s)
            kw={} if proto=="random" else{"groups":g}
            bf.append(cross_val_score(mdl(False),df,y,cv=cv,scoring="f1_macro",**kw).mean())
            bt.append(top3(cross_val_predict(mdl(False),df,y,cv=cv,method="predict_proba",**kw)))
            df1.append(cross_val_score(mdl(True),df,y,cv=cv,scoring="f1_macro",**kw).mean())
            dt.append(top3(cross_val_predict(mdl(True),df,y,cv=cv,method="predict_proba",**kw)))
        _,pf=stats.ttest_rel(df1,bf)
        print(f"  F1-macro : baseline={np.mean(bf):.4f} | +dom_TE={np.mean(df1):.4f} | d={np.mean(df1)-np.mean(bf):+.4f} | p={pf:.3f}")
        print(f"  TOP-3 acc: baseline={np.mean(bt):.4f} | +dom_TE={np.mean(dt):.4f} | d={np.mean(dt)-np.mean(bt):+.4f}\n")
        rows.append({"cv":proto,"f1_base":np.mean(bf),"f1_te":np.mean(df1),"f1_delta":np.mean(df1)-np.mean(bf),"f1_p":float(pf),"top3_base":np.mean(bt),"top3_te":np.mean(dt),"top3_delta":np.mean(dt)-np.mean(bt)})
    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"target_encode_v2_summary.csv",index=False)
    print("Salvo: target_encode_v2_summary.csv")

if __name__=="__main__":main()
