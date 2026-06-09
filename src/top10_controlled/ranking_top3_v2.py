"""
Reformulacao TOP-3: em vez de prever o melhor (top-1) ruidoso, prever a
performance de cada algoritmo e avaliar por TOP-3 OVERLAP (quantos dos 3 previstos
estao nos 3 reais), top-1 e Spearman. Base V2, complete-6, CV agrupada+aleatoria,
3 sementes. Compara baseline (stats) vs +dominio.
Saida: data/top10_controlled/ranking_top3_v2_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data"
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
SEEDS = [42, 7, 123]


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()

DOM = {
 "health":["health","medical","clinical","patient","disease","cancer","heart","thyroid","cleveland","cholesterol","dermatolog","obesity","ilpd","hepatit","diabet"],
 "finance":["financ","credit","bank","loan","fraud","stock","insurance","bankrupt","forex","currency","betting"],
 "biology":["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli","qsar","mushroom","soybean"],
 "image":["image","pixel","mnist","svhn","digit","face","ocr","letter","mfeat","texture","optdigit"],
 "text":["text","document","news","review","sentiment","spam","nlp","authorship","lyrics","corpus"],
 "sensor_signal":["sensor","signal","seismic","robot","fault","vibration","electric","grid","vowel","water","wind"],
 "education":["student","school","education","exam","grade","univers","dropout"],
 "social":["social","census","survey","employee","churn","customer","compas","income","adult","baseball"],
}


def assign(name, desc):
    blob=" "+norm(name)+" | "+norm(desc)+" "; best,bn="other",0
    for d,ks in DOM.items():
        n=sum(1 for k in ks if k in blob)
        if n>bn: best,bn=d,n
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


def metrics(Yt, Yp):
    t1, t3, sp = [], [], []
    for t, p in zip(Yt, Yp):
        true3 = set(np.argsort(t)[-3:]); pred3 = set(np.argsort(p)[-3:])
        t3.append(len(true3 & pred3)/3.0)
        t1.append(int(np.argmax(p)==np.argmax(t)))
        sp.append(stats.spearmanr(p, t).correlation)
    return np.mean(t1), np.mean(t3), np.nanmean(sp)


def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv"); perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did"]+ALL6], on="did", how="inner")
    Y=df[ALL6].to_numpy(float); keep=~np.isnan(Y).any(axis=1)
    df=df.loc[keep].reset_index(drop=True); Y=Y[keep]
    df["dom"]=[assign(nm, txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    groups=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in ("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1]
    df[num]=df[num].replace([np.inf,-np.inf],np.nan)
    print(f"Complete-6: {len(df)} datasets")
    # baselines triviais top-3 overlap
    rnd=np.mean([len(set(np.random.default_rng(i).choice(6,3,replace=False))&set(np.argsort(Y[i])[-3:]))/3 for i in range(len(Y))])
    print(f"Top-3 overlap aleatorio (~0.5 esperado): {rnd:.3f}\n")

    stat=("stat",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num)
    def mdl(dom):
        t=[stat]
        if dom: t.append(("dom",OneHotEncoder(handle_unknown="ignore"),["dom"]))
        return Pipeline([("pre",ColumnTransformer(t,remainder="drop")),("r",MultiOutputRegressor(RandomForestRegressor(n_estimators=200,random_state=42)))])

    rows=[]
    for proto in ["random","grouped"]:
        print(f"=== CV {proto} ===")
        for name,dom in [("baseline_stats",False),("+dominio",True)]:
            t1s,t3s,sps=[],[],[]
            for s in SEEDS:
                cv=KFold(5,shuffle=True,random_state=s) if proto=="random" else GroupKFold(5)
                kw={} if proto=="random" else {"groups":groups}
                Yp=cross_val_predict(mdl(dom), df, Y, cv=cv, **kw)
                a,b,c=metrics(Y,Yp); t1s.append(a); t3s.append(b); sps.append(c)
            print(f"  {name:16s} top1={np.mean(t1s):.3f}  top3_overlap={np.mean(t3s):.3f}  spearman={np.mean(sps):.3f}")
            rows.append({"cv":proto,"config":name,"top1":np.mean(t1s),"top3_overlap":np.mean(t3s),"spearman":np.mean(sps)})
        # delta top3
        b=[r for r in rows if r["cv"]==proto and r["config"]=="baseline_stats"][0]
        d=[r for r in rows if r["cv"]==proto and r["config"]=="+dominio"][0]
        print(f"  -> ganho do dominio em top3_overlap: {d['top3_overlap']-b['top3_overlap']:+.4f}\n")

    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"ranking_top3_v2_summary.csv", index=False)
    print("Salvo: ranking_top3_v2_summary.csv")


if __name__ == "__main__":
    main()
