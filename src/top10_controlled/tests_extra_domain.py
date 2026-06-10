"""
Testes para tentar revelar um efeito MAIOR do dominio:
 #1 Analise POR DOMINIO: onde o dominio ajuda mais (delta de acuracia por dominio).
 #2 Alvo = FAMILIA de algoritmo (tree/linear/kernel/neural/instance), rank encoding.
 #3 Dominio -> REGRET: o dominio reduz a perda vs oraculo?
Tudo com CV agrupada, multi-semente. Saida: data/top10_controlled/tests_extra_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

DATA=Path(__file__).resolve().parents[2]/"data"
ALL6=["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
FAMILY={"DecisionTree":"tree","LogisticRegression":"linear","Perceptron":"linear","SVM":"kernel","MLP":"neural","KNN":"instance"}
FAMS=["tree","linear","kernel","neural","instance"]
SEEDS=[42,7,123,2024,11,99,7777,2025,1,13]
DOMKW={"health":["health","medical","clinical","patient","disease","cancer","heart","thyroid","cleveland","cholesterol","dermatolog","obesity","ilpd","diabet"],"finance":["financ","credit","bank","loan","fraud","stock","insurance","forex","currency","betting"],"biology":["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli","qsar","mushroom"],"image":["image","pixel","mnist","svhn","digit","face","ocr","letter","mfeat","texture"],"text":["text","document","news","review","sentiment","spam","nlp","authorship","lyrics"],"sensor_signal":["sensor","signal","seismic","robot","fault","electric","grid","vowel","water","wind"],"education":["student","school","education","exam","grade","univers","dropout"],"social":["social","census","survey","employee","churn","customer","compas","income","adult","baseball"]}

def norm(t):
    t=unicodedata.normalize("NFKD",str(t).lower());return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()
def assign(nm,ds):
    b=" "+norm(nm)+" | "+norm(ds)+" ";best,bn="other",0
    for d,ks in DOMKW.items():
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

def enc_rank(cats_tr, Ymat_tr, cols):
    """dominio -> rank medio de cada coluna (algoritmo/familia), suavizado."""
    prior=np.nanmean(Ymat_tr,axis=0)
    d=pd.DataFrame(Ymat_tr,columns=cols); d["c"]=cats_tr; tab={}
    for c,g in d.groupby("c"):
        r=pd.DataFrame(g[cols]).rank(axis=1).to_numpy(float); v=np.nanmean(r,axis=0)
        n=len(g); tab[c]=(n*v+5.0*prior)/(n+5.0)
    return tab,prior
def apply_enc(cats,tab,prior): return np.array([tab.get(c,prior) for c in cats])

def load():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]+[a for a in ALL6 if a in perf.columns]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[assign(nm,txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1]
    Xs=df[num].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    Y6=df[ALL6].to_numpy(float)
    return df,Xs,Y6,np.array(df["dom"]),g

def rf(): return RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")

def main():
    df,Xs,Y6,dom,g=load()
    rows=[]

    # ===== #2 ALVO = FAMILIA =====
    Yfam=np.column_stack([np.nanmax(Y6[:,[i for i,a in enumerate(ALL6) if FAMILY[a]==fam]],axis=1) for fam in FAMS])
    bestfam=np.array([FAMS[i] for i in np.nanargmax(Yfam,axis=1)])
    yf=LabelEncoder().fit_transform(bestfam)
    print(f"[#2] Alvo FAMILIA | dist={pd.Series(bestfam).value_counts().to_dict()}")
    bacc,dacc,bf1,df1=[],[],[],[]
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        ab,ad,fb,fd=[],[],[],[]
        for tr,te in cv.split(Xs,yf,groups=g):
            imp=SimpleImputer(strategy="median").fit(Xs[tr])
            Xtr,Xte=imp.transform(Xs[tr]),imp.transform(Xs[te])
            m=rf().fit(Xtr,yf[tr]); pb=m.predict(Xte)
            tab,prior=enc_rank(dom[tr],Yfam[tr],FAMS)
            Xtr2=np.hstack([Xtr,apply_enc(dom[tr],tab,prior)]); Xte2=np.hstack([Xte,apply_enc(dom[te],tab,prior)])
            md=rf().fit(Xtr2,yf[tr]); pd_=md.predict(Xte2)
            ab.append(accuracy_score(yf[te],pb)); ad.append(accuracy_score(yf[te],pd_))
            fb.append(f1_score(yf[te],pb,average="macro")); fd.append(f1_score(yf[te],pd_,average="macro"))
        bacc.append(np.mean(ab)); dacc.append(np.mean(ad)); bf1.append(np.mean(fb)); df1.append(np.mean(fd))
    _,pa=stats.ttest_rel(dacc,bacc); _,pf=stats.ttest_rel(df1,bf1)
    print(f"[#2] Acuracia: {np.mean(bacc):.4f}->{np.mean(dacc):.4f} (d={np.mean(dacc)-np.mean(bacc):+.4f}, p={pa:.3f}) | F1: {np.mean(bf1):.4f}->{np.mean(df1):.4f} (d={np.mean(df1)-np.mean(bf1):+.4f}, p={pf:.3f})\n")
    rows.append({"teste":"#2 familia","metrica":"acuracia","base":np.mean(bacc),"dom":np.mean(dacc),"delta":np.mean(dacc)-np.mean(bacc),"p":pa})
    rows.append({"teste":"#2 familia","metrica":"f1","base":np.mean(bf1),"dom":np.mean(df1),"delta":np.mean(df1)-np.mean(bf1),"p":pf})

    # ===== #1 por dominio + #3 regret (alvo = algoritmo, rank encoding) =====
    y=LabelEncoder().fit(df["best_classifier"].astype(str)); ycodes=y.transform(df["best_classifier"].astype(str)); classes=y.classes_
    complete=~np.isnan(Y6).any(axis=1)
    perdom_b={d:[] for d in set(dom)}; perdom_d={d:[] for d in set(dom)}
    reg_b,reg_d=[],[]
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        predb=np.empty(len(df),dtype=object); predd=np.empty(len(df),dtype=object)
        for tr,te in cv.split(Xs,ycodes,groups=g):
            imp=SimpleImputer(strategy="median").fit(Xs[tr]); Xtr,Xte=imp.transform(Xs[tr]),imp.transform(Xs[te])
            mb=rf().fit(Xtr,ycodes[tr]); predb[te]=classes[mb.predict(Xte)]
            tab,prior=enc_rank(dom[tr],Y6[tr],ALL6)
            Xtr2=np.hstack([Xtr,apply_enc(dom[tr],tab,prior)]); Xte2=np.hstack([Xte,apply_enc(dom[te],tab,prior)])
            md=rf().fit(Xtr2,ycodes[tr]); predd[te]=classes[md.predict(Xte2)]
        truth=df["best_classifier"].astype(str).to_numpy()
        for d in set(dom):
            mask=(dom==d)
            perdom_b[d].append(accuracy_score(truth[mask],predb[mask]))
            perdom_d[d].append(accuracy_score(truth[mask],predd[mask]))
        # regret (so complete-6)
        def regret(pred):
            idx={a:i for i,a in enumerate(ALL6)}
            r=[Y6[i].max()-Y6[i,idx[pred[i]]] for i in range(len(df)) if complete[i]]
            return np.mean(r)
        reg_b.append(regret(predb)); reg_d.append(regret(predd))

    print("[#1] Ganho de ACURACIA por dominio (media sobre sementes):")
    print(f"  {'dominio':14s} {'n':>4s} {'base':>7s} {'+dom':>7s} {'delta':>8s}")
    for d in sorted(set(dom), key=lambda x:-(np.mean(perdom_d[x])-np.mean(perdom_b[x]))):
        n=int((dom==d).sum()); b=np.mean(perdom_b[d]); dd=np.mean(perdom_d[d])
        print(f"  {d:14s} {n:>4d} {b:>7.3f} {dd:>7.3f} {dd-b:>+8.4f}")
        rows.append({"teste":"#1 por dominio","metrica":d,"base":b,"dom":dd,"delta":dd-b,"p":np.nan})

    _,prg=stats.ttest_rel(reg_d,reg_b)
    print(f"\n[#3] REGRET (menor=melhor, complete-6): baseline={np.mean(reg_b):.4f} -> +dom={np.mean(reg_d):.4f} | d={np.mean(reg_d)-np.mean(reg_b):+.4f} | p={prg:.3f} | reduz em {-(np.mean(reg_d)-np.mean(reg_b))/np.mean(reg_b)*100:.1f}%")
    rows.append({"teste":"#3 regret","metrica":"regret","base":np.mean(reg_b),"dom":np.mean(reg_d),"delta":np.mean(reg_d)-np.mean(reg_b),"p":prg})

    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"tests_extra_summary.csv",index=False)
    print("\nSalvo: tests_extra_summary.csv")

if __name__=="__main__":main()
