"""
Fortalecimento do achado #1: o ganho do dominio, medido apenas em datasets de
DOMINIO REAL (excluindo 'other' sintetico) e por dominio com significancia.
30 sementes, CV agrupada, codificacao rank. Saida: real_domain_subset_summary.csv
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
import sys; sys.path.insert(0,str(Path(__file__).parent))
from tests_extra_domain import load, enc_rank, apply_enc, ALL6, rf

DATA=Path(__file__).resolve().parents[2]/"data"
SEEDS=[42,7,123,2024,11,99,7777,2025,1,13,500,808,314,271,1000,
       3,17,55,128,256,640,911,1234,2222,4096,5,88,777,9001,12345]

def main():
    df,Xs,Y6,dom,g=load()
    le=LabelEncoder().fit(df["best_classifier"].astype(str)); yc=le.transform(df["best_classifier"].astype(str)); classes=le.classes_
    truth=df["best_classifier"].astype(str).to_numpy()
    real=(dom!="other"); fin=(dom=="finance")
    print(f"Datasets: {len(df)} | dominio real: {real.sum()} | other: {(~real).sum()} | finance: {fin.sum()}\n")

    # acumula metricas por semente
    acc_all_b,acc_all_d,f1_all_b,f1_all_d=[],[],[],[]
    acc_real_b,acc_real_d,f1_real_b,f1_real_d=[],[],[],[]
    acc_fin_b,acc_fin_d=[],[]
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        pb=np.empty(len(df),dtype=object); pdd=np.empty(len(df),dtype=object)
        for tr,te in cv.split(Xs,yc,groups=g):
            imp=SimpleImputer(strategy="median").fit(Xs[tr]); Xtr,Xte=imp.transform(Xs[tr]),imp.transform(Xs[te])
            pb[te]=classes[rf().fit(Xtr,yc[tr]).predict(Xte)]
            tab,prior=enc_rank(dom[tr],Y6[tr],ALL6)
            Xtr2=np.hstack([Xtr,apply_enc(dom[tr],tab,prior)]); Xte2=np.hstack([Xte,apply_enc(dom[te],tab,prior)])
            pdd[te]=classes[rf().fit(Xtr2,yc[tr]).predict(Xte2)]
        acc_all_b.append(accuracy_score(truth,pb)); acc_all_d.append(accuracy_score(truth,pdd))
        f1_all_b.append(f1_score(truth,pb,average="macro")); f1_all_d.append(f1_score(truth,pdd,average="macro"))
        acc_real_b.append(accuracy_score(truth[real],pb[real])); acc_real_d.append(accuracy_score(truth[real],pdd[real]))
        f1_real_b.append(f1_score(truth[real],pb[real],average="macro")); f1_real_d.append(f1_score(truth[real],pdd[real],average="macro"))
        acc_fin_b.append(accuracy_score(truth[fin],pb[fin])); acc_fin_d.append(accuracy_score(truth[fin],pdd[fin]))

    def rep(name,b,d):
        b,d=np.array(b),np.array(d); _,p=stats.ttest_rel(d,b)
        npos=int((d-b>0).sum())
        print(f"  {name:28s} {b.mean():.4f} -> {d.mean():.4f} | d={d.mean()-b.mean():+.4f} | p={p:.4f} | +{npos}/{len(b)}")
        return {"grupo":name,"base":b.mean(),"dom":d.mean(),"delta":d.mean()-b.mean(),"p":float(p),"pos":f"{npos}/{len(b)}"}

    print(f"30 sementes, CV agrupada, codificacao rank:")
    rows=[]
    rows.append(rep("Acuracia (TODOS)",acc_all_b,acc_all_d))
    rows.append(rep("Acuracia (DOMINIO REAL)",acc_real_b,acc_real_d))
    rows.append(rep("Acuracia (FINANCE)",acc_fin_b,acc_fin_d))
    rows.append(rep("F1 (TODOS)",f1_all_b,f1_all_d))
    rows.append(rep("F1 (DOMINIO REAL)",f1_real_b,f1_real_d))
    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"real_domain_subset_summary.csv",index=False)
    print("\nSalvo: real_domain_subset_summary.csv")

if __name__=="__main__":main()
