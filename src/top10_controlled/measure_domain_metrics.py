"""Mede F1-macro, ACURACIA (top-1) e top-3 accuracy para baseline vs dominio
(rank encoding), 30 sementes, CV agrupada. Responde: o +0.69pp e so de F1 ou
tambem de acuracia?"""
import warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import sys; sys.path.insert(0, str(Path(__file__).parent))
from target_encode_advanced_v2 import norm, jacc, kw_assign, enc_perf, apply_enc, DOMKW, ALL6

DATA=Path(__file__).resolve().parents[2]/"data"
SEEDS=[42,7,123,2024,11,99,7777,2025,1,13,500,808,314,271,1000,
       3,17,55,128,256,640,911,1234,2222,4096,5,88,777,9001,12345]

def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]+[a for a in ALL6 if a in perf.columns]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[kw_assign(nm,txt.get(int(d),{}).get("description",""),DOMKW) for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1]
    Xs=df[num].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    y=LabelEncoder().fit_transform(df["best_classifier"].astype(str)); classes=np.unique(y)
    Y6=df[[a for a in ALL6 if a in df.columns]].to_numpy(float)
    dom=df["dom"].to_numpy()

    res={k:{"f1":[],"acc":[],"t3":[]} for k in ["baseline","+dominio(rank)"]}
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        for name,use_dom in [("baseline",False),("+dominio(rank)",True)]:
            f1f,accf,t3f=[],[],[]
            for tr,te in cv.split(Xs,y,groups=g):
                Xtr,Xte=Xs[tr],Xs[te]
                if use_dom:
                    tab,prior=enc_perf(dom[tr],Y6[tr],"rank")
                    Xtr=np.hstack([Xtr,apply_enc(dom[tr],tab,prior)]); Xte=np.hstack([Xte,apply_enc(dom[te],tab,prior)])
                imp=SimpleImputer(strategy="median").fit(Xtr); Xtr=imp.transform(Xtr); Xte=imp.transform(Xte)
                clf=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced").fit(Xtr,y[tr])
                yte=y[te]; pred=clf.predict(Xte); proba=clf.predict_proba(Xte)
                f1f.append(f1_score(yte,pred,average="macro")); accf.append(accuracy_score(yte,pred))
                t3f.append(np.mean([int(yte[i] in set(classes[np.argsort(proba[i])[-3:]])) for i in range(len(yte))]))
            res[name]["f1"].append(np.mean(f1f)); res[name]["acc"].append(np.mean(accf)); res[name]["t3"].append(np.mean(t3f))

    print(f"30 sementes, CV agrupada | dominio codificado por rank medio\n")
    print(f"{'metrica':14s} {'baseline':>9s} {'+dominio':>9s} {'delta':>8s} {'p':>7s} {'+/30':>6s}")
    for m,lab in [("f1","F1-macro"),("acc","Acuracia"),("t3","Top-3 acc")]:
        b=np.array(res["baseline"][m]); d=np.array(res["+dominio(rank)"][m])
        _,p=stats.ttest_rel(d,b)
        print(f"{lab:14s} {b.mean():>9.4f} {d.mean():>9.4f} {d.mean()-b.mean():>+8.4f} {p:>7.3f} {int((d-b>0).sum()):>3d}/30")

if __name__=="__main__":main()
