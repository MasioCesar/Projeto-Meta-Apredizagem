"""Robustez do unico sinal borderline: dominio target-encoded sob CV agrupada,
8 sementes, F1 + top-3, t-test + Wilcoxon. Decide se +1.7pp e real ou ruido."""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys; sys.path.insert(0, str(Path(__file__).parent))
from target_encode_domain_v2 import assign, jacc, norm, DomainTargetEncoder

DATA=Path(__file__).resolve().parents[2]/"data"
SEEDS=[42,7,123,2024,11,99,7777,2025,1,13,500,808,314,271,1000,
       3,17,55,128,256,640,911,1234,2222,4096,5,88,777,9001,12345]

def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[assign(nm,txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values
    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1];df[num]=df[num].replace([np.inf,-np.inf],np.nan)
    y=LabelEncoder().fit_transform(df["best_classifier"].astype(str));classes=np.unique(y)
    stat=("s",Pipeline([("i",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num)
    def mdl(te):
        t=[stat]
        if te:t.append(("te",DomainTargetEncoder(),["dom"]))
        return Pipeline([("p",ColumnTransformer(t,remainder="drop")),("c",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
    def top3(pr): return np.mean([int(y[i] in set(classes[np.argsort(pr[i])[-3:]])) for i in range(len(y))])
    print(f"Dominio target-encoded | CV AGRUPADA | {len(SEEDS)} sementes\n")
    fb,ft,tb,tt=[],[],[],[]
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        fb.append(cross_val_score(mdl(False),df,y,cv=cv,groups=g,scoring="f1_macro").mean())
        ft.append(cross_val_score(mdl(True),df,y,cv=cv,groups=g,scoring="f1_macro").mean())
        tb.append(top3(cross_val_predict(mdl(False),df,y,cv=cv,groups=g,method="predict_proba")))
        tt.append(top3(cross_val_predict(mdl(True),df,y,cv=cv,groups=g,method="predict_proba")))
    fb,ft,tb,tt=map(np.array,(fb,ft,tb,tt))
    df1=ft-fb; dt=tt-tb
    _,pf_t=stats.ttest_rel(ft,fb); _,pf_w=stats.wilcoxon(ft,fb)
    _,pt_t=stats.ttest_rel(tt,tb)
    print(f"F1-macro : base={fb.mean():.4f} +TE={ft.mean():.4f} | d={df1.mean():+.4f} | t-p={pf_t:.3f} wil-p={pf_w:.3f} | +em {(df1>0).sum()}/{len(SEEDS)}")
    print(f"TOP-3 acc: base={tb.mean():.4f} +TE={tt.mean():.4f} | d={dt.mean():+.4f} | t-p={pt_t:.3f} | +em {(dt>0).sum()}/{len(SEEDS)}")
    rep = (df1>0).all() and pf_t<0.05
    print(f"\nVEREDITO: ganho de F1 {'REAL (replica nas 8, p<0.05)' if rep else 'NAO confirmado'} | sempre positivo? {'SIM' if (df1>0).all() else 'NAO'}")
    pd.DataFrame({"seed":SEEDS,"f1_base":fb,"f1_te":ft,"top3_base":tb,"top3_te":tt}).to_csv(DATA/"top10_controlled"/"target_encode_robust_summary.csv",index=False)

if __name__=="__main__":main()
