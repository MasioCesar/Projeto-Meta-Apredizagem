"""
#4 Dominio FINO (subdominios) via taxonomia expandida + rank encoding.
Testa se uma granularidade maior de dominio supera as 8 categorias.
30 sementes, CV agrupada, em TODOS e em DOMINIO REAL.
Saida: finer_domain_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys; sys.path.insert(0,str(Path(__file__).parent))
from tests_extra_domain import load, enc_rank, apply_enc, ALL6, rf, assign as assign8

DATA=Path(__file__).resolve().parents[2]/"data"
SEEDS=[42,7,123,2024,11,99,7777,2025,1,13,500,808,314,271,1000,
       3,17,55,128,256,640,911,1234,2222,4096,5,88,777,9001,12345]

# subdominios (granularidade fina)
SUB={
 "fin_credito":["credit","loan","creditab","default","mortgage"],
 "fin_fraude":["fraud","phishing"],
 "fin_mercado":["stock","forex","currency","trading","market","betting","bwin","valuation"],
 "fin_bancario":["bank","bankrupt","insurance"],
 "bio_geneexp":["gene","genom","microarray","expression","probe"],
 "bio_molecular":["molecul","qsar","protein","dna","rna","chemical","compound"],
 "bio_organismo":["yeast","ecoli","mushroom","soybean","species","plant"],
 "img_digitos":["mnist","svhn","digit","optdigit","pendigit","mfeat","letter"],
 "img_natural":["image","pixel","face","cifar","texture","object","vision"],
 "saude_clinico":["patient","clinical","cleveland","cholesterol","heart","thyroid","ilpd","hepatit"],
 "saude_diag":["cancer","tumor","diagnos","disease","dermatolog","mammograph","obesity","diabet"],
 "texto":["text","document","news","review","sentiment","spam","nlp","authorship","lyrics","corpus"],
 "sinal":["sensor","signal","seismic","vibration","accelerom","vowel","robot","fault","wind","water","grid","electric"],
 "educacao":["student","school","education","exam","grade","univers","dropout"],
 "social":["social","census","survey","employee","churn","customer","compas","income","adult","baseball","marketing"],
}
def norm(t):
    t=unicodedata.normalize("NFKD",str(t).lower());return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()
def assign_fine(nm,ds):
    b=" "+norm(nm)+" | "+norm(ds)+" ";best,bn="other",0
    for d,ks in SUB.items():
        n=sum(1 for k in ks if k in b)
        if n>bn:best,bn=d,n
    return best if bn else "other"

def main():
    df,Xs,Y6,dom8,g=load()
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    domf=np.array([assign_fine(nm,txt.get(int(d),{}).get("description","")) for d,nm in zip(df["did"],df["name"])])
    print(f"Subdominios distintos: {len(set(domf))} | 'other': {(domf=='other').sum()} (vs {(dom8=='other').sum()} no de 8 cat.)")
    le=LabelEncoder().fit(df["best_classifier"].astype(str)); yc=le.transform(df["best_classifier"].astype(str)); classes=le.classes_
    truth=df["best_classifier"].astype(str).to_numpy(); real8=(dom8!="other")

    def run(domvar):
        ab,ad,arb,ard=[],[],[],[]
        for s in SEEDS:
            cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
            pb=np.empty(len(df),dtype=object); pdd=np.empty(len(df),dtype=object)
            for tr,te in cv.split(Xs,yc,groups=g):
                imp=SimpleImputer(strategy="median").fit(Xs[tr]); Xtr,Xte=imp.transform(Xs[tr]),imp.transform(Xs[te])
                pb[te]=classes[rf().fit(Xtr,yc[tr]).predict(Xte)]
                tab,prior=enc_rank(domvar[tr],Y6[tr],ALL6)
                Xtr2=np.hstack([Xtr,apply_enc(domvar[tr],tab,prior)]); Xte2=np.hstack([Xte,apply_enc(domvar[te],tab,prior)])
                pdd[te]=classes[rf().fit(Xtr2,yc[tr]).predict(Xte2)]
            ab.append(accuracy_score(truth,pb)); ad.append(accuracy_score(truth,pdd))
            arb.append(accuracy_score(truth[real8],pb[real8])); ard.append(accuracy_score(truth[real8],pdd[real8]))
        return np.array(ab),np.array(ad),np.array(arb),np.array(ard)

    print("\n30 sementes, CV agrupada | acuracia:")
    rows=[]
    for nome,domvar in [("Dominio 8 categorias",dom8),("Dominio FINO (subdominios)",domf)]:
        ab,ad,arb,ard=run(domvar)
        _,p1=stats.ttest_rel(ad,ab); _,p2=stats.ttest_rel(ard,arb)
        print(f"  [{nome}]")
        print(f"    TODOS:        {ab.mean():.4f}->{ad.mean():.4f}  d={ad.mean()-ab.mean():+.4f} p={p1:.4f}")
        print(f"    DOMINIO REAL: {arb.mean():.4f}->{ard.mean():.4f}  d={ard.mean()-arb.mean():+.4f} p={p2:.4f}")
        rows.append({"taxonomia":nome,"grupo":"todos","delta":ad.mean()-ab.mean(),"p":float(p1)})
        rows.append({"taxonomia":nome,"grupo":"dominio_real","delta":ard.mean()-arb.mean(),"p":float(p2)})
    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"finer_domain_summary.csv",index=False)
    print("\nSalvo: finer_domain_summary.csv")

if __name__=="__main__":main()
