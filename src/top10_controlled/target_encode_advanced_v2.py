"""
IDEIAS #1 e #2 para FORTALECER o sinal de contexto (base V2, 6 algoritmos).

#1 Target encoding de CLUSTERS SEMANTICOS aprendidos (embeddings da descricao ->
   KMeans), em vez do dominio por keyword. "Dominio aprendido dos dados".
#2 Codificacao mais RICA: alem de P(algoritmo e o melhor | categoria), tambem
   desempenho medio e rank medio de cada algoritmo por categoria.

Protocolo rigoroso: loop de CV manual, codificacao ajustada SO no treino (sem
vazamento); CV AGRUPADA por similaridade de colunas; N sementes; teste pareado.
Metricas: F1-macro e top-3 accuracy. Compara tudo contra baseline (so estatistica)
e contra o atual campeao (dominio TE-best).
Saida: data/top10_controlled/target_encode_advanced_summary.csv
"""
import warnings, json, re, unicodedata
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parents[2] / "data"
ALL6 = ["DecisionTree","KNN","LogisticRegression","MLP","Perceptron","SVM"]
SEEDS = [42,7,123,2024,11,99,7777,2025,1,13,500,808,314,271,1000,
         3,17,55,128,256,640,911,1234,2222,4096,5,88,777,9001,12345]  # 30
EMB_CACHE = DATA / "v2_desc_embeddings.npy"
N_CLUSTERS = 12
SMOOTH = 5.0

DOMKW = {"health":["health","medical","clinical","patient","disease","cancer","heart","thyroid","cleveland","cholesterol","dermatolog","obesity","ilpd","diabet"],"finance":["financ","credit","bank","loan","fraud","stock","insurance","forex","currency","betting"],"biology":["gene","genom","protein","dna","rna","microarray","molecul","yeast","ecoli","qsar","mushroom"],"image":["image","pixel","mnist","svhn","digit","face","ocr","letter","mfeat","texture"],"text":["text","document","news","review","sentiment","spam","nlp","authorship","lyrics"],"sensor_signal":["sensor","signal","seismic","robot","fault","electric","grid","vowel","water","wind"],"education":["student","school","education","exam","grade","univers","dropout"],"social":["social","census","survey","employee","churn","customer","compas","income","adult","baseball"]}
MODKW = {"image":["image","pixel","mnist","svhn","digit","face","ocr","mfeat","texture","cifar"],"text":["text","document","review","sentiment","nlp","corpus","authorship","lyrics","news"],"signal":["sensor","signal","seismic","vibration","accelerom","vowel","speech","audio","wave","eeg","ecg"],"highdim_bio":["gene","genom","microarray","protein","dna","rna","probe","expression"]}


def norm(t):
    t=unicodedata.normalize("NFKD",str(t).lower());return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ","".join(c for c in t if not unicodedata.combining(c)))).strip()
def kw_assign(nm,ds,table,default="other"):
    b=" "+norm(nm)+" | "+norm(ds)+" ";best,bn=default,0
    for d,ks in table.items():
        n=sum(1 for k in ks if k in b)
        if n>bn:best,bn=d,n
    return best
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


# ---- codificadores (ajustados no treino) ----
def enc_best(cats_tr, y_tr, classes):
    prior = pd.Series(y_tr).value_counts(normalize=True).reindex(classes, fill_value=0).values
    tab={}
    d=pd.DataFrame({"c":cats_tr,"y":y_tr})
    for c,g in d.groupby("c"):
        vc=g["y"].value_counts(normalize=True).reindex(classes,fill_value=0).values
        n=len(g); tab[c]=(n*vc+SMOOTH*prior)/(n+SMOOTH)
    return tab, prior
def enc_perf(cats_tr, Y_tr, kind="mean"):
    prior=np.nanmean(Y_tr,axis=0)
    tab={}
    d=pd.DataFrame(Y_tr,columns=ALL6); d["c"]=cats_tr
    for c,g in d.groupby("c"):
        if kind=="mean": v=np.nanmean(g[ALL6].to_numpy(float),axis=0)
        else:  # rank medio (maior=melhor)
            r=pd.DataFrame(g[ALL6]).rank(axis=1).to_numpy(float); v=np.nanmean(r,axis=0)
        n=len(g); tab[c]=(n*v+SMOOTH*prior)/(n+SMOOTH)
    return tab, prior
def apply_enc(cats, tab, prior):
    return np.array([tab.get(c, prior) for c in cats])


def main():
    meta=pd.read_csv(DATA/"metafeatures_v2.csv");perf=pd.read_csv(DATA/"performance_matrix_v2.csv")
    txt={r["did"]:r for r in json.loads((DATA/"v2_descriptions.json").read_text(encoding="utf-8"))}
    df=meta.merge(perf[["did","best_classifier"]+[a for a in ALL6 if a in perf.columns]],on="did",how="inner").dropna(subset=["best_classifier"]).reset_index(drop=True)
    df["dom"]=[kw_assign(nm,txt.get(int(d),{}).get("description",""),DOMKW) for d,nm in zip(df["did"],df["name"])]
    df["mod"]=[kw_assign(nm,txt.get(int(d),{}).get("description",""),MODKW,"tabular") for d,nm in zip(df["did"],df["name"])]
    g=df["did"].astype(int).map(jacc({int(d):set(norm(txt.get(int(d),{}).get("feature_names","")).split())-{""} for d in df["did"]})).values

    # embeddings + cluster (NAO supervisionado: usa so descricao -> sem vazamento de alvo)
    descs=[norm(txt.get(int(d),{}).get("description","")) for d in df["did"]]
    if EMB_CACHE.exists():
        emb=np.load(EMB_CACHE)
    else:
        from sentence_transformers import SentenceTransformer
        emb=np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(descs, normalize_embeddings=True),dtype=float)
        np.save(EMB_CACHE, emb)
    df["clu"]=KMeans(n_clusters=N_CLUSTERS,random_state=42,n_init=10).fit_predict(emb).astype(str)
    # faixa de tamanho (escala): bucket de instancias x bucket de atributos
    qi=pd.qcut(df["nr_inst"],4,labels=False,duplicates="drop").astype("Int64").astype(str)
    qa=pd.qcut(df["nr_attr"],3,labels=False,duplicates="drop").astype("Int64").astype(str)
    df["scale"]=(qi+"_"+qa).values

    num=[c for c in meta.columns if c not in("did","name") and pd.api.types.is_numeric_dtype(meta[c])]
    num=[c for c in num if df[c].nunique(dropna=True)>1]
    Xs=df[num].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    y=LabelEncoder().fit_transform(df["best_classifier"].astype(str)); classes=np.unique(y)
    Y6=df[[a for a in ALL6 if a in df.columns]].to_numpy(float)
    print(f"Base V2: {len(df)} datasets | {N_CLUSTERS} clusters | {len(SEEDS)} sementes | CV agrupada\n")

    # cada config = lista de (coluna_categoria, tipo_encoding)
    CONFIGS = {
        "baseline (stats)":                    [],
        "+ dominio (rank)":                    [("dom","rank")],
        "+ modalidade (rank)":                 [("mod","rank")],
        "+ tamanho/escala (rank)":             [("scale","rank")],
        "+ dom + mod (rank)":                  [("dom","rank"),("mod","rank")],
        "+ dom + mod + escala (#3, rank)":     [("dom","rank"),("mod","rank"),("scale","rank")],
    }

    def build_feats(spec, tr_idx, te_idx):
        parts_tr=[Xs[tr_idx]]; parts_te=[Xs[te_idx]]
        for col,kind in spec:
            cats=df[col].to_numpy()
            if kind=="best":
                tab,prior=enc_best(cats[tr_idx], y[tr_idx], classes)
            else:
                tab,prior=enc_perf(cats[tr_idx], Y6[tr_idx], "mean" if kind=="perf" else "rank")
            parts_tr.append(apply_enc(cats[tr_idx],tab,prior)); parts_te.append(apply_enc(cats[te_idx],tab,prior))
        return np.hstack(parts_tr), np.hstack(parts_te)

    def top3(proba): return np.mean([int(y_te[i] in set(classes[np.argsort(proba[i])[-3:]])) for i in range(len(y_te))])

    results={k:{"f1":[],"t3":[]} for k in CONFIGS}
    for s in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=s)
        for k,spec in CONFIGS.items():
            f1s=[]; t3s=[]
            for tr,te in cv.split(Xs,y,groups=g):
                Xtr,Xte=build_feats(spec,tr,te)
                imp=SimpleImputer(strategy="median").fit(Xtr); Xtr=imp.transform(Xtr); Xte=imp.transform(Xte)
                clf=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced").fit(Xtr,y[tr])
                y_te=y[te]; pred=clf.predict(Xte); proba=clf.predict_proba(Xte)
                f1s.append(f1_score(y_te,pred,average="macro"))
                t3s.append(np.mean([int(y_te[i] in set(classes[np.argsort(proba[i])[-3:]])) for i in range(len(y_te))]))
            results[k]["f1"].append(np.mean(f1s)); results[k]["t3"].append(np.mean(t3s))

    bf=np.array(results["baseline (stats)"]["f1"]); bt=np.array(results["baseline (stats)"]["t3"])
    print(f"{'configuracao':32s} {'F1':>7s} {'dF1':>8s} {'p':>6s} {'+/n':>6s} {'top3':>7s} {'dtop3':>8s}")
    rows=[]
    for k in CONFIGS:
        f=np.array(results[k]["f1"]); t=np.array(results[k]["t3"])
        if k=="baseline (stats)":
            print(f"{k:32s} {f.mean():>7.4f} {'-':>8s} {'-':>6s} {'-':>6s} {t.mean():>7.4f} {'-':>8s}")
        else:
            _,p=stats.ttest_rel(f,bf); npos=int((f-bf>0).sum())
            print(f"{k:32s} {f.mean():>7.4f} {f.mean()-bf.mean():>+8.4f} {p:>6.3f} {npos:>3d}/{len(SEEDS)} {t.mean():>7.4f} {t.mean()-bt.mean():>+8.4f}")
        rows.append({"config":k,"f1":f.mean(),"f1_delta":f.mean()-bf.mean(),"top3":t.mean(),"top3_delta":t.mean()-bt.mean()})
    pd.DataFrame(rows).to_csv(DATA/"top10_controlled"/"target_encode_advanced_summary.csv",index=False)
    print("\nSalvo: target_encode_advanced_summary.csv")


if __name__=="__main__":main()
