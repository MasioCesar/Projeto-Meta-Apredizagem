"""Busca descricao + nomes de colunas dos 547 datasets da base V2.
Usado para: (1) dominio limpo via keywords na DESCRICAO; (2) agrupamento por
similaridade de colunas (CV agrupada). Resumivel. Saida: data/v2_descriptions.json"""
import json, time
from pathlib import Path
import openml
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
META = DATA / "metafeatures_v2.csv"
OUT = DATA / "v2_descriptions.json"


def main():
    dids = pd.read_csv(META)["did"].dropna().astype(int).tolist()
    cache = {}
    if OUT.exists():
        cache = {r["did"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
    pending = [d for d in dids if d not in cache]
    print(f"V2: {len(dids)} | em cache: {len(cache)} | pendentes: {len(pending)}", flush=True)

    out = list(cache.values())
    for i, did in enumerate(pending, 1):
        desc, fnames = "", ""
        try:
            ds = openml.datasets.get_dataset(did, download_data=False,
                                             download_qualities=False, download_features_meta_data=True)
            desc = (ds.description or "").strip()[:2000]
            try:
                fnames = ", ".join(list({f.name for f in ds.features.values()})[:60])
            except Exception:
                fnames = ""
        except Exception as e:
            print(f"[{i}/{len(pending)}] did={did} FALHA: {str(e)[:40]}", flush=True)
        out.append({"did": did, "description": desc, "feature_names": fnames})
        if i % 25 == 0:
            OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            print(f"[{i}/{len(pending)}] ok", flush=True)
        time.sleep(0.03)

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    nonempty = sum(1 for r in out if len(r.get("description", "")) > 20)
    print(f"\nSalvo: {OUT} | descricoes uteis: {nonempty}/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
