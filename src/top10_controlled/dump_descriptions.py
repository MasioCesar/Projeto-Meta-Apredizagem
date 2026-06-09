"""Baixa e salva nome + descricao + nomes de features dos 116 datasets,
para anotacao de dominio por LLM (Claude). Saida: data/dataset_descriptions.json"""
import json
import time
from pathlib import Path

import openml
import pandas as pd

from common import INPUT_METAFEATURES

OUT = Path(INPUT_METAFEATURES).parent / "dataset_descriptions.json"
MAX_DESC = 1500  # trunca para manter legivel


def main():
    meta = pd.read_csv(INPUT_METAFEATURES)
    info = meta.set_index("did")[["name", "nr_attr", "nr_inst", "nr_class"]].to_dict("index")
    dids = meta["did"].dropna().astype(int).tolist()
    print(f"Baixando descricoes de {len(dids)} datasets...")

    out = []
    for i, did in enumerate(dids, 1):
        desc, fnames = "", ""
        try:
            ds = openml.datasets.get_dataset(did, download_data=False,
                                             download_qualities=False,
                                             download_features_meta_data=True)
            desc = (ds.description or "").strip()
            try:
                fnames = ", ".join(list(ds.features and {f.name for f in ds.features.values()} or [])[:40])
            except Exception:
                fnames = ""
        except Exception as e:
            print(f"  [{i}] did={did} FALHA: {str(e)[:50]}")
        meta_row = info.get(did, {})
        out.append({
            "did": did,
            "name": meta_row.get("name", ""),
            "nr_attr": int(meta_row.get("nr_attr", 0)),
            "nr_inst": int(meta_row.get("nr_inst", 0)),
            "nr_class": int(meta_row.get("nr_class", 0)),
            "description": desc[:MAX_DESC],
            "feature_names_sample": fnames[:600],
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(dids)}] ok")
        time.sleep(0.05)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSalvo: {OUT} ({len(out)} datasets)")
    nonempty = sum(1 for r in out if len(r["description"]) > 20)
    print(f"Descricoes uteis (>20 chars): {nonempty}/{len(out)}")


if __name__ == "__main__":
    main()
