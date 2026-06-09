"""
Anotacao de dominio por LLM (Claude), lendo as descricoes isoladas dos datasets.
NAO e keyword-matching: e atribuicao de CATEGORIA por entendimento semantico.
  - llm_domain: 8 categorias (health, finance, biology, image, text,
                sensor_signal, education, social)
  - llm_confidence: 1-5, quao inequivoco e o dominio (NAO conta palavras)
  - llm_modality: modalidade do dado

Familia AP_*/OVA_* = microarray de expressao genica (GEMLeR) -> biology, conf 5.
Saida: data/llm_domain_annotations.csv
"""
import json
from pathlib import Path
from common import INPUT_METAFEATURES

JSON_IN = Path(INPUT_METAFEATURES).parent / "dataset_descriptions.json"
OUT = Path(INPUT_METAFEATURES).parent / "llm_domain_annotations.csv"

# (domain, confidence, modality) anotado manualmente por Claude lendo a descricao
ANN = {
    46: ("biology", 5, "sequence"), 40670: ("biology", 5, "sequence"),
    1494: ("biology", 5, "molecular"), 46827: ("biology", 5, "molecular"),
    4134: ("biology", 5, "molecular"), 1412: ("biology", 4, "gene_expression"),
    46850: ("health", 5, "clinical"), 151: ("sensor_signal", 4, "signal"),
    316: ("biology", 4, "tabular_bio"), 1011: ("biology", 5, "tabular_bio"),
    46874: ("health", 3, "survey"), 1508: ("education", 5, "tabular"),
    46960: ("education", 5, "tabular"), 43255: ("education", 5, "tabular"),
    43890: ("education", 4, "tabular"), 46940: ("social", 5, "tabular"),
    43895: ("social", 4, "tabular"), 1461: ("finance", 5, "tabular"),
    46526: ("finance", 5, "tabular"), 42477: ("finance", 5, "tabular"),
    46372: ("finance", 5, "tabular"), 46382: ("finance", 5, "tabular"),
    46455: ("finance", 5, "tabular"), 31: ("finance", 5, "tabular"),
    44097: ("finance", 5, "tabular"), 44098: ("finance", 5, "tabular"),
    46962: ("finance", 5, "tabular"), 1495: ("finance", 5, "tabular"),
    49: ("health", 5, "clinical"), 15: ("health", 5, "clinical"),
    13: ("health", 5, "clinical"), 45557: ("health", 5, "clinical"),
    46846: ("health", 4, "clinical"), 40497: ("health", 5, "clinical"),
    35: ("health", 5, "clinical"), 481: ("health", 4, "clinical"),
    44149: ("health", 5, "clinical"), 53: ("health", 5, "clinical"),
    40475: ("health", 5, "clinical"), 40476: ("health", 5, "clinical"),
    40477: ("health", 5, "clinical"), 40478: ("health", 5, "clinical"),
    46879: ("health", 5, "survey"), 1464: ("health", 4, "tabular"),
    36: ("image", 5, "image_pixels"), 182: ("image", 5, "image_pixels"),
    6: ("image", 5, "image_pixels"), 32: ("image", 5, "image_pixels"),
    28: ("image", 5, "image_pixels"), 40499: ("image", 5, "image_pixels"),
    20: ("image", 5, "image_pixels"), 1459: ("image", 4, "image_shape"),
    40926: ("image", 5, "image_pixels"), 1462: ("image", 4, "image_derived"),
    40496: ("image", 3, "synthetic_digit"), 44698: ("image", 5, "image_pixels"),
    1478: ("sensor_signal", 5, "signal"), 1497: ("sensor_signal", 5, "signal"),
    40: ("sensor_signal", 5, "signal"), 1504: ("sensor_signal", 4, "signal"),
    45562: ("sensor_signal", 5, "signal"), 1476: ("sensor_signal", 5, "signal"),
    4538: ("sensor_signal", 5, "signal"), 46280: ("social", 5, "tabular"),
    1590: ("social", 5, "tabular"), 40701: ("social", 5, "tabular"),
    46667: ("text", 4, "text"), 1119: ("social", 5, "tabular"),
    46938: ("finance", 4, "tabular"), 46281: ("finance", 4, "tabular"),
    46371: ("social", 4, "tabular"), 46911: ("social", 4, "tabular"),
    44232: ("social", 5, "tabular"), 45545: ("social", 4, "tabular"),
    373: ("social", 3, "sequence"), 1457: ("text", 5, "text"),
    46806: ("text", 5, "text"), 46937: ("social", 5, "tabular"),
    46941: ("health", 5, "clinical"),
    # adicionados (anotados pelo nome, dominio inequivoco)
    46752: ("education", 4, "tabular"), 43098: ("education", 5, "tabular"),
    45748: ("education", 5, "tabular"), 46745: ("sensor_signal", 4, "signal"),
    46502: ("finance", 5, "tabular"), 43595: ("finance", 5, "tabular"),
    46605: ("health", 5, "clinical"), 37: ("health", 5, "clinical"),
    46601: ("health", 5, "clinical"), 46733: ("health", 5, "survey"),
    46600: ("health", 5, "clinical"), 46602: ("health", 5, "clinical"),
    42178: ("social", 5, "tabular"), 44226: ("social", 4, "tabular"),
    46652: ("text", 3, "text"),
}


def main():
    data = json.loads(JSON_IN.read_text(encoding="utf-8"))
    rows = []
    missing = []
    for r in data:
        did, name = r["did"], str(r["name"])
        if did in ANN:
            dom, conf, mod = ANN[did]
        elif name.startswith("AP_") or name.startswith("OVA_"):
            dom, conf, mod = "biology", 5, "gene_expression"
        else:
            missing.append((did, name))
            dom, conf, mod = "unknown", 1, "unknown"
        rows.append({"did": did, "name": name, "llm_domain": dom,
                     "llm_confidence": conf, "llm_modality": mod})

    import csv
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["did", "name", "llm_domain", "llm_confidence", "llm_modality"])
        w.writeheader(); w.writerows(rows)

    print(f"Salvo: {OUT} ({len(rows)} datasets)")
    if missing:
        print(f"\nATENCAO: {len(missing)} dids sem anotacao explicita (precisam ser adicionados):")
        for did, name in missing:
            print(f"  did={did} name={name}")
    else:
        print("Todos os datasets anotados.")
    # distribuicao
    from collections import Counter
    print("\nDistribuicao de llm_domain:")
    for d, c in Counter(r["llm_domain"] for r in rows).most_common():
        print(f"  {d:14s} {c}")


if __name__ == "__main__":
    main()
