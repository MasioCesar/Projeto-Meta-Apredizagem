import re, os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
tex = open("artigo_sbc.tex", encoding="utf-8").read()
from collections import Counter
begins = Counter(re.findall(r"\\begin\{([A-Za-z]+\*?)\}", tex))
ends = Counter(re.findall(r"\\end\{([A-Za-z]+\*?)\}", tex))
bad = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}
print("Ambientes desbalanceados:", bad or "nenhum")
figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
for f in figs:
    print(("OK   " if os.path.exists(f) else "FALTA"), f)
bib = open("referencias.bib", encoding="utf-8").read()
keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
cites = set()
for c in re.findall(r"\\cite\{([^}]+)\}", tex):
    cites |= {x.strip() for x in c.split(",")}
print("Citacoes sem entrada no bib:", (cites - keys) or "nenhuma")
print("Tabelas:", len(re.findall(r"\\begin\{tabular\}", tex)), "| Figuras:", len(figs))
