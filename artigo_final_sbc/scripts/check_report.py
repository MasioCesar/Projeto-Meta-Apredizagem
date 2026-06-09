import re, os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
s = open("relatorio_completo.tex", encoding="utf-8").read()
from collections import Counter
b = Counter(re.findall(r"\\begin\{([A-Za-z]+\*?)\}", s))
e = Counter(re.findall(r"\\end\{([A-Za-z]+\*?)\}", s))
bad = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
print("Ambientes desbalanceados:", bad or "nenhum")
figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", s)
for f in figs:
    print(("OK   " if os.path.exists(f) else "FALTA"), f)
print("Figuras:", len(figs), "| Tabelas:", len(re.findall(r"\\begin\{tabular\}", s)))
