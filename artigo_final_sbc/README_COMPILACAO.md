# Como compilar o artigo

O arquivo principal e:

```text
artigo_sbc.tex
```

Foi escrito no formato LaTeX usado pelo template SBC:

```latex
\documentclass[12pt]{article}
\usepackage{sbc-template}
```

Inclui um `sbc-template.sty` minimo para compilacao local. Se voce tiver o
arquivo oficial da SBC, pode substituir este `sbc-template.sty` pelo oficial.

Com LaTeX instalado, compile com:

```powershell
pdflatex artigo_sbc.tex
bibtex artigo_sbc
pdflatex artigo_sbc.tex
pdflatex artigo_sbc.tex
```

Os graficos ficam em:

```text
figuras/
```

Para recriar os graficos:

```powershell
venv\Scripts\python.exe artigo_final_sbc\scripts\gerar_figuras.py
```

