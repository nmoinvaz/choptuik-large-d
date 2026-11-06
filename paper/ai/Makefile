# Build the paper on how the computation was carried out with Claude. It needs tectonic, which fetches revtex4-2, TikZ
# and pgfplots on the first run (tectonic -X compile main.tex && cp main.pdf claude_workflow.pdf), or pdflatex with
# revtex4-2 (texlive-publishers) and pgfplots (make pdflatex). The figures are drawn in TikZ inside main.tex, and every
# number in the text is listed with its source in SOURCES.md.
TECTONIC ?= tectonic

claude_workflow.pdf: main.tex
	$(TECTONIC) -X compile main.tex
	cp main.pdf claude_workflow.pdf

pdflatex: main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf claude_workflow.pdf

clean:
	rm -f main.aux main.log main.out main.pdf main.toc mainNotes.bib

.PHONY: pdflatex clean
