# Build the mathematics paper: needs pdflatex with amsart (texlive-latex-base, texlive-latex-recommended), or with
# tectonic: tectonic -X compile main.tex && cp main.pdf dss_math.pdf.
# The figures are shared with the other papers and live in ../figures/.
dss_math.pdf: main.tex ../figures/selection.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf dss_math.pdf

clean:
	rm -f main.aux main.log main.out main.pdf main.toc

.PHONY: clean
