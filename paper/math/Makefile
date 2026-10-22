# Build the mathematics paper: needs pdflatex with amsart (texlive-latex-base, texlive-latex-recommended).
# The figures are shared with the other papers and live in ../figures/.
dss_math.pdf: main.tex ../figures/selection.pdf ../figures/spectrum.pdf
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf dss_math.pdf

clean:
	rm -f main.aux main.log main.out main.pdf main.toc

.PHONY: clean
