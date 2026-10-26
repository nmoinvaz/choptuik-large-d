# Build the physics paper: needs pdflatex with revtex4-2 (texlive-publishers). The figures are shared with the other
# papers in ../figures/ (PDF conversions of the repo's SVG figures; see ../Makefile).
dss_physics.pdf: main.tex ../figures/*.pdf
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	cp main.pdf dss_physics.pdf

clean:
	rm -f main.aux main.log main.out main.pdf main.toc mainNotes.bib build*.log
