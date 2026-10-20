# Build the paper: needs pdflatex with revtex4-2 (texlive-publishers). Figures in figures/ are PDF conversions of the
# repo's SVG figures (python3 -c "import cairosvg; cairosvg.svg2pdf(url='../solver/echo.svg', write_to='figures/echo.pdf')").
dss_large_D.pdf: main.tex figures/*.pdf
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	cp main.pdf dss_large_D.pdf

clean:
	rm -f main.aux main.log main.out main.pdf mainNotes.bib build*.log
