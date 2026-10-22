# Build both papers: needs pdflatex with revtex4-2 (texlive-publishers) and amsart. The figures in figures/ are PDF
# conversions of the repo's SVG figures, shared by both papers; rebuild them after changing an SVG with make figures.
all:
	$(MAKE) -C physics
	$(MAKE) -C math

figures:
	for f in solver/delta solver/echo solver/gamma hp/heldout solver/physics solver/selection solver/spectrum; do \
	  python3 -c "import cairosvg; cairosvg.svg2pdf(url='../$$f.svg', write_to='figures/'+'$$f'.split('/')[1]+'.pdf')"; done

clean:
	$(MAKE) -C physics clean
	$(MAKE) -C math clean

.PHONY: all figures clean
