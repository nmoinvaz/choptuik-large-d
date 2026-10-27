# Build both papers: needs pdflatex with revtex4-2 (texlive-publishers) and amsart, or tectonic (see physics/Makefile).
# The figures in figures/ are PDF conversions of print layouts of the repo's SVG figures (each plot script's --print:
# about 620 px wide, 480 px for selection, so their 10-11 px text sets at 7.5-8 pt at the text width), shared by both
# papers, their text set in FIG_FONT (svg2pdf.py). After changing a figure: make print-svgs, then make figures.
# On macOS cairosvg finds Homebrew's cairo through DYLD_FALLBACK_LIBRARY_PATH, set on the python command itself
# (macOS drops DYLD_ variables passed through make).
PYTHON ?= python3
FIG_FONT ?= Arial Unicode MS

all:
	$(MAKE) -C physics
	$(MAKE) -C math

print-svgs:
	cd ../solver/data && $(PYTHON) ../plot_delta.py --print --dd_max 61.03
	cd ../solver && $(PYTHON) plot_echo.py --print && $(PYTHON) plot_gamma.py --print
	cd ../solver && $(PYTHON) plot_selection.py --print && $(PYTHON) plot_spectrum.py --print
	cd ../hp && $(PYTHON) plot_heldout.py --print --dmax=60.1

figures:
	for f in solver/delta_print solver/echo_print solver/gamma_print hp/heldout_print solver/selection_print \
	         solver/spectrum_print solver/physics; do \
	  n=$${f#*/}; DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib $(PYTHON) svg2pdf.py ../$$f.svg figures/$${n%_print}.pdf \
	    --font "$(FIG_FONT)"; done

clean:
	$(MAKE) -C physics clean
	$(MAKE) -C math clean

.PHONY: all print-svgs figures clean
