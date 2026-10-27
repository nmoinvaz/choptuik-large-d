#!/usr/bin/env python3
"""An SVG figure to PDF with cairosvg, its text set in one font that has every glyph the figures use (Greek, ∞, →,
super- and subscripts). cairosvg takes one font per text element with no per-glyph fallback, so the figures' generic
"system-ui, sans-serif" can resolve to a font without, say, the arrow, which then prints as a box.

    python3 svg2pdf.py IN.svg OUT.pdf [--font "Arial Unicode MS"]
"""
import argparse
import re

import cairosvg

ap = argparse.ArgumentParser()
ap.add_argument("svg")
ap.add_argument("pdf")
ap.add_argument("--font", default="Arial Unicode MS")
a = ap.parse_args()
text = re.sub(r'font-family="[^"]*"', f'font-family="{a.font}"', open(a.svg, encoding="utf-8").read())
cairosvg.svg2pdf(bytestring=text.encode("utf-8"), write_to=a.pdf)
