#!/usr/bin/env python3
"""The published gamma and Delta below D = 14 that the 2025 review cites, against ours, in
data/gamma_published_compare.csv (#620; notes/REVIEW_GHMG25.md, section 8).

    cd hp && python3 compare_gamma_published.py                # writes the CSV
    cd hp && python3 compare_gamma_published.py --verify DIR   # every quote against the papers' arXiv sources

Each value is copied from the paper's arXiv source (the version in REFS), never from the review, with a locator and a
verbatim quote of the source line that holds it (whitespace runs collapsed). `--verify DIR` reads the unpacked sources
from DIR/<arXiv id with "/" as "_"><version>/ (for example DIR/gr-qc_9908044v1/, made by
`curl -L https://arxiv.org/src/gr-qc/9908044v1 | tar -xz -C DIR/gr-qc_9908044v1`), checks that every quote is in
them and that every published value and stated uncertainty is in a quote, and exits non-zero otherwise.

Quantities, each in its paper's convention as read there (the column convention says where):
  gamma       the length-scale exponent: a radius scales as |p - p*|^gamma, the maximum curvature as |p - p*|^(-2 gamma);
  gamma_mass  the mass exponent, (D - 3) gamma;
  Delta       the full echoing period, the period of the scalar field;
  Delta_half  half of it, the period of the metric and the curvature;
  T           Delta/(2 gamma), the period of the scaling law's wiggle in ln|p - p*|;
  T_log10     Delta/(2 ln(10) gamma), the same period in log10|p - p*|.
Ours is the same quantity from our gamma and Delta: gamma from data/delta_dd_lowD.csv (D <= 8.25, the largest M with
a gamma) or data/gamma_dd.jsonl and data/local/gamma_dd.jsonl (D >= 9, M 48), Delta from data/delta_all.csv. Our errors (below 1e-8, and unknown
for gamma(3.5), a single M) are negligible against every published uncertainty.

Columns: D, quantity, published_ref, conditions (family, cosmological constant, method), published_lo and
published_hi (as printed; equal unless the paper gives a range), published_unc (as printed: an absolute +-, a percent,
"range" or "none stated"), unc_abs and unc_kind (the absolute uncertainty used: the stated one, a percent of the value,
the half-width of a range, or, where none is stated, half a unit of the last printed digit, a rounding bound only),
convention, locator, ours, ours_source, diff (ours minus the published value, the midpoint of a range) and within
("yes" when |diff| <= unc_abs).
"""
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REFS = {
    "GCD1999": ("gr-qc/9908044v1", "D. Garfinkle, C. Cutler and G. C. Duncan, PRD 60, 104007 (1999)"),
    "Birukou2002": ("gr-qc/0201026v2", "M. Birukou, V. Husain, G. Kunstatter, E. Vaz and M. Olivier, PRD 65, 104036 (2002)"),
    "Husain2003": ("gr-qc/0210011v2", "V. Husain, G. Kunstatter, B. Preston and M. Birukou, CQG 20, L23 (2003)"),
    "SO2005": ("hep-th/0502034v2", "E. Sorkin and Y. Oren, PRD 71, 124005 (2005)"),
    "Bland2005": ("gr-qc/0507088v2", "J. Bland, B. Preston, M. Becker, G. Kunstatter and V. Husain, CQG 22, 5355 (2005)"),
    "BK2007": ("hep-th/0702226v1", "J. Bland and G. Kunstatter, PRD 75, 101501 (2007)"),
    "TK2011": ("1105.0878v2", "T. Taves and G. Kunstatter, PRD 84, 044034 (2011)"),
    "PVG2022": ("2209.08404v1", "B. Porto Veronese and C. Gundlach, PRD 106, 104044 (2022)"),
}

CONV = {
    ("GCD1999", "gamma"): "length: subcritical maximum curvature ~ (p* - p)^(-2 gamma) (Secs. II, V)",
    ("GCD1999", "gamma_mass"): "mass: M_BH ~ (p - p*)^((n - 3) gamma) (Sec. II), the supercritical slope",
    ("GCD1999", "Delta"): "full: period of h, linear in phi, in T = -ln(u* - u); e^Delta ~ 21 per period (Sec. V)",
    ("Birukou2002", "gamma"): "length: R_ah ~ (a - a*)^gamma, apparent-horizon radius, supercritical (Sec. III)",
    ("Husain2003", "gamma"): "length: R_ah ~ (a - a*)^gamma, apparent-horizon radius, supercritical (text before Table 1)",
    ("Husain2003", "Delta"): "full: set beside Delta = 3.44 at Lambda = 0 (text after Fig. 1)",
    ("SO2005", "gamma"): "length: |p - p*|^gamma has dimensions of length; subcritical R_max ~ (p* - p)^(-2 gamma) (Secs. I, II)",
    ("SO2005", "Delta"): "full: period of the scalar field on the axis in log(T* - T) (Fig. 3)",
    ("SO2005", "Delta_half"): "half: period of a metric function or the curvature, stated as half the scalar field's (Figs. 4, 5)",
    ("Bland2005", "gamma"): "length: R_ah = (a - a*)^gamma, supercritical (Eq. (1))",
    ("Bland2005", "Delta"): "full: chi(r,t) = chi(r e^Delta, t e^Delta) (Eq. (2)); Delta = 2 T gamma from the residuals' period T (Sec. III)",
    ("BK2007", "gamma"): "length: R_AH ~ (a - a*)^gamma (Eq. (1))",
    ("BK2007", "Delta"): "full: chi(r,t) = chi(r e^Delta, t e^Delta) (Eq. (2)); period of the field at the origin in -ln(u* - u) (Fig. 2)",
    ("TK2011", "gamma"): "length: the areal radius of the apparent horizon on formation (Sec. 1, Eq. (1.1))",
    ("TK2011", "T"): "Delta/(2 gamma): period T of f in ln|A - A*| (Eq. (1.1)); Table 1 lists Bland et al.'s and Sorkin-Oren's Delta as Delta/(2 gamma) (inferred)",
    ("PVG2022", "gamma"): "length: R_ah ~ (p - p*)^gamma and R_Psi^(-1/2) (Fig. 9)",
    ("PVG2022", "Delta_half"): "half: Delta_res = Delta_Psi/(ln(10) gamma) (Figs. 10, 11), where the scalar field's period gives Delta/(2 ln(10) gamma)",
    ("PVG2022", "T_log10"): "Delta/(2 ln(10) gamma): period of the residuals in log10|p - p*| (Fig. 10)",
}

# (ref, D, quantity, conditions, lo, hi, unc, locator, quotes). lo, hi and unc as printed; unc is "+-x", "x%",
# "range" or "none stated". The quotes are verbatim from the arXiv source, whitespace runs collapsed.
ROWS = [
    ("GCD1999", 6, "gamma", "subcritical; phi = p r^2 exp[-(r - r0)^2/sigma^2]; 300 points", "0.424", "0.424", "none stated", "Sec. V",
     [r"The result is $\gamma = 0.424$, so $M_{BH} \propto (p-p*)^{1.27}$."]),
    ("GCD1999", 6, "gamma_mass", "supercritical; same family", "1.27", "1.27", "none stated", "Sec. V",
     [r"the slope of the line is $\approx 1.27$."]),
    ("GCD1999", 6, "Delta", "near-critical evolution", "3.03", "3.03", "none stated", "Sec. V",
     [r"The period of $h$ is $\Delta = 3.03$."]),

    ("Birukou2002", 4, "gamma", "Gaussian; Lambda = 0", "0.36", "0.36", "none stated", "Sec. IV, Fig. 1",
     [r"slope of $\gamma = 0.36$, in good agreement $(\sim 4\%)$ with earlier studies"]),
    ("Birukou2002", 4, "gamma", "tanh; Lambda = 0", "0.35", "0.35", "none stated", "Sec. IV",
     [r"for the tanh initial data are $\gamma = 0.35$ and $\gamma = 0.41$ respectively."]),
    ("Birukou2002", 5, "gamma", "tanh; Lambda = 0", "0.41", "0.41", "none stated", "Sec. IV, Fig. 3",
     [r"With $\Lambda =0$, for the tanh initial data, we find a critical exponent of $\gamma=0.41$ (Figure 3)."]),
    ("Birukou2002", 5, "gamma", "Gaussian; Lambda = 0", "0.52", "0.52", "none stated", "Sec. IV, Fig. 5",
     [r"initial data yielded $\gamma = 0.52$ for $\Lambda = 0$, in comparison to $\gamma = 0.41$ for the tanh initial data (Figure 5)."]),
    ("Birukou2002", 5, "gamma", "Gaussian; Lambda = -1", "0.49", "0.49", "none stated", "Sec. IV, Fig. 4",
     [r"With $\Lambda = -1$, and Gaussian initial data, we find a critical exponent of $\gamma =0.49$ (Figure 4)."]),
    ("Birukou2002", 6, "gamma", "Gaussian; Lambda = 0", "0.44", "0.44", "none stated", "Sec. IV, Fig. 2",
     [r"Our result for the critical exponent is $\gamma = 0.44$."]),
    ("Birukou2002", 6, "gamma", "tanh; Lambda = 0", "0.41", "0.41", "none stated", "Sec. IV",
     [r"for the tanh initial data are $\gamma = 0.35$ and $\gamma = 0.41$ respectively."]),

    ("Husain2003", 4, "gamma", "Gaussian; Lambda = -0.001", "0.370", "0.375", "range", "Table 1", ["0.370-0.375"]),
    ("Husain2003", 4, "gamma", "Gaussian; Lambda = -5", "0.37", "0.38", "range", "Table 1", ["-5 && 0.37-0.38"]),
    ("Husain2003", 4, "gamma", "Gaussian; Lambda = -10", "0.37", "0.39", "range", "Table 1", ["-10 && 0.37-0.39"]),
    ("Husain2003", 4, "gamma", "Gaussian; Lambda = -20", "0.36", "0.38", "range", "Table 1", ["-20 && 0.36-0.38"]),
    ("Husain2003", 4, "gamma", "cosh; Lambda = -5", "0.37", "0.38", "range", "Table 2", [r"-5 && $\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $0.37-0.38"]),
    ("Husain2003", 4, "gamma", "cosh; Lambda = -20", "0.36", "0.38", "range", "Table 2", ["-20 && 0.36-0.38"]),
    ("Husain2003", 4, "gamma", "cosh; Lambda = -50", "0.37", "0.40", "range", "Table 2", ["-50 && 0.37-0.40"]),
    ("Husain2003", 4, "gamma", "Lambda = -5; least-squares fit of Fig. 1", "0.3738", "0.3738", "none stated", "Fig. 1",
     [r"The line is the least squares fit to the points giving $\gamma = 0.3738$"]),
    ("Husain2003", 4, "Delta", "Lambda = -5; two oscillations of Fig. 1, a rough estimate", "3.5", "3.5", "none stated", "text after Fig. 2",
     [r"it gives $\Delta \sim 3.5$."]),

    ("SO2005", 4, "Delta", "subcritical evolutions", "3.37", "3.37", "2%", "Table I", [r"4 & $3.37 \pm 2\%$ &$0.372 \pm 1\%$"]),
    ("SO2005", 4, "gamma", "subcritical evolutions", "0.372", "0.372", "1%", "Table I", [r"4 & $3.37 \pm 2\%$ &$0.372 \pm 1\%$"]),
    ("SO2005", 5, "Delta", "subcritical evolutions", "3.19", "3.19", "2%", "Table I", [r"5 & $3.19 \pm 2\% $ &$0.408 \pm 2\%$"]),
    ("SO2005", 5, "gamma", "subcritical evolutions", "0.408", "0.408", "2%", "Table I", [r"5 & $3.19 \pm 2\% $ &$0.408 \pm 2\%$"]),
    ("SO2005", 5, "Delta_half", "the metric function alpha on the axis", "1.6", "1.6", "none stated", "Fig. 4",
     [r"oscillations whose period ($\simeq 1.6$) is half the period of the scalar field ($\simeq 3.19$)."]),
    ("SO2005", 6, "Delta", "subcritical evolutions", "3.01", "3.01", "2%", "Table I", [r"6 & $3.01 \pm 2\%$ &$0.422 \pm 2\%$"]),
    ("SO2005", 6, "gamma", "subcritical evolutions", "0.422", "0.422", "2%", "Table I", [r"6 & $3.01 \pm 2\%$ &$0.422 \pm 2\%$"]),
    ("SO2005", 7, "Delta", "subcritical evolutions", "2.83", "2.83", "2%", "Table I", [r"7 & $2.83 \pm 2\%$ &$0.429 \pm 2\%$"]),
    ("SO2005", 7, "gamma", "subcritical evolutions", "0.429", "0.429", "2%", "Table I", [r"7 & $2.83 \pm 2\%$ &$0.429 \pm 2\%$"]),
    ("SO2005", 7, "Delta_half", "the curvature on the axis, last six echoes", "1.41", "1.41", "none stated", "Fig. 5",
     [r"equal to $ \Delta/2 \simeq 1.41$."]),
    ("SO2005", 8, "Delta", "subcritical evolutions", "2.70", "2.70", "3%", "Table I", [r"8 & $2.70 \pm 3\%$ &$0.436 \pm 2\%$"]),
    ("SO2005", 8, "gamma", "subcritical evolutions", "0.436", "0.436", "2%", "Table I", [r"8 & $2.70 \pm 3\%$ &$0.436 \pm 2\%$"]),
    ("SO2005", 9, "Delta", "subcritical evolutions", "2.61", "2.61", "3%", "Table I", [r"9 & $2.61 \pm 3\%$ &$0.442 \pm 2\%$"]),
    ("SO2005", 9, "gamma", "subcritical evolutions", "0.442", "0.442", "2%", "Table I", [r"9 & $2.61 \pm 3\%$ &$0.442 \pm 2\%$"]),
    ("SO2005", 10, "Delta", "subcritical evolutions", "2.55", "2.55", "3%", "Table I", [r"10 & $2.55 \pm 3\%$ &$0.447 \pm 3\% $"]),
    ("SO2005", 10, "gamma", "subcritical evolutions", "0.447", "0.447", "3%", "Table I", [r"10 & $2.55 \pm 3\%$ &$0.447 \pm 3\% $"]),
    ("SO2005", 11, "Delta", "subcritical evolutions", "2.51", "2.51", "3%", "Table I", [r"11 & $2.51 \pm 3\%$ &$0.44 \pm 3\% $"]),
    ("SO2005", 11, "gamma", "from the wiggle period and Delta (Fig. 8)", "0.44", "0.44", "3%", "Table I", [r"11 & $2.51 \pm 3\%$ &$0.44 \pm 3\% $"]),

    ("Bland2005", 3.5, "gamma", "Gaussian; amplitude varied", "0.349", "0.349", "+-0.003", "Table I", [r"3.5& $0.349\pm 0.003$"]),
    ("Bland2005", 4, "gamma", "Gaussian; amplitude varied", "0.374", "0.374", "+-0.002", "Table I", [r"4& $0.374\pm 0.002$ &$0.372 \pm 0.004$ & $3.40$"]),
    ("Bland2005", 4, "Delta", "residuals' period", "3.40", "3.40", "+-0.1", "Table I", [r"4& $0.374\pm 0.002$ &$0.372 \pm 0.004$ & $3.40$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 5, "gamma", "Gaussian; amplitude varied", "0.412", "0.412", "+-0.004", "Table I", [r"5& $0.412\pm 0.004$ & $0.408 \pm 0.008$&$3.10$"]),
    ("Bland2005", 5, "Delta", "residuals' period", "3.10", "3.10", "+-0.1", "Table I", [r"5& $0.412\pm 0.004$ & $0.408 \pm 0.008$&$3.10$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 6, "gamma", "Gaussian; amplitude varied", "0.430", "0.430", "+-0.003", "Table I", [r"6& $0.430\pm 0.003$ & $0.422 \pm 0.008$&$2.98$"]),
    ("Bland2005", 6, "Delta", "residuals' period", "2.98", "2.98", "+-0.1", "Table I", [r"6& $0.430\pm 0.003$ & $0.422 \pm 0.008$&$2.98$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 7, "gamma", "Gaussian; amplitude varied", "0.441", "0.441", "+-0.004", "Table I", [r"7& $0.441\pm 0.004$ & $0.429 \pm 0.009$&$2.96$"]),
    ("Bland2005", 7, "Delta", "residuals' period", "2.96", "2.96", "+-0.1", "Table I", [r"7& $0.441\pm 0.004$ & $0.429 \pm 0.009$&$2.96$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 8, "gamma", "Gaussian; amplitude varied", "0.446", "0.446", "+-0.004", "Table I", [r"8& $0.446\pm 0.004$ & $0.436 \pm 0.009$&$2.77$"]),
    ("Bland2005", 8, "Delta", "residuals' period", "2.77", "2.77", "+-0.1", "Table I", [r"8& $0.446\pm 0.004$ & $0.436 \pm 0.009$&$2.77$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 9, "gamma", "Gaussian; amplitude varied", "0.453", "0.453", "+-0.003", "Table I", [r"9& $0.453\pm 0.003$ & $0.442 \pm 0.009$&$2.63$"]),
    ("Bland2005", 9, "Delta", "residuals' period", "2.63", "2.63", "+-0.1", "Table I", [r"9& $0.453\pm 0.003$ & $0.442 \pm 0.009$&$2.63$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 10, "gamma", "Gaussian; amplitude varied", "0.456", "0.456", "+-0.004", "Table I", [r"10& $0.456\pm0.004$ & $0.447 \pm 0.013$&$2.50$"]),
    ("Bland2005", 10, "Delta", "residuals' period", "2.50", "2.50", "+-0.1", "Table I", [r"10& $0.456\pm0.004$ & $0.447 \pm 0.013$&$2.50$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 11, "gamma", "Gaussian; amplitude varied", "0.459", "0.459", "+-0.004", "Table I", [r"11& $0.459\pm 0.004$ & $0.44 \pm 0.013$&$2.46$"]),
    ("Bland2005", 11, "Delta", "residuals' period", "2.46", "2.46", "+-0.1", "Table I", [r"11& $0.459\pm 0.004$ & $0.44 \pm 0.013$&$2.46$", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 12, "gamma", "Gaussian; amplitude varied", "0.462", "0.462", "+-0.005", "Table I", [r"12& $0.462 \pm 0.005$ & &2.44 &"]),
    ("Bland2005", 12, "Delta", "residuals' period", "2.44", "2.44", "+-0.1", "Table I", [r"12& $0.462 \pm 0.005$ & &2.44 &", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 13, "gamma", "Gaussian; amplitude varied", "0.463", "0.463", "+-0.004", "Table I", [r"13& $0.463 \pm 0.004$ & &2.40 &"]),
    ("Bland2005", 13, "Delta", "residuals' period", "2.40", "2.40", "+-0.1", "Table I", [r"13& $0.463 \pm 0.004$ & &2.40 &", r"$\Delta (\pm 0.1)$"]),
    ("Bland2005", 14, "gamma", "Gaussian; amplitude varied", "0.465", "0.465", "+-0.004", "Table I", [r"14& $0.465 \pm 0.004$"]),

    ("BK2007", 5, "gamma", "Gaussian; amplitude varied; N = 2048", "0.4131", "0.4131", "+-0.0001", "Eq. (3)",
     [r"0.4131\pm 0.0001 \hspace*{25pt} \Delta_{5d} = 3.225 \pm 0.003."]),
    ("BK2007", 5, "Delta", "field at the origin; N = 2048", "3.225", "3.225", "+-0.003", "Eq. (3)",
     [r"0.4131\pm 0.0001 \hspace*{25pt} \Delta_{5d} = 3.225 \pm 0.003."]),
    ("BK2007", 5, "gamma", "N = 512", "0.4130", "0.4130", "+-0.0001", "Table I", [r"$512$ & $0.0048$ & $0.4130 \pm 0.0001$ & $3.2264 \pm 0.0005$"]),
    ("BK2007", 5, "Delta", "N = 512", "3.2264", "3.2264", "+-0.0005", "Table I", [r"$512$ & $0.0048$ & $0.4130 \pm 0.0001$ & $3.2264 \pm 0.0005$"]),
    ("BK2007", 5, "gamma", "N = 800", "0.4130", "0.4130", "+-0.0001", "Table I", [r"$800$ & $0.003072$ & $0.4130 \pm 0.0001$ & $3.2230 \pm 0.0006$"]),
    ("BK2007", 5, "Delta", "N = 800", "3.2230", "3.2230", "+-0.0006", "Table I", [r"$800$ & $0.003072$ & $0.4130 \pm 0.0001$ & $3.2230 \pm 0.0006$"]),
    ("BK2007", 5, "gamma", "N = 1024", "0.4131", "0.4131", "+-0.0001", "Table I", [r"$1024$ & $0.0024$ & $0.4131 \pm 0.0001$ & $3.2223 \pm 0.0006$"]),
    ("BK2007", 5, "Delta", "N = 1024", "3.2223", "3.2223", "+-0.0006", "Table I", [r"$1024$ & $0.0024$ & $0.4131 \pm 0.0001$ & $3.2223 \pm 0.0006$"]),
    ("BK2007", 5, "gamma", "N = 1536", "0.4131", "0.4131", "+-0.0001", "Table I", [r"$1536$ & $0.0016$ & $0.4131 \pm 0.0001$ & $3.2264 \pm 0.0004$"]),
    ("BK2007", 5, "Delta", "N = 1536", "3.2264", "3.2264", "+-0.0004", "Table I", [r"$1536$ & $0.0016$ & $0.4131 \pm 0.0001$ & $3.2264 \pm 0.0004$"]),
    ("BK2007", 5, "gamma", "N = 2048", "0.4131", "0.4131", "+-0.0001", "Table I", [r"$2048$ & $0.0012$ & $0.4131 \pm 0.0001$ & $3.2246 \pm 0.0005$"]),
    ("BK2007", 5, "Delta", "N = 2048", "3.2246", "3.2246", "+-0.0005", "Table I", [r"$2048$ & $0.0012$ & $0.4131 \pm 0.0001$ & $3.2246 \pm 0.0005$"]),
    ("BK2007", 5, "gamma", "width sigma varied; N = 2048", "0.4131", "0.4131", "+-0.0003", "Table II", [r"$\sigma$ & $0.4131 \pm 0.0003$"]),
    ("BK2007", 5, "gamma", "centre phi_0 varied; N = 2048", "0.4133", "0.4133", "+-0.0004", "Table II", [r"$\phi_0$ & $0.4133 \pm 0.0004$"]),

    ("TK2011", 4, "gamma", "A varied", "0.378", "0.378", "+-0.002", "Table 1", [r"4 & A varied & 0.378 $\pm$ 0.002 & 4.4 $\pm$ 0.3"]),
    ("TK2011", 4, "T", "A varied", "4.4", "4.4", "+-0.3", "Table 1", [r"4 & A varied & 0.378 $\pm$ 0.002 & 4.4 $\pm$ 0.3"]),
    ("TK2011", 4, "gamma", "B varied", "0.379", "0.379", "+-0.002", "Table 1", [r"& B varied & 0.379 $\pm$ 0.002 & 4.4 $\pm$ 0.2"]),
    ("TK2011", 4, "T", "B varied", "4.4", "4.4", "+-0.2", "Table 1", [r"& B varied & 0.379 $\pm$ 0.002 & 4.4 $\pm$ 0.2"]),
    ("TK2011", 5, "gamma", "A varied", "0.413", "0.413", "+-0.002", "Table 1", [r"5 & A varied & 0.413 $\pm$ 0.002 & 3.9 $\pm$ 0.8"]),
    ("TK2011", 5, "T", "A varied", "3.9", "3.9", "+-0.8", "Table 1", [r"5 & A varied & 0.413 $\pm$ 0.002 & 3.9 $\pm$ 0.8"]),
    ("TK2011", 5, "gamma", "B varied", "0.416", "0.416", "+-0.002", "Table 1", [r"& B varied & 0.416 $\pm$ 0.002 & 3.7 $\pm$ 0.4"]),
    ("TK2011", 5, "T", "B varied", "3.7", "3.7", "+-0.4", "Table 1", [r"& B varied & 0.416 $\pm$ 0.002 & 3.7 $\pm$ 0.4"]),
    ("TK2011", 6, "gamma", "A varied", "0.429", "0.429", "+-0.003", "Table 1", [r"6 & A varied & 0.429 $\pm$ 0.003 & 3.4 $\pm$ 0.3"]),
    ("TK2011", 6, "T", "A varied", "3.4", "3.4", "+-0.3", "Table 1", [r"6 & A varied & 0.429 $\pm$ 0.003 & 3.4 $\pm$ 0.3"]),
    ("TK2011", 6, "gamma", "B varied", "0.428", "0.428", "+-0.002", "Table 1", [r"& B varied & 0.428 $\pm$ 0.002 & 3.3 $\pm$ 0.2"]),
    ("TK2011", 6, "T", "B varied", "3.3", "3.3", "+-0.2", "Table 1", [r"& B varied & 0.428 $\pm$ 0.002 & 3.3 $\pm$ 0.2"]),
    ("TK2011", 7, "gamma", "A varied", "0.440", "0.440", "+-0.005", "Table 1", [r"7 & A varied & 0.440 $\pm$ 0.005 & 3.1 $\pm$ 0.1"]),
    ("TK2011", 7, "T", "A varied", "3.1", "3.1", "+-0.1", "Table 1", [r"7 & A varied & 0.440 $\pm$ 0.005 & 3.1 $\pm$ 0.1"]),
    ("TK2011", 7, "gamma", "B varied", "0.440", "0.440", "+-0.006", "Table 1", [r"& B varied & 0.440 $\pm$ 0.006 & 3.1 $\pm$ 0.4"]),
    ("TK2011", 7, "T", "B varied", "3.1", "3.1", "+-0.4", "Table 1", [r"& B varied & 0.440 $\pm$ 0.006 & 3.1 $\pm$ 0.4"]),
    ("TK2011", 8, "gamma", "A varied; first two periods only, tentative", "0.44", "0.44", "+-0.02", "Sec. 3.2",
     [r"then we find $\gamma = 0.44 \pm 0.02$ and $T = 3.0 \pm 0.1$."]),
    ("TK2011", 8, "T", "A varied; first two periods only, tentative", "3.0", "3.0", "+-0.1", "Sec. 3.2",
     [r"then we find $\gamma = 0.44 \pm 0.02$ and $T = 3.0 \pm 0.1$."]),

    ("PVG2022", 5, "gamma", "q = 0 (pure scalar field); Gaussian", "0.415", "0.415", "none stated", "Sec. V B",
     [r"exponents $\gamma_{\Psi} \simeq 0.415$ and $\gamma_{B} \simeq 0.164$ respectively"]),
    ("PVG2022", 5, "Delta_half", "q = 0; period of the highest Fourier peak of Psi, fitted by eye", "1.6", "1.6", "none stated", "Sec. V B",
     [r"We determined $\Delta_{\Psi} \simeq 1.6$ and $\Delta_B \simeq 0.47$"]),
    ("PVG2022", 5, "gamma", "q = 1e-6; Gaussian", "0.413", "0.413", "none stated", "Sec. V D",
     [r"$\gamma_\Psi \simeq 0.413$ was calculated numerically from the scaling laws for the radius of apparent horizon formation and for the Ricci scalar"]),
    ("PVG2022", 5, "gamma", "q = 1e-6; Gaussian; fitted slopes", "0.4131", "0.4131", "none stated", "Fig. 9",
     [r"The slope of the lines fitted against $R_\text{ah}$ and $(R_\Psi)^{-1/2}$ were $0.4131$."]),
    ("PVG2022", 5, "T_log10", "q = 1e-6; residuals of the R_ah fit", "1.7", "1.7", "none stated", "Fig. 10",
     [r"the fitted period of the residuals is $\Delta_\text{res} = 1.7$, which is related to the echoing period of the critical solution by $\Delta_\text{res} = \Delta_\Psi/(\ln(10)\gamma)$, resulting in $\Delta_\Psi \simeq 1.6$"]),
    ("PVG2022", 5, "gamma", "q = 0; Gaussian and Gaussian-derivative families", "0.41", "0.41", "none stated", "Sec. V E",
     [r"$\gamma \simeq 0.41$ for $q=0$"]),
]


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def ours_sources():
    """gamma(D) and Delta(D) from the repository's data files, with where each came from."""
    gam = {}
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "data", "delta_dd_lowD.csv")) if not l.startswith("#"))]
    for r in rows:
        if r["gamma"] and (float(r["D"]) not in gam or int(r["M"]) > gam[float(r["D"])][2]):
            gam[float(r["D"])] = (float(r["gamma"]), f"data/delta_dd_lowD.csv M{r['M']}", int(r["M"]))
    for fn in ("gamma_dd.jsonl", "local/gamma_dd.jsonl"):
        for line in open(os.path.join(HERE, "data", fn)):
            r = json.loads(line)
            if "gamma" in r and r["D"] not in gam:
                gam[r["D"]] = (r["gamma"], f"data/{fn} M{r['M']}", r["M"])
    dl = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))):
        dl[float(r["D"])] = (sum(float(r[f"Delta_q{i}"]) for i in range(4) if r[f"Delta_q{i}"] != ""), "data/delta_all.csv")
    return gam, dl


def ours(q, D, gam, dl):
    g, gs, _ = gam[float(D)]
    d, ds = dl[float(D)]
    return {"gamma": (g, gs), "gamma_mass": ((D - 3) * g, gs), "Delta": (d, ds), "Delta_half": (d / 2, ds),
            "T": (d / (2 * g), f"{ds}; {gs}"), "T_log10": (d / (2 * math.log(10) * g), f"{ds}; {gs}")}[q]


def uncertainty(lo, hi, unc):
    v = (float(lo) + float(hi)) / 2
    if unc == "range":
        return v, (float(hi) - float(lo)) / 2, "range stated"
    if unc == "none stated":
        return v, 0.5 * 10 ** -decimals(lo), "rounding bound (none stated)"
    if unc.endswith("%"):
        return v, float(unc[:-1]) / 100 * v, "stated (percent)"
    return v, float(unc.lstrip("+-")), "stated"


def build():
    gam, dl = ours_sources()
    out = []
    for ref, D, q, cond, lo, hi, unc, loc, _ in ROWS:
        v, ua, uk = uncertainty(lo, hi, unc)
        o, src = ours(q, D, gam, dl)
        diff = o - v
        out.append(dict(D=repr(float(D)), quantity=q, published_ref=ref, conditions=cond, published_lo=lo, published_hi=hi,
                        published_unc=unc, unc_abs=format(ua, ".3g"), unc_kind=uk, convention=CONV[(ref, q)],
                        locator=f"arXiv:{REFS[ref][0]}, {loc}", ours=format(o, ".10f"), ours_source=src,
                        diff=format(diff, ".2e"), within="yes" if abs(diff) <= ua * (1 + 1e-9) else "no"))
    return out


def norm(s):
    return re.sub(r"\s+", " ", s)


def verify(root):
    bad = 0
    for ref, (ax, _) in REFS.items():
        d = os.path.join(root, ax.replace("/", "_"))
        tex = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".tex")] if os.path.isdir(d) else []
        if not tex:
            print(f"  {ref}: no .tex files in {d}")
            bad += 1
            continue
        src = norm(" ".join(open(f, encoding="latin-1").read() for f in tex))
        for r in (r for r in ROWS if r[0] == ref):
            _, D, q, cond, lo, hi, unc, loc, quotes = r
            missing = [s for s in quotes if norm(s) not in src]
            strings = [lo, hi] + ([] if unc in ("range", "none stated") else
                                  [unc[:-1] + "\\%"] if unc.endswith("%") else [unc.lstrip("+-")])
            absent = [s for s in strings if not any(re.search(r"(?<![0-9.])" + re.escape(s) + r"(?![0-9])", qt) for qt in quotes)]
            if missing or absent:
                bad += 1
                print(f"  FAIL {ref} D {D} {q} ({cond}): quotes not in the source {missing}; values not in a quote {absent}")
    print(f"verify: {len(ROWS) - bad} of {len(ROWS)} rows have every quote in their arXiv source" if not bad else
          f"verify: {bad} failures")
    return bad == 0


def main():
    if "--verify" in sys.argv:
        sys.exit(0 if verify(sys.argv[sys.argv.index("--verify") + 1]) else 1)
    out = build()
    path = os.path.join(HERE, "data", "gamma_published_compare.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} published values -> {path}")
    for ref in REFS:
        rs = [o for o in out if o["published_ref"] == ref]
        ok = sum(o["within"] == "yes" for o in rs)
        worst = max(rs, key=lambda o: abs(float(o["diff"])))
        print(f"  {ref}: {len(rs)} values, {ok} within their uncertainty; largest |diff| {abs(float(worst['diff'])):.2e}"
              f" ({worst['quantity']} at D {worst['D']}, {worst['conditions']})")


if __name__ == "__main__":
    main()
