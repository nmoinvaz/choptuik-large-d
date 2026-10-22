#!/usr/bin/env python3
"""Compare two series pickles written by expand.py (e.g. the same run with sympy's integrator and with the exact
integrator). Exit 1 if any coefficient differs.

    python3 cmp_series.py logs/expand_K2_zero_css_sympyint.pkl logs/expand_K2_zero_css.pkl
"""
import pickle
import sys
import sympy as sp

a, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
same = all(sp.expand(sp.sympify(a[k][j]) - sp.sympify(b[k][j])) == 0 for k in "YCUS" for j in range(len(a[k])))
print("identical series:" if same else "DIFFERENT series:", sys.argv[1], sys.argv[2])
sys.exit(0 if same else 1)
