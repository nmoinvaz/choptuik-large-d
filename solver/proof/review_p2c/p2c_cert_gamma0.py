#!/usr/bin/env python3
"""P2c review (#988): the Perron root of the certificate's saved Gamma0 (solver/proof/b8m_lcj.json), certified with
p2c_theta0.certify, against the certificate's printed theta0. A comparison, not part of the check.

    python3 p2c_cert_gamma0.py
"""
import json
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
import p2c_theta0 as P        # noqa: E402

d = json.load(open(os.path.join(os.path.dirname(HERE), "b8m_lcj.json")))
order = ("S", "V", "b", "c")
G = [[Fr(repr(d["Gamma0"][i][j])) for j in order] for i in order]
ev, theta, w = P.certify(G)
print(f"certificate's Gamma0 (rows S, V, b, c): printed theta0 {d['theta0']:.6e}; its Perron root {ev:.6e}, "
      f"certified here {float(theta):.6e} (weight {[float(a) for a in w]})")
