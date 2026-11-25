"""Track A's form-S guess at large D (hp/form_s.py jump's fit, as in I1088-mac-a/form_s/hprobe_mob.py): the chain's
states from D 226 to DMAX fitted in 1/n at degree DEG in jump.scale's variables, evaluated at D and mapped to form S.
Written: m_h (M = 32) per D, for hp_mob_check.py --guess (#1090). Uses Track A's hp/jump.py, form_s.py and
octo_corrector.py from FORMS_HP (read only).

Usage: python3 make_guess.py RAW OUT.json --Ds=2000,5000,10000 [--deg=2] [--dmax=252.5]
  RAW: a folder with I903-macm3/chain_k135/ and I966-dh370/ks_chain3b/ of the raw-data repository.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("FORMS_HP", ""))
import jump  # noqa: E402
from form_s import a_to_s, groups_s  # noqa: E402
from octo_corrector import load_o  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
raw, out = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
deg = int(opts.get("deg", 2))
dmax = float(opts.get("dmax", 252.5))
paths = sorted(glob.glob(raw + "/I903-macm3/chain_k135/anm_pin3_D*.npz") + glob.glob(raw + "/I966-dh370/ks_chain3b/*/anm_pin3_D*.npz"))
rows = []
for p in paths:
    z, m = load_o(p)
    if 226 <= m["D"] <= dmax:
        rows.append((m["D"], z, m))
rows.sort(key=lambda r: r[0])
M, nodes = 32, rows[-1][2]["nodes"]
K = len(nodes)
Ds = np.array([r[0] for r in rows])
Y = np.array([jump.scale(sum(np.asarray(p) for p in r[1][:2]), r[0], M, K) for r in rows])
c = np.polyfit(1.0 / (Ds - 2.0), Y, deg)
res = {"states": len(rows), "D_range": [float(Ds[0]), float(Ds[-1])], "deg": deg, "nodes": list(map(float, nodes)), "guess": {}}
for D in [float(v) for v in opts.get("Ds", "2000,5000,10000").split(",")]:
    y = np.array([np.polyval(c[:, j], 1.0 / (D - 2.0)) for j in range(Y.shape[1])])
    zA = jump.unscale(y, D, M, K)
    zs = a_to_s([zA], M, K)[0]
    mh = zs[groups_s(M, K)["mh"]]
    res["guess"]["%g" % D] = {"mh": list(map(float, mh)), "Delta_fit": float(zs[-1])}
    print("D %g: %d states D %.2f-%.2f, degree %d; max|m_h| %.4e, Delta_fit %.12f" % (D, len(rows), Ds[0], Ds[-1], deg, np.abs(mh).max(), zs[-1]))
json.dump(res, open(out, "w"), indent=1)
