"""gamma of the sigma-model DSS solution minus gamma of the l = 1 multiplet (eta = infinity) at the same D, and fits of
the difference in powers of 1/n. The l = 1 values are the l-multiplet agent's k_c = 23 growth rates
(data/sigdss_l1_reference.jsonl, copied); the sigma-model values are this page's k_c = 15 ones (D = 6 and 9 checked at
k_c = 23). Writes data/sigdss_diff.json."""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D_ = os.path.join(HERE, "data")
ref = {r["D"]: r["lam"] for r in map(json.loads, open(os.path.join(D_, "sigdss_l1_reference.jsonl")))}
lam = {}
for f in glob.glob(os.path.join(D_, "sigdss_*.jsonl")):
    if "reference" in f or "refine" in f:
        continue
    for line in open(f):
        r = json.loads(line)
        if not r.get("lam") or r.get("kc") != 15:
            continue
        if r.get("last_step") is not None and r["last_step"] > 1e-3:
            continue
        lam[(round(r["eps"], 3), r["D"])] = r["lam"]
out = {}
for eps, name in ((1.0, "eta = 1"), (3.333, "eta = 0.3")):
    Ds = sorted(D for (e, D) in lam if e == eps and D in ref)
    dg = np.array([1 / lam[(eps, D)] - 1 / ref[D] for D in Ds])
    print(f"{name}: D {Ds}\n  gamma_sigma - gamma_l1 = {np.round(dg, 5).tolist()}")
    n = np.array(Ds) - 2
    fits = []
    for Dmin in (5, 6, 7):
        sel = np.array(Ds) >= Dmin
        for m in (0, 1, 2):
            if sel.sum() < m + 2:
                continue
            A = np.stack([n[sel] ** -k for k in range(m + 1)], 1)
            c, *_ = np.linalg.lstsq(A, dg[sel], rcond=None)
            rms = float(np.sqrt(np.mean((A @ c - dg[sel]) ** 2)))
            fits.append(dict(Dmin=Dmin, m=m, npts=int(sel.sum()), d_inf=float(c[0]), rms=rms))
            print(f"  D >= {Dmin}, {m} powers of 1/n: d_inf = {c[0]:+.5f} (rms {rms:.1e}, {int(sel.sum())} points)")
    out[name] = dict(D=Ds, dgamma=dg.tolist(), fits=fits)
json.dump(out, open(os.path.join(D_, "sigdss_diff.json"), "w"), indent=1)
