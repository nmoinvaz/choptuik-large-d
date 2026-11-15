#!/usr/bin/env python3
"""The amplitude of #782's near singularities in lam_g's and Delta's real series, against the fold's (#821): does #820's
bound, |D*(lam_g) - D*(Delta)| < 1e-6 when that amplitude is at most 0.01, apply?

    cd hp && python3 near_amp.py planted RAWDIR [--json=OUT]   plants of known amplitude at each base (exits 1 on a fail)
    cd hp && python3 near_amp.py real RAWDIR --planted=P.json [--json=OUT]   the real series at the usable bases

RAWDIR is a clone of the raw-data repository (I603-win2/chain_nsr, as hp/fold_cancel.py reads it); the bases are
#818's 26 (hp/data/lam_singularity.json).

The fit, at each base, s = 259.9033 - D0, orders k = 4..24 by least squares, each order weighted by 1/|c_k|:
    c_k = sum_{j = 1, 3, 5} b_j s^(j/2) binom(j/2, k) (-1)^k s^-k                (the fold, as fold_cancel.coeffs)
          + A2 binom(1/2, k) (-1)^k s2^-k + 2 Re[A3 binom(1/2, k) (-1)^k s3^-k]   (#782's near singularities)
with s2 = 260.17 - D0, s3 = 259.47 + 1.17i - D0, and A3 complex: eight real parameters. The amplitudes are read as
#820's plant defines them, against the fold's: a2 = |A2|/(b1 sqrt(s)) and a3 = |A3|/(b1 sqrt(s)).

planted (fixed before the real series were fitted by this script): at each base, each series built from
fold_cancel's median b (hp/data/fold_cancel_coeffs.json), the near singularities at a = 0, 0.003, 0.01, 0.03 and 0.3
of the fold's amplitude (A2 = A3 = a b1 sqrt(s), #820's plant), a regular part (lam_g: lam_singularity.regular, 10.4's
reference fit and #815's entire term; Delta: the same divided by r = 0.020646, so that its size against the fold is
lam_g's), and relative noise 1e-12. A base is usable when every a > 0 is recovered to 30% (a2 and a3 within 0.3 a of a)
in both series. planted passes when 4 or more bases are usable; exit 1 otherwise. a = 0 is reported, the fit's floor.

real: a2 and a3 of both series at the usable bases. #820's conditional bound applies when every one of them is at
most 0.01; otherwise it does not apply. Reported beside it: a2 at every base where a2 alone is recovered.

The lowest order (#821, plants only, before the real fit): with the regular part in the plant, orders 4-24 (the issue's
text) recover no base, since the near terms absorb the regular part's low orders (a = 0 reads a2 up to 0.2 and a3 0.5
to 2.8; without the regular part every plant is exact). --kmin=K starts the fit at order K: K = 8 and 10 give 0 and 1
usable bases, K = 12 gives 8 (259.4928 to 259.6462) and K = 14 gives 7. The registered fit is K = 12.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

import fold_cancel as FC
import lam_singularity as LS

HERE = os.path.dirname(os.path.abspath(__file__))
DSTAR = FC.DSTAR
KS = range(12, 25)          # the registered orders; the issue's text had 4-24 (see the docstring)
AMPS = (0.003, 0.01, 0.03, 0.3)
TOL = 0.3
LIMIT = 0.01
R_MED = 0.020646
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]


def near_funcs(D0, k):
    h = mp.mpf(1) / 2
    s2 = mp.mpf(260.17 - D0)
    s3 = mp.mpc(259.47 - D0, 1.17)
    g = mp.binomial(h, k) * (-1) ** k
    return g / s2 ** k, g / s3 ** k


def fit(c, D0):
    """(b1, b3, b5, A2, A3, a2, a3, largest relative residual) from c_1..c_24"""
    s = DSTAR - mp.mpf(D0)
    rows, y = [], []
    for k in KS:
        f2, f3 = near_funcs(D0, k)
        w = 1 / abs(c[k - 1])
        rows.append([float(FC.basis(j, k, s) * w) for j in FC.JS] + [float(f2 * w), float(2 * f3.real * w),
                                                                        float(-2 * f3.imag * w)])
        y.append(float(c[k - 1] * w))
    A, y = np.array(rows), np.array(y)
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    b1 = p[0]
    A3 = complex(p[4], p[5])
    norm = abs(b1) * float(mp.sqrt(s))
    return dict(b=[float(x) for x in p[:3]], A2=float(p[3]), A3=[A3.real, A3.imag], a2=abs(p[3]) / norm,
                a3=abs(A3) / norm, res=float(np.max(np.abs(A @ p - y))))


def plant(D0, b, a, reg, seed):
    rng = np.random.default_rng(seed)
    s = DSTAR - mp.mpf(D0)
    f = FC.fold_terms(b, s)
    amp = a * b[0] * mp.sqrt(s)
    out = []
    for k in range(1, 25):
        f2, f3 = near_funcs(D0, k)
        out.append((f[k - 1] + amp * (f2 + 2 * f3.real) + reg[k - 1]) * (1 + 1e-12 * rng.standard_normal()))
    return out


def set_kmin():
    """--kmin=K: the fit's lowest order (12 registered, 4 in the issue's text)"""
    global KS
    if "kmin" in args:
        KS = range(int(args["kmin"]), 25)


def cmd_planted(raw):
    set_kmin()
    mp.mp.dps = 200
    print("fit orders %d-%d" % (KS[0], KS[-1]))
    co = {float(k): v for k, v in json.load(open(os.path.join(HERE, "data", "fold_cancel_coeffs.json"))).items()}
    med = lambda key, i: float(np.median([v[key][i] for v in co.values()]))      # noqa: E731
    bl = [med("lam", i) for i in range(3)]
    bd = [med("delta", i) for i in range(3)]
    out, usable = {}, []
    print("plants: near singularities at a = 0, %s of the fold's amplitude; recovered when a2 and a3 are within %.0f%%"
          % (", ".join(str(x) for x in AMPS), 100 * TOL))
    for D0 in sorted(co):
        with mp.workdps(60):
            reg_l = LS.regular(D0, 24)
        reg_d = [x / R_MED for x in reg_l]
        res, ok = {}, True
        for name, b, reg in (("lam", bl, reg_l), ("delta", bd, reg_d)):
            for a in (0.0,) + AMPS:
                f = fit(plant(D0, b, a, reg, 821 + int(round(D0 * 1e4)) % 1000), D0)
                good = a == 0 or (abs(f["a2"] - a) <= TOL * a and abs(f["a3"] - a) <= TOL * a)
                ok &= good
                res["%s:%g" % (name, a)] = dict(f, recovered=bool(good))
        out[D0] = dict(res, usable=bool(ok))
        if ok:
            usable.append(D0)
        cell = lambda n: " ".join("%.3g/%.3g" % (res["%s:%g" % (n, a)]["a2"], res["%s:%g" % (n, a)]["a3"])   # noqa: E731
                                  for a in (0.0,) + AMPS)
        print("  D0 %.4f s %.3f  lam_g a2/a3: %s | Delta: %s | %s" % (D0, float(DSTAR) - D0, cell("lam"), cell("delta"),
                                                                     "usable" if ok else "not usable"))
    passed = len(usable) >= LS.NCOUNT
    print("planted: %d of %d bases usable (%s): %s" % (len(usable), len(out), ", ".join("%.4f" % x for x in usable),
                                                      "PASS" if passed else "FAIL"))
    if "json" in args:
        json.dump(dict(usable=usable, bases=out), open(args["json"], "w"), indent=1)
    return 0 if passed else 1


def cmd_real(raw):
    set_kmin()
    mp.mp.dps = 200
    print("fit orders %d-%d" % (KS[0], KS[-1]))
    usable = json.load(open(args["planted"]))["usable"]
    out, worst = {}, 0.0
    print("real: a2 (D 260.17) and a3 (the pair 259.47 +- 1.17i) against the fold's amplitude, at the usable bases")
    for D0 in sorted(json.load(open(os.path.join(HERE, "data", "fold_cancel_coeffs.json"))), key=float):
        D0 = float(D0)
        lam, dl = FC.series(raw, D0)
        fl, fd = fit(lam[1:], D0), fit(dl[1:], D0)
        use = any(abs(D0 - u) < 1e-9 for u in usable)
        out[D0] = dict(lam=fl, delta=fd, usable=use)
        if use:
            worst = max(worst, fl["a2"], fl["a3"], fd["a2"], fd["a3"])
        print("  D0 %.4f  lam_g a2 %.3e a3 %.3e (res %.1e) | Delta a2 %.3e a3 %.3e (res %.1e)%s" % (
            D0, fl["a2"], fl["a3"], fl["res"], fd["a2"], fd["a3"], fd["res"], "" if use else " (not usable)"))
    applies = bool(usable) and worst <= LIMIT
    v = ("#820's conditional bound applies: every amplitude at the usable bases is at most %.2f (largest %.3e)" % (LIMIT, worst)
         if applies else "#820's conditional bound does not apply (largest amplitude at the usable bases %.3e)" % worst)
    print("real:", v)
    if "json" in args:
        json.dump(dict(largest=worst, applies=applies, verdict=v, bases=out), open(args["json"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(planted=cmd_planted, real=cmd_real).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))
