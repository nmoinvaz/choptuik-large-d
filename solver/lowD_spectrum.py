#!/usr/bin/env python3
"""Harmonic content of the low-D solutions (notes/LOW_D.md): the odd-harmonic spectrum |F_k| of the centre slope
F(tau) = sqrt(D - 2) dp/dx at x = 0, its decay rate per harmonic, and the largest harmonic-k amplitude over every field
and node (the content the tau discretisation has to carry).

    python3 lowD_spectrum.py STATE [STATE ...] [--fit=KLO:KHI] [--json=out.jsonl]

STATE: an f64 solution (solver/data/neB_kc{kc}_..._D{D}.npy: x_K = 0.6 layout; kc read from the name) or a
double-double state (hp/data/lowD_states/*.npz, hp/data/states/*.npz). The decay rate beta is the slope of
-ln|F_k| against k by least squares over KLO <= k <= KHI (default: the odd k from 9 to the last harmonic above 1e3 times
the state's noise floor, 1e-15 for f64 and 1e-28 for double-double, relative to |F_1|). Prints per state: D, |F_1|,
|F_3|/|F_1|, beta, and |F_k| at k = 15, 23, 31, ...
--selftest: the same code path on planted spectra (a pure exponential e^{-beta k} with noise, and a sum of two poles)
must return the planted beta to 1%; it exits non-zero if not.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def f64_fields(path):
    from spec_long import make_long
    m = re.search(r"kc(\d+)", os.path.basename(path))
    kc = int(m.group(1)) if m else 23
    D = float(re.findall(r"D([0-9.]+)\.npy", path)[0])
    z = np.load(path)
    S = make_long(kc, D - 2, nodes=NL, Ks=60)
    if S.layout()[2] != len(z):
        raise ValueError(f"{path}: length {len(z)} is not the kc {kc} x_K 0.6 layout ({S.layout()[2]})")
    parts, _ = S.split(z[None, :])
    pars = S.parities()
    out = []
    for c, p in zip(parts, pars):
        c = c[0]
        if p == "odd":
            mk = len(S.kodd)
            amp = np.zeros(kc + 1)
            amp[S.kodd] = 2 * np.abs(c[:mk] + 1j * c[mk:])
        else:
            mk = len(S.keven) - 1
            amp = np.zeros(kc + 1)
            amp[0] = abs(c[0])
            amp[S.keven[1:]] = 2 * np.abs(c[1:1 + mk] + 1j * c[1 + mk:])
        out.append((p, amp))
    return D, kc, float(z[-1]), out, 1e-15


def dd_fields(path):
    d = np.load(path)
    zh, zl, meta = d["zh"], d["zl"], json.loads(str(d["meta"]))
    M, K = meta["M"], len(meta["nodes"])
    z = zh + zl
    pars = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    out = []
    for j, p in enumerate(pars):
        f = z[j * M:(j + 1) * M]
        full = np.r_[f, -f] if p == "odd" else np.r_[f, f]
        c = np.fft.rfft(full) / M
        amp = np.abs(c[:M])
        amp[0] /= 2
        out.append((p, amp))
    return meta["D"], M, float(zh[-1] + zl[-1]), out, 1e-28


def decay(amp, floor_rel, klo=None, khi=None):
    k = np.arange(len(amp))
    odd = (k % 2 == 1) & (amp > 0)
    a1 = amp[1]
    if khi is None:
        good = odd & (amp > 1e3 * floor_rel * a1)
        khi = int(k[good].max()) if good.any() else 1
    if klo is None:
        klo = 9
    sel = odd & (k >= klo) & (k <= khi)
    if sel.sum() < 3:
        return None, klo, khi
    c = np.polyfit(k[sel], np.log(amp[sel]), 1)
    return float(-c[0]), klo, khi


def selftest():
    ok = True
    rng = np.random.default_rng(1)
    for beta in (0.3, 0.2, 0.12):
        k = np.arange(128)
        amp = np.where(k % 2 == 1, 0.8 * np.exp(-beta * k) * (1 + 0.05 * rng.standard_normal(128)) + 1e-17, 0.0)
        b, lo, hi = decay(amp, 1e-15)
        good = abs(b - beta) < 0.01 * beta
        ok &= good
        print(f"planted e^(-{beta} k) + noise 1e-17: beta {b:.4f} over k {lo}..{hi}  {'ok' if good else 'FAIL'}")
    # control that must fire: a wrong floor (fit into the noise) biases beta low
    amp = np.where(k % 2 == 1, 0.8 * np.exp(-0.3 * k) + 1e-12, 0.0)
    b, lo, hi = decay(amp, 1e-30)
    fired = abs(b - 0.3) > 0.01 * 0.3
    ok &= fired
    print(f"control (fit window into a 1e-12 noise floor): beta {b:.4f} over k {lo}..{hi}  {'fires' if fired else 'DID NOT FIRE'}")
    sys.exit(0 if ok else 1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if "selftest" in opts:
        selftest()
    klo = khi = None
    if "fit" in opts:
        klo, khi = (int(v) for v in opts["fit"].split(":"))
    files = []
    for a in args:
        files += sorted(glob.glob(a)) or [a]
    recs = []
    for f in files:
        D, res, Delta, fields, floor = (dd_fields if f.endswith(".npz") else f64_fields)(f)
        F = np.sqrt(D - 2) * fields[0][1]          # p1 = dp/dx at x = 0
        b, lo, hi = decay(F, floor, klo, khi)
        allmax = np.max([a[:len(F)] if len(a) >= len(F) else np.r_[a, np.zeros(len(F) - len(a))] for _, a in fields], axis=0)
        ball, _, _ = decay(allmax, floor, klo, khi)
        ks = [k for k in (15, 23, 31, 39, 47, 55, 63, 79, 95, 111, 127) if k < len(F)]
        print(f"{os.path.basename(f):34s} D {D:.6f} res {res:3d} Delta {Delta:.12f}  |F1| {F[1]:.4f}  F3/F1 {F[3] / F[1]:.4f}  "
              f"beta {b if b is None else round(b, 4)} (k {lo}..{hi})  beta_all {ball if ball is None else round(ball, 4)}  "
              + "  ".join(f"F{k} {F[k]:.1e}" for k in ks))
        recs.append(dict(file=os.path.basename(f), D=D, res=res, Delta=Delta, F1=float(F[1]), F3_F1=float(F[3] / F[1]),
                         beta=b, fit=[lo, hi], beta_all=ball, F_harm=[float(f"{v:.4e}") for v in F[1::2]],
                         all_harm=[float(f"{v:.4e}") for v in allmax[1::2]]))
    if "json" in opts:
        with open(opts["json"], "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
