#!/usr/bin/env python3
"""lambda_1 on a stored state with M's rounding floor lifted (#1025), and the checks that the lift is sound.

M = DSSEll.mode_matrix (real lambda) has singular values below 1e-14 sigma_max: centre modes that decay outward and reach
the outer conditions only below rounding, so the sign of det M, which ell_refine's root finder reads, is noise past
D 17.5 at k_c 55 (raw I1025-mac detscan_kc55_D18.log, signtest.log). Here the root is the sign change of
det(M + W_A), W_A = ell_eigs.floor_lift(M(ref_A)), found from the guess by ell_refine's widening grid and refined by
brentq (xtol 1e-10). Each M(lambda) is built once and serves three determinants: M, M + W_A and M + W_B (a second
reference). Checks at the root lam_A:
  raw: sign of det M at lam_A -/+ d_raw (1e-8): opposite signs put M's own root within d_raw of lam_A (meaningful
       only where M's sign is resolved, which the rotation test reports);
  ref: sign of det(M + W_B) at lam_A -/+ d_ref (1e-6): opposite signs put the B root within d_ref of the A root;
  rot: at both ends, the sign of det(M + W_A) from LU(Q (M + W_A)) sign(det Q) for 3 random orthogonal Q agrees with
       LU(M + W_A)'s (the sign is resolved), and the same for M itself.
--scan lo:hi:step first evaluates the three determinants on a grid and lists every sign change (real roots near
lambda_1: a neighbour shows there).
usage: python ell_lam.py STATE D kc guess --refs RA,RB [--record LAM] [--scan lo:hi:step] [--d_raw 1e-8]
                         [--d_ref 1e-6] [--out FILE]
       (from solver/matter; --refs are the reference lambda, real, away from the roots; --record a stored lambda)"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD
from ell_eigs import floor_lift


def rot_signs(M, nq=3, seed=11):
    g = np.random.default_rng(seed)
    out = [float(np.linalg.slogdet(M)[0])]
    for _ in range(nq):
        Q = np.linalg.qr(g.standard_normal(M.shape))[0]
        out.append(float(np.linalg.slogdet(Q @ M)[0] * np.linalg.slogdet(Q)[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int)
    ap.add_argument("guess", type=float)
    ap.add_argument("--refs", required=True); ap.add_argument("--record", type=float, default=None)
    ap.add_argument("--d_raw", type=float, default=1e-8); ap.add_argument("--d_ref", type=float, default=1e-6)
    ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--u0", type=float, default=0.04)
    ap.add_argument("--Ks", type=int, default=30); ap.add_argument("--out", default=None)
    ap.add_argument("--scan", default=None)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.state)
    S = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=1, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
    S.freeze_from(z, a.kappa)
    refs = [float(v) for v in a.refs.split(",")]
    rec = dict(D=a.D, kc=a.kc, state=os.path.basename(a.state), guess=a.guess, refs=refs, record=a.record)
    built = [0]

    def Mof(l):
        built[0] += 1
        with np.errstate(all="ignore"):
            return S.mode_matrix(z, l)
    W, info = [], []
    for r in refs:
        w, i = floor_lift(Mof(r))
        W.append(w); info.append(i)
        print(f"reference {r}: lifted {i['k']} (floor top {i['floor_top']:.1e}, next {i['above']:.1e} of sigma_max "
              f"{i['sigma_max']:.3e})", flush=True)
    rec["lift"] = info
    cache = {}

    def dets(l):
        if l not in cache:
            M = Mof(l)
            cache[l] = [np.linalg.slogdet(M)] + [np.linalg.slogdet(M + w) for w in W]
        return cache[l]
    sgA = lambda l: dets(l)[1][0]
    if a.scan:
        lo_, hi_, st_ = (float(v) for v in a.scan.split(":"))
        sc = [float(x) for x in np.arange(lo_, hi_ + 1e-12, st_)]
        tabs = [[dets(l)[j][0] for l in sc] for j in range(len(W) + 1)]
        chg = [[f"{sc[i]:.4f}-{sc[i + 1]:.4f}" for i in range(len(sc) - 1) if t[i] != t[i + 1]] for t in tabs]
        rec.update(scan=sc, scan_signs=tabs, scan_changes=chg)
        for nm, t, c in zip(["M", "M + W_A", "M + W_B"], tabs, chg):
            print(f"scan {nm}: signs {''.join('+' if v > 0 else '-' if v < 0 else '0' for v in t)}; changes {c}", flush=True)
    root, nch = None, 0
    for wd in (0.02, 0.06, 0.15, 0.35):
        grid = a.guess * (1 + wd * np.array([-1.0, 0.0, 1.0]))
        sg = [sgA(l) for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            nch = len(ch)
            i = ch[0] if len(ch) == 1 else min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - a.guess))
            ref = 0.5 * (dets(grid[i])[1][1] + dets(grid[i + 1])[1][1])
            f = lambda l: dets(l)[1][0] * np.exp(dets(l)[1][1] - ref)
            root = brentq(f, grid[i], grid[i + 1], xtol=1e-10, rtol=1e-14)
            rec.update(grid=[float(x) for x in grid], grid_signs_A=[float(x) for x in sg])
            break
    rec.update(lam=root, n_signchanges=nch)
    if root is None:
        print(json.dumps(rec), flush=True)
        sys.exit(1)
    lo, hi = root - a.d_raw, root + a.d_raw
    raw = [dets(lo)[0][0], dets(hi)[0][0]]
    lo2, hi2 = root - a.d_ref, root + a.d_ref
    refB = [dets(lo2)[2][0], dets(hi2)[2][0]] if len(W) > 1 else None
    rot = {}
    for tag, l in (("lo", lo2), ("hi", hi2)):
        M = Mof(l)
        rot[tag] = dict(raw=rot_signs(M), A=rot_signs(M + W[0]))
    rec.update(raw_signs=raw, raw_within=bool(raw[0] != raw[1]), refB_signs=refB,
               refB_within=bool(refB[0] != refB[1]) if refB else None, rot=rot,
               rotA_agree=all(len(set(rot[t]["A"])) == 1 for t in rot), rotraw_agree=all(len(set(rot[t]["raw"])) == 1 for t in rot),
               A_opposite=bool(rot["lo"]["A"][0] != rot["hi"]["A"][0]),
               minus_record=(root - a.record) if a.record is not None else None, n_built=built[0],
               seconds=round(time.time() - t0))
    print(f"D {a.D} k_c {a.kc}: lambda_1 {root:.12f} (lift A at {refs[0]}); M's own root within {a.d_raw:g}: "
          f"{rec['raw_within']} (M's sign resolved there: {rec['rotraw_agree']}); B root within {a.d_ref:g}: "
          f"{rec['refB_within']}; lifted sign resolved: {rec['rotA_agree']}"
          + (f"; minus the record {rec['minus_record']:+.2e}" if a.record is not None else ""), flush=True)
    print(json.dumps(rec), flush=True)
    out = a.out or os.path.join(HERE, "data", "ell_lam_lift_1025.jsonl")
    open(out, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
