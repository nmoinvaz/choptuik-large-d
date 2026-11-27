#!/usr/bin/env python3
"""#1098: the point-to-point scatter of the low-D rows of Delta(D) (sing_data.py's file), which sets
sing_core.sigma_eff. Each point is predicted by a polynomial of degree K in e = 1/(D - 2) through its H neighbours on
each side (itself left out, 50 digits); its deviation is data - prediction, divided by sqrt(1 + h), h the prediction's
variance in units of the noise variance ([(A^T A)^-1]_00), so that it estimates the noise itself. Reported per kind
and D band: the median and rms of the scaled deviations (points above --thr left out of both, and listed), beside the
table's err.

    cd hp && python3 sing_noise.py DATA [--H=7] [--K=8] [--thr=3e-16] [--planted=SIZE]

Junctions: for each pair of consecutive runs (the source's run name), one polynomial of degree K in e through the last
J points of the first run and the first J points of the second (J = --J, 12), plus a step: the second run's offset,
with its 1-sigma error from the scatter of the fit's residuals. An offset beyond 4 of its sigma and above 3e-16 is
flagged: a systematic difference between the runs, which local scatter does not show.

--planted=SIZE replaces every value by T1 (sing_core.t1) plus Gaussian noise of that size (seed 1): the check that the
estimator recovers a known scatter. It exits 1 unless the rms of every band with at least 20 points is within a
factor 1.25 of SIZE, for bands where the local polynomial's own truncation is below SIZE (D >= 16; at the defaults,
the settings of sigma_eff; at --K=6 --H=6 it fails for D 16-22, where that polynomial is too short for the gaps between
runs, and its truncation of 1e-14 at D 23 reads as scatter).
"""
import json
import math
import random
import statistics
import sys

import mpmath as mp

import sing_core as S

mp.mp.dps = 50
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
H, K, THR = int(args.get("H", "7")), int(args.get("K", "8")), float(args.get("thr", "3e-16"))
BANDS = [(8, 16), (16, 22), (22, 30), (30, 36), (36, 50), (50, 60), (60, 68), (68, 74), (74, 120), (120, 200),
         (200, 246)]


def deviations(pts):
    e = [1 / (mp.mpf(p["D"]) - 2) for p in pts]
    y = [mp.mpf(p["Delta"]) for p in pts]
    out = []
    for i in range(len(pts)):
        nb = [j for j in range(max(0, i - H), min(len(pts), i + H + 1)) if j != i]
        if len(nb) < K + 3:
            continue
        c = e[i]
        h = max(abs(e[j] - c) for j in nb)
        A = mp.matrix([[((e[j] - c) / h) ** k for k in range(K + 1)] for j in nb])
        b = mp.matrix([y[j] for j in nb])
        N = A.T * A
        x = mp.lu_solve(N, A.T * b)
        h = mp.inverse(N)[0, 0]
        out.append((float(pts[i]["D"]), float((y[i] - x[0]) / mp.sqrt(1 + h)), pts[i]))
    return out


def junctions(pts):
    J = int(args.get("J", "12"))
    runs = []
    for p in pts:
        name = p["kind"] + ":" + (p["src"].split(":")[-1] if p["kind"] == "dd" else p["src"].split("/")[0])
        if not runs or runs[-1][0] != name:
            runs.append((name, []))
        runs[-1][1].append(p)
    out = []
    for (na, A), (nb, B) in zip(runs, runs[1:]):
        sel = A[-J:] + B[:J]
        if len(A) < 3 or len(B) < 3 or len(sel) < K + 4:
            continue
        e = [1 / (mp.mpf(p["D"]) - 2) for p in sel]
        y = [mp.mpf(p["Delta"]) for p in sel]
        c = (e[0] + e[-1]) / 2
        h = max(abs(x - c) for x in e)
        nA = len(A[-J:])
        rows = [[((e[i] - c) / h) ** k for k in range(K + 1)] + [mp.mpf(1 if i >= nA else 0)] for i in range(len(sel))]
        Am = mp.matrix(rows)
        Nm = Am.T * Am
        x = mp.lu_solve(Nm, Am.T * mp.matrix(y))
        r = [y[i] - sum(rows[i][k] * x[k] for k in range(K + 2)) for i in range(len(sel))]
        dof = len(sel) - K - 2
        s2 = sum(v * v for v in r) / dof if dof > 0 else mp.mpf(0)
        sd = mp.sqrt(s2 * mp.inverse(Nm)[K + 1, K + 1])
        off = x[K + 1]
        flag = abs(off) > 4 * sd and abs(off) > 3e-16
        out.append((na, nb, float(A[-1]["D"]), float(B[0]["D"]), float(off), float(sd), flag))
    return out


def main():
    pts = [p for p in json.load(open(sys.argv[1]))["points"] if p["kind"] in ("dd", "qd", "chain_fit")]
    pts.sort(key=lambda p: float(p["D"]))
    size = float(args["planted"]) if "planted" in args else None
    if size:
        rng = random.Random(1)
        for p in pts:
            p["Delta"] = mp.nstr(S.t1(1 / (mp.mpf(p["D"]) - 2)) + mp.mpf(rng.gauss(0, size)), 40)
        print("PLANTED: T1 plus Gaussian noise of %.1e" % size)
    dev = deviations(pts)
    bad = [d for d in dev if abs(d[1]) > THR]
    print("%d points; %d with |deviation| > %.1e (D < 16.5: the local polynomial's truncation, not noise)" % (
        len(dev), len(bad), THR))
    for D, v, p in bad:
        if D >= 16.5:
            print("  D %.6f deviation %+.2e (%s, err %.1e) %s" % (D, v, p["kind"], p["sigma"], p["src"]))
    ok = True
    expect = size or 0
    for kind in ("dd", "qd", "chain_fit"):
        for lo, hi in BANDS:
            sel = [d for d in dev if d[2]["kind"] == kind and lo <= d[0] < hi]
            v = [abs(d[1]) for d in sel if abs(d[1]) <= THR]
            if not v:
                continue
            rms = math.sqrt(sum(x * x for x in v) / len(v))
            errs = sorted(d[2]["sigma"] for d in sel)
            line = "  %-9s D %3d-%3d: %4d points (%d above thr), median |dev| %.2e, rms %.2e; table err %.1e-%.1e" % (
                kind, lo, hi, len(sel), len(sel) - len(v), statistics.median(v), rms, errs[0], errs[-1])
            if size and lo >= 16 and len(v) >= 20:
                good = expect / 1.25 <= rms <= expect * 1.25
                ok &= good
                line += "  [planted: expect %.2e, %s]" % (expect, "PASS" if good else "FAIL")
            print(line)
    print("junctions (second run's offset from the first, J %s points each side, degree %d):" % (args.get("J", "12"), K))
    for na, nb, Da, Db, off, sd, flag in junctions(pts):
        print("  %-22s D %8.3f | %-22s D %8.3f: offset %+.2e +- %.1e%s" % (na, Da, nb, Db, off, sd,
                                                                         "  FLAGGED" if flag else ""))
    if size:
        print("planted check: %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


main()
