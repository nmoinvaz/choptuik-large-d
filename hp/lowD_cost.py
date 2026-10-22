#!/usr/bin/env python3
"""Cost of approaching D = 3 (notes/LOW_D.md, Sec. 2): the tau modes needed for a target error in Delta and the time per
point, extrapolated from the measured spectra and timings. An estimate under stated assumptions, not a bound.

    python3 lowD_cost.py [--target=1e-10] [--selftest]

Model: Delta's truncation error at resolution K (f64 kc, or double-double M - 1) is A exp(-2 beta(D) K) (at D = 4 the
measured steps shrink by 0.30-0.32 per mode, twice beta = 0.157). A is calibrated at D = 4 from kc 55: error 2.6e-8
against Martin-Garcia & Gundlach's 3.445452402 (T15's kc 55 value 3.4454524280). Two assumptions for other D bracket the
answer: A constant, or A growing like |F_1|^2 (the amplitude). beta(D): the measured f64 kc 47 values (k = 21-35;
solver/data/lowD_spectrum.jsonl) for D >= 3.7, then two continuations to D = 3.046, where Ecker et al.'s exported field
gives beta about 0.032 (hp/lowD_ecker.py --fields; indicative only): linear in D, and a power of (D - 3) through the
D = 4 and 3.046 values. |F_1|: (D - 3)^(-1.09) through the 3.7 and 3.046 values.
Time per Newton iteration: t(M) = t48 (M / 48)^p with t48 and p from the double-double records in hp/data/lowD_dd.jsonl
(seconds / iterations at M = 48 and 64, same node set); quoted for this cloud machine under its load and to be scaled.
--selftest: synthetic errors A exp(-2 b K) with known A, b, calibrated through the same functions from two K, must give
the planted K for the target to within 1 mode; a control with a wrong b must miss.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BETA = [(4.0, 0.157), (3.7938, 0.1411), (3.7, 0.1341)]   # f64 kc 47, k = 21-35 (solver/data/lowD_spectrum.jsonl)
BETA_ECKER = (3.046, 0.032)                                # hp/lowD_ecker.py --fields (antiperiodicity error 4%)
F1 = [(4.0, 16.23), (3.7, 22.65), (3.046, 437.2)]


def calibrate(K1, e1, b):
    return e1 * math.exp(2 * b * K1)


def modes_needed(A, b, target):
    return math.log(A / target) / (2 * b)


def beta_of(D, model):
    if D >= 3.7:
        Ds = [d for d, _ in BETA][::-1]
        bs = [b for _, b in BETA][::-1]
        for i in range(len(Ds) - 1):
            if Ds[i] <= D <= Ds[i + 1]:
                t = (D - Ds[i]) / (Ds[i + 1] - Ds[i])
                return bs[i] + t * (bs[i + 1] - bs[i])
        return bs[-1]
    D0, b0 = 3.7, BETA[-1][1]
    D1, b1 = BETA_ECKER
    if model == "linear":
        return b0 + (D - D0) * (b1 - b0) / (D1 - D0)
    q = math.log(BETA[0][1] / b1) / math.log((4.0 - 3) / (D1 - 3))    # beta = beta4 (D - 3)^q
    return BETA[0][1] * (D - 3) ** q


def F1_of(D):
    q = math.log(F1[2][1] / F1[1][1]) / math.log((F1[1][0] - 3) / (F1[2][0] - 3))
    if D >= 3.7:
        return F1[0][1] + (D - 4.0) * (F1[1][1] - F1[0][1]) / (3.7 - 4.0)
    return F1[1][1] * ((D - 3) / 0.7) ** (-q)


def timings():
    f = os.path.join(HERE, "data", "lowD_dd.jsonl")
    per = {}
    if os.path.exists(f):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("converged") and len(r["nodes"]) == 11 and r.get("its"):
                    per.setdefault(r["M"], []).append(r["seconds"] / r["its"])
    if 48 in per and 64 in per:
        t48, t64 = min(per[48]), min(per[64])
        return t48, math.log(t64 / t48) / math.log(64 / 48)
    return None, None


def selftest():
    ok = True
    for A, b, target in ((0.55, 0.157, 1e-10), (2.0, 0.12, 1e-12), (0.1, 0.05, 1e-8)):
        K1 = 47
        e1 = A * math.exp(-2 * b * K1)
        Kt = modes_needed(calibrate(K1, e1, b), b, target)
        Ktrue = math.log(A / target) / (2 * b)
        good = abs(Kt - Ktrue) < 1
        ok &= good
        print(f"planted A {A} b {b}: modes for {target:g} {Kt:.1f} (true {Ktrue:.1f})  {'ok' if good else 'FAIL'}")
    A, b = 0.55, 0.157
    e1 = A * math.exp(-2 * b * 47)
    Kw = modes_needed(calibrate(47, e1, 0.8 * b), 0.8 * b, 1e-10)
    fired = abs(Kw - math.log(A / 1e-10) / (2 * b)) > 1
    ok &= fired
    print(f"control (beta 20% low): {Kw:.1f} modes  {'fires' if fired else 'DID NOT FIRE'}")
    sys.exit(0 if ok else 1)


def main():
    if "--selftest" in sys.argv:
        selftest()
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    target = float(opts.get("target", 1e-10))
    A4 = calibrate(55, 2.6e-8, BETA[0][1])
    t48, p = timings()
    print(f"calibration at D = 4: A = {A4:.3g} (kc 55 error 2.6e-8, beta 0.157); target {target:g}")
    if t48:
        print(f"double-double time per Newton iteration here: {t48:.0f} s at M = 48, exponent p = {p:.2f} from M = 64")
    print("    D   beta(lin)  beta(pow)  |F_1|   modes: A const      A ~ F_1^2        time/iteration at that M")
    for D in (4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.05):
        bl, bp = beta_of(D, "linear"), beta_of(D, "power")
        f1 = F1_of(D)
        ks = []
        for b in (bl, bp):
            ks.append(modes_needed(A4, b, target))
            ks.append(modes_needed(A4 * (f1 / 16.23) ** 2, b, target))
        lo, hi = min(ks), max(ks)
        tt = ""
        if t48:
            tt = f"{t48 * (lo / 48) ** p / 60:.0f}-{t48 * (hi / 48) ** p / 60:.0f} min"
        print(f"  {D:4.2f}  {bl:.3f}      {bp:.3f}     {f1:6.1f}   {modes_needed(A4, bl, target):5.0f}-{modes_needed(A4, bp, target):5.0f}"
              f"     {modes_needed(A4 * (f1 / 16.23) ** 2, bl, target):5.0f}-{modes_needed(A4 * (f1 / 16.23) ** 2, bp, target):5.0f}"
              f"      {tt}")


if __name__ == "__main__":
    main()
