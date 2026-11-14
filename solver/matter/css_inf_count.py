#!/usr/bin/env python3
"""Box count of css_inf_jump.py's Stokes-jump determinant D(lambda) at D = infinity, kappa = 0 (#781).

D is entire in lambda (its formal series are polynomial in lambda, and optimal truncation changes it only at
e^-44 of its size), so the winding of D around the box is its number of zeros inside. The full boundary of
0.05 < Re lambda < 5, |Im lambda| < 9 is traced counterclockwise from initial steps of h0 = 0.25, each step bisected
until |change of arg| <= 0.5 rad and the two halves agree with the whole to 1e-6. The conjugation symmetry is not
assumed. It is checked on the evaluated points instead: D(conj lam) = conj D(lam).

    python3 css_inf_count.py selftest [--mutate=coarse]     a synthetic D with known zeros, three of them 0.02-0.05
                                                            from the edges; coarse (one sample per edge) must fail
    python3 css_inf_count.py count [--r0=0.1 --r1=0.3 --alpha=0.375 --rtol=1e-13 --N=160] [--plant=3.1+5j]
                                   [--box=RE0,RE1,IM0,IM1] [--evlog=F] [--out=F]
                                   --plant multiplies D by (lam - p)(lam - conj p); --box defaults to 0.05,5,-9,9
    python3 css_inf_count.py locate LAM0 [LAM0 ...] [--target=T,...] [--atol=A] [setting options] [--out=F]
                                   zeros of D by css_inf_jump's secant from each LAM0 (and LAM0 + 1e-3); --atol sets
                                   DOP853's absolute tolerance (css_inf_jump.ATOL), needed next to lambda = +-i omega
--evlog keeps every D(lam) (jsonl) and is read back first, so a plant reuses the plain count's evaluations.
Exit codes: selftest 0 pass, 1 fail; count 0 when the winding is an integer and no bisection limit is hit; locate 0.
"""
import json
import math
import os
import sys
import time

BOX = (0.05, 5.0, -9.0, 9.0)
H0, STEP, DEPTH = 0.25, 0.5, 14


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Arg:
    """arg of g(lam) = D(lam) (lam - p)(lam - conj p), D cached by lam"""

    def __init__(self, D, plant=None, cache=None, log=None):
        self.D, self.plant, self.cache, self.log = D, plant, ({} if cache is None else cache), log
        self.nev = 0

    def value(self, lam):
        key = (round(lam.real, 12), round(lam.imag, 12))
        if key not in self.cache:
            t = time.time()
            d = self.D(complex(*key))
            self.cache[key] = d
            self.nev += 1
            if self.log:
                self.log(dict(kind="eval", lam=list(key), D=[d.real, d.imag], s=round(time.time() - t, 1)))
        return self.cache[key]

    def __call__(self, lam):
        a = math.atan2(self.value(lam).imag, self.value(lam).real)
        if self.plant is not None:
            p = self.plant
            a += math.atan2((lam - p).imag, (lam - p).real) + math.atan2((lam - p.conjugate()).imag,
                                                                         (lam - p.conjugate()).real)
        return a


def track(arg, a, c, K, coarse=False):
    """unwrapped change of arg from a to c over K initial steps, each bisected until |step| <= STEP and the halves agree
    with the whole; returns (change, largest step, intervals, bisection-limit hits)"""
    tot, big, nint, hits = 0.0, 0.0, 0, 0
    pt = lambda t: a + (c - a) * t

    def incr(t0, t1, depth):
        nonlocal big, nint, hits
        d = wrap(arg(pt(t1)) - arg(pt(t0)))
        if coarse:
            big = max(big, abs(d))
            nint += 1
            return d
        tm = 0.5 * (t0 + t1)
        d1 = wrap(arg(pt(tm)) - arg(pt(t0)))
        d2 = wrap(arg(pt(t1)) - arg(pt(tm)))
        if abs(d1 + d2 - d) > 1e-6 or abs(d1) > STEP or abs(d2) > STEP:
            if depth < DEPTH:
                return incr(t0, tm, depth + 1) + incr(tm, t1, depth + 1)
            hits += 1
        big = max(big, abs(d1), abs(d2))
        nint += 2
        return d1 + d2
    for k in range(K):
        tot += incr(k / K, (k + 1) / K, 0)
    return tot, big, nint, hits


def count_box(arg, box=BOX, coarse=False):
    re0, re1, im0, im1 = box
    corners = [complex(re0, im0), complex(re1, im0), complex(re1, im1), complex(re0, im1), complex(re0, im0)]
    tot, big, nint, hits = 0.0, 0.0, 0, 0
    for a, c in zip(corners[:-1], corners[1:]):
        K = 1 if coarse else max(1, int(math.ceil(abs(c - a) / H0)))
        d, b, n, h = track(arg, a, c, K, coarse)
        tot += d
        big, nint, hits = max(big, b), nint + n, hits + h
    return dict(winding=tot / (2 * math.pi), largest_step=big, intervals=nint, bisection_limit_hits=hits)


def selftest(coarse=False):
    om = 3.978969507370554
    zl = 0.5365223424 + 7.9320241697j
    inside = [1.0, 2.0, zl, zl.conjugate(), 4.98 + 0.3j, 4.98 - 0.3j]
    outside = [0.0, 1j * om, -1j * om, 0.03 + 9.02j, 3.0 + 9.5j]

    def D(lam):
        v = complex(math.exp(4 * lam.real), 0) * complex(math.cos(4 * lam.imag), math.sin(4 * lam.imag))
        for z in inside + outside:
            v *= lam - z
        return v
    r = count_box(Arg(D), coarse=coarse)
    rp = count_box(Arg(D, plant=3.1 + 5j), coarse=coarse)
    ok = round(r["winding"]) == len(inside) and abs(r["winding"] - round(r["winding"])) < 1e-9 \
        and round(rp["winding"]) == len(inside) + 2 and r["largest_step"] <= STEP and r["bisection_limit_hits"] == 0
    print("selftest%s: zeros inside %d (1, 2, the lambda_osc pair, 4.98 +- 0.3i; outside 0, +-i omega, 0.03 + 9.02i, "
          "3 + 9.5i): winding %.12f, planted %.12f, largest step %.3f rad, %d intervals: %s" % (
              " (coarse)" if coarse else "", len(inside), r["winding"], rp["winding"], r["largest_step"],
              r["intervals"], "PASS" if ok else "FAIL"))
    strip = (-0.05, 0.05, -9.0, 9.0)                # #791: 0 and +-i omega inside, 0.03 + 9.02i just outside
    s = count_box(Arg(D), box=strip, coarse=coarse)
    sp = count_box(Arg(D, plant=0.02 + 6j), box=strip, coarse=coarse)
    ok2 = round(s["winding"]) == 3 and abs(s["winding"] - 3) < 1e-9 and round(sp["winding"]) == 5 \
        and s["largest_step"] <= STEP and s["bisection_limit_hits"] == 0
    print("selftest%s, strip -0.05 < Re < 0.05: winding %.12f (3: 0, +-i omega), planted 0.02 +- 6i %.12f, largest "
          "step %.3f rad, %d intervals: %s" % (" (coarse)" if coarse else "", s["winding"], sp["winding"],
                                                s["largest_step"], s["intervals"], "PASS" if ok2 else "FAIL"))
    return ok and ok2


def main():
    args = sys.argv[1:]
    opts = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--")}
    pos = [a for a in args if not a.startswith("--")]
    if pos and pos[0] == "selftest":
        sys.exit(0 if selftest(coarse=opts.get("mutate") == "coarse") else 1)
    if not pos or pos[0] not in ("count", "locate"):
        raise SystemExit(__doc__)
    import css_inf_jump as cj
    st = dict(r0=float(opts.get("r0", 0.1)), r1=float(opts.get("r1", 0.3)), alpha=float(opts.get("alpha", 0.375)),
              rtol=float(opts.get("rtol", 1e-13)), N=int(opts.get("N", 160)))
    t0 = time.time()
    P = cj.Problem(r0=st["r0"], r1=st["r1"], alpha=st["alpha"] * math.pi, rtol=st["rtol"], N=st["N"])
    x, j = P.select(log=lambda s: None)
    if pos[0] == "locate":
        cj.ATOL = float(opts["atol"]) if "atol" in opts else None
        om = x[1]
        names = {"0": 0j, "iw": 1j * om, "-iw": -1j * om}
        targets = [names[t] for t in opts.get("target", "").split(",")] if "target" in opts else [None] * len(pos[1:])
        for s0, tg in zip(pos[1:], targets):
            t = time.time()
            lam, it = P.root(complex(s0), log=lambda s: None)
            d = P.det(lam)
            rec = dict(kind="locate", **st, atol=cj.ATOL, f0=x[0], omega=om, bg_jump=j, start=[complex(s0).real,
                       complex(s0).imag], lam=[lam.real, lam.imag], iterations=it, D_at_root=abs(d),
                       target=None if tg is None else [tg.real, tg.imag],
                       dist_to_target=None if tg is None else abs(lam - tg), seconds=round(time.time() - t, 1))
            print(json.dumps(rec), flush=True)
            if "out" in opts:
                with open(opts["out"], "a") as fo:
                    fo.write(json.dumps(rec) + "\n")
        return
    box = tuple(float(v) for v in opts["box"].split(",")) if "box" in opts else BOX
    evlog = opts.get("evlog")
    cache = {}
    if evlog and os.path.exists(evlog):
        for line in open(evlog):
            r = json.loads(line)
            cache[tuple(r["lam"])] = complex(*r["D"])

    def log(r):
        if evlog:
            with open(evlog, "a") as fo:
                fo.write(json.dumps(r) + "\n")
    plant = complex(opts["plant"]) if "plant" in opts else None
    arg = Arg(P.det, plant=plant, cache=cache, log=log)
    res = count_box(arg, box=box)
    sym = [abs(arg.cache[k].conjugate() - arg.cache[(k[0], -k[1])]) / abs(arg.cache[k])
           for k in arg.cache if k[1] > 0 and (k[0], -k[1]) in arg.cache]
    rec = dict(kind="count", **st, box=list(box), plant=[plant.real, plant.imag] if plant else None, f0=x[0],
               omega=x[1], bg_jump=j, **res, new_evaluations=arg.nev, cached=len(arg.cache) - arg.nev,
               conj_pairs=len(sym), conj_symmetry_max=max(sym) if sym else None, seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    if "out" in opts:
        with open(opts["out"], "a") as fo:
            fo.write(json.dumps(rec) + "\n")
    ok = abs(res["winding"] - round(res["winding"])) < 1e-6 and res["bisection_limit_hits"] == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
