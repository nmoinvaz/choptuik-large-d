"""#1076: a zero of the M = 3 multiplier R_1(A, w_f) located by Newton in (A, w_f) on its extended-precision values.

    python zero_mp.py A0 WF0 CACHE OUT.jsonl [--r1=0.17] [--procs=5] [--iters=5] [--hA=1e-7] [--hw=1e-8] [--f3=RE,IM]
    python zero_mp.py selftest [--mutate=sign]      a planted analytic map through the same Newton; exit 0 iff it lands
    python zero_mp.py A0 WF0 - OUT.jsonl --M=5 --v0=B3,C3,B5,C5 [...]   M harmonics (#1080): the harmonics v of the
        inner solve (value_mp.value_v) start from v0 and then from the previous point's solved v; no cache

R_1 at each point is value_mp.value_point (34 and 42 digits, chord steps), started from the F_3 of the nearest valued
cache record and then from the previous point's tuned F_3. The Jacobian is central differences in A and w_f (4 points,
run in parallel with the centre); a step is dx = -J^-1 R. It stops when |R| < 10^3 sigma_mp, when |dx| < 1e-13, or after
--iters steps. Each iteration writes a row: A, w_f, R, sigma_mp, |R|/sigma_mp, J, the orientation sign(det J) (the
argument principle's +1 or -1), dx. A zero is reported as located when the last |R| is below 10^3 sigma_mp, or when the
next step is below the float64 coordinate grid (1e-13) and |R| is what that step would remove (within 10x)."""
import json
import math
import sys
import time

import numpy as np

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), None)


def _value(args):
    import value_mp as VM
    (A, wf, f3re, f3im), r1 = args
    row = VM.value_point(repr(A), repr(wf), repr(f3re), repr(f3im), r1=r1)
    return dict(A=A, wf=wf, R=complex(row["re"], row["im"]), sig=row["sig"], ok=row["valued"],
                F3=(row["F3re"], row["F3im"]), info=row["info"], sec=row["seconds"])


def _value_v(args):
    import value_mp as VM
    (A, wf, v), r1, M = args[:3]
    kw = args[3] if len(args) > 3 else {}
    row = VM.value_v(repr(A), repr(wf), list(v), M, r1=r1, **kw)
    return dict(A=A, wf=wf, R=complex(row["re"], row["im"]), sig=row["sig"], ok=row["valued"],
                F3=tuple(row["v"]), info=row["info"], sec=row["seconds"], J=row.get("J"))


def _planted(args):
    (A, wf, f3re, f3im), r1 = args
    a0, w0 = 3.5265381234, 0.2114823456             # the planted zero
    R = complex((A - a0) * 40 + (wf - w0) * 300 + 2 * (A - a0) ** 2, -(A - a0) * 25 + (wf - w0) * 900)
    return dict(A=A, wf=wf, R=R, sig=1e-18, ok=True, F3=(f3re, f3im), info=0.0, sec=0.0)


def newton(A, wf, f3, fn, iters, hA, hw, procs, out=None, M=3, J0=None):
    """f3 is F_3 = (re, im) at M = 3, and the harmonics v = (B_3, C_3, ..., B_M, C_M) (strings) with fn = _value_v"""
    import multiprocessing as mp_
    pool = mp_.get_context("spawn").Pool(procs) if procs > 1 and fn in (_value, _value_v) else None
    run = (lambda xs: list(pool.map(fn, xs))) if pool else (lambda xs: list(map(fn, xs)))
    rows = []
    for it in range(iters + 1):
        pts = [(A, wf), (A + hA, wf), (A - hA, wf), (A, wf + hw), (A, wf - hw)]
        if fn is _value_v:
            # the centre first, at both precisions, from the last centre's Jacobian; then the offsets from the centre's
            # solved harmonics and Jacobian, at one precision (they only form the outer Jacobian's differences)
            c0 = fn(((A, wf, tuple(f3)), "0.17", M, dict(J=J0, want_J=True)))
            J0 = c0["J"]
            res = [c0] + run([((a, w, tuple(c0["F3"])), "0.17", M, dict(J=J0, single=True)) for a, w in pts[1:]])
        else:
            res = run([((a, w, f3[0], f3[1]), "0.17") for a, w in pts])
        c = res[0]
        dRA = (res[1]["R"] - res[2]["R"]) / (2 * hA)
        dRw = (res[3]["R"] - res[4]["R"]) / (2 * hw)
        J = np.array([[dRA.real, dRw.real], [dRA.imag, dRw.imag]])
        if MUT == "sign":                               # the control: a wrong-signed Jacobian
            J = -J
        dx = -np.linalg.solve(J, np.array([c["R"].real, c["R"].imag]))
        row = dict(it=it, A=A, wf=wf, re=c["R"].real, im=c["R"].imag, sig=c["sig"], R_over_sig=abs(c["R"]) / c["sig"],
                   valued=[r["ok"] for r in res], J=J.tolist(), orientation=int(np.sign(np.linalg.det(J))),
                   dA=float(dx[0]), dwf=float(dx[1]), sec=[r["sec"] for r in res], F3=list(c["F3"]))
        if fn is _value_v:
            row["J_inner"] = c.get("J")                 # the centre's inner Jacobian: --j0 restarts a walk from it
        rows.append(row)
        if out:
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"it {it}: A {A:.15f} wf {wf:.15f} |R| {abs(c['R']):.3e} (|R|/sigma_mp {row['R_over_sig']:.2e}), "
              f"orientation {row['orientation']:+d}, dx ({dx[0]:+.2e}, {dx[1]:+.2e})", flush=True)
        if abs(c["R"]) < 1e3 * c["sig"] or max(abs(dx[0]), abs(dx[1])) < 1e-13 or it == iters:
            break
        A, wf, f3 = A + float(dx[0]), wf + float(dx[1]), c["F3"]
    if pool:
        pool.close()
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    iters, hA, hw = int(o.get("iters", 5)), float(o.get("hA", 1e-7)), float(o.get("hw", 1e-8))
    if args and args[0] == "selftest":
        rows = newton(3.5265, 0.2115, (0.0, 0.0), _planted, 6, hA, hw, 1)
        last = rows[-1]
        ok = abs(last["A"] - 3.5265381234) < 1e-10 and abs(last["wf"] - 0.2114823456) < 1e-10 and last["orientation"] == 1
        print(f"selftest: {'PASS' if ok else 'FAIL'} (landed at {last['A']:.12f}, {last['wf']:.12f})")
        return 0 if ok else 1
    if len(args) < 4:
        print(__doc__)
        return 2
    A0, wf0, cache, out = float(args[0]), float(args[1]), args[2], args[3]
    if "M" in o and int(o["M"]) > 3:                 # M harmonics from --v0 (#1080)
        M = int(o["M"])
        v0 = o["v0"].split(",")
        assert len(v0) == M - 1, "--v0 needs B_3, C_3, ..., B_M, C_M"
        J0 = None
        if "j0" in o:                               # --j0=ROWS.jsonl: the last row's J_inner (restart of a walk)
            J0 = [json.loads(line) for line in open(o["j0"]) if line.strip()][-1].get("J_inner")
        print(f"start ({A0}, {wf0}); M = {M}, v0 = {v0}, inner Jacobian {'given' if J0 else 'fresh'}", flush=True)
        t0 = time.time()
        rows = newton(A0, wf0, v0, _value_v, iters, hA, hw, int(o.get("procs", 5)), out, M=M, J0=J0)
        return report(rows, t0)
    if "f3" in o:                                   # --f3=RE,IM: an explicit start (a point outside the cache's box)
        f3 = tuple(float(t) for t in o["f3"].split(","))
        print(f"start ({A0}, {wf0}); F_3 given {f3}", flush=True)
        t0 = time.time()
        rows = newton(A0, wf0, f3, _value, iters, hA, hw, int(o.get("procs", 5)), out)
        return report(rows, t0)
    best, bd = None, 1e300
    for line in open(cache):
        r = json.loads(line)
        if r.get("info") is not None and isinstance(r["info"], float) and r["info"] < 1e-8 and math.isfinite(r.get("F3re", float("nan"))):
            d = (r["A"] - A0) ** 2 + (r["wf"] - wf0) ** 2
            if d < bd:
                best, bd = r, d
    f3 = (best["F3re"], best["F3im"])
    print(f"start ({A0}, {wf0}); F_3 from the cache record at ({best['A']}, {best['wf']}), distance {math.sqrt(bd):.2e}",
          flush=True)
    t0 = time.time()
    rows = newton(A0, wf0, f3, _value, iters, hA, hw, int(o.get("procs", 5)), out)
    return report(rows, t0)


def report(rows, t0):
    last = rows[-1]
    J = np.array(last["J"])
    step = max(abs(last["dA"]), abs(last["dwf"]))
    lin = float(np.linalg.norm(J @ np.array([last["dA"], last["dwf"]])))    # the |R| the unapplied step would remove
    R = math.hypot(last["re"], last["im"])
    # located: |R| below 1e3 sigma_mp, or the next step below the float64 coordinate grid (1e-13) with |R| what that
    # step removes (within a factor 10): near a zero one ulp of A moves R by about |dR/dA| 4e-16, above 1e3 sigma_mp
    located = last["R_over_sig"] < 1e3 or (step < 1e-13 and R <= 10 * max(lin, 1e-300))
    print(f"{'located' if located else 'NOT located'}: A {last['A']:.15f} wf {last['wf']:.15f} (unapplied step "
          f"{last['dA']:+.2e}, {last['dwf']:+.2e}), |R| {R:.3e} against the step's {lin:.3e}, |R|/sigma_mp "
          f"{last['R_over_sig']:.2e}, orientation {last['orientation']:+d}; {time.time() - t0:.0f} s", flush=True)
    return 0 if located else 1


if __name__ == "__main__":
    sys.exit(main())
