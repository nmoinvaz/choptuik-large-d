#!/usr/bin/env python3
"""#1082: the decisive test of the quad-double mode matrix at D 3.044375 (M 96).

    python3 mode_qd_decide.py OUTDIR STATE.npz [--threads=6] [--hmax=0.05] [--lams=L1,L2,...] [--skip_dd]
                              [--allow_unraw]
    python3 mode_qd_decide.py verdict OUTDIR
    python3 mode_qd_decide.py selftest [--mutate=accept_any]

At D 3.044375 the double-double entries of M(lam) flip det M's sign every 3e-13 in lam (#1078 comment 7120204793:
signs +, +, -, + and ln|det| jumps of 4 to 8 at the four rates below, factored by a quad-double LU). This script
evaluates det M at those rates twice, each factored by the quad-double LU of mode_qd.qd_det: with double-double
entries (Solver.mode_matrix, the control) and with quad-double entries (Solver.mode_matrix_q). The rows go to
OUTDIR/dd.jsonl and OUTDIR/qd.jsonl as each rate finishes; a rerun skips the rates already there, so an interrupted
run resumes. OUTDIR must be a task folder of the raw-data repository (hp/raw_out.py) unless --allow_unraw.

The criterion, fixed before the run: over a window of 9e-13 det M(lam) is linear in lam, so it has at most one root
there. The four rows (lam_i, sign_i, ln|det|_i) pass when the line a + b (lam - lam_1) fitted to the signed values
sign_i exp(ln_i - max ln) by least squares relative to each value gives every row's sign and its ln|det| within 1e-6.
(Far from a root ln|det| moves by about 15 per unit lam there, 1e-11 over the window.)

Verdict: RESOLVED when the quad-double rows pass and the double-double rows (the control) fail; exit 0 then, else 1.
The rows' jsonl and OUTDIR/verdict.json hold the numbers. The fit is also shown as the coordinator read the
double-double rows: the line through the two middle rows, its prediction at the outer two.

selftest: planted rows through the same criterion. A line with its root between the middle rates and a constant must
pass; the double-double rows of comment 7120204793, a line with one ln|det| moved by 1e-5, and a sign pattern of two
roots must fail. --mutate=accept_any makes the criterion pass everything; the selftest must then fail. Exit 1 on
failure.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# the four rates of #1078 comment 7120204793 (dh370, state blob 6a0f83d2 of the raw-data repository)
LAMS = [5.1408197887857, 5.140819788786, 5.1408197887863025, 5.1408197887866]
TOL = 1e-6
# the double-double-entry rows of that comment (sign, ln|det|): the selftest's jagged case
JAGGED = [(1.0, 3600.328311), (1.0, 3604.297374), (-1.0, 3601.178104), (1.0, 3596.436366)]
MUTATE = {"accept_any": False}


def linear_check(rows, tol=TOL):
    """the criterion on rows of (lam, sign, ln|det|): the least-squares line in lam through sign exp(ln - max ln),
    each residual relative to its value; returns dict(ok, worst, per-row predictions, middle-pair predictions)"""
    rows = sorted(rows, key=lambda r: r[0])
    lam = np.array([r[0] for r in rows])
    sg = np.array([r[1] for r in rows])
    ln = np.array([r[2] for r in rows])
    ref = ln.max()
    y = sg * np.exp(ln - ref)
    x = lam - lam[0]                                           # exact: the rates are within a factor 2
    w = 1.0 / np.abs(y)
    A = np.stack([np.ones_like(x), x / max(x.max(), 1e-300)], axis=1) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, y * w, rcond=None)
    f = coef[0] + coef[1] * x / max(x.max(), 1e-300)
    dev = [abs(math.log(abs(fi)) - (li - ref)) if fi != 0 and math.copysign(1.0, fi) == si else math.inf
           for fi, si, li in zip(f, sg, ln)]
    pred = [(math.copysign(1.0, fi), float(math.log(abs(fi)) + ref) if fi != 0 else -math.inf) for fi in f]
    mid = None
    if len(rows) == 4:
        # the line through rows 2 and 3, at rows 1 and 4
        s = (y[2] - y[1]) / (x[2] - x[1])
        at = lambda xi: y[1] + s * (xi - x[1])
        mid = [(math.copysign(1.0, at(x[i])), float(math.log(abs(at(x[i]))) + ref)) for i in (0, 3)]
    worst = max(dev)
    ok = worst <= tol or MUTATE["accept_any"]
    return dict(ok=bool(ok), worst=float(worst), dev=[float(d) for d in dev], pred=pred, mid_pred=mid,
                lam=lam.tolist(), sign=sg.tolist(), ln=ln.tolist())


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def show(name, chk, log):
    log(f"{name}: lam, sign, ln|det| (fit sign, ln|det|; |dev|)")
    for l, s, la, (ps, pl), d in zip(chk["lam"], chk["sign"], chk["ln"], chk["pred"], chk["dev"]):
        log(f"  {l!r:20} {s:+.0f} {la:.9f}   fit {ps:+.0f} {pl:.9f}   {d:.2e}")
    if chk["mid_pred"]:
        (s1, l1), (s4, l4) = chk["mid_pred"]
        log(f"  the line through the middle two at the outer two: {s1:+.0f} {l1:.6f}, {s4:+.0f} {l4:.6f}")
    log(f"  worst |dev| {chk['worst']:.2e} against {TOL:g}: {'linear (consistent)' if chk['ok'] else 'NOT linear (jagged)'}")


def verdict(outdir, lams, log=print):
    out = {}
    for e in ("dd", "qd"):
        rows = [r for r in read_rows(os.path.join(outdir, f"{e}.jsonl")) if r["lam"] in lams]
        have = {r["lam"] for r in rows}
        if have != set(lams):
            log(f"{e}: rows for {len(have)} of {len(lams)} rates")
            out[e] = None
            continue
        last = {r["lam"]: r for r in rows}                     # a rerun's row replaces an earlier one
        chk = linear_check([(l, last[l]["sign"], last[l]["ln"]) for l in lams])
        show(f"{e} entries" + (" (the control)" if e == "dd" else ""), chk, log)
        out[e] = chk
    qd_ok = out.get("qd") is not None and out["qd"]["ok"]
    dd_jag = out.get("dd") is not None and not out["dd"]["ok"]
    res = "RESOLVED" if qd_ok and dd_jag else "NOT RESOLVED"
    if out.get("dd") is not None and out["dd"]["ok"]:
        log("the control did not fire: the double-double rows are linear, unlike #1078's; the test cannot decide")
    log(f"verdict: {res} (quad-double rows {'linear' if qd_ok else 'not linear or missing'}; double-double rows "
        f"{'jagged' if dd_jag else 'not jagged or missing'})")
    with open(os.path.join(outdir, "verdict.json"), "w") as fh:
        json.dump(dict(verdict=res, tol=TOL, lams=lams, qd=out.get("qd"), dd=out.get("dd")), fh, indent=1)
    return 0 if res == "RESOLVED" else 1


def selftest(log=print):
    lam = LAMS
    root = 0.5 * (lam[1] + lam[2])
    K = 3600.0
    linear = [(l, math.copysign(1.0, l - root), K + math.log(abs(l - root) / 1e-13)) for l in lam]
    const = [(l, -1.0, K + 1.5e-11 * i) for i, l in enumerate(lam)]
    moved = [(l, s, la + (1e-5 if i == 3 else 0.0)) for i, (l, s, la) in enumerate(linear)]
    jagged = [(l, s, la) for l, (s, la) in zip(lam, JAGGED)]
    two = [(l, s, K) for l, s in zip(lam, (1.0, -1.0, -1.0, 1.0))]
    cases = [("a line with its root between the middle rates", linear, True), ("a constant", const, True),
             ("#1078's double-double rows", jagged, False), ("a line with one ln|det| moved by 1e-5", moved, False),
             ("two roots' signs", two, False)]
    ok = True
    for name, rows, want in cases:
        c = linear_check(rows)
        good = c["ok"] == want
        ok &= good
        log(f"selftest: {name}: worst |dev| {c['worst']:.2e}, {'passes' if c['ok'] else 'fails'} "
            f"(must {'pass' if want else 'fail'}): {'OK' if good else 'WRONG'}")
    log(f"selftest: {'PASS' if ok else 'FAIL'}" + (" (mutate accept_any)" if MUTATE["accept_any"] else ""))
    return 0 if ok else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if o.get("mutate") == "accept_any":
        MUTATE["accept_any"] = True
    lams = [float(v) for v in o["lams"].split(",")] if "lams" in o else LAMS
    if args and args[0] == "selftest":
        return selftest()
    if len(args) == 2 and args[0] == "verdict":
        return verdict(args[1], lams)
    if len(args) != 2:
        print(__doc__)
        return 2
    from mode_qd import det_rows
    from raw_out import require_raw
    outdir, state = args
    os.makedirs(outdir, exist_ok=True)
    require_raw(os.path.join(outdir, "qd.jsonl"))
    log = lambda s: print(s, flush=True)
    kinds = ("qd",) if "skip_dd" in o else ("dd", "qd")
    for e in kinds:
        path = os.path.join(outdir, f"{e}.jsonl")
        done = {r["lam"] for r in read_rows(path)}
        todo = [l for l in lams if l not in done]
        log(f"{e} entries: {len(lams) - len(todo)} of {len(lams)} rates done, {len(todo)} to run")
        if todo:
            det_rows(path, state, todo, entries=e, threads=o.get("threads", 6), hmax=o.get("hmax", 0.05), log=log)
    return verdict(outdir, lams)


if __name__ == "__main__":
    sys.exit(main())
