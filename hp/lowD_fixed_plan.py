#!/usr/bin/env python3
"""The low-D predictors on fixed plans (#927, notes/LOW_D_ANM.md 4): from a stored Taylor series in D (z_0..z_N at a
converged base, hp/lowD_plan_probe.py's coeffs.npz) the partial sums P_n(D) = sum_{k<=n} z_k h^k, and from stored
converged states lowD_highM.py's Lagrange predictor (hpkernel.driver.predict), each evaluated at a target D on the
plans named, without planning from the predicted state (anm_run.py's rule: a prediction is evaluated and corrected on
the base's plan, notes/QUAD_DOUBLE.md 13.16). Where a converged state exists at the target, the predictions are
compared with it. No Jacobian is formed: every number is a residual evaluation or a sum.

    python3 lowD_fixed_plan.py RAWDIR OUTDIR --coeffs I798-win/probe/coeffs.npz
        --targets 3.04335,3.04315 [--orders 0,1,2,3] [--plans base,I343-win2/states/lowD343_D3.043350_M384.npz]
        [--refs I343-win2/states/lowD343_D3.043350_M384.npz] [--hist A.npz,B.npz,C.npz] [--threads 4]
        [--mutate none|drop2]

RAWDIR is a clone of the raw-data repository (every path is relative to it); OUTDIR lies in a task folder of it
(raw_out.py). --plans: "base" is the stored plan of the series' base state; any other entry is a state whose stored
plan is used. --hist: converged states at the same M and nodes, oldest first; at each target the Lagrange predictor
through the last <= 4 of them whose D lies above the target (the leg's history at that point). --refs: converged states
at targets, for max |P - ref| and Delta(P) - Delta(ref). Writes OUTDIR/fixed_plan.jsonl, one record per case.
The control --mutate drop2 evaluates the partial sums without z_2: the agreement of P_n (n >= 2) with a reference must
then fail by the size of z_2 h^2.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from anm_lowd import evaluate, fmax, solver_for  # noqa: E402
from hpkernel.driver import load, predict  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--orders", default="0,1,2,3")
    ap.add_argument("--plans", default="base")
    ap.add_argument("--refs", default="")
    ap.add_argument("--hist", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mutate", choices=("none", "drop2"), default="none")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    path = os.path.join(a.out, "fixed_plan.jsonl")
    require_raw(path)
    os.makedirs(a.out, exist_ok=True)
    jf = open(path, "a")

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    c = np.load(os.path.join(a.raw, a.coeffs))
    D0, base = float(c["D0"]), str(c["state"])
    Zh, Zl = [np.asarray(z) for z in c["Zh"]], [np.asarray(z) for z in c["Zl"]]
    if a.mutate == "drop2":
        Zh[2], Zl[2] = np.zeros_like(Zh[2]), np.zeros_like(Zl[2])
    bh, bl, bm = load(os.path.join(a.raw, base))
    if not (np.array_equal(bh, Zh[0]) and np.array_equal(bl, Zl[0])):
        raise SystemExit(f"z_0 of {a.coeffs} is not the state {base}")
    plans = {}
    for p in [p for p in a.plans.split(",") if p]:
        m = bm if p == "base" else load(os.path.join(a.raw, p))[2]
        if m["M"] != bm["M"] or list(m["nodes"]) != list(bm["nodes"]) or (m["kc"], m["ks"]) != (bm["kc"], bm["ks"]):
            raise SystemExit(f"{p}: M, nodes or series orders differ from the base's")
        plans[p] = m["segs"]
    refs = {}
    for p in [p for p in a.refs.split(",") if p]:
        h, l, m = load(os.path.join(a.raw, p))
        refs[round(float(m["D"]), 9)] = (p, h, l)
    hist = []
    for p in [p for p in a.hist.split(",") if p]:
        h, l, m = load(os.path.join(a.raw, p))
        hist.append((float(m["D"]), p, np.asarray(h) + np.asarray(l)))
    orders = [int(n) for n in a.orders.split(",")]
    put(stage="start", coeffs=a.coeffs, base=base, D0=D0, M=bm["M"], N=len(Zh) - 1, plans={k: sum(len(g) - 1 for g in v)
        for k, v in plans.items()}, refs=sorted(refs), hist=[(d, p) for d, p, _ in hist], mutate=a.mutate,
        threads=a.threads, norms=[float(np.abs(z).max()) for z in Zh], delta_k=[float(z[-1]) for z in Zh])
    print(f"series at D {D0} ({base}), M {bm['M']}, orders 0..{len(Zh) - 1}, mutate {a.mutate}; plans "
          + ", ".join(f"{k} ({sum(len(g) - 1 for g in v)} steps)" for k, v in plans.items()), flush=True)
    for Dt in [float(x) for x in a.targets.split(",") if x]:
        h = Dt - D0
        cases = [(f"anm{n}", evaluate(Zh[:n + 1], Zl[:n + 1], h)) for n in orders]
        prior = [(d, z) for d, _, z in hist if d > Dt + 1e-12][-4:]
        if prior:
            zp = np.asarray(predict(prior, Dt), float)
            cases.append((f"lagrange{len(prior)}", (zp, np.zeros_like(zp))))
        ref = refs.get(round(Dt, 9))
        for lab, (ph, pl) in cases:
            r = dict(stage="case", D=Dt, h=h, pred=lab)
            if ref is not None:
                r.update(ref=ref[0], max_err=float(np.abs((ph - ref[1]) + (pl - ref[2])).max()),
                         dDelta=float((ph[-1] - ref[1][-1]) + (pl[-1] - ref[2][-1])))
            for pname, segs in plans.items():
                t0 = time.time()
                s = solver_for(bm, Dt, a.threads, segs)
                with np.errstate(all="ignore"):
                    F = fmax(s, ph, pl)
                r[f"F[{pname}]"] = F if np.isfinite(F) else str(F)
                r[f"t[{pname}]"] = round(time.time() - t0, 1)
            put(**r)
            fmt = lambda v: f"{v:.2e}" if isinstance(v, float) else v  # noqa: E731
            print(f"D {Dt} (h {h:+.5f}) {lab}: " + ", ".join(f"|F| {fmt(r[f'F[{p}]'])} on {p.split('/')[-1]}" for p in plans)
                  + (f"; vs ref max {r['max_err']:.2e}, Delta {r['dDelta']:+.2e}" if ref is not None else "")
                  + f" ({sum(r[f't[{p}]'] for p in plans):.0f} s)", flush=True)
    put(stage="end")


if __name__ == "__main__":
    main()
