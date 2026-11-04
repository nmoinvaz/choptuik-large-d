#!/usr/bin/env python3
"""Checks of the mapped tau grid (hp/tau_map.py; notes/LOW_D_SPEEDUP.md, step 5).

(1) Control: the identity map (L = 1) gives the uniform grid's inputs (driver.dmat_odd, phase_weights) bit for bit,
    and the mapped solver's residual of a stored state equals make_solver's bit for bit.
(2) A known answer: the stored M = 48 state at D = 4 (hp/data/lowD_states/lowD_D4.0000_M48.npz) is under-resolved,
    its Delta 4e-7 above the published 3.445452402(3) (Martin-Garcia and Gundlach 2003; README "Relation to published
    work"). The state is mapped onto grids with L < 1 centred on its centre field's pulse, at the same M = 48, and
    solved there; a map that saves modes puts Delta nearer the published value at the same M. Pass: (1) exact; (2) at
    least one L brings the error below a tenth of the uniform grid's (judged on the printed table, recorded).

    python3 tau_map_check.py [--Ls 0.7,0.5,0.35] [--threads 1] [--out F.json]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from hpkernel.driver import dmat_odd, load, make_solver, newton_solve, phase_weights  # noqa: E402
from tau_map import make_solver_mapped, map_state, mapped_inputs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA4 = 3.4454524022278213   # the double-double value at D 4 (README "Relation to published work"; 3.445452402(3) published)


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="0.7,0.5,0.35")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    zh, zl, meta = load(os.path.join(HERE, "data", "lowD_states", "lowD_D4.0000_M48.npz"))
    M, nodes, K = meta["M"], meta["nodes"], len(meta["nodes"])
    rec = dict(state="lowD_D4.0000_M48.npz", M=M)
    # (1) the control
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    mh, ml, qh, ql = mapped_inputs(M, 1.83, 1)
    diff = int(np.sum(dh != mh) + np.sum(dl != ml) + np.sum(ph != qh) + np.sum(pl != ql))
    su = make_solver(meta["D"], M, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=a.threads)
    segs = su.plan(zh, zl, 1e-32, 0.05)
    su.set_segs(segs)
    sm = make_solver_mapped(meta["D"], M, 1.83, 1, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], segs=segs, threads=a.threads)
    rh, rl = su.residual(zh, zl)
    qh2, ql2 = sm.residual(zh, zl)
    rdiff = int(np.sum(np.asarray(rh) != np.asarray(qh2)) + np.sum(np.asarray(rl) != np.asarray(ql2)))
    print(f"(1) identity map: {diff} differing input words, {rdiff} differing residual words "
          f"({'PASS' if diff == 0 and rdiff == 0 else 'FAIL'})", flush=True)
    rec.update(control_inputs=diff, control_residual=rdiff)
    # the uniform grid's own answer at this M
    hu, lu, hist = newton_solve(su, zh, zl, tol=1e-28, it_max=12, verbose=False, jac="f64")
    Du = float(hu[-1]) + float(lu[-1])
    # the centre field's pulse, mod pi, on a fine grid
    z = np.asarray(hu) + np.asarray(lu)
    full = np.r_[z[:M], -z[:M]]
    c = np.fft.rfft(full) / (2 * M)
    th = np.linspace(0, 2 * np.pi, 8 * 2 * M, endpoint=False)
    k = np.arange(len(c)); w = np.where((k == 0) | (k == M), 1.0, 2.0)
    p1 = (w[None, :] * c[None, :] * np.exp(1j * np.outer(th, k))).real.sum(1)
    s0 = float(th[np.argmax(np.abs(p1))] % np.pi)
    print(f"uniform M {M}: Delta {Du:.12f}, error {Du - DELTA4:+.2e}; |F| {fmax(su, hu, lu):.1e}; centre pulse at s0 = {s0:.3f}", flush=True)
    rows = [dict(L=1.0, Delta=Du, err=Du - DELTA4, F=fmax(su, hu, lu))]
    for L in [float(v) for v in a.Ls.split(",")]:
        zm, zml = map_state(hu, lu, M, M, K, None, (s0, L))
        sm = make_solver_mapped(meta["D"], M, s0, L, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=a.threads)
        sm.set_segs(sm.plan(zm, zml, 1e-32, 0.05))
        F0 = fmax(sm, zm, zml)
        hm, lm, hist = newton_solve(sm, zm, zml, tol=1e-28, it_max=12, verbose=False, jac="f64")
        Dm = float(hm[-1]) + float(lm[-1])
        rows.append(dict(L=L, Delta=Dm, err=Dm - DELTA4, F=fmax(sm, hm, lm), F_start=F0, calls=len(hist)))
        print(f"mapped L {L}: start |F| {F0:.1e}, {len(hist)} Newton calls, |F| {rows[-1]['F']:.1e}, Delta {Dm:.12f}, "
              f"error {Dm - DELTA4:+.2e} (uniform {Du - DELTA4:+.2e})", flush=True)
    best = min(rows[1:], key=lambda r: abs(r["err"]))
    ok2 = abs(best["err"]) < 0.1 * abs(rows[0]["err"])
    print(f"(2) best L {best['L']}: error {best['err']:+.2e} against the uniform {rows[0]['err']:+.2e} "
          f"({'below' if ok2 else 'NOT below'} a tenth)", flush=True)
    rec.update(s0=s0, rows=rows, pass_control=diff == 0 and rdiff == 0, pass_known=ok2)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    sys.exit(0 if (diff == 0 and rdiff == 0) else 1)


if __name__ == "__main__":
    main()
