#!/usr/bin/env python3
"""Conditioning of the f64 problem at low D (notes/LOW_D.md): the singular values of the exact Jacobian (Rust kernel)
at a converged f64 solution, how many fall in the ne_solve "weak" band (3e-14 <= s/s_max < 1e-10) and "junk" band
(< 3e-14), and for the weakest directions their Delta component and the block, field and tau-harmonic that dominate
the right singular vector.

    python3 lowD_svd.py STATE.npy [STATE.npy ...] [--rk=2] [--threads=2] [--n=24] [--json=out.jsonl]

STATE: solver/data/neB_kc{kc}_rk{rk}_D{D}.npy (x_K = 0.6 nodes). kc 55 / 63 need the rustkernel built from this tree
(the 96-lane size, cf79fba). Also prints the residual of the state in this layout.
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rust_solver import make  # noqa: E402

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def labels(S):
    """(block, field, harmonic) of every unknown in the layout"""
    sizes, K, L = S.layout()
    names = ["p1(centre)", "c0(centre)"] + [f"{f}@x{x}" for x in S.nodes for f in ("p", "m", "a", "c")] + ["mS(horizon)", "aS(horizon)"]
    pars = S.parities()
    out = []
    for b, (n, nm, par) in enumerate(zip(sizes, names, pars)):
        if par == "odd":
            ks = list(S.kodd) * 2
        else:
            ks = [0] + list(S.keven[1:]) * 2
        out += [(b, nm, int(k)) for k in ks]
    out.append((len(sizes), "Delta", 0))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    rk = int(opts.get("rk", 2))
    threads = int(opts.get("threads", 2))
    nshow = int(opts.get("n", 24))
    for f in args:
        m = re.search(r"kc(\d+)", os.path.basename(f))
        kc = int(m.group(1)) if m else 23
        mr = re.search(r"_rk(\d+)_", os.path.basename(f))
        rk_f = int(mr.group(1)) if mr else 1
        D = float(re.findall(r"D([0-9.]+)\.npy", f)[0])
        z = np.load(f)
        t0 = time.time()
        S, R = make(D, z[-1], nodes=NL, threads=threads, kc=kc, rk=rk)
        F, J = R.jacobian(z)
        U, s, Vt = np.linalg.svd(J)
        r = s / s[0]
        lab = labels(S)
        nweak = int(((r < 1e-10) & (r >= 3e-14)).sum())
        njunk = int((r < 3e-14).sum())
        n16 = int((r < 1e-16).sum())
        # dDelta/dF = row of J^-1 for Delta: y = sum_i V[-1, i] u_i / s_i (all directions; and with the junk band
        # (s/s_max < 3e-14) dropped, as ne_solve effectively does)
        cvec = Vt[:, -1] / s
        y_all = U @ cvec
        y_cut = U[:, r >= 3e-14] @ cvec[r >= 3e-14]
        sens_all, sens_cut = float(np.abs(y_all).sum()), float(np.abs(y_cut).sum())
        print(f"{os.path.basename(f)} (rk of state {rk_f}, Jacobian at rk {rk}): D {D} kc {kc} L {len(z)} |F| {np.abs(F).max():.1e} "
              f"s_max {s[0]:.2e} s_min/s_max {r[-1]:.1e}  weak band {nweak}  junk band {njunk}  below 1e-16 {n16}  "
              f"|dDelta/dF|_1 {sens_all:.1e} (junk dropped {sens_cut:.1e})  ({time.time() - t0:.0f} s)", flush=True)
        rows = []
        for i in range(len(s) - 1, max(len(s) - 1 - nshow, -1), -1):
            v = Vt[i]
            j = int(np.argmax(np.abs(v)))
            # harmonic content of the vector: the weighted mean harmonic over its unknowns
            w = v ** 2
            kbar = float(sum(w[t] * lab[t][2] for t in range(len(v))) / w.sum())
            rows.append(dict(r=float(r[i]), delta=float(abs(v[-1])), top=lab[j][1], k_top=lab[j][2], kbar=round(kbar, 1)))
            print(f"   s/s_max {r[i]:.2e}  |Delta comp| {abs(v[-1]):.1e}  largest entry {lab[j][1]} k={lab[j][2]} ({abs(v[j]):.2f})  mean k {kbar:.1f}")
        if "json" in opts:
            with open(opts["json"], "a") as fh:
                fh.write(json.dumps(dict(state=os.path.basename(f), D=D, kc=kc, rk_jac=rk, L=len(z), F=float(np.abs(F).max()),
                                         s_max=float(s[0]), n_weak=nweak, n_junk=njunk, n_below_1e16=n16,
                                         sens_all=sens_all, sens_junk_dropped=sens_cut,
                                         sv_ladder=[float(f"{x:.3e}") for x in r[-60:]], weakest=rows)) + "\n")


if __name__ == "__main__":
    main()
