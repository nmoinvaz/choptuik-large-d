"""Check of the d/dtheta matrices' f64 remainders (Solver.set_dmat_corr, HP_EXACT_DMAT, #1098) on a saved state.

The kernel's quad- and octo-double lanes apply D_dd + c, c = D - D_dd from 80-digit values. hp/dmat_perturb.py moves
the double-double matrices to D_dd - S c instead, through their low parts and with no kernel change. To first order
the octo-double residual then moves by -S times what the remainders move it by, so the two mechanisms check each other:

    F(D_dd + c) - F(D_dd)  =  -(F(D_dd - S c) - F(D_dd)) / S  (1 + O(1/S)).

PASS when the largest difference is at most --rel (1e-3) of the remainders' change, which must itself be above
--floor (1e-45). The control applies 2c and must fail by about a factor 2.

    python3 hp/dmat_check.py STATE [--S 65536] [--rel 1e-3] [--floor 1e-45] [--kfreeze 13]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def opts(argv):
    o = dict(S="65536", rel="1e-3", floor="1e-45", kfreeze="13", resid="", corr_scale="1", perturb="0")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    return pos, o


def resid(path, o):
    """one residual, saved as its eight parts (run in its own process: dmat_perturb patches the matrix functions)"""
    if o["perturb"] == "1":
        import dmat_perturb  # noqa: F401
    import form_s
    from mpmath import mp
    from octo_corrector import load_o
    z8, meta = load_o(path)
    with mp.workdps(60):
        D = mp.mpf(meta["D_str"])
    kw = dict(kt=int(meta.get("kt", 36)), kc=int(meta["kc"]), ks=int(meta["ks"]), threads=2, segs=meta["segs"],
              mob=float(meta.get("hmob", 0.0) or 0.0))
    s = form_s.make_solver_s(D, int(meta["M"]), list(meta["nodes"]), kfreeze=int(o["kfreeze"]), **kw)
    if form_s.exact_dmat() and float(o["corr_scale"]) != 1.0:
        co, ce = form_s.dmat_corr(int(meta["M"]))
        k = float(o["corr_scale"])
        s.set_dmat_corr(np.ascontiguousarray(k * co), np.ascontiguousarray(k * ce))
    F = s.residual_s([np.ascontiguousarray(z8[q]) for q in range(8)])
    np.save(o["resid"], np.array([np.asarray(f) for f in F]))


def run(path, td, name, env, extra):
    out = os.path.join(td, name + ".npy")
    subprocess.run([sys.executable, os.path.abspath(__file__), path, "--resid", out] + extra,
                   env=dict(os.environ, **env), check=True)
    return np.load(out)


def main():
    pos, o = opts(sys.argv[1:])
    path = pos[0]
    if o["resid"]:
        resid(path, o)
        return 0
    from mpmath import mp
    S = float(o["S"])
    kf = ["--kfreeze", o["kfreeze"]]
    with tempfile.TemporaryDirectory() as td:
        F0 = run(path, td, "dd", dict(HP_EXACT_DMAT="0"), kf)
        F1 = run(path, td, "corr", dict(HP_EXACT_DMAT="1"), kf)
        FS = run(path, td, "moved", dict(HP_EXACT_DMAT="0", HP_DMAT_SCALE=o["S"]), kf + ["--perturb", "1"])
        F2 = run(path, td, "corr2", dict(HP_EXACT_DMAT="1"), kf + ["--corr_scale", "2"])
    mp.dps = 160

    def val(F, i):
        return mp.fsum(mp.mpf(float(F[q][i])) for q in range(F.shape[0]))

    n = F0.shape[1]
    d1 = [val(F1, i) - val(F0, i) for i in range(n)]
    dS = [-(val(FS, i) - val(F0, i)) / S for i in range(n)]
    d2 = [val(F2, i) - val(F0, i) for i in range(n)]
    big = max(abs(v) for v in d1)
    err = max(abs(a - b) for a, b in zip(d1, dS))
    err2 = max(abs(a - b) for a, b in zip(d2, dS))
    print(f"max |F| without the remainders {mp.nstr(max(abs(val(F0, i)) for i in range(n)), 4)}; the remainders move "
          f"F by {mp.nstr(big, 4)}; against -(F(D_dd - S c) - F)/S, S = {S:g}: {mp.nstr(err, 4)} "
          f"({mp.nstr(err / big, 3)} of it); the control (2c) {mp.nstr(err2, 4)} ({mp.nstr(err2 / big, 3)})", flush=True)
    ok = big > float(o["floor"]) and err <= float(o["rel"]) * big and err2 > float(o["rel"]) * big
    print(f"  {'PASS' if ok else 'FAIL'}: the two mechanisms agree to {mp.nstr(err / big, 3)} (bar {o['rel']}); the "
          f"control is off by {mp.nstr(err2 / big, 3)} (must exceed the bar)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
