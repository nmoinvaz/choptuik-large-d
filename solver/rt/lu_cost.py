"""Cost of a dense multiprecision LU solve on this machine (python-flint arb_mat.solve, algorithm "approx", one
thread), for the cost estimate of notes/RT_FORMULATION.md: seconds / L^3 at each precision.

    python3 solver/rt/lu_cost.py OUT.jsonl [--sizes 150,300] [--precs 128,256,512,1024,2048]

Control: the solve of a random well-conditioned system is checked against its planted solution (relative error below
2^(-prec/2)); a failure exits 1. --mutate=perturb solves with one entry of A changed by 1e-10 and must exit 1.
"""
import sys
import json
import time
import numpy as np
import flint


def main():
    out = sys.argv[1]
    sizes = [150, 300]
    precs = [128, 256, 512, 1024, 2048]
    for a in sys.argv[2:]:
        if a.startswith("--sizes"):
            sizes = [int(v) for v in a.split("=")[1].split(",")]
        if a.startswith("--precs"):
            precs = [int(v) for v in a.split("=")[1].split(",")]
    rng = np.random.default_rng(5)
    bad = False
    for L in sizes:
        A0 = rng.normal(size=(L, L)) + L ** 0.5 * np.eye(L)
        x0 = rng.normal(size=L)
        for P in precs:
            flint.ctx.prec = P
            A = flint.arb_mat(L, L, [float(v) for v in A0.ravel()])
            xs = flint.arb_mat(L, 1, [float(v) for v in x0])
            b = A * xs
            t0 = time.time()
            if "--mutate=perturb" in sys.argv:
                A[0, 0] = A[0, 0] + flint.arb(1e-10)
            x = A.solve(b, algorithm="approx")
            dt = time.time() - t0
            err = max(abs(float((x[i, 0] - xs[i, 0]).mid())) for i in range(L)) / max(abs(x0))
            ok = err < 2.0 ** (-P / 2)
            bad |= not ok
            rec = dict(L=L, prec=P, seconds=dt, per_L3=dt / L ** 3, err=float(err), ok=bool(ok))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"L {L} prec {P}: {dt:.2f} s, {dt / L ** 3:.2e} s per L^3, error {err:.1e} {'ok' if ok else 'FAIL'}",
                  flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
