"""#1025: the conditioning of the l-multiplet DSS solve at a state, and where two states of the same D and k_c differ.

    python ell_cond.py STATE_A STATE_B kc l D [--kappa 0.7] [--Ks 30] [--u0 0.04] [--rk 1] [--k 8] [--out f.json]
                       [--mutate=plant]

Builds the solver at STATE_A as ell_refine.py does (dss_ell.DSSEll, freeze_from at kappa), prints |F| at both states
with that freeze, forms the Jacobian at STATE_A (dss_ell.jacobian, complex step), its k smallest singular values (with
the largest) and right singular vectors, and the components of dz = STATE_B - STATE_A along them, the part of dz
outside them, and dz's Delta entry. --mutate=plant replaces STATE_B by STATE_A + 1e-6 v_min (v_min the smallest right
singular vector): the components must read 1e-6 on v_min and nothing elsewhere, or the check is broken (exit 1).
Read-only: it writes nothing but --out."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("kc", type=int); ap.add_argument("l", type=int)
    ap.add_argument("D", type=float)
    ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--Ks", type=int, default=30)
    ap.add_argument("--u0", type=float, default=0.04); ap.add_argument("--rk", type=int, default=1)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--out"); ap.add_argument("--mutate")
    a = ap.parse_args()
    t0 = time.time()
    zA, zB = np.load(a.A), np.load(a.B)
    assert zA.shape == zB.shape, "the two states must have the same layout (same k_c)"
    S = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=a.l, Delta_ref=zA[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
    steps = S.freeze_from(zA, a.kappa)
    FA = float(np.linalg.norm(S.residual(zA[None, :])[0]))
    F, J = S.jacobian(zA)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    k = min(a.k, len(s))
    V = Vt[-k:][::-1]                       # the k smallest, smallest first
    sk = s[-k:][::-1]
    if a.mutate == "plant":
        zB = zA + 1e-6 * V[0]
    FB = float(np.linalg.norm(S.residual(zB[None, :])[0]))
    dz = zB - zA
    comp = V @ dz
    rest = float(np.linalg.norm(dz - V.T @ comp))
    lin = float(np.linalg.norm(J @ dz))
    dF = float(np.linalg.norm(S.residual(zB[None, :])[0] - S.residual(zA[None, :])[0]))
    print(f"states: {a.A} (A), {a.B} (B){' [mutation plant: B = A + 1e-6 v_min]' if a.mutate == 'plant' else ''}")
    print(f"layout {len(zA)} unknowns, J {J.shape}, frozen steps {int(sum(steps.values()))}; |F(A)| {FA:.4e}, "
          f"|F(B)| {FB:.4e} (A's freeze)")
    print(f"singular values: largest {s[0]:.4e}; smallest {', '.join(f'{x:.4e}' for x in sk)}; "
          f"condition {s[0] / s[-1]:.3e}")
    print(f"dz = B - A: |dz| {np.linalg.norm(dz):.4e}, Delta entry {dz[-1]:+.4e}")
    for i in range(k):
        print(f"   along v_{i} (sigma {sk[i]:.3e}): {comp[i]:+.4e}  ({abs(comp[i]) / max(np.linalg.norm(dz), 1e-300):.3f} of |dz|);"
              f" v_{i}'s Delta entry {V[i][-1]:+.3e}")
    print(f"   outside these {k}: {rest:.4e} ({rest / max(np.linalg.norm(dz), 1e-300):.3f} of |dz|)")
    print(f"|J dz| {lin:.4e} against |F(B) - F(A)| {dF:.4e} (the linearization)")
    ok = True
    if a.mutate == "plant":
        ok = abs(comp[0] - 1e-6) < 1e-9 and np.all(np.abs(comp[1:]) < 1e-9) and rest < 1e-9
        print(f"plant check: {'PASS' if ok else 'FAIL'} (v_min component {comp[0]:.6e}, others max "
              f"{np.max(np.abs(comp[1:])) if k > 1 else 0:.1e}, outside {rest:.1e})")
    if a.out:
        json.dump(dict(A=a.A, B=a.B, kc=a.kc, l=a.l, D=a.D, kappa=a.kappa, Ks=a.Ks, u0=a.u0, rk=a.rk,
                       n=len(zA), FA=FA, FB=FB, smax=float(s[0]), smin=[float(x) for x in sk],
                       dz=float(np.linalg.norm(dz)), dDelta=float(dz[-1]), comp=[float(x) for x in comp], rest=rest,
                       vDelta=[float(v[-1]) for v in V], lin=lin, dF=dF, mutate=a.mutate,
                       seconds=round(time.time() - t0)), open(a.out, "w"), indent=1)
    print(f"{time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
