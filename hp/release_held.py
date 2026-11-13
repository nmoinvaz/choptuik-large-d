#!/usr/bin/env python3
"""Release the held harmonics: re-converge a chain state with p1's harmonics 5-13 free, the full system (kfreeze 13),
in quad-double at the state's own D, and set the full Delta against the held state's (#640). Above D 73 the chain holds
those harmonics on a 1/n fit of the full solutions (--held_fit, notes/HELD_HARMONICS.md), because the full system
stopped converging there; the exact freeze projector (#523) and better starts may change that, and where the release
converges, Delta_full - Delta_held is the fitted method's own error at that D.

    cd hp && python3 release_held.py OUT.jsonl STATE.npz [--kfreeze=13] [--threads=N] [--tol=1e-50]
        [--tol_accept=1e-44] [--it_max=12] [--perturb=0] [--held_ks=5,7,9,11,13] [--allow_unraw]

The state is converged as scan_qd.py converges a start: make_solver at the state's D (its D_str when present), M,
nodes, kt, kc and ks from its metadata, the Taylor steps planned from the state (Solver.plan, tol 1e-32, hmax 0.05),
and driver.newton_qd. It is converged when max|F| <= --tol, or when Newton stalls with max|F| <= --tol_accept. The
record holds D, the start's Delta (all its parts), the full system's |F| at the start, the converged Delta and |F|,
their difference, the Newton steps, whether it converged, and p1's coefficients of --held_ks at the start and at the
end (pinning.band_coeffs). The Taylor steps are planned afresh, so the discrete system differs from the one the state
was converged in. At T81's full state at D 73.0 the release without perturbation moves Delta by -8.4e-28 with a kernel
from before #523 (the rounded freeze projector T81 was converged with, on macOS and Windows alike) and by -2.8e-23 with
one from #523 on (the exact projector, where the state starts at |F| 2.6e-34). That move is the floor of any comparison
with the state's own Delta.

--perturb=EPS is the planted control: before the release, the cos and sin coefficients of p1's harmonics --held_ks
are moved by EPS each (pinning.band_field), as a wrong fit would put them. From a full-system state, the release from
the perturbed start must reach the release from the unperturbed one: Delta within 1e-25 and the coefficients within
1e-20. --it_max=0 (no Newton step, nothing released) leaves the coefficients EPS off, and that comparison must fail.
Exit 1 when the release does not converge.
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import load_q, make_solver, newton_qd, save_q  # noqa: E402
from pinning import band_coeffs, band_field, dd_add  # noqa: E402
from raw_out import require_raw  # noqa: E402

mp.mp.dps = 80


def delta_of(z):
    return sum((mp.mpf(float(p[-1])) for p in z), mp.mpf(0))


def max_f(s, z):
    F = s.residual_q(*z)
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if len(args) != 2:
        sys.exit(__doc__)
    out, path = args
    require_raw(out, allow="allow_unraw" in opts)
    kf = int(opts.get("kfreeze", 13))
    tol, tol_acc = float(opts.get("tol", 1e-50)), float(opts.get("tol_accept", 1e-44))
    eps = float(opts.get("perturb", 0))
    ks = [int(k) for k in opts.get("held_ks", "5,7,9,11,13").split(",")]
    z, m = load_q(path)
    z = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z]
    M = int(m["M"])
    D = m.get("D_str", repr(float(m["D"])))
    d0 = delta_of(z)
    if eps:
        bh, bl = band_field([(eps, eps)] * len(ks), ks, M)
        z[0][:M], z[1][:M] = dd_add(z[0][:M], z[1][:M], np.asarray(bh), np.asarray(bl))
    s = make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=kf)
    s.set_segs(s.plan(z[0], z[1], 1e-32, 0.05))
    f0 = max_f(s, z)
    t0 = time.time()
    print(f"{os.path.basename(path)}: D {float(m['D']):.6f}, M {M}, kfreeze {kf}, perturb {eps:g}; full-system |F| at the "
          f"start {f0:.2e}", flush=True)
    it_max = int(opts.get("it_max", 12))
    if it_max > 0:
        z1, hist, conv = newton_qd(s, [np.ascontiguousarray(p) for p in z], tol=tol, it_max=it_max)
    else:
        z1, hist, conv = z, [], False
    f1 = hist[-1].get("F_new", hist[-1]["F"]) if hist else f0
    stalled = not conv and bool(hist) and hist[-1].get("lam") == 0.0
    ok = conv or (stalled and f1 <= tol_acc) or it_max == 0

    def coeffs(zz):
        return [[float(a), float(b)] for a, b in band_coeffs(zz[0][:M], zz[1][:M], M, ks)]
    d1 = delta_of(z1)
    rec = dict(state=path, D=float(m["D"]), D_str=str(D), M=M, kfreeze=kf, perturb=eps, held_ks=ks, F_start=f0,
               F=f1, converged=bool(conv), stalled=bool(stalled), ok=bool(ok), steps=len(hist),
               Delta_start=mp.nstr(d0, 40), Delta_full=mp.nstr(d1, 40), diff=float(d1 - d0),
               coeffs_start=coeffs(z), coeffs_end=coeffs(z1), it_max=it_max,
               seconds=round(time.time() - t0, 1))
    word = "not released (--it_max=0)" if it_max == 0 else "converged" if ok else "NOT converged"
    print(f"  {word}: |F| {f1:.2e} after {len(hist)} steps; Delta_full - Delta_start = "
          f"{rec['diff']:+.3e} ({rec['seconds']} s)", flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if ok:
        save_q(os.path.join(os.path.dirname(os.path.abspath(out)), f"released_kf{kf}_D{float(m['D']):.4f}"
                            + (f"_p{eps:g}" if eps else "") + ".npz"), z1, dict(m, kfreeze=kf, released=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
