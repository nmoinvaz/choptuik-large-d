"""Half-period symmetry check for the sigma-model DSS solution.

The solver imposes p, m, psi odd and a, c even under tau -> tau + Delta/2 (u -> -u after half a period). Here the
symmetric solution is transferred to the full layout (every field carries all harmonics 0..kc, no symmetry imposed):
 1. its residual there (must be the same, ~1e-15);
 2. the forbidden harmonics are perturbed (relative size `pert`) and Newton is run in the full layout: if the
    symmetric solution is an isolated solution of the unrestricted problem, they return to zero;
 3. real growth rates of the full mode matrix in [lo, hi] (both parity sectors) against those of the symmetric one
    (same-parity sector only): an extra root would be an unstable mode that breaks the symmetry.
usage: python sigdss_parity.py SOL D eps kc [--pert 1e-3] [--lo 0.3] [--hi 12] [--ngrid 13] [--kappa 0.7]
Appends to data/sigdss_parity.jsonl."""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigdss_solver import SigDSS, NODES_STD, lm, real_roots    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sol"); ap.add_argument("D", type=float); ap.add_argument("eps", type=float); ap.add_argument("kc", type=int)
ap.add_argument("--pert", type=float, default=1e-3); ap.add_argument("--lo", type=float, default=0.3)
ap.add_argument("--hi", type=float, default=12.0); ap.add_argument("--ngrid", type=int, default=13)
ap.add_argument("--kappa", type=float, default=0.7); ap.add_argument("--Ks", type=int, default=16)
ap.add_argument("--u0", type=float, default=0.01); ap.add_argument("--no-roots", action="store_true")
a = ap.parse_args()

z = np.load(a.sol)
kw = dict(kc=a.kc, nodes=NODES_STD, n=a.D - 2, eps=a.eps, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
Ss = SigDSS(**kw); Ss.freeze_from(z, a.kappa)
Sf = SigDSS(full=True, **kw); Sf.steps_map = dict(Ss.steps_map)
zf = Sf.to_full(z, Ss)
Fs = float(np.linalg.norm(Ss.residual(z[None, :])[0]))
Ff = float(np.linalg.norm(Sf.residual(zf[None, :])[0]))
bad0, good0 = Sf.wrong_parity_norm(zf)
print(f"symmetric |F| {Fs:.3e}; full layout |F| {Ff:.3e}; forbidden harmonics {bad0:.2e} (allowed {good0:.3e}); "
      f"unknowns {Ss.layout()[2]} -> {Sf.layout()[2]}", flush=True)
rec = dict(sol=os.path.basename(a.sol), D=a.D, eps=a.eps, kc=a.kc, Delta=float(z[-1]), F_sym=Fs, F_full=Ff,
           n_sym=Ss.layout()[2], n_full=Sf.layout()[2])
# 2. perturb the forbidden harmonics and re-solve without symmetry
rng = np.random.default_rng(1)
parts, _ = Sf.split(zf[None, :])
kc = a.kc
k = np.concatenate([[0], np.arange(1, kc + 1), np.arange(1, kc + 1)])
sym = ["odd", "even"] + ["odd", "odd", "even", "even", "odd"] * len(NODES_STD) + ["odd", "even", "odd"]
zp = []
for c, par in zip(parts, sym):
    c = c[0].copy()
    wrong = (k % 2 == 0) if par == "odd" else (k % 2 == 1)
    scale = a.pert * max(np.max(np.abs(c)), 1e-3)
    c[wrong] += scale * rng.standard_normal(int(wrong.sum()))
    zp.append(c)
zp = np.concatenate(zp + [zf[-1:]])
zp[Sf.phase_index] = 0.0
bad1, _ = Sf.wrong_parity_norm(zp)
Fp = float(np.linalg.norm(Sf.residual(zp[None, :])[0]))
print(f"perturbed: forbidden harmonics {bad1:.2e}, |F| {Fp:.3e}", flush=True)
t0 = time.time()
zs, nF, _ = lm(Sf, zp, tol=1e-12, maxjac=10, verbose=True)
bad2, good2 = Sf.wrong_parity_norm(zs)
print(f"re-solved without symmetry: |F| {nF:.3e}, forbidden harmonics {bad2:.2e}, Delta {zs[-1]:.12f} "
      f"(symmetric {z[-1]:.12f}) ({time.time() - t0:.0f} s)", flush=True)
rec.update(pert=a.pert, forbidden_before=bad1, F_perturbed=Fp, F_resolved=float(nF), forbidden_after=bad2,
           Delta_resolved=float(zs[-1]), solve_seconds=round(time.time() - t0))
# 3. real growth rates in both sectors
if not a.no_roots:
    t0 = time.time()
    rs, ns = real_roots(Ss, z, a.lo, a.hi, a.ngrid)
    rf, nf = real_roots(Sf, zf, a.lo, a.hi, a.ngrid)
    print(f"real roots in [{a.lo}, {a.hi}]: same-parity sector {rs}; full (both sectors) {rf} "
          f"({time.time() - t0:.0f} s)", flush=True)
    rec.update(lo=a.lo, hi=a.hi, ngrid=a.ngrid, roots_sym=rs, roots_full=rf, roots_seconds=round(time.time() - t0))
print(json.dumps(rec), flush=True)
open(os.path.join(HERE, "data", "sigdss_parity.jsonl"), "a").write(json.dumps(rec) + "\n")
