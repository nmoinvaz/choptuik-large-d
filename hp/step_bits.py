#!/usr/bin/env python3
"""The octo-double path's bits against the process's settings (#831): reduced_cubic's Real (#823's stored system) is
built in fresh interpreters under several settings, and the bits of its residual at z0, of the Newton step re-solved
through its LU (-J^-1 F(z0)) and of a second LU's step are compared with the first setting's.

    cd hp && python3 step_bits.py RAW DZ0 BASE [--threads=3] [--settings=blas1,blasN,reseed_first,kernel1]

The settings: blas1 (OPENBLAS_NUM_THREADS=1), blasN (unset: OpenBLAS's default, one thread per core), reseed_first
(reseed imported after reduced_cubic and before Real is built, as #828's probe did) and kernel1 (the kernel's own
threads 1, not --threads). Each prints the working precision, the stored-step check (|-J^-1 F(z0) - dz0| / |dz0|) and
the three hashes, the bits of eight-part f64 arrays (sha256, first 16 hex digits). Exit 1 when any setting's hashes
differ from the first's. The octo-double path calls no BLAS: its kernels link only libSystem, and its Python layer no
numpy linear algebra. Before #831, reseed set 50 digits when imported, and reseed_first read the check at 3.6e-35 with
the solve's bits changed (raw I831-mac). --mutate=dps50 sets 50 digits where reseed_first imports reseed, as the old
import did, and must fail.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = r"""
import hashlib, os, sys
sys.path.insert(0, {here!r})
first = {first!r}
import numpy as np
import reduced_cubic as rc
import mpmath
if first:
    import reseed               # noqa: F401  (after reduced_cubic, before Real, as #828's probe did)
    if {dps50!r}:
        mpmath.mp.dps = 50      # the control: what reseed's import did before #831


def h(parts):
    return hashlib.sha256(b"".join(np.ascontiguousarray(p, dtype=float).tobytes() for p in parts)).hexdigest()[:16]


sysm = rc.Real({raw!r}, {dzf!r}, {base!r}, {threads}, lambda s: None, fast=True)
F = sysm.so.residual_o(sysm.z0)
x = sysm.s.olu_solve(list(sysm.parts8(sysm.F())), False)
st = sysm.s.newton_o(list(sysm.z0))
print("BITS", mpmath.mp.dps, "%.3e" % float(sysm.dz0_re), h(F), h(x), h(st[0]), "%.6e" % st[3])
"""


def run(setting, raw, dzf, base, threads, mutate=""):
    env = dict(os.environ)
    first = setting == "reseed_first"
    thr = 1 if setting == "kernel1" else threads
    if setting == "blasN":
        env.pop("OPENBLAS_NUM_THREADS", None)
    else:
        env["OPENBLAS_NUM_THREADS"] = "1"
    code = PROBE.format(here=HERE, first=first, raw=raw, dzf=dzf, base=base, threads=thr, dps50=mutate == "dps50")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, capture_output=True, text=True)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("BITS")), None)
    if line is None:
        return None, (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    _, dps, re_, hF, hx, hd, piv = line.split()
    return dict(dps=int(dps), dz0_re=re_, F=hF, solve=hx, newton=hd, pivot=piv), None


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if len(args) != 3:
        sys.exit(__doc__)
    raw, dzf, base = args
    threads = int(opts.get("threads", 3))
    settings = opts.get("settings", "blas1,blasN,reseed_first,kernel1").split(",")
    ref, ok = None, True
    for st in settings:
        r, err = run(st, raw, dzf, base, threads, opts.get("mutate", ""))
        if r is None:
            print(f"  {st}: not run ({err})")
            ok = False
            continue
        same = ref is None or all(r[k] == ref[k] for k in ("F", "solve", "newton"))
        ref = ref or r
        ok &= same
        print(f"  {st}: dps {r['dps']}, the stored-step check {r['dz0_re']}, F {r['F']}, solve {r['solve']}, second "
              f"LU's step {r['newton']}, smallest pivot {r['pivot']}: {'same bits' if same else 'DIFFERENT bits'}",
              flush=True)
    print(f"step_bits: {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
