"""Delta's error from the double-double storage of form S's spectral d/dtheta (#1098): driver.dmat_odd and
form_s.dmat_even are rounded once to double-double, D_dd = D - e with |e| at most half a unit in the last place of the
low part, a fixed perturbation of the operator that no step plan or series order removes (the kfreeze projector is not
affected: the frozen rows use multishoot's octo-double one, model.rs check_freeze). This runs a form_s.py command line
with each matrix's low part moved to D_dd - S e (e from 80-digit values, S = HP_DMAT_SCALE, default 2^16), so that
Delta moves by -S Delta'(e) to first order: a release_ls --harm 13 from a converged state then gives the double-double
term Delta_dd - Delta_exact = (Delta_S - Delta_dd) / S, good to about 1/S (lo's own rounding after the move). The
linearity control is a second scale, whose change must be in the ratio of the scales.

    HP_DMAT_SCALE=65536 python3 hp/dmat_perturb.py release_ls STATE OUT --harm 13 --kmax 13 ...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpkernel.driver as drv  # noqa: E402
from mpmath import mp  # noqa: E402

import form_s  # noqa: E402

SCALE = float(os.environ.get("HP_DMAT_SCALE", str(2.0 ** 16)))
MOVED = {}


def exact_odd(M):
    N = 2 * M
    out = []
    for i in range(M):
        for j in range(M):
            v = mp.mpf(0)
            for jj, sgn in ((j, 1), (j + M, -1)):
                if jj != i:
                    v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
            out.append(v)
    return out


def exact_even(M):
    N = 2 * M
    out = []
    for i in range(M):
        for j in range(M):
            v = mp.mpf(0)
            for jj in (j, j + M):
                if jj != i:
                    v += mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
            out.append(v)
    return out


def moved(f, exact, name):
    def g(M, *a, **k):
        hi, lo = f(M, *a, **k)
        if (name, M) not in MOVED:
            with mp.workdps(80):
                ex = exact(M)
                e = np.array([float(v - mp.mpf(float(h)) - mp.mpf(float(lw)))
                              for v, h, lw in zip(ex, hi.ravel(), lo.ravel())]).reshape(hi.shape)
            MOVED[name, M] = lo - SCALE * e
            print(f"dmat_perturb: {name}({M}) lo moved to D_dd - {SCALE:g} e, max |e| {np.abs(e).max():.2e}, "
                  f"max |lo change| {np.abs(MOVED[name, M] - lo).max():.2e}", flush=True)
        return hi, MOVED[name, M]
    return g


drv.dmat_odd = moved(drv.dmat_odd, exact_odd, "dmat_odd")
form_s.dmat_even = moved(form_s.dmat_even, exact_even, "dmat_even")

if __name__ == "__main__":
    sys.argv = [form_s.__file__] + sys.argv[1:]
    rc = form_s.main()
    if not MOVED:
        print("dmat_perturb: no matrix was built through the patched functions", flush=True)
        sys.exit(3)
    sys.exit(rc)
