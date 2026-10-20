"""2-D grid of the M = 1 Galerkin inner Stokes constant S_+(A, omega) (mode with Im gamma < 0, normalized by its
eigenvector's harmonic-1 component), to locate zeros by the winding of S_+ around each grid cell.
    python3 inner_grid.py A_lo A_hi nA wf_lo wf_hi nw out.txt [Mmax J]
"""
import sys
import mpmath as mp
from inner_scan import stokes
from inner_stokes import W0
a0, a1, na, w0, w1, nw, out = sys.argv[1:8]
Mmax = int(sys.argv[8]) if len(sys.argv) > 8 else 1
J = int(sys.argv[9]) if len(sys.argv) > 9 else 200
a0, a1, w0, w1 = map(mp.mpf, (a0, a1, w0, w1)); na, nw = int(na), int(nw)
with open(out, "w") as fh:
    for i in range(na):
        A = a0 + (a1 - a0) * i / (na - 1)
        for k in range(nw):
            wf = w0 + (w1 - w0) * k / (nw - 1)
            gam, S = stokes(A, wf * W0, Mmax, J, mus=(1,))
            fh.write(f"{float(A)} {float(wf)} " + " ".join(f"{float(mp.re(g))} {float(mp.im(g))} {mp.nstr(mp.re(s), 17)} {mp.nstr(mp.im(s), 17)}"
                                                       for g, s in zip(gam, S[0])) + "\n")
            fh.flush()
