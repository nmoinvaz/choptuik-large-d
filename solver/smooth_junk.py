"""Fix the junk (near-null) directions by smoothness: choose their coordinates so that the high harmonics of the
centre data are as small as possible (the physical solution's centre spectrum decays; the junk coordinates are
otherwise left at arbitrary values ~1e-2 that F cannot see but Delta can, at the 1e-5 level)."""
import numpy as np


def centre_high_index(S, kmin=9):
    sizes, K, L = S.layout()
    h = sizes[0] // 2                       # centre p: cos / sin of odd k = 1, 3, ..., (2h-1)
    ks = 2 * np.arange(h) + 1
    sel = np.where(ks >= kmin)[0]
    return np.r_[sel, h + sel]


def smooth_junk(R, S, z, cut_junk=3e-14, kmin=9, J=None):
    if J is None:
        _, J = R.jacobian(z)
    U, s, Vt = np.linalg.svd(J)
    Vj = Vt[s / s[0] < cut_junk].T
    idx = centre_high_index(S, kmin)
    M = Vj[idx]
    a = -np.linalg.lstsq(M, z[idx], rcond=1e-10)[0]
    return z + Vj @ a, float(np.linalg.norm(z[idx])), float(np.linalg.norm((z + Vj @ a)[idx]))
