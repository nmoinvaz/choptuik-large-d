"""Form S with a fine tau grid and few unknown harmonics (#1088): products resolved, unknowns truncated.

The collocation layout of fseries_s.py ties the number of unknown harmonics to the grid, so its top harmonic carries
the products' aliasing. Here the series run on a grid of Mg points (fseries_s.FactoredS or fseries_sdd.FactoredSDD at
M = Mg), the unknowns are the Fourier coefficients of p1, Q0 (odd harmonics 1..ju) and c0 (even harmonics 0..ju+1),
and the rows are the matching residuals of U, V (odd harmonics 1..ju) and c (even 0..ju+1), projected, with the phase
row the sin(theta) coefficient of p1. Higher harmonics of the data are zero; those of the fields come from the
nonlinear terms. (hp's pinned layout drops the high p-rows in the same way; ALGORITHM.md section 3.)

Coefficient order: odd field: [a_1, b_1, a_3, b_3, ...] for a_j cos(j th) + b_j sin(j th); even field: [a_0, a_2, b_2,
..., a_{ju+1}, b_{ju+1}] (with the Nyquist sine dropped if ju + 1 = Mg).
"""
import numpy as np


def bases(Mg, ju, mp_dps=None):
    th = np.pi * np.arange(Mg) / Mg
    odd = [j for j in range(1, ju + 1, 2)]
    even = [j for j in range(0, ju + 2, 2)]
    So = []
    names_o = []
    for j in odd:
        So.append(np.cos(j * th))
        So.append(np.sin(j * th))
        names_o += [("cos", j), ("sin", j)]
    Se = []
    names_e = []
    for j in even:
        Se.append(np.cos(j * th))
        names_e.append(("cos", j))
        if j > 0 and j < Mg:
            Se.append(np.sin(j * th))
            names_e.append(("sin", j))
    So = np.array(So).T  # (Mg, No)
    Se = np.array(Se).T
    # analysis: least squares on the grid (exact for these harmonics on the half grid)
    Ao = np.linalg.pinv(So)
    Ae = np.linalg.pinv(Se)
    return So, Se, Ao, Ae, names_o, names_e


class GalerkinS:
    """Wraps a grid residual (fseries_s.FactoredS or its dd twin) with coefficient unknowns and projected rows."""

    def __init__(self, inner, Mg, ju, dd=False):
        self.inner = inner
        self.Mg = Mg
        self.ju = ju
        self.So, self.Se, self.Ao, self.Ae, self.no, self.ne = bases(Mg, ju)
        self.No = self.So.shape[1]
        self.Ne = self.Se.shape[1]
        self.L = 2 * self.No + self.Ne + 1
        self.dd = dd
        if dd:
            import ddnp as d
            from ddnp import DD
            import mpmath as mp
            with mp.workdps(50):
                th = [mp.pi * i / Mg for i in range(Mg)]

                def mat(names, analysis):
                    cols = []
                    for kind, j in names:
                        cols.append([mp.cos(j * t) if kind == "cos" else mp.sin(j * t) for t in th])
                    S = mp.matrix(cols).T  # Mg x N
                    if analysis:
                        A = (S.T * S) ** -1 * S.T
                        return d.from_mp([A[a, b] for a in range(A.rows) for b in range(A.cols)], (A.rows, A.cols))
                    return d.from_mp([S[a, b] for a in range(S.rows) for b in range(S.cols)], (S.rows, S.cols))
                self.So_dd = mat(self.no, False)
                self.Se_dd = mat(self.ne, False)
                self.Ao_dd = mat(self.no, True)
                self.Ae_dd = mat(self.ne, True)
            self.d = d
            self.DD = DD

    def to_grid(self, yc):
        No, Ne = self.No, self.Ne
        p1c, c0c, Q0c, Delta = yc[0:No], yc[No:No + Ne], yc[No + Ne:2 * No + Ne], yc[2 * No + Ne]
        if self.dd:
            d = self.d
            mv = d.matvec
            parts = [mv(self.So_dd, p1c), mv(self.Se_dd, c0c), mv(self.So_dd, Q0c)]
            Mg = self.Mg
            out = d.zeros_like_kind((3 * Mg + 1,), isinstance(yc, d.Dual))
            for i, p in enumerate(parts):
                out[i * Mg:(i + 1) * Mg] = p
            out[3 * Mg] = Delta
            return out
        return np.concatenate([self.So @ p1c, self.Se @ c0c, self.So @ Q0c, [Delta]])

    def residual(self, yc):
        Mg, No, Ne = self.Mg, self.No, self.Ne
        Rg = self.inner.residual(self.to_grid(yc))
        self.last = self.inner.last
        if self.dd:
            d = self.d
            out = d.zeros_like_kind((self.L,), isinstance(yc, d.Dual))
            out[0:No] = d.matvec(self.Ao_dd, Rg[0:Mg])
            out[No:2 * No] = d.matvec(self.Ao_dd, Rg[Mg:2 * Mg])
            out[2 * No:2 * No + Ne] = d.matvec(self.Ae_dd, Rg[2 * Mg:3 * Mg])
            out[2 * No + Ne] = yc[1]  # the sin(theta) coefficient of p1
            return out
        return np.concatenate([self.Ao @ Rg[0:Mg], self.Ao @ Rg[Mg:2 * Mg], self.Ae @ Rg[2 * Mg:3 * Mg], [yc[1]]])

    def jacobian(self, yc, h=1e-30):
        L = self.L
        if self.dd:
            d, DD = self.d, self.DD
            J = d.zeros((L, L))
            for i in range(L):
                e = np.zeros(L)
                e[i] = 1.0
                col = self.residual(d.Dual(yc, DD(e))).t
                J.h[:, i] = col.h
                J.l[:, i] = col.l
            self.inner.dual = False
            return J
        J = np.zeros((L, L))
        for i in range(L):
            z = yc.astype(complex)
            z[i] += 1j * h
            J[:, i] = self.residual(z).imag / h
        return J

    def from_grid(self, yg, Mg_src):
        """Coefficients of a grid state (p1, c0, Q0 on Mg_src points, Delta)."""
        _, _, Ao, Ae, _, _ = bases(Mg_src, self.ju)
        M = Mg_src
        return np.concatenate([Ao @ yg[0:M], Ae @ yg[M:2 * M], Ao @ yg[2 * M:3 * M], [yg[3 * M]]])
