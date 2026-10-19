"""
Dealiased multiple-shooting solver.

Unknowns are Fourier coefficients up to harmonic kc (odd harmonics for p, m; even for a, c), products are
formed on a grid of N >= 3 kc points, and every right-hand-side evaluation is truncated back to k <= kc.
This removes the spurious high-frequency content the plain collocation version developed.
"""
import numpy as np
from dss import DSSMulti


class DSSSpec(DSSMulti):
    def __init__(self, kc, nodes, **kw):
        N = kw.pop("N", None) or int(4 * np.ceil((3 * kc + 2) / 4))
        super().__init__(nodes=nodes, N=N, **kw)
        self.kc = kc
        k = np.arange(self.N // 2 + 1)
        self.mask = (k <= kc).astype(float)
        self.kodd = np.arange(1, kc + 1, 2)
        self.keven = np.arange(0, kc + 1, 2)
        self.n_odd = 2 * len(self.kodd)
        self.n_even = 1 + 2 * (len(self.keven) - 1)

    # ---------- spectral <-> physical ----------
    def trunc(self, f):
        return np.fft.irfft(np.fft.rfft(f, axis=-1) * self.mask, n=self.N, axis=-1)

    def phys(self, c, parity):
        B = c.shape[0]
        F = np.zeros((B, self.N // 2 + 1), complex)
        if parity == "odd":
            m = len(self.kodd)
            F[:, self.kodd] = c[:, :m] + 1j * c[:, m:]
        else:
            m = len(self.keven) - 1
            F[:, 0] = c[:, 0]
            F[:, self.keven[1:]] = c[:, 1:1 + m] + 1j * c[:, 1 + m:]
        return np.fft.irfft(F * self.N, n=self.N, axis=-1)

    def spec(self, f, parity):
        F = np.fft.rfft(f, axis=-1) / self.N
        if parity == "odd":
            return np.concatenate([F[:, self.kodd].real, F[:, self.kodd].imag], axis=1)
        return np.concatenate([F[:, :1].real, F[:, self.keven[1:]].real, F[:, self.keven[1:]].imag], axis=1)

    def rhs(self, x, s, omega):
        return self.trunc(super().rhs(x, s, omega))

    # ---------- layout ----------
    def layout(self):
        K = len(self.nodes)
        no, ne = self.n_odd, self.n_even
        sizes = [no, ne] + [no, no, ne, ne] * K + [no, ne]
        L = sum(sizes) + 1
        return sizes, K, L

    def parities(self):
        K = len(self.nodes)
        return ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]

    def split(self, Zb):
        sizes, K, L = self.layout()
        out, o = [], 0
        for s in sizes:
            out.append(Zb[:, o:o + s]); o += s
        return out, Zb[:, o]

    def residual(self, Zb):
        sizes, K, L = self.layout()
        parts, Dl = self.split(Zb)
        pars = self.parities()
        F = [self.phys(c, p) for c, p in zip(parts, pars)]
        omega = 2 * np.pi / Dl
        p1, c0 = F[0], F[1]
        states = [np.stack(F[2 + 4 * i:6 + 4 * i]) for i in range(K)]
        mS, aS = F[-2], F[-1]
        blocks = []
        s = self._centre_to_node0(p1, c0, omega)
        blocks.append(s - states[0])
        for i in range(K - 1):
            s = self.seg(states[i], self.nodes[i], self.nodes[i + 1], omega)
            blocks.append(s - states[i + 1])
        s = self._ssh_to_last(mS, aS, omega)
        blocks.append(s - states[-1])
        mis = np.concatenate([np.concatenate([self.spec(b[0], "odd"), self.spec(b[1], "odd"),
                                              self.spec(b[2], "even"), self.spec(b[3], "even")], axis=1) for b in blocks], axis=1)
        phase = parts[0][:, len(self.kodd)]          # Im of harmonic 1 of p1
        return np.concatenate([mis, phase[:, None]], axis=1)

    def _centre_to_node0(self, p1, c0, omega):
        P, M, A, C = self.centre_series(p1, c0, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.x0) for S_ in (P, M, A, C)]))
        return self.integrate_log(s, self.x0, self.nodes[0], self.log_steps, omega)

    def _ssh_to_last(self, mS, aS, omega):
        P, M, A, C = self.ssh_series(mS, aS, omega)
        s = self.trunc(np.stack([self.eval_series(S_, self.u0) for S_ in (P, M, A, C)]))
        return self.seg(s, 1 - self.u0, self.nodes[-1], omega)

    # ---------- block-structured Jacobian / mode matrix ----------
    def _blocks(self, z, lam=None, central=False, eps=1e-7):
        """Yields (row_block_index, col_offset, dR/dcols) for each shooting segment, plus Delta derivative."""
        sizes, K, L = self.layout()
        parts, Dl = self.split(z[None, :])
        parts = [p[0] for p in parts]; D = float(Dl[0])
        pars = self.parities()
        offs = np.concatenate([[0], np.cumsum(sizes)])
        hD = eps * max(1.0, D)

        def batch(vecs):
            base = np.concatenate(vecs); nf = len(base)
            if central:
                X = np.repeat(base[None, :], 1 + 2 * nf, axis=0)
                X[1:1 + nf] += eps * np.eye(nf); X[1 + nf:] -= eps * np.eye(nf)
                om = np.full(len(X), 2 * np.pi / D)
            else:
                X = np.repeat(base[None, :], 2 + nf, axis=0)
                X[1:1 + nf] += eps * np.eye(nf)
                om = np.full(len(X), 2 * np.pi / D); om[-1] = 2 * np.pi / (D + hD)
            return X, om, nf

        def tospec(s):
            return np.concatenate([self.spec(s[0], "odd"), self.spec(s[1], "odd"), self.spec(s[2], "even"), self.spec(s[3], "even")], axis=1)

        def deriv(R, nf):
            if central:
                return ((R[1:1 + nf] - R[1 + nf:]) / (2 * eps)).T, None
            return ((R[1:1 + nf] - R[0]) / eps).T, (R[-1] - R[0]) / hD

        if lam is not None:
            self.lam = lam
        try:
            # centre
            X, om, nf = batch([parts[0], parts[1]])
            no, ne = sizes[0], sizes[1]
            s = self._centre_to_node0(self.phys(X[:, :no], "odd"), self.phys(X[:, no:], "even"), om)
            yield 0, [0], deriv(tospec(s), nf)
            for i in range(K - 1):
                j = 2 + 4 * i
                X, om, nf = batch(parts[j:j + 4])
                o = np.concatenate([[0], np.cumsum(sizes[j:j + 4])])
                st = np.stack([self.phys(X[:, o[q]:o[q + 1]], pars[j + q]) for q in range(4)])
                s = self.seg(st, self.nodes[i], self.nodes[i + 1], om)
                yield i + 1, [j], deriv(tospec(s), nf)
            X, om, nf = batch([parts[-2], parts[-1]])
            s = self._ssh_to_last(self.phys(X[:, :sizes[-2]], "odd"), self.phys(X[:, sizes[-2]:], "even"), om)
            yield K, [len(sizes) - 2], deriv(tospec(s), nf)
        finally:
            self.lam = 0.0

    def jacobian(self, z, eps=1e-7):
        sizes, K, L = self.layout()
        offs = np.concatenate([[0], np.cumsum(sizes)])
        F0 = self.residual(z[None, :])[0]
        J = np.zeros((L, L))
        blk = sum(sizes[2:6])                         # rows per block (one full state)
        node0 = offs[2]
        for b, cols, (dR, dD) in self._blocks(z, eps=eps):
            r0 = b * blk
            c0 = offs[cols[0]]
            J[r0:r0 + blk, c0:c0 + dR.shape[1]] = dR
            J[r0:r0 + blk, -1] = dD
            # minus identity on the node state this block lands on
            tgt = node0 + blk * (b if b < K else K - 1)
            J[r0:r0 + blk, tgt:tgt + blk] -= np.eye(blk)
        J[-1, offs[0] + len(self.kodd)] = 1.0
        return F0, J

    def mode_matrix(self, z, lam, eps=1e-6):
        sizes, K, L = self.layout()
        offs = np.concatenate([[0], np.cumsum(sizes)])
        nv = L - 1
        M = np.zeros((nv, nv))
        blk = sum(sizes[2:6]); node0 = offs[2]
        for b, cols, (dR, _) in self._blocks(z, lam=lam, central=True, eps=eps):
            r0 = b * blk; c0 = offs[cols[0]]
            M[r0:r0 + blk, c0:c0 + dR.shape[1]] = dR
            tgt = node0 + blk * (b if b < K else K - 1)
            M[r0:r0 + blk, tgt:tgt + blk] -= np.eye(blk)
        return M

    # ---------- conversion from the collocation (half-sample) layout ----------
    def from_halfsamples(self, zh, Nh, K):
        H = Nh // 2
        pars = self.parities()
        out = []
        for b, par in enumerate(pars):
            h = zh[b * H:(b + 1) * H]
            f = np.concatenate([h, -h]) if par == "odd" else np.concatenate([h, h])
            F = np.fft.rfft(f) / Nh
            G = np.zeros(self.N // 2 + 1, complex)
            m = min(len(F), len(G)); G[:m] = F[:m]
            out.append(self.spec(np.fft.irfft(G * self.N, n=self.N)[None, :], par)[0])
        return np.concatenate(out + [zh[-1:]])

    def regrid(self, z, other):
        """Convert solution vector z of another DSSSpec instance (different kc) to this one."""
        parts, Dl = other.split(z[None, :])
        out = []
        for c, par in zip(parts, other.parities()):
            f = other.phys(c, par)[0]
            F = np.fft.rfft(f) / other.N
            G = np.zeros(self.N // 2 + 1, complex)
            m = min(len(F), len(G)); G[:m] = F[:m]
            out.append(self.spec(np.fft.irfft(G * self.N, n=self.N)[None, :], par)[0])
        return np.concatenate(out + [Dl])
