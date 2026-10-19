"""Rust port of the DSSSpec multiple-shooting residual F(z) (spec_long.make_long variant with
spec_c.freeze_steps step counts).

    from rustkernel import RustResidual
    R = RustResidual(S)                 # S = freeze_steps(make_long(...), Delta_ref)
    F = R(Zb)                           # Zb: (B, L) or (L,) float64 -> same shape as S.residual(Zb)
    F = R(Zb, threads=1, backend="auto")
    Fc = R.residual_complex(Zb_complex) # complex-step capable (generic backend)

backend="simd" (default) runs the best explicit SIMD kernels this CPU supports, picked at runtime
(AVX-512 on x86_64); "auto" runs plain loops compiled for the best instruction set, "baseline" plain
loops without extra features, and an instruction set name ("avx512") forces those kernels.
available_backends() lists what this CPU can run, and RUSTKERNEL_ISA=avx2 (or baseline) caps the
instruction set for the whole process.

folded=True (default) uses the half-period parity of the fields (p, m odd; a, c even under
tau -> tau + Delta/2) to work on 36 instead of 72 tau points; folded=False works on all 72 points
like the Python code. Both agree with the Python reference to rounding level.
"""
import numpy as np

from ._rustkernel import Kernel, available_backends, build_info, simd_available

__all__ = ["RustResidual", "Kernel", "kernel_args", "step_counts", "simd_available", "available_backends", "build_info"]


def _closure_var(fn, name):
    f = getattr(fn, "__func__", fn)
    if f.__closure__ is None:
        return None
    for n, c in zip(f.__code__.co_freevars, f.__closure__):
        if n == name:
            return c.cell_contents
    return None


def step_counts(S):
    """RK4 step counts of the interior segments, exactly as S.seg uses them (S must be frozen with
    spec_c.freeze_steps, so the counts do not depend on omega). Returns (steps, is_u)."""
    om_ref = _closure_var(S.integrate_u, "om_ref")
    if om_ref is None or _closure_var(S.nseg, "om_ref") is None:
        raise ValueError("S must be frozen with spec_c.freeze_steps (step counts independent of omega)")
    steps, is_u = [], []
    for xa, xb in zip(S.nodes[:-1], S.nodes[1:]):
        if min(xa, xb) >= 0.4:  # DSSMulti.seg -> integrate_u (frozen formula from spec_c.freeze_steps)
            ya, yb = np.log(1 - xa), np.log(1 - xb)
            hy = min(0.01, 0.45 / (0.5 * S.N * om_ref))
            steps.append(max(4, int(np.ceil(abs(yb - ya) / hy))))
            is_u.append(True)
        else:  # DSSMulti.seg -> integrate with the frozen nseg
            steps.append(int(S.nseg(xa, xb, None)))
            is_u.append(False)
    return steps, is_u


def _matrices(S, folded):
    N = S.N
    H = N // 2
    E = np.eye(N)
    Dm = np.asarray(S._D_unit(E), float)  # f_tau = f @ Dm  (omega = 1)
    T = np.asarray(S.trunc(E), float)  # trunc(f) = f @ T
    no, ne = S.n_odd, S.n_even
    Po = S.phys(np.eye(no), "odd")  # c @ Po  (no, N)
    Pe = S.phys(np.eye(ne), "even")
    So = S.spec(E, "odd")  # f @ So  (N, no)
    Se = S.spec(E, "even")
    if not folded:
        return dict(nh=N, d_odd=Dm, t_odd=T, t_even=T, phys_odd=Po, phys_even=Pe, spec_odd=So, spec_even=Se)

    def fold(M, s):  # f = [h, s h]:  f @ M restricted to the first half
        return M[:H, :H] + s * M[H:, :H]

    return dict(
        nh=H,
        d_odd=fold(Dm, -1.0),
        t_odd=fold(T, -1.0),
        t_even=fold(T, 1.0),
        phys_odd=Po[:, :H],
        phys_even=Pe[:, :H],
        spec_odd=So[:H] - So[H:],
        spec_even=Se[:H] + Se[H:],
    )


def kernel_args(S, folded=True):
    """Positional arguments for Kernel(...) extracted from a DSSSpec instance S."""
    qn = getattr(getattr(S._ssh_to_last, "__func__", S._ssh_to_last), "__qualname__", "")
    if "make_long" not in qn:
        raise NotImplementedError("only the spec_long.make_long variant (_ssh_to_last without integration) is ported")
    if getattr(S, "lam", 0.0) != 0.0:
        raise NotImplementedError("perturbation mode (lam != 0) is not ported")
    steps, is_u = step_counts(S)
    sizes, K, L = S.layout()
    M = _matrices(S, folded)
    c = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    return (
        float(S.n), int(M["nh"]),
        c(M["d_odd"]), c(M["t_odd"]), c(M["t_even"]),
        c(M["phys_odd"]), c(M["phys_even"]), c(M["spec_odd"]), c(M["spec_even"]),
        float(S.x0), [float(x) for x in S.nodes], int(S.log_steps), steps, is_u,
        float(S.u0), int(S.Kc), int(S.Ks), [int(s) for s in sizes], int(len(S.kodd)),
    )


class RustResidual:
    """Callable drop-in for S.residual (lam = 0)."""

    def __init__(self, S, folded=True, threads=2, backend="simd", dedup=False, kernel_cls=None):
        self.folded = folded
        self.threads = threads
        self.dedup = dedup
        self.backend = backend if (backend != "simd" or simd_available()) else "auto"
        self.kernel = (kernel_cls or Kernel)(*kernel_args(S, folded))
        self.L = S.layout()[2]

    def __call__(self, Zb, threads=None, backend=None, dedup=None):
        Zb = np.asarray(Zb)
        one = Zb.ndim == 1
        Z2 = np.ascontiguousarray(Zb[None, :] if one else Zb, dtype=np.float64)
        F = self.kernel.residual(Z2, self.threads if threads is None else threads, backend or self.backend,
                                 self.dedup if dedup is None else dedup)
        return F[0] if one else F

    def residual_complex(self, Zb, threads=None, dedup=None):
        Zb = np.asarray(Zb)
        one = Zb.ndim == 1
        Z2 = np.ascontiguousarray(Zb[None, :] if one else Zb, dtype=np.complex128)
        F = self.kernel.residual_complex(Z2, self.threads if threads is None else threads,
                                         self.dedup if dedup is None else dedup)
        return F[0] if one else F
