//! The DSSSpec multiple-shooting residual (spec_long variant with frozen RK4 step counts).
//!
//! Everything is generic over the scalar type T (f64 or Complex64) and over a `Backend` that
//! supplies the hot kernels (matrix-vector products, the ODE right-hand side, RK4 vector updates,
//! Cauchy products, LU row updates). `Generic` is plain Rust loops (auto-vectorised or scalar,
//! depending on the instruction set it is compiled for); `avx512::Avx512` implements the same
//! operations, in the same order, with explicit AVX-512 intrinsics.
//!
//! Everything from `block_state` / `block_out` down is `#[inline(always)]`, so it is compiled into
//! the `#[target_feature]` entry points of `dispatch` with the instruction set chosen at runtime.
//!
//! Fields are stored on N tau points (N = 72, or N = 40 for the parity-folded representation:
//! 36 half-period samples padded to a multiple of 8). The parity-folded representation uses that p, m
//! are odd and a, c even under tau -> tau + Delta/2, so every 72-point field is [h, +-h]; the
//! circulant D and trunc matrices then fold exactly into 36x36 matrices (and the periodic solve
//! decouples into a 36x36 system).

use crate::dispatch::Exec;
use crate::scalar::{zero_state, Dual, Fld, Sc, State};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------------------------------
// elementwise Fld arithmetic (shared by all backends; cheap parts of the computation)
// ---------------------------------------------------------------------------------------------
macro_rules! fld_binop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl<T: Sc, const N: usize> $tr for Fld<T, N> {
            type Output = Fld<T, N>;
            #[inline(always)]
            fn $f(self, o: Fld<T, N>) -> Fld<T, N> {
                let mut r = self;
                for j in 0..N { r.0[j] = self.0[j] $op o.0[j]; }
                r
            }
        }
        impl<T: Sc, const N: usize> $tr<f64> for Fld<T, N> {
            type Output = Fld<T, N>;
            #[inline(always)]
            fn $f(self, o: f64) -> Fld<T, N> {
                let mut r = self;
                for j in 0..N { r.0[j] = self.0[j] $op o; }
                r
            }
        }
    };
}
fld_binop!(Add, add, +);
fld_binop!(Sub, sub, -);
fld_binop!(Mul, mul, *);
fld_binop!(Div, div, /);

impl<T: Sc, const N: usize> Neg for Fld<T, N> {
    type Output = Fld<T, N>;
    #[inline(always)]
    fn neg(self) -> Fld<T, N> {
        let mut r = self;
        for j in 0..N {
            r.0[j] = -self.0[j];
        }
        r
    }
}

impl<T: Sc, const N: usize> Fld<T, N> {
    #[inline(always)]
    pub fn scale_t(self, s: T) -> Self {
        let mut r = self;
        for j in 0..N {
            r.0[j] = self.0[j] * s;
        }
        r
    }
}

// ---------------------------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------------------------
#[derive(Clone)]
pub struct Ctx<const N: usize> {
    /// number of logical tau points stored (36 folded, 72 full)
    pub nh: usize,
    pub n: f64,
    /// d/dtau for omega = 1 in the odd sector: f_tau = f @ d_odd  (nh rows)
    pub d_odd: Vec<Fld<f64, N>>,
    /// spectral truncation k <= kc: f @ t_odd / f @ t_even
    pub t_odd: Vec<Fld<f64, N>>,
    pub t_even: Vec<Fld<f64, N>>,
    /// spectral coefficients -> physical samples (no / ne rows)
    pub phys_odd: Vec<Fld<f64, N>>,
    pub phys_even: Vec<Fld<f64, N>>,
    /// physical samples -> spectral coefficients, stored as rows: c[k] = <spec_t[k], f>
    pub spec_odd_t: Vec<Fld<f64, N>>,
    pub spec_even_t: Vec<Fld<f64, N>>,
    pub x0: f64,
    pub nodes: Vec<f64>,
    pub log_steps: usize,
    pub seg_steps: Vec<usize>,
    pub seg_u: Vec<bool>,
    pub u0: f64,
    pub kc_terms: usize,
    pub ks_terms: usize,
    pub no: usize,
    pub ne: usize,
    pub offs: Vec<usize>,
    pub len: usize,
    pub phase_idx: usize,
    /// Growth rate of the perturbation mode (perturbations ~ exp(lam tau)). Only tangent scalars
    /// (`Dual`) see it: their tau-derivatives become D + lam on the tangent. 0 for the residual.
    pub lam: f64,
}

impl<const N: usize> Ctx<N> {
    pub fn nblocks(&self) -> usize {
        self.nodes.len() + 1
    }
    pub fn block_len(&self) -> usize {
        2 * self.no + 2 * self.ne
    }
}

// ---------------------------------------------------------------------------------------------
// backend trait
// ---------------------------------------------------------------------------------------------
pub trait Backend<T: Sc, const N: usize> {
    /// out = (sum_i h[i] * m[i]) [* scale]   (rows of m, accumulated in order i = 0.. with FMA)
    fn matvec(m: &[Fld<f64, N>], h: &[T], out: &mut Fld<T, N>);
    /// two vectors sharing one matrix pass; optional real scale applied after accumulation
    fn matvec2(m: &[Fld<f64, N>], h0: &[T], h1: &[T], o0: &mut Fld<T, N>, o1: &mut Fld<T, N>, scale: f64);
    /// Cauchy product coefficient sum_{j=0..k} x[j] * y[k-j]
    fn cs(x: &[Fld<T, N>], y: &[Fld<T, N>], k: usize) -> Fld<T, N>;
    /// weighted sum sum_{i=0..k} (i * x[i]) * y[off - i]
    fn wsum(x: &[Fld<T, N>], y: &[Fld<T, N>], k: usize, off: usize) -> Fld<T, N>;
    /// trunc(DSS.rhs(x, s, omega)) * scale
    fn rhs(ctx: &Ctx<N>, x: f64, scale: f64, omega: T, s: &State<T, N>, out: &mut State<T, N>);
    /// out = s + k * a
    fn axpy(s: &State<T, N>, a: f64, k: &State<T, N>, out: &mut State<T, N>);
    /// s = s + h6 * (((k1 + 2 k2) + 2 k3) + k4)
    fn rk_update(s: &mut State<T, N>, h6: f64, k1: &State<T, N>, k2: &State<T, N>, k3: &State<T, N>, k4: &State<T, N>);
    /// Solve A x = b, A given column-major (cols[i] = column i, nh x nh, padded rows zero) by LU with
    /// partial pivoting (right-looking, LAPACK dgetf2 order: one reciprocal per pivot). A is destroyed.
    fn lu_solve(cols: &mut [Fld<T, N>], b: &Fld<T, N>, nh: usize) -> Fld<T, N>;
}

/// Row interchange k <-> p in the columns still needed (j >= k: L is not stored, the system's
/// right-hand side is eliminated on the fly) and in b. Shared by all backends.
#[inline(always)]
pub fn swap_rows<T: Sc, const N: usize>(cols: &mut [Fld<T, N>], b: &mut Fld<T, N>, k: usize, p: usize, nh: usize) {
    if p != k {
        for c in cols[k..nh].iter_mut() {
            c.0.swap(k, p);
        }
        b.0.swap(k, p);
    }
}

/// Pivot row: first index of the maximum magnitude in column k, rows k..nh (as LAPACK i?amax).
#[inline(always)]
pub fn pivot_scalar<T: Sc, const N: usize>(ck: &Fld<T, N>, k: usize, nh: usize) -> usize {
    let mut p = k;
    let mut best = ck.0[k].mag();
    for i in k + 1..nh {
        let v = ck.0[i].mag();
        if v > best {
            best = v;
            p = i;
        }
    }
    p
}

/// Plain-loop backend (auto-vectorised with target-cpu=native; scalar when vectorisation is off).
pub struct Generic;

impl<T: Sc, const N: usize> Backend<T, N> for Generic {
    #[inline(always)]
    fn matvec(m: &[Fld<f64, N>], h: &[T], out: &mut Fld<T, N>) {
        let mut acc = [T::zero(); N];
        for (i, row) in m.iter().enumerate() {
            let hi = h[i];
            for j in 0..N {
                acc[j] = hi.fma_r(row.0[j], acc[j]);
            }
        }
        out.0 = acc;
    }

    #[inline(always)]
    fn matvec2(m: &[Fld<f64, N>], h0: &[T], h1: &[T], o0: &mut Fld<T, N>, o1: &mut Fld<T, N>, scale: f64) {
        let mut a0 = [T::zero(); N];
        let mut a1 = [T::zero(); N];
        for (i, row) in m.iter().enumerate() {
            let b0 = h0[i];
            let b1 = h1[i];
            for j in 0..N {
                a0[j] = b0.fma_r(row.0[j], a0[j]);
                a1[j] = b1.fma_r(row.0[j], a1[j]);
            }
        }
        if scale != 1.0 {
            for j in 0..N {
                a0[j] = a0[j] * scale;
                a1[j] = a1[j] * scale;
            }
        }
        o0.0 = a0;
        o1.0 = a1;
    }

    #[inline(always)]
    fn cs(x: &[Fld<T, N>], y: &[Fld<T, N>], k: usize) -> Fld<T, N> {
        let mut acc = [T::zero(); N];
        for j in 0..=k {
            let (xa, yb) = (&x[j], &y[k - j]);
            for l in 0..N {
                acc[l] = xa.0[l].fma(yb.0[l], acc[l]);
            }
        }
        Fld(acc)
    }

    #[inline(always)]
    fn wsum(x: &[Fld<T, N>], y: &[Fld<T, N>], k: usize, off: usize) -> Fld<T, N> {
        let mut acc = [T::zero(); N];
        for i in 0..=k {
            let w = i as f64;
            let (xa, yb) = (&x[i], &y[off - i]);
            for l in 0..N {
                acc[l] = (xa.0[l] * w).fma(yb.0[l], acc[l]);
            }
        }
        Fld(acc)
    }

    #[inline(always)]
    fn rhs(ctx: &Ctx<N>, x: f64, scale: f64, omega: T, s: &State<T, N>, out: &mut State<T, N>) {
        let nh = ctx.nh;
        let mut dp = Fld::<T, N>::zero();
        let mut dm = Fld::<T, N>::zero();
        Self::matvec2(&ctx.d_odd, &s[0].0[..nh], &s[1].0[..nh], &mut dp, &mut dm, 1.0);
        let n = ctx.n;
        let n1 = n - 1.0;
        let ix = 1.0 / x;
        let i2x = 1.0 / (2.0 * x);
        let one = T::re(1.0);
        let xt = T::re(x);
        let mut f: State<T, N> = zero_state();
        for j in 0..N {
            let (p, m, a, c) = (s[0].0[j], s[1].0[j], s[2].0[j], s[3].0[j]);
            let a2 = a * a;
            let cx = ((c * n1) * (a2 - 1.0)) * ix;
            let ax = (a * ((p * p + m * m) + (one - a2) * n1)) * i2x;
            let pc = ((c * n) * (p + m)) * i2x;
            let mut dpj = dp.0[j] * omega;
            let mut dmj = dm.0[j] * omega;
            if T::TANGENT && ctx.lam != 0.0 {
                // perturbation mode: d/dtau -> d/dtau + lam on the tangent (DSS.D with lam)
                dpj = dpj + p.tangent() * ctx.lam;
                dmj = dmj + m.tangent() * ctx.lam;
            }
            f[0].0[j] = (((-dpj + cx * p) - (c * p) * ix) + pc) / (xt - c);
            f[1].0[j] = (((-dmj - cx * m) + (c * m) * ix) - pc) / (xt + c);
            f[2].0[j] = ax;
            f[3].0[j] = cx;
        }
        let (o01, o23) = out.split_at_mut(2);
        let (o0, o1) = o01.split_at_mut(1);
        let (o2, o3) = o23.split_at_mut(1);
        Self::matvec2(&ctx.t_odd, &f[0].0[..nh], &f[1].0[..nh], &mut o0[0], &mut o1[0], scale);
        Self::matvec2(&ctx.t_even, &f[2].0[..nh], &f[3].0[..nh], &mut o2[0], &mut o3[0], scale);
    }

    #[inline(always)]
    fn axpy(s: &State<T, N>, a: f64, k: &State<T, N>, out: &mut State<T, N>) {
        for q in 0..4 {
            for j in 0..N {
                out[q].0[j] = s[q].0[j] + k[q].0[j] * a;
            }
        }
    }

    #[inline(always)]
    fn rk_update(s: &mut State<T, N>, h6: f64, k1: &State<T, N>, k2: &State<T, N>, k3: &State<T, N>, k4: &State<T, N>) {
        for q in 0..4 {
            for j in 0..N {
                let t = ((k1[q].0[j] + k2[q].0[j] * 2.0) + k3[q].0[j] * 2.0) + k4[q].0[j];
                s[q].0[j] = s[q].0[j] + t * h6;
            }
        }
    }

    #[inline(always)]
    fn lu_solve(cols: &mut [Fld<T, N>], b: &Fld<T, N>, nh: usize) -> Fld<T, N> {
        let mut b = *b;
        for k in 0..nh {
            let p = pivot_scalar(&cols[k], k, nh);
            swap_rows(cols, &mut b, k, p, nh);
            let inv = T::re(1.0) / cols[k].0[k];
            // multipliers l[i] = a[i][k] / a[k][k] for i > k, zero elsewhere
            let mut l = Fld::<T, N>::zero();
            for i in k + 1..N {
                l.0[i] = cols[k].0[i] * inv;
            }
            let c0 = (k + 1) / 8;
            let (left, right) = cols.split_at_mut(k + 1);
            let _ = left;
            for cj in right[..nh - k - 1].iter_mut() {
                let nu = -cj.0[k];
                for i in 8 * c0..N {
                    cj.0[i] = l.0[i].fma(nu, cj.0[i]);
                }
            }
            let nbk = -b.0[k];
            for i in 8 * c0..N {
                b.0[i] = l.0[i].fma(nbk, b.0[i]);
            }
        }
        // back substitution, column oriented
        let mut x = Fld::<T, N>::zero();
        for k in (0..nh).rev() {
            let xk = b.0[k] / cols[k].0[k];
            x.0[k] = xk;
            let nxk = -xk;
            for i in 0..k {
                b.0[i] = cols[k].0[i].fma(nxk, b.0[i]);
            }
        }
        x
    }
}

// ---------------------------------------------------------------------------------------------
// building blocks
// ---------------------------------------------------------------------------------------------
#[inline(always)]
fn dtau<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, f: &Fld<T, N>, omega: T) -> Fld<T, N> {
    let mut o = Fld::zero();
    B::matvec(&ctx.d_odd, &f.0[..ctx.nh], &mut o);
    let o = o.scale_t(omega);
    if T::TANGENT && ctx.lam != 0.0 {
        let mut o = o;
        for j in 0..N {
            o.0[j] = o.0[j] + f.0[j].tangent() * ctx.lam;
        }
        return o;
    }
    o
}

#[inline(always)]
fn trunc_state<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, s: &State<T, N>) -> State<T, N> {
    let nh = ctx.nh;
    let mut o: State<T, N> = zero_state();
    let (o01, o23) = o.split_at_mut(2);
    let (o0, o1) = o01.split_at_mut(1);
    let (o2, o3) = o23.split_at_mut(1);
    B::matvec2(&ctx.t_odd, &s[0].0[..nh], &s[1].0[..nh], &mut o0[0], &mut o1[0], 1.0);
    B::matvec2(&ctx.t_even, &s[2].0[..nh], &s[3].0[..nh], &mut o2[0], &mut o3[0], 1.0);
    o
}

#[inline(always)]
fn phys<T: Sc, B: Backend<T, N>, const N: usize>(m: &[Fld<f64, N>], c: &[T]) -> Fld<T, N> {
    let mut o = Fld::zero();
    B::matvec(m, c, &mut o);
    o
}

#[inline(always)]
fn spec<T: Sc, const N: usize>(mt: &[Fld<f64, N>], nh: usize, f: &Fld<T, N>, out: &mut [T]) {
    for (k, row) in mt.iter().enumerate() {
        let mut acc = T::zero();
        for j in 0..nh {
            acc = f.0[j].fma_r(row.0[j], acc);
        }
        out[k] = acc;
    }
}

/// Horner evaluation sum_k S[k] x^k (as DSS.eval_series)
#[inline(always)]
fn eval_series<T: Sc, const N: usize>(s: &[Fld<T, N>], x: f64) -> Fld<T, N> {
    let mut out = Fld::<T, N>::zero();
    for k in (0..s.len()).rev() {
        out = out * x + s[k];
    }
    out
}

/// Solve (omega D + diag(kappa)) f = r for periodic f in the odd sector (DSS.solve_periodic).
/// Column i of the system matrix is omega * d_odd[i] (+ kappa_i on the diagonal).
#[inline(always)]
pub fn solve_periodic<T: Sc, B: Backend<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    kappa: &Fld<T, N>,
    r: &Fld<T, N>,
    omega: T,
    a: &mut Vec<Fld<T, N>>,
) -> Fld<T, N> {
    if T::TANGENT {
        return solve_periodic_tangent(ctx, kappa, r, omega);
    }
    let nh = ctx.nh;
    a.clear();
    for i in 0..nh {
        let mut col = Fld::<T, N>::zero();
        let dr = &ctx.d_odd[i];
        for j in 0..N {
            col.0[j] = omega * dr.0[j];
        }
        col.0[i] = col.0[i] + kappa.0[i];
        a.push(col);
    }
    B::lu_solve(a, r, nh)
}

/// `solve_periodic` for tangent scalars. With A0 = omega_v D + diag(kappa_v):
///   value:    A0 f_v = r_v                                     (bitwise the f64 plain-loop solve)
///   tangent:  (A0 + lam) f_d = r_d - (omega_d D + diag(kappa_d)) f_v
/// which is the first-order part of DSS.solve_periodic's perturbation rows (lam on the diagonal,
/// lam * background added to the right-hand side).
#[inline(always)]
fn solve_periodic_tangent<T: Sc, const N: usize>(ctx: &Ctx<N>, kappa: &Fld<T, N>, r: &Fld<T, N>, omega: T) -> Fld<T, N> {
    let nh = ctx.nh;
    let (om_v, om_d) = omega.parts();
    let split = |f: &Fld<T, N>| {
        let (mut v, mut d) = (Fld::<f64, N>::zero(), Fld::<f64, N>::zero());
        for j in 0..N {
            let (a, b) = f.0[j].parts();
            v.0[j] = a;
            d.0[j] = b;
        }
        (v, d)
    };
    let (k_v, k_d) = split(kappa);
    let (r_v, r_d) = split(r);
    let mut cols = Vec::with_capacity(nh);
    let build = |cols: &mut Vec<Fld<f64, N>>, shift: f64| {
        cols.clear();
        for i in 0..nh {
            let mut col = Fld::<f64, N>::zero();
            let dr = &ctx.d_odd[i];
            for j in 0..N {
                col.0[j] = om_v * dr.0[j];
            }
            col.0[i] = col.0[i] + k_v.0[i];
            if shift != 0.0 {
                col.0[i] = col.0[i] + shift;
            }
            cols.push(col);
        }
    };
    build(&mut cols, 0.0);
    let f_v = <Generic as Backend<f64, N>>::lu_solve(&mut cols, &r_v, nh);
    // right-hand side of the tangent system
    let mut df = Fld::<f64, N>::zero();
    <Generic as Backend<f64, N>>::matvec(&ctx.d_odd, &f_v.0[..nh], &mut df);
    let mut rhs = Fld::<f64, N>::zero();
    for j in 0..nh {
        rhs.0[j] = r_d.0[j] - om_d * df.0[j] - k_d.0[j] * f_v.0[j];
    }
    build(&mut cols, ctx.lam);
    let f_d = <Generic as Backend<f64, N>>::lu_solve(&mut cols, &rhs, nh);
    let mut out = Fld::<T, N>::zero();
    for j in 0..N {
        out.0[j] = T::from_parts(f_v.0[j], f_d.0[j]);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// centre series (DSS.centre_series), evaluated at x0 and truncated
// ---------------------------------------------------------------------------------------------
#[inline(always)]
fn centre_state<T: Sc, B: Backend<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    p1: &Fld<T, N>,
    c0: &Fld<T, N>,
    omega: T,
) -> State<T, N> {
    let n = ctx.n;
    let kmax = ctx.kc_terms;
    let z = Fld::<T, N>::zero();
    let mut pp = vec![z; kmax + 1];
    let mut mm = vec![z; kmax + 1];
    let mut aa = vec![z; kmax + 1];
    let mut cc = vec![z; kmax + 1];
    aa[0] = Fld::splat(T::re(1.0));
    cc[0] = *c0;
    pp[1] = *p1;
    mm[1] = -*p1;
    for k in 2..=kmax {
        let kf = k as f64;
        // Plain loops rather than map/collect closures, which LLVM may leave outlined and so compile
        // without the dispatched target features.
        let mut pm = Vec::with_capacity(kmax + 1);
        for j in 0..=kmax {
            pm.push(pp[j] + mm[j]);
        }
        let mut q2 = Vec::with_capacity(k + 1);
        for j in 0..=k {
            q2.push(B::cs(&pp, &pp, j) + B::cs(&mm, &mm, j));
        }
        let s = B::cs(&aa, &q2, k);
        let mut a2 = Vec::with_capacity(k + 1);
        for j in 0..=k {
            a2.push(B::cs(&aa, &aa, j));
        }
        let t3 = B::cs(&aa, &a2, k);
        aa[k] = (s * 0.5 - t3 * (0.5 * (n - 1.0))) / (kf + n - 1.0);
        let mut a2m1 = Vec::with_capacity(k + 1);
        for j in 0..=k {
            a2m1.push(B::cs(&aa, &aa, j));
        }
        a2m1[0] = a2m1[0] - 1.0;
        cc[k] = (B::cs(&cc, &a2m1, k) * (n - 1.0)) / kf;
        let dp = dtau::<T, B, N>(ctx, &pp[k - 1], omega);
        let dm = dtau::<T, B, N>(ctx, &mm[k - 1], omega);
        // X[j], Y[j] for j = 1..=k
        let xj = |j: usize| -> Fld<T, N> {
            if j == 1 {
                -cc[0]
            } else {
                -cc[j - 1] + if j == 2 { 1.0 } else { 0.0 }
            }
        };
        let yj = |j: usize| -> Fld<T, N> {
            if j == 1 {
                cc[0]
            } else {
                cc[j - 1] + if j == 2 { 1.0 } else { 0.0 }
            }
        };
        let mut xpx = z;
        let mut xmx = z;
        let mut xcxp = z;
        let mut xcxm = z;
        for j in 1..=k {
            let w = (k - j + 1) as f64;
            xpx = xpx + (xj(j) * w) * pp[k - j + 1];
            xmx = xmx + (yj(j) * w) * mm[k - j + 1];
            xcxp = xcxp + (cc[j] * (j as f64)) * pp[k - j];
            xcxm = xcxm + (cc[j] * (j as f64)) * mm[k - j];
        }
        let cp = B::cs(&cc, &pp, k);
        let cm = B::cs(&cc, &mm, k);
        let cpm = B::cs(&cc, &pm, k);
        let r3 = (((dp + xpx) - xcxp) + cp) - cpm * (0.5 * n);
        let r4 = (((dm + xmx) + xcxm) - cm) + cpm * (0.5 * n);
        let c = cc[0];
        let m11 = c * (1.0 - kf) - c * (0.5 * n);
        let m12 = c * (-0.5 * n);
        let m21 = c * (0.5 * n);
        let m22 = c * (kf - 1.0) + c * (0.5 * n);
        let det = m11 * m22 - m12 * m21;
        pp[k] = ((-r3) * m22 + r4 * m12) / det;
        mm[k] = ((-r4) * m11 + r3 * m21) / det;
    }
    let s = [eval_series(&pp, ctx.x0), eval_series(&mm, ctx.x0), eval_series(&aa, ctx.x0), eval_series(&cc, ctx.x0)];
    trunc_state::<T, B, N>(ctx, &s)
}

// ---------------------------------------------------------------------------------------------
// horizon (SSH) series, O(K^2) form of fast.ssh_series_fast, evaluated at u0 and truncated
// ---------------------------------------------------------------------------------------------
#[inline(always)]
fn ssh_state<T: Sc, B: Backend<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    m0: &Fld<T, N>,
    a0: &Fld<T, N>,
    omega: T,
) -> State<T, N> {
    let (pp, mm, aa, cc) = ssh_series::<T, B, N>(ctx, m0, a0, omega);
    let u0 = ctx.u0;
    let s = [eval_series(&pp, u0), eval_series(&mm, u0), eval_series(&aa, u0), eval_series(&cc, u0)];
    trunc_state::<T, B, N>(ctx, &s)
}

/// Horizon series coefficients P, M, A, C (orders 0..=Ks each).
#[allow(clippy::type_complexity)]
#[inline(always)]
pub fn ssh_series<T: Sc, B: Backend<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    m0: &Fld<T, N>,
    a0: &Fld<T, N>,
    omega: T,
) -> (Vec<Fld<T, N>>, Vec<Fld<T, N>>, Vec<Fld<T, N>>, Vec<Fld<T, N>>) {
    let n = ctx.n;
    let kk_max = ctx.ks_terms;
    let z = Fld::<T, N>::zero();
    let len = kk_max + 2;
    let mut pp = vec![z; len];
    let mut mm = vec![z; len];
    let mut aa = vec![z; len];
    let mut cc = vec![z; len];
    let mut a2 = vec![z; len];
    let mut q2 = vec![z; len];
    let mut dp = vec![z; len];
    let mut dm = vec![z; len];
    let mut pm = vec![z; len];
    let mut vv = vec![z; len];
    let mut ww = vec![z; len];
    let mut zc = vec![z; len];
    let mut lu = Vec::with_capacity(ctx.nh);
    aa[0] = *a0;
    cc[0] = Fld::splat(T::re(1.0));
    mm[0] = *m0;
    a2[0] = aa[0] * aa[0];
    dm[0] = dtau::<T, B, N>(ctx, &mm[0], omega);
    // w_i = -C_i (+1 for i = 0, -1 for i = 1);  zc_i = C_i (+1 for i = 0, -1 for i = 1)
    let wfun = |c: &Vec<Fld<T, N>>, i: usize| -> Fld<T, N> {
        match i {
            0 => -c[0] + 1.0,
            1 => -c[1] - 1.0,
            _ => -c[i],
        }
    };
    let zfun = |c: &Vec<Fld<T, N>>, i: usize| -> Fld<T, N> {
        match i {
            0 => c[0] + 1.0,
            1 => c[1] - 1.0,
            _ => c[i],
        }
    };
    ww[0] = wfun(&cc, 0);
    zc[0] = zfun(&cc, 0);
    for kk in 0..=kk_max {
        let kf = kk as f64;
        // c_{K+1}
        let t = B::cs(&cc, &a2, kk);
        cc[kk + 1] = (cc[kk] * kf - (t - cc[kk]) * (n - 1.0)) / (kf + 1.0);
        // W[kk+1], V[kk], Zc[kk+1] (the earlier entries do not change)
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk);
        vv[kk] = cc[kk + 1] * ((kk + 1) as f64) - cc[kk] * kf;
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk);
        // p_K
        pm[kk] = pp[kk] + mm[kk];
        let wpu = B::wsum(&pp, &ww, kk, kk + 1);
        let tau_term = if kk >= 1 { -dp[kk - 1] } else { z };
        let r = (((tau_term - wpu) + B::cs(&vv, &pp, kk)) + B::cs(&cc, &pp, kk)) - B::cs(&cc, &pm, kk) * (0.5 * n);
        let kappa = (((cc[1] + 1.0) * kf + cc[1]) + 1.0) - 0.5 * n;
        pp[kk] = solve_periodic::<T, B, N>(ctx, &kappa, &(-r), omega, &mut lu);
        dp[kk] = dtau::<T, B, N>(ctx, &pp[kk], omega);
        q2[kk] = B::cs(&pp, &pp, kk) + B::cs(&mm, &mm, kk);
        if kk == kk_max {
            break;
        }
        // a_{K+1}
        let a3 = B::cs(&aa, &a2, kk);
        let ag = B::cs(&aa, &q2, kk) + (aa[kk] - a3) * (n - 1.0);
        aa[kk + 1] = (aa[kk] * kf - ag * 0.5) / (kf + 1.0);
        a2[kk + 1] = B::cs(&aa, &aa, kk + 1);
        // m_{K+1}
        let tau_term = if kk >= 1 { dm[kk] - dm[kk - 1] } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = B::wsum(&mm, &zc, kk, kk + 1);
        let rest = (((tau_term - zmu) - B::cs(&vv, &mm, kk)) - B::cs(&cc, &mm, kk)) + B::cs(&cc, &pm, kk) * (0.5 * n);
        mm[kk + 1] = rest / (zc[0] * (kf + 1.0));
        dm[kk + 1] = dtau::<T, B, N>(ctx, &mm[kk + 1], omega);
    }
    let k1 = kk_max + 1;
    pp.truncate(k1);
    mm.truncate(k1);
    aa.truncate(k1);
    cc.truncate(k1);
    (pp, mm, aa, cc)
}

/// Debug helper: horizon series of batch row z (orders 0..=Ks), as (4, Ks+1, nh) row-major values.
#[inline(always)]
pub fn ssh_series_row<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T]) -> Vec<T> {
    let omega = T::re(2.0 * std::f64::consts::PI) / z[ctx.len - 1];
    let ns = ctx.offs.len() - 1;
    let ms = part_phys::<T, B, N>(ctx, z, ns - 2, true);
    let a_s = part_phys::<T, B, N>(ctx, z, ns - 1, false);
    let (pp, mm, aa, cc) = ssh_series::<T, B, N>(ctx, &ms, &a_s, omega);
    let mut out = Vec::new();
    for q in [&pp, &mm, &aa, &cc] {
        for f in q.iter() {
            out.extend_from_slice(&f.0[..ctx.nh]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// RK4 integrators (DSS.integrate, DSSMulti.integrate_log, frozen integrate_u)
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Copy)]
enum Var {
    X,   // integration variable is x itself
    Log, // y = ln x:        f = exp(y) * rhs(exp(y))
    U,   // y = ln(1 - x):   f = -exp(y) * rhs(1 - exp(y))
}

#[inline(always)]
fn xmap(v: Var, y: f64) -> (f64, f64) {
    match v {
        Var::X => (y, 1.0),
        Var::Log => {
            let e = y.exp();
            (e, e)
        }
        Var::U => {
            let e = y.exp();
            (1.0 - e, -e)
        }
    }
}

#[inline(always)]
fn rk4<T: Sc, B: Backend<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    s: &mut State<T, N>,
    var: Var,
    y0: f64,
    h: f64,
    steps: usize,
    omega: T,
) {
    let mut k1: State<T, N> = zero_state();
    let mut k2: State<T, N> = zero_state();
    let mut k3: State<T, N> = zero_state();
    let mut k4: State<T, N> = zero_state();
    let mut tmp: State<T, N> = zero_state();
    let mut y = y0;
    let h2 = h / 2.0;
    let h6 = h / 6.0;
    for _ in 0..steps {
        let (x, sc) = xmap(var, y);
        B::rhs(ctx, x, sc, omega, s, &mut k1);
        B::axpy(s, h2, &k1, &mut tmp);
        let (x, sc) = xmap(var, y + h2);
        B::rhs(ctx, x, sc, omega, &tmp, &mut k2);
        B::axpy(s, h2, &k2, &mut tmp);
        B::rhs(ctx, x, sc, omega, &tmp, &mut k3);
        B::axpy(s, h, &k3, &mut tmp);
        let (x, sc) = xmap(var, y + h);
        B::rhs(ctx, x, sc, omega, &tmp, &mut k4);
        B::rk_update(s, h6, &k1, &k2, &k3, &k4);
        y += h;
    }
}

// ---------------------------------------------------------------------------------------------
// one residual block (one shooting segment) of one batch row
// ---------------------------------------------------------------------------------------------
#[inline(always)]
fn part_phys<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], idx: usize, odd: bool) -> Fld<T, N> {
    let c = &z[ctx.offs[idx]..ctx.offs[idx + 1]];
    if odd {
        phys::<T, B, N>(&ctx.phys_odd, c)
    } else {
        phys::<T, B, N>(&ctx.phys_even, c)
    }
}

#[inline(always)]
fn node_state<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], i: usize) -> State<T, N> {
    let j = 2 + 4 * i;
    [
        part_phys::<T, B, N>(ctx, z, j, true),
        part_phys::<T, B, N>(ctx, z, j + 1, true),
        part_phys::<T, B, N>(ctx, z, j + 2, false),
        part_phys::<T, B, N>(ctx, z, j + 3, false),
    ]
}

/// Range of z (part indices -> offsets) that block b's integration / series depends on (besides Delta).
pub fn block_input_range<const N: usize>(ctx: &Ctx<N>, b: usize) -> (usize, usize) {
    let kn = ctx.nodes.len();
    let ns = ctx.offs.len() - 1;
    if b == 0 {
        (ctx.offs[0], ctx.offs[2])
    } else if b < kn {
        (ctx.offs[2 + 4 * (b - 1)], ctx.offs[2 + 4 * b])
    } else {
        (ctx.offs[ns - 2], ctx.offs[ns])
    }
}

/// The expensive part of block b: centre series + log integration (b = 0), RK4 segment from node b-1
/// (1 <= b < K), or horizon series at u0 (b = K). Depends only on block_input_range(b) and Delta.
#[inline(always)]
pub fn block_state<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N> {
    let kn = ctx.nodes.len();
    let omega = T::re(2.0 * std::f64::consts::PI) / z[ctx.len - 1];
    if b >= kn {
        let ns = ctx.offs.len() - 1;
        let ms = part_phys::<T, B, N>(ctx, z, ns - 2, true);
        let a_s = part_phys::<T, B, N>(ctx, z, ns - 1, false);
        return ssh_state::<T, B, N>(ctx, &ms, &a_s, omega);
    }
    // One rk4 call site, so the integrator (and the plain-loop rhs inside it) is inlined once.
    let (mut s, var, y0, h, steps) = if b == 0 {
        let p1 = part_phys::<T, B, N>(ctx, z, 0, true);
        let mut c0 = part_phys::<T, B, N>(ctx, z, 1, false);
        for j in ctx.nh..N {
            c0.0[j] = T::re(1.0); // padding lanes: keep the centre-series determinant non-zero
        }
        let s = centre_state::<T, B, N>(ctx, &p1, &c0, omega);
        let (xa, xb) = (ctx.x0, ctx.nodes[0]);
        let h = (xb.ln() - xa.ln()) / ctx.log_steps as f64;
        (s, Var::Log, xa.ln(), h, ctx.log_steps)
    } else {
        let s = node_state::<T, B, N>(ctx, z, b - 1);
        let (xa, xb) = (ctx.nodes[b - 1], ctx.nodes[b]);
        let steps = ctx.seg_steps[b - 1];
        if ctx.seg_u[b - 1] {
            let (ya, yb) = ((1.0 - xa).ln(), (1.0 - xb).ln());
            (s, Var::U, ya, (yb - ya) / steps as f64, steps)
        } else {
            (s, Var::X, xa, (xb - xa) / steps as f64, steps)
        }
    };
    rk4::<T, B, N>(ctx, &mut s, var, y0, h, steps, omega);
    s
}

/// The cheap part: mismatch with the target node state, converted to spectral coefficients.
#[inline(always)]
pub fn block_out<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize, s: &State<T, N>, out: &mut [T]) {
    let kn = ctx.nodes.len();
    let tgt = node_state::<T, B, N>(ctx, z, b.min(kn - 1));
    let (no, ne, nh) = (ctx.no, ctx.ne, ctx.nh);
    spec(&ctx.spec_odd_t, nh, &(s[0] - tgt[0]), &mut out[0..no]);
    spec(&ctx.spec_odd_t, nh, &(s[1] - tgt[1]), &mut out[no..2 * no]);
    spec(&ctx.spec_even_t, nh, &(s[2] - tgt[2]), &mut out[2 * no..2 * no + ne]);
    spec(&ctx.spec_even_t, nh, &(s[3] - tgt[3]), &mut out[2 * no + ne..2 * no + 2 * ne]);
}

/// Residual block b (0: centre -> node 0, 1..K-1: node b-1 -> node b, K: horizon series - node K-1)
/// of one batch row z; writes block_len() spectral values to out.
#[inline(always)]
pub fn block<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize, out: &mut [T]) {
    let s = block_state::<T, B, N>(ctx, z, b);
    block_out::<T, B, N>(ctx, z, b, &s, out);
}

/// Full residual of a batch (rows of length ctx.len) using the given rayon pool (None: sequential).
/// dedup: evaluate each block's expensive part once per distinct (block input, Delta) bit pattern
/// (bitwise-identical results; large savings for finite-difference Jacobian batches).
/// e: the backend and instruction set the blocks run with (see `dispatch`).
pub fn residual_batch<T: Sc, E: Exec<T, N>, const N: usize>(
    ctx: &Ctx<N>,
    zb: &[T],
    out: &mut [T],
    pool: Option<&rayon::ThreadPool>,
    dedup: bool,
    e: &E,
) {
    use rayon::prelude::*;
    let l = ctx.len;
    let bl = ctx.block_len();
    let nb = ctx.nblocks();
    let nrows = zb.len() / l;
    if dedup && nrows > 1 {
        use std::collections::HashMap;
        let mut tasks: Vec<(usize, usize)> = Vec::new(); // (block, representative row)
        let mut assign = vec![0usize; nrows * nb];
        for b in 0..nb {
            let (lo, hi) = block_input_range(ctx, b);
            let mut map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(nrows);
            for r in 0..nrows {
                let z = &zb[r * l..(r + 1) * l];
                let mut key = Vec::with_capacity(2 * (hi - lo + 1));
                for v in z[lo..hi].iter().chain(std::iter::once(&z[l - 1])) {
                    v.push_bits(&mut key);
                }
                let idx = *map.entry(key).or_insert_with(|| {
                    tasks.push((b, r));
                    tasks.len() - 1
                });
                assign[r * nb + b] = idx;
            }
        }
        if tasks.len() < nrows * nb {
            let comp = |&(b, r): &(usize, usize)| e.block_state(ctx, &zb[r * l..(r + 1) * l], b);
            let states: Vec<State<T, N>> = match pool {
                None => tasks.iter().map(comp).collect(),
                Some(p) => p.install(|| tasks.par_iter().map(comp).collect()),
            };
            let fill = |(r, (z, o)): (usize, (&[T], &mut [T]))| {
                for (b, ob) in o[..nb * bl].chunks_mut(bl).enumerate() {
                    e.block_out(ctx, z, b, &states[assign[r * nb + b]], ob);
                }
                o[l - 1] = z[ctx.phase_idx];
            };
            match pool {
                None => zb.chunks(l).zip(out.chunks_mut(l)).enumerate().for_each(fill),
                Some(p) => p.install(|| zb.par_chunks(l).zip(out.par_chunks_mut(l)).enumerate().for_each(fill)),
            }
            return;
        }
    }
    let row = |z: &[T], o: &mut [T], par: bool| {
        if par {
            o[..nb * bl].par_chunks_mut(bl).enumerate().for_each(|(b, ob)| e.block(ctx, z, b, ob));
        } else {
            for (b, ob) in o[..nb * bl].chunks_mut(bl).enumerate() {
                e.block(ctx, z, b, ob);
            }
        }
        o[l - 1] = z[ctx.phase_idx];
    };
    match pool {
        None => {
            for (z, o) in zb.chunks(l).zip(out.chunks_mut(l)) {
                row(z, o, false);
            }
        }
        Some(p) => p.install(|| {
            zb.par_chunks(l).zip(out.par_chunks_mut(l)).for_each(|(z, o)| row(z, o, true));
        }),
    }
}

/// Exact linearization at z by forward-mode differentiation: one `Dual` evaluation of block b for
/// each unknown block b depends on (`block_input_range`), plus, for the Jacobian, one per block for
/// Delta. Returns a row-major square matrix:
///
/// * `with_delta = true`: the L x L Jacobian dF/dz of `residual_batch`, phase row and Delta column
///   included (`ctx.lam` should be 0).
/// * `with_delta = false`: the (L-1) x (L-1) mode matrix M(lam) of DSSSpec.mode_matrix, the
///   linearization for perturbations ~ exp(lam tau) with lam = `ctx.lam` (no phase row, no Delta
///   column). Its determinant vanishes at the growth rate of the unstable mode, gamma = 1 / lam.
///
/// Each block's mismatch is spec(s_b(inputs) - phys(node_tgt)), so its derivative with respect to the
/// target node is the constant -spec o phys of each field (the identity for the real, orthogonal
/// Fourier matrices), added directly rather than differentiated.
pub fn linearization<E: Exec<Dual, N>, const N: usize>(
    ctx: &Ctx<N>,
    z: &[f64],
    with_delta: bool,
    pool: Option<&rayon::ThreadPool>,
    e: &E,
) -> Vec<f64> {
    use rayon::prelude::*;
    let l = ctx.len;
    let bl = ctx.block_len();
    let nb = ctx.nblocks();
    let kn = ctx.nodes.len();
    let m = if with_delta { l } else { l - 1 };
    let mut tasks: Vec<(usize, usize)> = Vec::new(); // (block, unknown)
    for b in 0..nb {
        let (lo, hi) = block_input_range(ctx, b);
        tasks.extend((lo..hi).map(|j| (b, j)));
        if with_delta {
            tasks.push((b, l - 1));
        }
    }
    let zd0: Vec<Dual> = z.iter().map(|&v| Dual::new(v, 0.0)).collect();
    let column = |&(b, j): &(usize, usize)| {
        let mut zd = zd0.clone();
        zd[j].d = 1.0;
        let mut ob = vec![Dual::new(0.0, 0.0); bl];
        e.block(ctx, &zd, b, &mut ob);
        ob.iter().map(|x| x.d).collect::<Vec<f64>>()
    };
    let cols: Vec<Vec<f64>> = match pool {
        None => tasks.iter().map(column).collect(),
        Some(p) => p.install(|| tasks.par_iter().map(column).collect()),
    };
    let mut a = vec![0.0f64; m * m];
    for (&(b, j), col) in tasks.iter().zip(&cols) {
        for (r, v) in col.iter().enumerate() {
            a[(b * bl + r) * m + j] = *v;
        }
    }
    // spec o phys for the odd and even fields: sp[k'][k] = sum_j phys[k][j] spec_t[k'][j]
    let sp = |phys: &[Fld<f64, N>], spec_t: &[Fld<f64, N>]| -> Vec<Vec<f64>> {
        spec_t
            .iter()
            .map(|st| phys.iter().map(|ph| (0..ctx.nh).map(|j| ph.0[j] * st.0[j]).sum::<f64>()).collect())
            .collect()
    };
    let (sp_odd, sp_even) = (sp(&ctx.phys_odd, &ctx.spec_odd_t), sp(&ctx.phys_even, &ctx.spec_even_t));
    let (no, ne) = (ctx.no, ctx.ne);
    let fields = [(0, &sp_odd), (no, &sp_odd), (2 * no, &sp_even), (2 * no + ne, &sp_even)];
    for b in 0..nb {
        let tgt = ctx.offs[2 + 4 * b.min(kn - 1)];
        for &(off, spq) in &fields {
            for (r, row) in spq.iter().enumerate() {
                for (k, v) in row.iter().enumerate() {
                    a[(b * bl + off + r) * m + tgt + off + k] -= *v;
                }
            }
        }
    }
    if with_delta {
        a[(l - 1) * m + ctx.phase_idx] = 1.0;
    }
    a
}
