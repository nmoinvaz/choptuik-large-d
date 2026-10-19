//! Explicit SIMD backends for AVX2 (4 f64 lanes) and NEON (2 lanes), written once over a lane type.
//! As in `avx512::Avx512`, every operation is performed in exactly the same order as in
//! `model::Generic`, so all backends give bitwise identical results.
//!
//! `simd_backend!` compiles the lane-generic kernels below inside `#[target_feature]` functions for
//! one lane type and implements `Backend<f64, N>` with safe shims around them. The generated types
//! must only be run through `dispatch`, which checks the CPU first.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::model::{pivot_scalar, swap_rows, Ctx};
use crate::scalar::{Fld, State};

/// One SIMD register of f64 lanes.
///
/// # Safety
/// The methods use the implementation's instruction set, so they may only run where it is
/// available. `load` and `store` need pointers aligned to the register width.
pub unsafe trait Lanes: Copy {
    /// f64 lanes per register (a divisor of 8)
    const L: usize;
    /// registers accumulated per pass in one-output (G1) and two-output (G2) reductions, sized so
    /// the accumulators, broadcasts and loads fit the register file
    const G1: usize;
    const G2: usize;
    unsafe fn zero() -> Self;
    unsafe fn splat(v: f64) -> Self;
    unsafe fn load(p: *const f64) -> Self;
    unsafe fn store(p: *mut f64, v: Self);
    unsafe fn add(a: Self, b: Self) -> Self;
    unsafe fn sub(a: Self, b: Self) -> Self;
    unsafe fn mul(a: Self, b: Self) -> Self;
    unsafe fn div(a: Self, b: Self) -> Self;
    /// a * b + c, fused
    unsafe fn fmadd(a: Self, b: Self, c: Self) -> Self;
    /// c - a * b, fused
    unsafe fn fnmadd(a: Self, b: Self, c: Self) -> Self;
    /// exact sign flip
    unsafe fn neg(a: Self) -> Self;
}

/// Evaluates `$body` with the constant `$G` equal to `$g` (1..=16), so accumulator arrays have a
/// size known at compile time and stay in registers.
macro_rules! with_group {
    ($g:expr, $G:ident => $body:block) => {
        match $g {
            1 => { const $G: usize = 1; $body }
            2 => { const $G: usize = 2; $body }
            3 => { const $G: usize = 3; $body }
            4 => { const $G: usize = 4; $body }
            5 => { const $G: usize = 5; $body }
            6 => { const $G: usize = 6; $body }
            7 => { const $G: usize = 7; $body }
            8 => { const $G: usize = 8; $body }
            9 => { const $G: usize = 9; $body }
            10 => { const $G: usize = 10; $body }
            11 => { const $G: usize = 11; $body }
            12 => { const $G: usize = 12; $body }
            13 => { const $G: usize = 13; $body }
            14 => { const $G: usize = 14; $body }
            15 => { const $G: usize = 15; $body }
            16 => { const $G: usize = 16; $body }
            _ => unreachable!("register group larger than 16"),
        }
    };
}

/// Splits `nv` registers into the fewest groups of at most `gmax`, evenly sized: (first, count).
#[inline(always)]
fn groups(nv: usize, gmax: usize) -> impl Iterator<Item = (usize, usize)> {
    let ng = nv.div_ceil(gmax);
    (0..ng).map(move |i| (i * nv / ng, (i + 1) * nv / ng - i * nv / ng))
}

/// out = sum_i h[i] * m[i]
#[inline(always)]
pub unsafe fn matvec<V: Lanes, const N: usize>(m: &[Fld<f64, N>], h: &[f64], out: &mut Fld<f64, N>) {
    const { assert!(N % V::L == 0) };
    assert!(h.len() >= m.len());
    let op = out.0.as_mut_ptr();
    for (c0, g) in groups(N / V::L, V::G1) {
        with_group!(g, G => {
            let mut a = [V::zero(); G];
            for i in 0..m.len() {
                let b = V::splat(*h.get_unchecked(i));
                let rp = m.get_unchecked(i).0.as_ptr().add(V::L * c0);
                for q in 0..G {
                    a[q] = V::fmadd(b, V::load(rp.add(V::L * q)), a[q]);
                }
            }
            for q in 0..G {
                V::store(op.add(V::L * (c0 + q)), a[q]);
            }
        })
    }
}

/// o0 = sum_i h0[i] * m[i], o1 = sum_i h1[i] * m[i], each times scale (when not 1)
#[inline(always)]
pub unsafe fn matvec2<V: Lanes, const N: usize>(
    m: &[Fld<f64, N>],
    h0: &[f64],
    h1: &[f64],
    o0: &mut Fld<f64, N>,
    o1: &mut Fld<f64, N>,
    scale: f64,
) {
    const { assert!(N % V::L == 0) };
    assert!(h0.len() >= m.len() && h1.len() >= m.len());
    let (p0, p1) = (o0.0.as_mut_ptr(), o1.0.as_mut_ptr());
    for (c0, g) in groups(N / V::L, V::G2) {
        with_group!(g, G => {
            let mut a0 = [V::zero(); G];
            let mut a1 = [V::zero(); G];
            for i in 0..m.len() {
                let b0 = V::splat(*h0.get_unchecked(i));
                let b1 = V::splat(*h1.get_unchecked(i));
                let rp = m.get_unchecked(i).0.as_ptr().add(V::L * c0);
                for q in 0..G {
                    let r = V::load(rp.add(V::L * q));
                    a0[q] = V::fmadd(b0, r, a0[q]);
                    a1[q] = V::fmadd(b1, r, a1[q]);
                }
            }
            if scale != 1.0 {
                let s = V::splat(scale);
                for q in 0..G {
                    a0[q] = V::mul(a0[q], s);
                    a1[q] = V::mul(a1[q], s);
                }
            }
            for q in 0..G {
                V::store(p0.add(V::L * (c0 + q)), a0[q]);
                V::store(p1.add(V::L * (c0 + q)), a1[q]);
            }
        })
    }
}

/// sum_{j=0..k} (w_j x[j]) * y[off-j], with w_j = j if weighted (cs: off = k, unweighted)
#[inline(always)]
pub unsafe fn sums<V: Lanes, const N: usize>(
    x: &[Fld<f64, N>],
    y: &[Fld<f64, N>],
    k: usize,
    off: usize,
    weighted: bool,
) -> Fld<f64, N> {
    const { assert!(N % V::L == 0) };
    assert!(x.len() > k && off >= k && y.len() > off);
    let mut out = Fld::<f64, N>::zero();
    let op = out.0.as_mut_ptr();
    for (c0, g) in groups(N / V::L, V::G1) {
        with_group!(g, G => {
            let mut a = [V::zero(); G];
            for j in 0..=k {
                let xp = x.get_unchecked(j).0.as_ptr().add(V::L * c0);
                let yp = y.get_unchecked(off - j).0.as_ptr().add(V::L * c0);
                if weighted {
                    let w = V::splat(j as f64);
                    for q in 0..G {
                        a[q] = V::fmadd(V::mul(V::load(xp.add(V::L * q)), w), V::load(yp.add(V::L * q)), a[q]);
                    }
                } else {
                    for q in 0..G {
                        a[q] = V::fmadd(V::load(xp.add(V::L * q)), V::load(yp.add(V::L * q)), a[q]);
                    }
                }
            }
            for q in 0..G {
                V::store(op.add(V::L * (c0 + q)), a[q]);
            }
        })
    }
    out
}

/// trunc(DSS.rhs(x, s, omega)) * scale
#[inline(always)]
pub unsafe fn rhs<V: Lanes, const N: usize>(
    ctx: &Ctx<N>,
    x: f64,
    scale: f64,
    omega: f64,
    s: &State<f64, N>,
    out: &mut State<f64, N>,
) {
    let nh = ctx.nh;
    assert!(nh <= N && ctx.d_odd.len() == nh && ctx.t_odd.len() == nh && ctx.t_even.len() == nh);
    let mut dp = Fld::<f64, N>::zero();
    let mut dm = Fld::<f64, N>::zero();
    let mut f = [Fld::<f64, N>::zero(); 4];
    matvec2::<V, N>(&ctx.d_odd, &s[0].0, &s[1].0, &mut dp, &mut dm, 1.0);
    let n = ctx.n;
    let vn1 = V::splat(n - 1.0);
    let vn = V::splat(n);
    let vix = V::splat(1.0 / x);
    let vi2x = V::splat(1.0 / (2.0 * x));
    let vx = V::splat(x);
    let one = V::splat(1.0);
    let vom = V::splat(omega);
    for c in 0..N / V::L {
        let o = V::L * c;
        let p = V::load(s[0].0.as_ptr().add(o));
        let m = V::load(s[1].0.as_ptr().add(o));
        let a = V::load(s[2].0.as_ptr().add(o));
        let cc = V::load(s[3].0.as_ptr().add(o));
        let a2 = V::mul(a, a);
        // cx = ((c (n-1)) (a2 - 1)) / x
        let cx = V::mul(V::mul(V::mul(cc, vn1), V::sub(a2, one)), vix);
        // ax = (a ((p p + m m) + (1 - a2)(n-1))) / (2x)
        let q = V::add(V::add(V::mul(p, p), V::mul(m, m)), V::mul(V::sub(one, a2), vn1));
        let ax = V::mul(V::mul(a, q), vi2x);
        // pc = ((c n)(p + m)) / (2x)
        let pc = V::mul(V::mul(V::mul(cc, vn), V::add(p, m)), vi2x);
        let dpv = V::mul(V::load(dp.0.as_ptr().add(o)), vom);
        let dmv = V::mul(V::load(dm.0.as_ptr().add(o)), vom);
        // px = (((-Dp + cx p) - (c p)/x) + pc) / (x - c)
        let px = V::div(
            V::add(V::sub(V::add(V::neg(dpv), V::mul(cx, p)), V::mul(V::mul(cc, p), vix)), pc),
            V::sub(vx, cc),
        );
        // mx = (((-Dm - cx m) + (c m)/x) - pc) / (x + c)
        let mx = V::div(
            V::sub(V::add(V::sub(V::neg(dmv), V::mul(cx, m)), V::mul(V::mul(cc, m), vix)), pc),
            V::add(vx, cc),
        );
        V::store(f[0].0.as_mut_ptr().add(o), px);
        V::store(f[1].0.as_mut_ptr().add(o), mx);
        V::store(f[2].0.as_mut_ptr().add(o), ax);
        V::store(f[3].0.as_mut_ptr().add(o), cx);
    }
    let (o01, o23) = out.split_at_mut(2);
    let (o0, o1) = o01.split_at_mut(1);
    let (o2, o3) = o23.split_at_mut(1);
    matvec2::<V, N>(&ctx.t_odd, &f[0].0, &f[1].0, &mut o0[0], &mut o1[0], scale);
    matvec2::<V, N>(&ctx.t_even, &f[2].0, &f[3].0, &mut o2[0], &mut o3[0], scale);
}

/// out = s + k * a
#[inline(always)]
pub unsafe fn axpy<V: Lanes, const N: usize>(s: &State<f64, N>, a: f64, k: &State<f64, N>, out: &mut State<f64, N>) {
    const { assert!(N % V::L == 0) };
    let va = V::splat(a);
    for q in 0..4 {
        for c in 0..N / V::L {
            let o = V::L * c;
            let v = V::add(V::load(s[q].0.as_ptr().add(o)), V::mul(V::load(k[q].0.as_ptr().add(o)), va));
            V::store(out[q].0.as_mut_ptr().add(o), v);
        }
    }
}

/// s = s + h6 * (((k1 + 2 k2) + 2 k3) + k4)
#[inline(always)]
pub unsafe fn rk_update<V: Lanes, const N: usize>(
    s: &mut State<f64, N>,
    h6: f64,
    k1: &State<f64, N>,
    k2: &State<f64, N>,
    k3: &State<f64, N>,
    k4: &State<f64, N>,
) {
    const { assert!(N % V::L == 0) };
    let vh = V::splat(h6);
    let two = V::splat(2.0);
    for q in 0..4 {
        for c in 0..N / V::L {
            let o = V::L * c;
            let t = V::add(
                V::add(
                    V::add(V::load(k1[q].0.as_ptr().add(o)), V::mul(V::load(k2[q].0.as_ptr().add(o)), two)),
                    V::mul(V::load(k3[q].0.as_ptr().add(o)), two),
                ),
                V::load(k4[q].0.as_ptr().add(o)),
            );
            let v = V::add(V::load(s[q].0.as_ptr().add(o)), V::mul(t, vh));
            V::store(s[q].0.as_mut_ptr().add(o), v);
        }
    }
}

/// LU solve with partial pivoting, as `Generic::lu_solve` (A given column-major, destroyed).
#[inline(always)]
pub unsafe fn lu_solve<V: Lanes, const N: usize>(cols: &mut [Fld<f64, N>], b: &Fld<f64, N>, nh: usize) -> Fld<f64, N> {
    const { assert!(N % 8 == 0 && 8 % V::L == 0) };
    assert!(nh <= N && cols.len() >= nh);
    let nv = N / V::L;
    let mut b = *b;
    for k in 0..nh {
        let p = pivot_scalar(&cols[k], k, nh);
        swap_rows(cols, &mut b, k, p, nh);
        let inv = V::splat(1.0 / cols[k].0[k]);
        // Updates start at the 8-lane chunk holding row k + 1, as in Generic, so lanes <= k of that
        // chunk take the multiplier 0.
        let c0 = 8 * ((k + 1) / 8) / V::L;
        // multipliers: lanes i > k of column k times 1/pivot, zero elsewhere
        let mut l = Fld::<f64, N>::zero();
        let ckp = cols[k].0.as_ptr();
        for c in c0..nv {
            V::store(l.0.as_mut_ptr().add(V::L * c), V::mul(V::load(ckp.add(V::L * c)), inv));
        }
        for i in V::L * c0..=k {
            l.0[i] = 0.0;
        }
        let lp = l.0.as_ptr();
        for j in k + 1..nh {
            let cj = cols.get_unchecked_mut(j).0.as_mut_ptr();
            let u = V::splat(*cj.add(k));
            for c in c0..nv {
                let o = V::L * c;
                V::store(cj.add(o), V::fnmadd(V::load(lp.add(o)), u, V::load(cj.add(o))));
            }
        }
        let u = V::splat(b.0[k]);
        let bp = b.0.as_mut_ptr();
        for c in c0..nv {
            let o = V::L * c;
            V::store(bp.add(o), V::fnmadd(V::load(lp.add(o)), u, V::load(bp.add(o))));
        }
    }
    // back substitution, column oriented: b[i] -= x_k a[i][k] for i < k (whole registers, then the
    // remaining lanes one by one)
    let mut x = Fld::<f64, N>::zero();
    let bp = b.0.as_mut_ptr();
    for k in (0..nh).rev() {
        let xk = *bp.add(k) / cols[k].0[k];
        x.0[k] = xk;
        let vx = V::splat(xk);
        let ckp = cols[k].0.as_ptr();
        let full = k / V::L;
        for c in 0..full {
            let o = V::L * c;
            V::store(bp.add(o), V::fnmadd(V::load(ckp.add(o)), vx, V::load(bp.add(o))));
        }
        for i in V::L * full..k {
            *bp.add(i) = (*ckp.add(i)).mul_add(-xk, *bp.add(i));
        }
    }
    x
}

/// Defines backend `$name`: the kernels above for lane type `$v`, compiled with
/// `#[target_feature(enable = $feat)]`, behind a `Backend<f64, N>` implementation.
macro_rules! simd_backend {
    ($name:ident, $v:ty, $feat:literal) => {
        pub struct $name;

        #[allow(clippy::too_many_arguments)]
        impl $name {
            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn matvec_k<const N: usize>(m: &[Fld<f64, N>], h: &[f64], out: &mut Fld<f64, N>) {
                unsafe { $crate::simd::matvec::<$v, N>(m, h, out) }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn matvec2_k<const N: usize>(
                m: &[Fld<f64, N>],
                h0: &[f64],
                h1: &[f64],
                o0: &mut Fld<f64, N>,
                o1: &mut Fld<f64, N>,
                scale: f64,
            ) {
                unsafe { $crate::simd::matvec2::<$v, N>(m, h0, h1, o0, o1, scale) }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn sums_k<const N: usize>(
                x: &[Fld<f64, N>],
                y: &[Fld<f64, N>],
                k: usize,
                off: usize,
                weighted: bool,
            ) -> Fld<f64, N> {
                unsafe { $crate::simd::sums::<$v, N>(x, y, k, off, weighted) }
            }

            #[target_feature(enable = $feat)]
            #[inline(never)]
            unsafe fn rhs_k<const N: usize>(
                ctx: &Ctx<N>,
                x: f64,
                scale: f64,
                omega: f64,
                s: &State<f64, N>,
                out: &mut State<f64, N>,
            ) {
                unsafe { $crate::simd::rhs::<$v, N>(ctx, x, scale, omega, s, out) }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn axpy_k<const N: usize>(s: &State<f64, N>, a: f64, k: &State<f64, N>, out: &mut State<f64, N>) {
                unsafe { $crate::simd::axpy::<$v, N>(s, a, k, out) }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn rk_update_k<const N: usize>(
                s: &mut State<f64, N>,
                h6: f64,
                k1: &State<f64, N>,
                k2: &State<f64, N>,
                k3: &State<f64, N>,
                k4: &State<f64, N>,
            ) {
                unsafe { $crate::simd::rk_update::<$v, N>(s, h6, k1, k2, k3, k4) }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn lu_solve_k<const N: usize>(cols: &mut [Fld<f64, N>], b: &Fld<f64, N>, nh: usize) -> Fld<f64, N> {
                unsafe { $crate::simd::lu_solve::<$v, N>(cols, b, nh) }
            }
        }

        // SAFETY (all methods): dispatch only runs this backend where its instruction set is enabled.
        impl<const N: usize> $crate::model::Backend<f64, N> for $name {
            #[inline(always)]
            fn matvec(m: &[Fld<f64, N>], h: &[f64], out: &mut Fld<f64, N>) {
                unsafe { Self::matvec_k(m, h, out) }
            }

            #[inline(always)]
            fn matvec2(m: &[Fld<f64, N>], h0: &[f64], h1: &[f64], o0: &mut Fld<f64, N>, o1: &mut Fld<f64, N>, scale: f64) {
                unsafe { Self::matvec2_k(m, h0, h1, o0, o1, scale) }
            }

            #[inline(always)]
            fn cs(x: &[Fld<f64, N>], y: &[Fld<f64, N>], k: usize) -> Fld<f64, N> {
                unsafe { Self::sums_k(x, y, k, k, false) }
            }

            #[inline(always)]
            fn wsum(x: &[Fld<f64, N>], y: &[Fld<f64, N>], k: usize, off: usize) -> Fld<f64, N> {
                unsafe { Self::sums_k(x, y, k, off, true) }
            }

            #[inline(always)]
            fn rhs(ctx: &Ctx<N>, x: f64, scale: f64, omega: f64, s: &State<f64, N>, out: &mut State<f64, N>) {
                unsafe { Self::rhs_k(ctx, x, scale, omega, s, out) }
            }

            #[inline(always)]
            fn axpy(s: &State<f64, N>, a: f64, k: &State<f64, N>, out: &mut State<f64, N>) {
                unsafe { Self::axpy_k(s, a, k, out) }
            }

            #[inline(always)]
            fn rk_update(
                s: &mut State<f64, N>,
                h6: f64,
                k1: &State<f64, N>,
                k2: &State<f64, N>,
                k3: &State<f64, N>,
                k4: &State<f64, N>,
            ) {
                unsafe { Self::rk_update_k(s, h6, k1, k2, k3, k4) }
            }

            #[inline(always)]
            fn lu_solve(cols: &mut [Fld<f64, N>], b: &Fld<f64, N>, nh: usize) -> Fld<f64, N> {
                unsafe { Self::lu_solve_k(cols, b, nh) }
            }
        }
    };
}

#[allow(unused_imports)] // unused on architectures without a lane type
pub(crate) use simd_backend;
