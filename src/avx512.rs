//! Explicit AVX-512 backend (f64 only). Every operation is performed in exactly the same order as in
//! `model::Generic`, so both backends give bitwise identical results; this one just pins the
//! vectorisation (8-lane zmm registers, FMA, accumulators held in registers across the reduction).
//!
//! The kernels are compiled with `#[target_feature(enable = "avx512f")]` whatever the build target,
//! and `dispatch` only selects this backend when the CPU reports AVX-512F at runtime. The trait
//! methods are safe shims around them, so this type must not be used outside `dispatch`.

use crate::model::{swap_rows, Backend, Ctx};
use crate::scalar::{Fld, State};
use core::arch::x86_64::*;

pub struct Avx512;

const MAXV: usize = 12; // up to 96 lanes

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn ld(p: *const f64) -> __m512d {
    _mm512_load_pd(p)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn st(p: *mut f64, v: __m512d) {
    _mm512_store_pd(p, v)
}
/// Sign flip by integer xor (AVX-512F; _mm512_xor_pd would need AVX-512DQ).
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn neg(v: __m512d) -> __m512d {
    _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v), _mm512_set1_epi64(i64::MIN)))
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn matvec2_impl<const N: usize>(
    m: &[Fld<f64, N>],
    h0: &[f64],
    h1: &[f64],
    o0: &mut Fld<f64, N>,
    o1: &mut Fld<f64, N>,
    scale: f64,
) {
    const { assert!(N % 8 == 0 && N <= 8 * MAXV) };
    let nv = N / 8;
    let mut a0 = [_mm512_setzero_pd(); MAXV];
    let mut a1 = [_mm512_setzero_pd(); MAXV];
    let nr = m.len();
    debug_assert!(h0.len() >= nr && h1.len() >= nr);
    for i in 0..nr {
        let b0 = _mm512_set1_pd(*h0.get_unchecked(i));
        let b1 = _mm512_set1_pd(*h1.get_unchecked(i));
        let rp = m.get_unchecked(i).0.as_ptr();
        for c in 0..nv {
            let r = ld(rp.add(8 * c));
            a0[c] = _mm512_fmadd_pd(b0, r, a0[c]);
            a1[c] = _mm512_fmadd_pd(b1, r, a1[c]);
        }
    }
    if scale != 1.0 {
        let s = _mm512_set1_pd(scale);
        for c in 0..nv {
            a0[c] = _mm512_mul_pd(a0[c], s);
            a1[c] = _mm512_mul_pd(a1[c], s);
        }
    }
    for c in 0..nv {
        st(o0.0.as_mut_ptr().add(8 * c), a0[c]);
        st(o1.0.as_mut_ptr().add(8 * c), a1[c]);
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn matvec_impl<const N: usize>(m: &[Fld<f64, N>], h: &[f64], out: &mut Fld<f64, N>) {
    const { assert!(N % 8 == 0 && N <= 8 * MAXV) };
    let nv = N / 8;
    let mut a = [_mm512_setzero_pd(); MAXV];
    assert!(h.len() >= m.len());
    for i in 0..m.len() {
        let b = _mm512_set1_pd(*h.get_unchecked(i));
        let rp = m.get_unchecked(i).0.as_ptr();
        for c in 0..nv {
            a[c] = _mm512_fmadd_pd(b, ld(rp.add(8 * c)), a[c]);
        }
    }
    for c in 0..nv {
        st(out.0.as_mut_ptr().add(8 * c), a[c]);
    }
}

// SAFETY (all methods): dispatch only runs this backend when the CPU has AVX-512F.
impl<const N: usize> Backend<f64, N> for Avx512 {
    #[inline(always)]
    fn matvec(m: &[Fld<f64, N>], h: &[f64], out: &mut Fld<f64, N>) {
        unsafe { matvec_impl(m, h, out) }
    }

    #[inline(always)]
    fn matvec2(m: &[Fld<f64, N>], h0: &[f64], h1: &[f64], o0: &mut Fld<f64, N>, o1: &mut Fld<f64, N>, scale: f64) {
        assert!(h0.len() >= m.len() && h1.len() >= m.len());
        unsafe { matvec2_impl(m, h0, h1, o0, o1, scale) }
    }

    #[inline(always)]
    fn cs(x: &[Fld<f64, N>], y: &[Fld<f64, N>], k: usize) -> Fld<f64, N> {
        assert!(x.len() > k && y.len() > k);
        let mut out = Fld::<f64, N>::zero();
        unsafe { grouped::<N>(|c0, g, o| sum_group(x, y, k, k, false, c0, g, o), out.0.as_mut_ptr()) };
        out
    }

    #[inline(always)]
    fn wsum(x: &[Fld<f64, N>], y: &[Fld<f64, N>], k: usize, off: usize) -> Fld<f64, N> {
        assert!(x.len() > k && off >= k && y.len() > off);
        let mut out = Fld::<f64, N>::zero();
        unsafe { grouped::<N>(|c0, g, o| sum_group(x, y, k, off, true, c0, g, o), out.0.as_mut_ptr()) };
        out
    }

    #[inline(always)]
    fn rhs(ctx: &Ctx<N>, x: f64, scale: f64, omega: f64, s: &State<f64, N>, out: &mut State<f64, N>) {
        unsafe { rhs_impl(ctx, x, scale, omega, s, out) }
    }

    #[inline(always)]
    fn axpy(s: &State<f64, N>, a: f64, k: &State<f64, N>, out: &mut State<f64, N>) {
        unsafe { axpy_impl(s, a, k, out) }
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
        unsafe { rk_update_impl(s, h6, k1, k2, k3, k4) }
    }

    #[inline(always)]
    fn lu_solve(cols: &mut [Fld<f64, N>], b: &Fld<f64, N>, nh: usize) -> Fld<f64, N> {
        unsafe { lu_solve_impl(cols, b, nh) }
    }
}

#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn rhs_impl<const N: usize>(
    ctx: &Ctx<N>,
    x: f64,
    scale: f64,
    omega: f64,
    s: &State<f64, N>,
    out: &mut State<f64, N>,
) {
    const { assert!(N % 8 == 0 && N <= 8 * MAXV) };
    let nh = ctx.nh;
    assert!(nh <= N && ctx.d_odd.len() == nh && ctx.t_odd.len() == nh && ctx.t_even.len() == nh);
    let mut dp = Fld::<f64, N>::zero();
    let mut dm = Fld::<f64, N>::zero();
    let mut f = [Fld::<f64, N>::zero(); 4];
    unsafe {
        matvec2_impl(&ctx.d_odd, &s[0].0, &s[1].0, &mut dp, &mut dm, 1.0);
        let n = ctx.n;
        let vn1 = _mm512_set1_pd(n - 1.0);
        let vn = _mm512_set1_pd(n);
        let vix = _mm512_set1_pd(1.0 / x);
        let vi2x = _mm512_set1_pd(1.0 / (2.0 * x));
        let vx = _mm512_set1_pd(x);
        let one = _mm512_set1_pd(1.0);
        let vom = _mm512_set1_pd(omega);
        for c in 0..N / 8 {
            let o = 8 * c;
            let p = ld(s[0].0.as_ptr().add(o));
            let m = ld(s[1].0.as_ptr().add(o));
            let a = ld(s[2].0.as_ptr().add(o));
            let cc = ld(s[3].0.as_ptr().add(o));
            let a2 = _mm512_mul_pd(a, a);
            // cx = ((c (n-1)) (a2 - 1)) / x
            let cx = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(cc, vn1), _mm512_sub_pd(a2, one)), vix);
            // ax = (a ((p p + m m) + (1 - a2)(n-1))) / (2x)
            let q = _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(p, p), _mm512_mul_pd(m, m)),
                _mm512_mul_pd(_mm512_sub_pd(one, a2), vn1),
            );
            let ax = _mm512_mul_pd(_mm512_mul_pd(a, q), vi2x);
            // pc = ((c n)(p + m)) / (2x)
            let pc = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(cc, vn), _mm512_add_pd(p, m)), vi2x);
            let dpv = _mm512_mul_pd(ld(dp.0.as_ptr().add(o)), vom);
            let dmv = _mm512_mul_pd(ld(dm.0.as_ptr().add(o)), vom);
            // px = (((-Dp + cx p) - (c p)/x) + pc) / (x - c)
            let px = _mm512_div_pd(
                _mm512_add_pd(
                    _mm512_sub_pd(
                        _mm512_add_pd(neg(dpv), _mm512_mul_pd(cx, p)),
                        _mm512_mul_pd(_mm512_mul_pd(cc, p), vix),
                    ),
                    pc,
                ),
                _mm512_sub_pd(vx, cc),
            );
            // mx = (((-Dm - cx m) + (c m)/x) - pc) / (x + c)
            let mx = _mm512_div_pd(
                _mm512_sub_pd(
                    _mm512_add_pd(
                        _mm512_sub_pd(neg(dmv), _mm512_mul_pd(cx, m)),
                        _mm512_mul_pd(_mm512_mul_pd(cc, m), vix),
                    ),
                    pc,
                ),
                _mm512_add_pd(vx, cc),
            );
            st(f[0].0.as_mut_ptr().add(o), px);
            st(f[1].0.as_mut_ptr().add(o), mx);
            st(f[2].0.as_mut_ptr().add(o), ax);
            st(f[3].0.as_mut_ptr().add(o), cx);
        }
        let (o01, o23) = out.split_at_mut(2);
        let (o0, o1) = o01.split_at_mut(1);
        let (o2, o3) = o23.split_at_mut(1);
        matvec2_impl(&ctx.t_odd, &f[0].0, &f[1].0, &mut o0[0], &mut o1[0], scale);
        matvec2_impl(&ctx.t_even, &f[2].0, &f[3].0, &mut o2[0], &mut o3[0], scale);
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn axpy_impl<const N: usize>(s: &State<f64, N>, a: f64, k: &State<f64, N>, out: &mut State<f64, N>) {
    const { assert!(N % 8 == 0) };
    unsafe {
        let va = _mm512_set1_pd(a);
        for q in 0..4 {
            for c in 0..N / 8 {
                let o = 8 * c;
                let v = _mm512_add_pd(ld(s[q].0.as_ptr().add(o)), _mm512_mul_pd(ld(k[q].0.as_ptr().add(o)), va));
                st(out[q].0.as_mut_ptr().add(o), v);
            }
        }
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn rk_update_impl<const N: usize>(
    s: &mut State<f64, N>,
    h6: f64,
    k1: &State<f64, N>,
    k2: &State<f64, N>,
    k3: &State<f64, N>,
    k4: &State<f64, N>,
) {
    const { assert!(N % 8 == 0) };
    unsafe {
        let vh = _mm512_set1_pd(h6);
        let two = _mm512_set1_pd(2.0);
        for q in 0..4 {
            for c in 0..N / 8 {
                let o = 8 * c;
                let t = _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_add_pd(ld(k1[q].0.as_ptr().add(o)), _mm512_mul_pd(ld(k2[q].0.as_ptr().add(o)), two)),
                        _mm512_mul_pd(ld(k3[q].0.as_ptr().add(o)), two),
                    ),
                    ld(k4[q].0.as_ptr().add(o)),
                );
                let v = _mm512_add_pd(ld(s[q].0.as_ptr().add(o)), _mm512_mul_pd(t, vh));
                st(s[q].0.as_mut_ptr().add(o), v);
            }
        }
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn lu_solve_impl<const N: usize>(cols: &mut [Fld<f64, N>], b: &Fld<f64, N>, nh: usize) -> Fld<f64, N> {
    const { assert!(N % 8 == 0 && N <= 8 * MAXV) };
    assert!(nh <= N && cols.len() >= nh);
    let nv = N / 8;
    let mut b = *b;
    unsafe {
        for k in 0..nh {
            let p = pivot_simd::<N>(&cols[k], k, nh);
            swap_rows(cols, &mut b, k, p, nh);
            let inv = _mm512_set1_pd(1.0 / cols[k].0[k]);
            let c0 = (k + 1) / 8;
            // multipliers: lanes i > k of column k times 1/pivot, zero elsewhere
            let mut l = [_mm512_setzero_pd(); MAXV];
            let ckp = cols[k].0.as_ptr();
            for c in c0..nv {
                let m = gt_mask(k, c);
                l[c] = _mm512_maskz_mul_pd(m, ld(ckp.add(8 * c)), inv);
            }
            for j in k + 1..nh {
                let cj = cols.get_unchecked_mut(j).0.as_mut_ptr();
                let u = _mm512_set1_pd(*cj.add(k));
                for c in c0..nv {
                    st(cj.add(8 * c), _mm512_fnmadd_pd(l[c], u, ld(cj.add(8 * c))));
                }
            }
            let u = _mm512_set1_pd(b.0[k]);
            let bp = b.0.as_mut_ptr();
            for c in c0..nv {
                st(bp.add(8 * c), _mm512_fnmadd_pd(l[c], u, ld(bp.add(8 * c))));
            }
        }
        // back substitution, column oriented: b[i] -= x_k a[i][k] for i < k
        let mut x = Fld::<f64, N>::zero();
        let bp = b.0.as_mut_ptr();
        for k in (0..nh).rev() {
            let xk = *bp.add(k) / cols[k].0[k];
            x.0[k] = xk;
            let vx = _mm512_set1_pd(xk);
            let ckp = cols[k].0.as_ptr();
            let full = k / 8;
            for c in 0..full {
                st(bp.add(8 * c), _mm512_fnmadd_pd(ld(ckp.add(8 * c)), vx, ld(bp.add(8 * c))));
            }
            let rem = k % 8;
            if rem > 0 {
                let o = 8 * full;
                let m: __mmask8 = ((1u32 << rem) - 1) as __mmask8;
                let v = _mm512_fnmadd_pd(ld(ckp.add(o)), vx, ld(bp.add(o)));
                st(bp.add(o), _mm512_mask_blend_pd(m, ld(bp.add(o)), v));
            }
        }
        x
    }
}

/// Same result as model::pivot_scalar (first index of the max |a_ik|, i in k..nh): exact max
/// reduction over the masked lanes, then the lowest lane index attaining it.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn pivot_simd<const N: usize>(ck: &Fld<f64, N>, k: usize, nh: usize) -> usize {
    let nv = N / 8;
    let p = ck.0.as_ptr();
    let c0 = k / 8;
    let c1 = (nh + 7) / 8;
    let mut vmax = _mm512_setzero_pd();
    let mut masks = [0u8; MAXV];
    for c in c0..c1.min(nv) {
        let m = range_mask(k, nh, c);
        masks[c] = m;
        let v = _mm512_abs_pd(ld(p.add(8 * c)));
        vmax = _mm512_mask_max_pd(vmax, m, vmax, v);
    }
    let best = _mm512_reduce_max_pd(vmax);
    let vb = _mm512_set1_pd(best);
    for c in c0..c1.min(nv) {
        let v = _mm512_abs_pd(ld(p.add(8 * c)));
        let hit = _mm512_mask_cmp_pd_mask::<_CMP_EQ_OQ>(masks[c], v, vb);
        if hit != 0 {
            return 8 * c + hit.trailing_zeros() as usize;
        }
    }
    k // all NaN: keep the diagonal, like the scalar search
}

/// lane mask of chunk c selecting global indices k <= i < nh
#[inline(always)]
fn range_mask(k: usize, nh: usize, c: usize) -> __mmask8 {
    let mut m: u32 = 0;
    for l in 0..8 {
        let i = 8 * c + l;
        if i >= k && i < nh {
            m |= 1 << l;
        }
    }
    m as __mmask8
}

/// lane mask of chunk c selecting global indices i > k
#[inline(always)]
fn gt_mask(k: usize, c: usize) -> __mmask8 {
    let base = 8 * c;
    if k + 1 <= base {
        0xff
    } else {
        let skip = k + 1 - base; // lanes 0..skip are <= k
        if skip >= 8 {
            0
        } else {
            (0xffu32 << skip) as __mmask8
        }
    }
}

/// Run `f(c0, g, out)` over the N/8 vector chunks in register-sized groups (g <= 5 chunks).
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn grouped<const N: usize>(mut f: impl FnMut(usize, usize, *mut f64), out: *mut f64) {
    let nv = N / 8;
    let mut c = 0;
    while c < nv {
        let g = (nv - c).min(5);
        f(c, g, out);
        c += g;
    }
}

/// out[c0..c0+g chunks] = sum_{j=0..k} (w_j x[j]) * y[off-j]  with w_j = j if weighted else 1 (no multiply)
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn sum_group<const N: usize>(
    x: &[Fld<f64, N>],
    y: &[Fld<f64, N>],
    k: usize,
    off: usize,
    weighted: bool,
    c0: usize,
    g: usize,
    out: *mut f64,
) {
    macro_rules! body {
        ($G:expr) => {{
            let mut a = [_mm512_setzero_pd(); $G];
            for j in 0..=k {
                let xp = x.get_unchecked(j).0.as_ptr().add(8 * c0);
                let yp = y.get_unchecked(off - j).0.as_ptr().add(8 * c0);
                if weighted {
                    let w = _mm512_set1_pd(j as f64);
                    for q in 0..$G {
                        a[q] = _mm512_fmadd_pd(_mm512_mul_pd(ld(xp.add(8 * q)), w), ld(yp.add(8 * q)), a[q]);
                    }
                } else {
                    for q in 0..$G {
                        a[q] = _mm512_fmadd_pd(ld(xp.add(8 * q)), ld(yp.add(8 * q)), a[q]);
                    }
                }
            }
            for q in 0..$G {
                st(out.add(8 * (c0 + q)), a[q]);
            }
        }};
    }
    match g {
        5 => body!(5),
        4 => body!(4),
        3 => body!(3),
        2 => body!(2),
        _ => body!(1),
    }
}
