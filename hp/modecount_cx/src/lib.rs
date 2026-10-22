//! det M(lambda) of the double-double DSS kernel (hp/) at complex lambda.
//!
//! hpkernel's `Solver.mode_det(zh, zl, lam_hi, lam_lo)` takes lambda as a real double-double (hi, lo), so the growth
//! rate can only be real there. This crate reuses hpkernel's arithmetic, fields, layout (`Ctx`) and LU, and repeats
//! the parts of `model.rs` where lambda enters with a complex lambda = lr + i li:
//!
//! * `Dc`: a value (double-double lanes) with a complex forward-mode tangent (re, im). The background is real, so the
//!   linearisation is real-linear in the tangent and every rule is the real dual rule applied to re and im separately;
//! * `dtau_c`: d/dtau = omega D, plus lambda (complex) times the tangent: re += lr re - li im, im += lr im + li re;
//! * `psolve_c` / `horizon_lus_c`: the horizon series' periodic solves (A + lambda) f_d = r_d - kappa_d f_v with
//!   A = omega D + diag kappa_kk, as the real 2M x 2M system [[A + lr, -li], [li, A + lr]];
//! * `mode_matrix_c`: the (L-1) x (L-1) mode matrix (Delta fixed, no phase row), as hpkernel's `jacobian` with
//!   `Ctx::lam` set, but complex;
//! * `clu_logdet`: complex double-double LU with partial pivoting (getf2 order, 8-row lane vectors):
//!   ln|det| and arg det.
//!
//! With li = 0 the matrix equals hpkernel's mode matrix (checked by hp/modecount_check.py).
use hpk::dd::DD;
use hpk::dv::{DotAcc, DV};
use hpk::field::{cs, cs_range, cs_w, horner, DMat, Fl, State};
use hpk::lane::Lane;
use hpk::linalg::{ld_for, DDLu};
use hpk::model::{self as hm, Ctx};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------------------------
// value + complex tangent
// ---------------------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct Dc {
    pub v: DV,
    pub r: DV,
    pub i: DV,
}

impl Add for Dc {
    type Output = Dc;
    #[inline(always)]
    fn add(self, o: Dc) -> Dc {
        Dc { v: self.v + o.v, r: self.r + o.r, i: self.i + o.i }
    }
}
impl Sub for Dc {
    type Output = Dc;
    #[inline(always)]
    fn sub(self, o: Dc) -> Dc {
        Dc { v: self.v - o.v, r: self.r - o.r, i: self.i - o.i }
    }
}
impl Neg for Dc {
    type Output = Dc;
    #[inline(always)]
    fn neg(self) -> Dc {
        Dc { v: -self.v, r: -self.r, i: -self.i }
    }
}
impl Mul for Dc {
    type Output = Dc;
    #[inline(always)]
    fn mul(self, o: Dc) -> Dc {
        let r = DotAcc::zero().add_prod(self.v, o.r).add_prod(self.r, o.v).finish();
        let i = DotAcc::zero().add_prod(self.v, o.i).add_prod(self.i, o.v).finish();
        Dc { v: self.v * o.v, r, i }
    }
}
impl Div for Dc {
    type Output = Dc;
    #[inline(always)]
    fn div(self, o: Dc) -> Dc {
        let q = self.v / o.v;
        Dc { v: q, r: (self.r - q * o.r) / o.v, i: (self.i - q * o.i) / o.v }
    }
}

impl Lane for Dc {
    type Acc = (DotAcc, DotAcc, DotAcc);
    const TANGENT: bool = true;
    #[inline(always)]
    fn zero() -> Self {
        Dc { v: DV::zero(), r: DV::zero(), i: DV::zero() }
    }
    #[inline(always)]
    fn cst(c: DV) -> Self {
        Dc { v: c, r: DV::zero(), i: DV::zero() }
    }
    #[inline(always)]
    fn mulc(self, c: DV) -> Self {
        Dc { v: self.v * c, r: self.r * c, i: self.i * c }
    }
    #[inline(always)]
    fn mulf(self, s: f64) -> Self {
        Dc { v: self.v.mul_f(s), r: self.r.mul_f(s), i: self.i.mul_f(s) }
    }
    #[inline(always)]
    fn addc(self, c: DV) -> Self {
        Dc { v: self.v + c, r: self.r, i: self.i }
    }
    #[inline(always)]
    fn bcast(self, j: usize) -> Self {
        Dc { v: self.v.bcast(j), r: self.r.bcast(j), i: self.i.bcast(j) }
    }
    #[inline(always)]
    fn val(self) -> DV {
        self.v
    }
    /// real part of the tangent (the generic helpers that use it are not called with Dc)
    #[inline(always)]
    fn tan(self) -> DV {
        self.r
    }
    #[inline(always)]
    fn from_vt(v: DV, t: DV) -> Self {
        Dc { v, r: t, i: DV::zero() }
    }
    #[inline(always)]
    fn acc0() -> Self::Acc {
        (DotAcc::zero(), DotAcc::zero(), DotAcc::zero())
    }
    #[inline(always)]
    fn acc(a: Self::Acc, x: Self, y: Self) -> Self::Acc {
        (
            a.0.add_prod(x.v, y.v),
            a.1.add_prod(x.v, y.r).add_prod(x.r, y.v),
            a.2.add_prod(x.v, y.i).add_prod(x.i, y.v),
        )
    }
    #[inline(always)]
    fn accc(a: Self::Acc, x: Self, c: DV) -> Self::Acc {
        (a.0.add_prod(x.v, c), a.1.add_prod(x.r, c), a.2.add_prod(x.i, c))
    }
    #[inline(always)]
    fn acc1(a: Self::Acc, x: Self) -> Self::Acc {
        (a.0.add(x.v), a.1.add(x.r), a.2.add(x.i))
    }
    #[inline(always)]
    fn fin(a: Self::Acc) -> Self {
        Dc { v: a.0.finish(), r: a.1.finish(), i: a.2.finish() }
    }
}

fn tan_re<const B: usize>(x: &Fl<Dc, B>) -> Fl<DV, B> {
    let mut r = [DV::zero(); B];
    for b in 0..B {
        r[b] = x.0[b].r;
    }
    Fl(r)
}
fn tan_im<const B: usize>(x: &Fl<Dc, B>) -> Fl<DV, B> {
    let mut r = [DV::zero(); B];
    for b in 0..B {
        r[b] = x.0[b].i;
    }
    Fl(r)
}
fn from_vri<const B: usize>(v: &Fl<DV, B>, re: &Fl<DV, B>, im: &Fl<DV, B>) -> Fl<Dc, B> {
    let mut r = [Dc::zero(); B];
    for b in 0..B {
        r[b] = Dc { v: v.0[b], r: re.0[b], i: im.0[b] };
    }
    Fl(r)
}

/// The layout (hpkernel's Ctx) and the complex growth rate.
pub struct Cx<'a, const B: usize> {
    pub ctx: &'a Ctx<B>,
    pub lr: DD,
    pub li: DD,
}

#[inline(always)]
fn recip_int(k: usize) -> DD {
    DD::ONE / DD::from_f64(k as f64)
}

/// d/dtau = omega D, plus lambda on the (complex) tangent.
#[inline(always)]
fn dtau_c<const B: usize>(cx: &Cx<B>, x: &Fl<Dc, B>, omega: Dc) -> Fl<Dc, B> {
    let mut r = cx.ctx.d.apply(x).mul_t(omega);
    let a = DV::splat(cx.lr);
    let b = DV::splat(cx.li);
    let nb = DV::splat(-cx.li);
    for k in 0..B {
        let t = x.0[k];
        let re = DotAcc::zero().add_prod(t.r, a).add_prod(t.i, nb).finish();
        let im = DotAcc::zero().add_prod(t.i, a).add_prod(t.r, b).finish();
        r.0[k].r = r.0[k].r + re;
        r.0[k].i = r.0[k].i + im;
    }
    r
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (hpkernel model.rs centre_series with dtau -> dtau_c)
// ---------------------------------------------------------------------------------------------------------------
fn centre_series_c<const B: usize>(cx: &Cx<B>, p1: &Fl<Dc, B>, c0: &Fl<Dc, B>, omega: Dc) -> [Vec<Fl<Dc, B>>; 4] {
    let ctx = cx.ctx;
    let n = ctx.n;
    let n1 = n - DD::ONE;
    let nh = n.mul_f(0.5);
    let kmax = ctx.kc;
    let z = Fl::<Dc, B>::zero();
    let mut pp = vec![z; kmax + 1];
    let mut mm = vec![z; kmax + 1];
    let mut aa = vec![z; kmax + 1];
    let mut cc = vec![z; kmax + 1];
    let mut q2 = vec![z; kmax + 1];
    let mut a2 = vec![z; kmax + 1];
    aa[0] = Fl::splat(DD::ONE);
    cc[0] = *c0;
    pp[1] = *p1;
    mm[1] = -*p1;
    a2[0] = aa[0] * aa[0];
    let invc = Fl::<Dc, B>::splat(DD::ONE) / cc[0];
    for k in 2..=kmax {
        let kf = k as f64;
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        a2[k] = cs(&aa, &aa, k);
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        aa[k] = (s.mulf(0.5) - t3.muls(n1.mul_f(0.5))).muls(DD::ONE / (DD::from_f64(kf) + n1));
        a2[k] = cs(&aa, &aa, k);
        cc[k] = cs_range(&cc, &a2, k, 0, k - 1).muls(n1 * recip_int(k));
        let dp = dtau_c(cx, &pp[k - 1], omega);
        let dm = dtau_c(cx, &mm[k - 1], omega);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0) + s1m;
        let xcxp = cs_w(&cc, &pp, k, 1, k, |j| j as f64);
        let xcxm = cs_w(&cc, &mm, k, 1, k, |j| j as f64);
        let cp = cs(&cc, &pp, k);
        let cm = cs(&cc, &mm, k);
        let cpm = cp + cm;
        let r3 = (((dp + xpx) - xcxp) + cp) - cpm.muls(nh);
        let r4 = (((dm + xmx) + xcxm) - cm) + cpm.muls(nh);
        let mu11 = DD::from_f64(1.0 - kf) - nh;
        let mu12 = -nh;
        let mu21 = nh;
        let mu22 = DD::from_f64(kf - 1.0) + nh;
        let idel = DD::ONE / (-(DD::from_f64(kf - 1.0) * (DD::from_f64(kf - 1.0) + n)));
        pp[k] = ((r4.muls(mu12) - r3.muls(mu22)) * invc).muls(idel);
        mm[k] = ((r3.muls(mu21) - r4.muls(mu11)) * invc).muls(idel);
    }
    [pp, mm, aa, cc]
}

fn centre_state_c<const B: usize>(cx: &Cx<B>, p1: &Fl<Dc, B>, c0: &Fl<Dc, B>, omega: Dc) -> State<Dc, B> {
    let s = centre_series_c(cx, p1, c0, omega);
    let x = DD::from_f64(cx.ctx.nodes[0]);
    [horner(&s[0], x), horner(&s[1], x), horner(&s[2], x), horner(&s[3], x)]
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor stepping in x (hpkernel model.rs taylor_coeffs / taylor_step / taylor_path with dtau -> dtau_c)
// ---------------------------------------------------------------------------------------------------------------
struct TWork<const B: usize> {
    p: Vec<Fl<Dc, B>>,
    m: Vec<Fl<Dc, B>>,
    a: Vec<Fl<Dc, B>>,
    c: Vec<Fl<Dc, B>>,
    a2: Vec<Fl<Dc, B>>,
    q2: Vec<Fl<Dc, B>>,
    g: Vec<Fl<Dc, B>>,
    h: Vec<Fl<Dc, B>>,
    pth: Vec<Fl<Dc, B>>,
    mth: Vec<Fl<Dc, B>>,
    ww: Vec<Fl<Dc, B>>,
    vv: Vec<Fl<Dc, B>>,
}

impl<const B: usize> TWork<B> {
    fn new(kt: usize) -> Self {
        let v = || vec![Fl::<Dc, B>::zero(); kt + 2];
        TWork { p: v(), m: v(), a: v(), c: v(), a2: v(), q2: v(), g: v(), h: v(), pth: v(), mth: v(), ww: v(), vv: v() }
    }
}

fn taylor_coeffs_c<const B: usize>(cx: &Cx<B>, s: &State<Dc, B>, x0: f64, omega: Dc, w: &mut TWork<B>) {
    let ctx = cx.ctx;
    let kt = ctx.kt;
    let n1 = ctx.n - DD::ONE;
    let nh = ctx.n.mul_f(0.5);
    let x0d = DD::from_f64(x0);
    let inv_x0 = DD::ONE / x0d;
    let x2 = [x0d * x0d, x0d.mul_f(2.0), DD::ONE];
    w.p[0] = s[0];
    w.m[0] = s[1];
    w.a[0] = s[2];
    w.c[0] = s[3];
    let mut iw0 = Fl::<Dc, B>::zero();
    let mut iv0 = Fl::<Dc, B>::zero();
    for k in 0..kt {
        let kf = k as f64;
        w.a2[k] = cs(&w.a, &w.a, k);
        w.q2[k] = cs(&w.p, &w.p, k) + cs(&w.m, &w.m, k);
        w.g[k] = (cs(&w.c, &w.a2, k) - w.c[k]).muls(n1);
        w.h[k] = cs(&w.a, &w.q2, k) + (w.a[k] - cs(&w.a, &w.a2, k)).muls(n1);
        w.pth[k] = dtau_c(cx, &w.p[k], omega);
        w.mth[k] = dtau_c(cx, &w.m[k], omega);
        let mut xp = w.pth[k].muls(x0d);
        let mut xm = w.mth[k].muls(x0d);
        if k > 0 {
            xp = xp + w.pth[k - 1];
            xm = xm + w.mth[k - 1];
        }
        let cp = cs(&w.c, &w.p, k);
        let cm = cs(&w.c, &w.m, k);
        let cpm = cp + cm;
        let rp = ((cs(&w.g, &w.p, k) - xp) - cp) + cpm.muls(nh);
        let rm = ((cm - xm) - cs(&w.g, &w.m, k)) - cpm.muls(nh);
        let ik = inv_x0 * recip_int(k + 1);
        w.c[k + 1] = (w.g[k] - w.c[k].mulf(kf)).muls(ik);
        w.a[k + 1] = (w.h[k].mulf(0.5) - w.a[k].mulf(kf)).muls(ik);
        let mut xc = w.c[k].muls(x0d);
        if k > 0 {
            xc = xc + w.c[k - 1];
        }
        if k <= 2 {
            w.ww[k] = (-xc).adds(x2[k]);
            w.vv[k] = xc.adds(x2[k]);
        } else {
            w.ww[k] = -xc;
            w.vv[k] = xc;
        }
        if k == 0 {
            iw0 = Fl::<Dc, B>::splat(DD::ONE) / w.ww[0];
            iv0 = Fl::<Dc, B>::splat(DD::ONE) / w.vv[0];
        }
        let ik1 = recip_int(k + 1);
        let (sp, sm) = if k >= 1 {
            (cs_w(&w.ww, &w.p, k + 1, 1, k, |j| (k + 1 - j) as f64), cs_w(&w.vv, &w.m, k + 1, 1, k, |j| (k + 1 - j) as f64))
        } else {
            (Fl::zero(), Fl::zero())
        };
        w.p[k + 1] = ((rp - sp) * iw0).muls(ik1);
        w.m[k + 1] = ((rm - sm) * iv0).muls(ik1);
    }
}

fn taylor_path_c<const B: usize>(cx: &Cx<B>, s0: &State<Dc, B>, xs: &[f64], omega: Dc, w: &mut TWork<B>) -> State<Dc, B> {
    let kt = cx.ctx.kt;
    let mut s = *s0;
    for i in 0..xs.len() - 1 {
        taylor_coeffs_c(cx, &s, xs[i], omega, w);
        let h = DD::from_f64(xs[i + 1]) - DD::from_f64(xs[i]);
        s = [horner(&w.p[..=kt], h), horner(&w.m[..=kt], h), horner(&w.a[..=kt], h), horner(&w.c[..=kt], h)];
    }
    s
}

// ---------------------------------------------------------------------------------------------------------------
// horizon series (hpkernel model.rs horizon_series with complex lambda in the tangent solves)
// ---------------------------------------------------------------------------------------------------------------

/// LU factors of the real 2M x 2M form of (omega D + diag kappa_kk + lambda), kk = 0..=ks:
/// [[A + lr, -li], [li, A + lr]] acting on (f_re, f_im).
fn horizon_lus_c<const B: usize>(ctx: &Ctx<B>, a0: &Fl<DV, B>, delta: DD, lr: DD, li: DD, par: bool) -> Result<Vec<DDLu>, String> {
    let m = Ctx::<B>::M;
    let m2 = 2 * m;
    let ld = ld_for(m2);
    let omega = DD::TWO_PI / delta;
    let n1 = ctx.n - DD::ONE;
    let c1 = (Fl::<DV, B>::splat(DD::ONE) - *a0 * *a0).muls(n1);
    let c1s: Vec<DD> = (0..m).map(|i| c1.get(i)).collect();
    let mut base_h = vec![0.0; ld * m2];
    let mut base_l = vec![0.0; ld * m2];
    for j in 0..m {
        let col = ctx.d.cols[j].muls(omega);
        let mut h = vec![0.0; m];
        let mut l = vec![0.0; m];
        col.store(&mut h, &mut l);
        for i in 0..m {
            // top-left and bottom-right blocks: omega D
            base_h[j * ld + i] = h[i];
            base_l[j * ld + i] = l[i];
            base_h[(j + m) * ld + i + m] = h[i];
            base_l[(j + m) * ld + i + m] = l[i];
        }
        // top-right: -li I (column j + m, row j); bottom-left: +li I (column j, row j + m)
        base_h[(j + m) * ld + j] = -li.hi;
        base_l[(j + m) * ld + j] = -li.lo;
        base_h[j * ld + j + m] = li.hi;
        base_l[j * ld + j + m] = li.lo;
    }
    let nh = ctx.n.mul_f(0.5);
    let one_kk = |kk: usize| -> Result<DDLu, String> {
        let mut h = base_h.clone();
        let mut l = base_l.clone();
        for i in 0..m {
            let kap = (c1s[i] + DD::ONE).mul_f(kk as f64) + c1s[i] + DD::ONE - nh + lr;
            for &ii in &[i, i + m] {
                let v = DD { hi: h[ii * ld + ii], lo: l[ii * ld + ii] } + kap;
                h[ii * ld + ii] = v.hi;
                l[ii * ld + ii] = v.lo;
            }
        }
        DDLu::factor(m2, ld, h, l, false)
    };
    if par {
        (0..=ctx.ks).into_par_iter().map(one_kk).collect()
    } else {
        (0..=ctx.ks).map(one_kk).collect()
    }
}

/// (omega D + diag kappa) f = r at the values (lu, real M x M); for the complex tangent
/// (A + lambda) f_d = r_d - kappa_d f_v (Delta fixed, so omega has no tangent), with lu2 the 2M x 2M real form.
fn psolve_c<const B: usize>(lu: &DDLu, lu2: &DDLu, kappa: &Fl<Dc, B>, rhs: &Fl<Dc, B>) -> Fl<Dc, B> {
    let m = 8 * B;
    let fv = lu.solve_lanes(&rhs.val());
    let rr = tan_re(rhs) - tan_re(kappa) * fv;
    let ri = tan_im(rhs) - tan_im(kappa) * fv;
    let mut h = vec![0.0; m];
    let mut l = vec![0.0; m];
    let mut b = vec![DD::ZERO; 2 * m];
    rr.store(&mut h, &mut l);
    for i in 0..m {
        b[i] = DD { hi: h[i], lo: l[i] };
    }
    ri.store(&mut h, &mut l);
    for i in 0..m {
        b[i + m] = DD { hi: h[i], lo: l[i] };
    }
    lu2.solve(&mut b);
    for i in 0..m {
        h[i] = b[i].hi;
        l[i] = b[i].lo;
    }
    let fr = Fl::<DV, B>::load(&h, &l);
    for i in 0..m {
        h[i] = b[i + m].hi;
        l[i] = b[i + m].lo;
    }
    let fi = Fl::<DV, B>::load(&h, &l);
    from_vri(&fv, &fr, &fi)
}

fn horizon_series_c<const B: usize>(cx: &Cx<B>, m0: &Fl<Dc, B>, a0: &Fl<Dc, B>, omega: Dc, lus: &[DDLu], lus2: &[DDLu]) -> [Vec<Fl<Dc, B>>; 4] {
    let ctx = cx.ctx;
    let n1 = ctx.n - DD::ONE;
    let nh = ctx.n.mul_f(0.5);
    let ks = ctx.ks;
    let len = ks + 2;
    let z = Fl::<Dc, B>::zero();
    let one = Fl::<Dc, B>::splat(DD::ONE);
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
    aa[0] = *a0;
    cc[0] = one;
    mm[0] = *m0;
    a2[0] = aa[0] * aa[0];
    dm[0] = dtau_c(cx, &mm[0], omega);
    let wfun = |c: &Vec<Fl<Dc, B>>, i: usize| -> Fl<Dc, B> {
        match i {
            0 => one - c[0],
            1 => -c[1] - one,
            _ => -c[i],
        }
    };
    let zfun = |c: &Vec<Fl<Dc, B>>, i: usize| -> Fl<Dc, B> {
        match i {
            0 => c[0] + one,
            1 => c[1] - one,
            _ => c[i],
        }
    };
    ww[0] = wfun(&cc, 0);
    zc[0] = zfun(&cc, 0);
    let izc0 = one / zc[0];
    for kk in 0..=ks {
        let kf = kk as f64;
        let ik1 = recip_int(kk + 1);
        let t = cs(&cc, &a2, kk);
        cc[kk + 1] = (cc[kk].mulf(kf) - (t - cc[kk]).muls(n1)).muls(ik1);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64);
        let tau = if kk >= 1 { -dp[kk - 1] } else { z };
        let r = (((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).muls(nh);
        let kappa = ((cc[1] + one).mulf(kf) + cc[1] + one).adds(-nh);
        pp[kk] = psolve_c(&lus[kk], &lus2[kk], &kappa, &(-r));
        dp[kk] = dtau_c(cx, &pp[kk], omega);
        q2[kk] = cs(&pp, &pp, kk) + cs(&mm, &mm, kk);
        if kk == ks {
            break;
        }
        let a3 = cs(&aa, &a2, kk);
        let ag = cs(&aa, &q2, kk) + (aa[kk] - a3).muls(n1);
        aa[kk + 1] = (aa[kk].mulf(kf) - ag.mulf(0.5)).muls(ik1);
        a2[kk + 1] = cs(&aa, &aa, kk + 1);
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1] } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64);
        let rest = (((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).muls(nh);
        mm[kk + 1] = (rest * izc0).muls(ik1);
        dm[kk + 1] = dtau_c(cx, &mm[kk + 1], omega);
    }
    pp.truncate(ks + 1);
    mm.truncate(ks + 1);
    aa.truncate(ks + 1);
    cc.truncate(ks + 1);
    [pp, mm, aa, cc]
}

fn block_state_c<const B: usize>(cx: &Cx<B>, b: usize, inp: &[Fl<Dc, B>], delta: Dc, lus: &[DDLu], lus2: &[DDLu]) -> State<Dc, B> {
    let ctx = cx.ctx;
    let omega = Dc::splat(DD::TWO_PI) / delta;
    let k = ctx.k();
    if b == 0 {
        centre_state_c(cx, &inp[0], &inp[1], omega)
    } else if b < k {
        let mut w = TWork::<B>::new(ctx.kt);
        taylor_path_c(cx, &[inp[0], inp[1], inp[2], inp[3]], &ctx.segs[b - 1], omega, &mut w)
    } else {
        let s = horizon_series_c(cx, &inp[0], &inp[1], omega, lus, lus2);
        let u = ctx.u0();
        [horner(&s[0], u), horner(&s[1], u), horner(&s[2], u), horner(&s[3], u)]
    }
}

/// Complex mode matrix M(lambda), (L-1) x (L-1), column-major with leading dimension ld_for(L-1):
/// (re_hi, re_lo, im_hi, im_lo, ld).
#[allow(clippy::type_complexity)]
pub fn mode_matrix_c<const B: usize>(cx: &Cx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize), String> {
    let ctx = cx.ctx;
    let l = ctx.len();
    if zh.len() != l || zl.len() != l {
        return Err(format!("z must have length {l}"));
    }
    let n = l - 1;
    let m = Ctx::<B>::M;
    let ld = ld_for(n);
    let delta = DD { hi: zh[l - 1], lo: zl[l - 1] };
    let ah = ctx.fld(zh, zl, ctx.off_ah());
    let lus = hm::horizon_lus(ctx, &ah, delta, par, DD::ZERO)?;
    let lus2 = horizon_lus_c(ctx, &ah, delta, cx.lr, cx.li, par)?;
    let nb = ctx.k() + 1;
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for b in 0..nb {
        let ins = ctx.block_inputs(b);
        for f in 0..ins.len() {
            for j in 0..m {
                tasks.push((b, f, j));
            }
        }
    }
    let column = |&(b, f, j): &(usize, usize, usize)| -> Vec<(DD, DD)> {
        let ins = ctx.block_inputs(b);
        let inp: Vec<Fl<Dc, B>> = ins
            .iter()
            .enumerate()
            .map(|(fi, &o)| {
                let v = ctx.fld(zh, zl, o);
                let mut t = Fl::<DV, B>::zero();
                if fi == f {
                    let mut th = vec![0.0; m];
                    let tl = vec![0.0; m];
                    th[j] = 1.0;
                    t = Fl::load(&th, &tl);
                }
                from_vri(&v, &t, &Fl::<DV, B>::zero())
            })
            .collect();
        let d = Dc::cst(DV::splat(delta));
        let s = block_state_c(cx, b, &inp, d, &lus, &lus2);
        let mut out = Vec::with_capacity(4 * m);
        for q in 0..4 {
            let mut t = s[q];
            if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
                t = lo.apply(&t);
            }
            let (tr, ti) = (tan_re(&t), tan_im(&t));
            for i in 0..m {
                out.push((tr.get(i), ti.get(i)));
            }
        }
        out
    };
    let cols: Vec<Vec<(DD, DD)>> = if par { tasks.par_iter().map(column).collect() } else { tasks.iter().map(column).collect() };
    let mut rh = vec![0.0; ld * n];
    let mut rl = vec![0.0; ld * n];
    let mut ih = vec![0.0; ld * n];
    let mut il = vec![0.0; ld * n];
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let c = ctx.block_inputs(b)[f] + j;
        for (r, (vr, vi)) in col.iter().enumerate() {
            let idx = c * ld + b * 4 * m + r;
            rh[idx] = vr.hi;
            rl[idx] = vr.lo;
            ih[idx] = vi.hi;
            il[idx] = vi.lo;
        }
    }
    let add_re = |rh: &mut Vec<f64>, rl: &mut Vec<f64>, idx: usize, v: DD| {
        let s = DD { hi: rh[idx], lo: rl[idx] } + v;
        rh[idx] = s.hi;
        rl[idx] = s.lo;
    };
    for b in 0..nb {
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            if let (true, Some((lo, hi))) = (b == ctx.k() && r < m, &ctx.freeze) {
                for j in 0..m {
                    add_re(&mut rh, &mut rl, (t + j) * ld + b * 4 * m + r, -lo.cols[j].get(r));
                    add_re(&mut rh, &mut rl, j * ld + b * 4 * m + r, hi.cols[j].get(r));
                }
                continue;
            }
            add_re(&mut rh, &mut rl, (t + r) * ld + b * 4 * m + r, -DD::ONE);
        }
    }
    Ok((rh, rl, ih, il, ld))
}

// ---------------------------------------------------------------------------------------------------------------
// complex double-double LU
// ---------------------------------------------------------------------------------------------------------------
#[inline(always)]
fn dd_at(h: &[f64], l: &[f64], i: usize) -> DD {
    DD { hi: h[i], lo: l[i] }
}

/// ln|det| and arg det (in (-pi, pi]) of the complex n x n matrix (column-major, leading dimension ld, ld % 8 == 0,
/// padding rows zero) by LU with partial pivoting; also min and max |u_kk| and the number of row interchanges.
#[allow(clippy::type_complexity)]
pub fn clu_logdet(n: usize, ld: usize, mut rh: Vec<f64>, mut rl: Vec<f64>, mut ih: Vec<f64>, mut il: Vec<f64>, par: bool) -> Result<(f64, f64, f64, f64), String> {
    assert!(ld % 8 == 0 && ld >= n);
    let mut la = 0.0f64;
    let mut arg = 0.0f64;
    let mut pmin = f64::INFINITY;
    let mut pmax = 0.0f64;
    let mut lrh = vec![0.0f64; ld];
    let mut lrl = vec![0.0f64; ld];
    let mut lih = vec![0.0f64; ld];
    let mut lil = vec![0.0f64; ld];
    for k in 0..n {
        let ck = k * ld;
        let mut p = k;
        let mut best = -1.0f64;
        for i in k..n {
            let v = rh[ck + i].hypot(ih[ck + i]);
            if v > best {
                best = v;
                p = i;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return Err(format!("singular or non-finite pivot at column {k} ({best})"));
        }
        if p != k {
            for j in 0..n {
                rh.swap(j * ld + k, j * ld + p);
                rl.swap(j * ld + k, j * ld + p);
                ih.swap(j * ld + k, j * ld + p);
                il.swap(j * ld + k, j * ld + p);
            }
            arg += std::f64::consts::PI;
        }
        let ur = dd_at(&rh, &rl, ck + k);
        let ui = dd_at(&ih, &il, ck + k);
        let mag = (ur.hi + ur.lo).hypot(ui.hi + ui.lo);
        // |u| to double-double accuracy is not needed for ln|u| beyond f64; the phase likewise
        la += mag.ln();
        arg += (ui.hi + ui.lo).atan2(ur.hi + ur.lo);
        pmin = pmin.min(mag);
        pmax = pmax.max(mag);
        if k + 1 == n {
            break;
        }
        // 1/u = conj(u) / |u|^2
        let den = ur * ur + ui * ui;
        let vr = ur / den;
        let vi = -(ui / den);
        for i in k + 1..n {
            let ar = dd_at(&rh, &rl, ck + i);
            let ai = dd_at(&ih, &il, ck + i);
            let xr = ar * vr - ai * vi;
            let xi = ar * vi + ai * vr;
            rh[ck + i] = xr.hi;
            rl[ck + i] = xr.lo;
            ih[ck + i] = xi.hi;
            il[ck + i] = xi.lo;
        }
        let r0 = 8 * ((k + 1) / 8);
        for i in r0..ld {
            if i > k && i < n {
                lrh[i] = rh[ck + i];
                lrl[i] = rl[ck + i];
                lih[i] = ih[ck + i];
                lil[i] = il[ck + i];
            } else {
                lrh[i] = 0.0;
                lrl[i] = 0.0;
                lih[i] = 0.0;
                lil[i] = 0.0;
            }
        }
        let ncols = n - k - 1;
        let (_, arh) = rh.split_at_mut((k + 1) * ld);
        let (_, arl) = rl.split_at_mut((k + 1) * ld);
        let (_, aih) = ih.split_at_mut((k + 1) * ld);
        let (_, ail) = il.split_at_mut((k + 1) * ld);
        let upd = |(((ch, cl), dh), dl): (((&mut [f64], &mut [f64]), &mut [f64]), &mut [f64])| {
            let tr = DD { hi: ch[k], lo: cl[k] };
            let ti = DD { hi: dh[k], lo: dl[k] };
            if tr.hi == 0.0 && ti.hi == 0.0 {
                return;
            }
            let ntr = DV::splat(-tr);
            let nti = DV::splat(-ti);
            let pti = DV::splat(ti);
            let mut i = r0;
            while i < ld {
                let ar = DV::load(&ch[i..i + 8], &cl[i..i + 8]);
                let ai = DV::load(&dh[i..i + 8], &dl[i..i + 8]);
                let lr = DV::load(&lrh[i..i + 8], &lrl[i..i + 8]);
                let li = DV::load(&lih[i..i + 8], &lil[i..i + 8]);
                // a -= l t:  re: ar - lr tr + li ti,  im: ai - lr ti - li tr
                let nr = DotAcc::zero().add(ar).add_prod(lr, ntr).add_prod(li, pti).finish();
                let ni = DotAcc::zero().add(ai).add_prod(lr, nti).add_prod(li, ntr).finish();
                nr.store(&mut ch[i..i + 8], &mut cl[i..i + 8]);
                ni.store(&mut dh[i..i + 8], &mut dl[i..i + 8]);
                i += 8;
            }
        };
        if par && ncols * (ld - r0) > 32768 {
            arh[..ncols * ld]
                .par_chunks_mut(ld)
                .zip(arl[..ncols * ld].par_chunks_mut(ld))
                .zip(aih[..ncols * ld].par_chunks_mut(ld))
                .zip(ail[..ncols * ld].par_chunks_mut(ld))
                .for_each(upd);
        } else {
            arh[..ncols * ld]
                .chunks_mut(ld)
                .zip(arl[..ncols * ld].chunks_mut(ld))
                .zip(aih[..ncols * ld].chunks_mut(ld))
                .zip(ail[..ncols * ld].chunks_mut(ld))
                .for_each(upd);
        }
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut a = arg.rem_euclid(two_pi);
    if a > std::f64::consts::PI {
        a -= two_pi;
    }
    Ok((la, a, pmin, pmax))
}

// ---------------------------------------------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------------------------------------------
type CMat = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize);

trait CApi: Send + Sync {
    fn len(&self) -> usize;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn mode_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<CMat, String>;
}

impl<const B: usize> CApi for Ctx<B> {
    fn len(&self) -> usize {
        Ctx::<B>::len(self)
    }
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String> {
        if segs.len() + 1 != self.k() {
            return Err(format!("need {} segments, got {}", self.k() - 1, segs.len()));
        }
        for (b, s) in segs.iter().enumerate() {
            let (from, to) = if self.seg_inward(b + 1) { (b + 1, b) } else { (b, b + 1) };
            if s.len() < 2 || s[0] != self.nodes[from] || *s.last().unwrap() != self.nodes[to] {
                return Err(format!("segment {} must run from node {} to node {}", b + 1, self.nodes[from], self.nodes[to]));
            }
        }
        self.segs = segs;
        Ok(())
    }
    fn mode_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<CMat, String> {
        let cx = Cx { ctx: self, lr, li };
        mode_matrix_c(&cx, zh, zl, par)
    }
}

fn freeze_pair<const B: usize>(p: &Option<(Vec<f64>, Vec<f64>)>) -> Option<(DMat<B>, DMat<B>)> {
    p.as_ref().map(|(h, l)| {
        let m = 8 * B;
        let lo = DMat::<B>::from_rows(h, l);
        let mut hh = vec![0.0; m * m];
        let mut hl = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                let v = DD::from_f64(if i == j { 1.0 } else { 0.0 }) - DD { hi: h[i * m + j], lo: l[i * m + j] };
                hh[i * m + j] = v.hi;
                hl[i * m + j] = v.lo;
            }
        }
        (lo, DMat::<B>::from_rows(&hh, &hl))
    })
}

macro_rules! make_ctx {
    ($b:expr, $n:expr, $dh:expr, $dl:expr, $ph:expr, $pl:expr, $nodes:expr, $kt:expr, $kc:expr, $ks:expr, $plow:expr, $mn:expr, [$($bb:literal),*]) => {
        match $b {
            $( $bb => Box::new(Ctx::<$bb> {
                n: $n,
                d: DMat::<$bb>::from_rows($dh, $dl),
                phase: Fl::<DV, $bb>::load($ph, $pl),
                nodes: $nodes,
                segs: vec![],
                kt: $kt,
                kc: $kc,
                ks: $ks,
                freeze: freeze_pair::<$bb>(&$plow),
                match_node: $mn,
                lam: DD::ZERO,
            }) as Box<dyn CApi>, )*
            _ => return Err(PyValueError::new_err("M must be one of 16, 32, 48, 64")),
        }
    };
}

fn slice<'a>(a: &'a PyReadonlyArray1<f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

/// Same constructor as hpkernel.Solver; segs must be given (take them from hpkernel's Solver.get_segs()).
#[pyclass]
struct CSolver {
    api: Box<dyn CApi>,
    pool: Arc<rayon::ThreadPool>,
}

#[pymethods]
impl CSolver {
    #[new]
    #[pyo3(signature = (m, n_hi, n_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc, ks, threads=1, plow_hi=None, plow_lo=None, match_node=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        m: usize,
        n_hi: f64,
        n_lo: f64,
        dmat_hi: PyReadonlyArray2<f64>,
        dmat_lo: PyReadonlyArray2<f64>,
        phase_hi: PyReadonlyArray1<f64>,
        phase_lo: PyReadonlyArray1<f64>,
        nodes: Vec<f64>,
        segs: Vec<Vec<f64>>,
        kt: usize,
        kc: usize,
        ks: usize,
        threads: usize,
        plow_hi: Option<PyReadonlyArray2<f64>>,
        plow_lo: Option<PyReadonlyArray2<f64>>,
        match_node: Option<usize>,
    ) -> PyResult<Self> {
        let dh = dmat_hi.as_slice().map_err(|_| PyValueError::new_err("dmat_hi must be C-contiguous"))?;
        let dl = dmat_lo.as_slice().map_err(|_| PyValueError::new_err("dmat_lo must be C-contiguous"))?;
        if m % 8 != 0 || dh.len() != m * m || dl.len() != m * m {
            return Err(PyValueError::new_err("dmat must be (M, M), M a multiple of 8"));
        }
        let ph = slice(&phase_hi, "phase_hi")?;
        let pl = slice(&phase_lo, "phase_lo")?;
        let n = DD::new(n_hi, n_lo);
        let plow: Option<(Vec<f64>, Vec<f64>)> = match (plow_hi, plow_lo) {
            (Some(a), Some(b)) => Some((
                a.as_slice().map_err(|_| PyValueError::new_err("plow"))?.to_vec(),
                b.as_slice().map_err(|_| PyValueError::new_err("plow"))?.to_vec(),
            )),
            _ => None,
        };
        let mn = match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1);
        let mut api: Box<dyn CApi> = make_ctx!(m / 8, n, dh, dl, ph, pl, nodes.clone(), kt, kc, ks, plow, mn, [2, 4, 6, 8]);
        api.set_segs(segs).map_err(PyValueError::new_err)?;
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap());
        Ok(CSolver { api, pool })
    }

    #[getter]
    fn size(&self) -> usize {
        self.api.len()
    }

    /// det M(lam) for lam = (lr_hi + lr_lo) + i (li_hi + li_lo): (ln|det|, arg det in (-pi, pi], min |u_kk|, max |u_kk|).
    #[pyo3(signature = (zh, zl, lr_hi, li_hi, lr_lo=0.0, li_lo=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn mode_det(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, lr_hi: f64, li_hi: f64, lr_lo: f64, li_lo: f64) -> PyResult<(f64, f64, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len() - 1;
        let (lr, li) = (DD::new(lr_hi, lr_lo), DD::new(li_hi, li_lo));
        py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let (rh, rl, ih, il, ld) = api.mode_matrix_c(h, l, lr, li, true)?;
                clu_logdet(n, ld, rh, rl, ih, il, true)
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// The complex mode matrix as four (L-1, L-1) arrays (re_hi, re_lo, im_hi, im_lo).
    #[allow(clippy::type_complexity)]
    fn mode_matrix<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len() - 1;
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.mode_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), true)))
            .map_err(PyValueError::new_err)?;
        let to2 = |v: Vec<f64>| {
            let mut a = Array2::<f64>::zeros((n, n));
            for c in 0..n {
                for r in 0..n {
                    a[[r, c]] = v[c * ld + r];
                }
            }
            a
        };
        Ok((to2(rh).into_pyarray(py), to2(rl).into_pyarray(py), to2(ih).into_pyarray(py), to2(il).into_pyarray(py)))
    }
}

/// ln|det| and arg det of a complex matrix given as four (n, n) double-double arrays (row index first): a test of
/// the complex LU against numpy / mpmath.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn cdet<'py>(rh: PyReadonlyArray2<'py, f64>, rl: PyReadonlyArray2<'py, f64>, ih: PyReadonlyArray2<'py, f64>, il: PyReadonlyArray2<'py, f64>) -> PyResult<(f64, f64, f64, f64)> {
    let a = rh.as_array();
    let n = a.shape()[0];
    let ld = ld_for(n);
    let mut v = [vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n]];
    for (q, src) in [rh.as_array(), rl.as_array(), ih.as_array(), il.as_array()].iter().enumerate() {
        for c in 0..n {
            for r in 0..n {
                v[q][c * ld + r] = src[[r, c]];
            }
        }
    }
    let [a0, a1, a2, a3] = v;
    clu_logdet(n, ld, a0, a1, a2, a3, false).map_err(PyValueError::new_err)
}

#[pymodule]
fn modecount_cx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CSolver>()?;
    m.add_function(wrap_pyfunction!(cdet, m)?)?;
    Ok(())
}
