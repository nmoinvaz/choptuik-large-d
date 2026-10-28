//! det M(lambda) of the double-double DSS kernel (hp/) at complex lambda, and the same for a test scalar field.
//!
//! hpkernel's `Solver.mode_det(zh, zl, lam_hi, lam_lo)` takes lambda as a real double-double (hi, lo), so the growth
//! rate can only be real there. This crate reuses hpkernel's arithmetic, fields, layout (`Ctx`) and LU, and repeats
//! the parts of `model.rs` where lambda enters with a complex lambda = lr + i li:
//!
//! * `Dc` (multidouble's lane, written here first): a value (double-double lanes) with a complex forward-mode tangent
//!   (re, im). The background is real, so the linearisation is real-linear in the tangent and every rule is the real
//!   dual rule applied to re and im separately;
//! * `dtau_c`: d/dtau = omega D, plus lambda (complex) times the tangent: re += lr re - li im, im += lr im + li re;
//! * `psolve_c` / `horizon_lus_c`: the horizon series' periodic solves (A + lambda) f_d = r_d - kappa_d f_v with
//!   A = omega D + diag kappa_kk, as the real 2M x 2M system [[A + lr, -li], [li, A + lr]] (li = 0: the M x M solves
//!   of hpkernel, for re and im separately);
//! * `centre_series_c`, `taylor_coeffs_c`, `horizon_series_c`: model.rs's series in the power-of-two scaled variables
//!   (model::series_scale), for the default layout (Ctx::horizon_alpha 0; no early series end, which is for f64 lanes);
//! * `mode_matrix_c`: the (L-1) x (L-1) mode matrix (Delta fixed, no phase row), as hpkernel's `jacobian` with
//!   `Ctx::lam` set, but complex;
//! * multidouble's `linalg::clu_logdet` (written here first): complex double-double LU with partial pivoting (getf2
//!   order, 8-row lane vectors): ln|det| and arg det.
//!
//! With li = 0 the matrix equals hpkernel's mode matrix bit for bit (checked by hp/modecount_check.py and
//! hp/modecount_test_check.py).
//!
//! **Test field** (`Cx::test`, notes/GHOST_ADMIXTURE.md): a second scalar field psi on the background, with no
//! back-reaction at linear order (a ghost admixture, or the imaginary part of a complex field). Its equations are the
//! p, m equations with a, c held at the background, so with test = TEST_ON the tangents of a and c are set to zero
//! wherever the series and the Taylor steps make them (inputs included): the tangents of p, m are then exactly the
//! test field's. `test_matrix_c` keeps the psi rows (p, m of every block) and the psi columns (p1, the nodes' p, m, m_h):
//! a 2M(K+1) square matrix whose determinant vanishes at the test field's growth rates. psi = the background's own
//! p, m solves it at lambda = 0 (the p, m equations are linear in p, m), so lambda = 0 is a zero.
//! TEST_MUT, the control, zeroes the tangents of c only: a's response to p, m stays in, and that identity breaks.
use hpk::dd::DD;
use hpk::dv::DV;
use hpk::field::{cs, cs_range, cs_w, horner, DMat, FieldSolve, Fl, State};
use hpk::lane::{BaseLane, Dc};
use hpk::linalg::{clu_logdet, ld_for, DDLu};
use hpk::model::{self as hm, series_scale, Ctx};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// `Cx::test`: the full linearisation
pub const TEST_OFF: u8 = 0;
/// `Cx::test`: the test field (a and c tangents zeroed)
pub const TEST_ON: u8 = 1;
/// `Cx::test`: the control (c tangents zeroed, a's kept)
pub const TEST_MUT: u8 = 2;

/// The layout (hpkernel's Ctx), the complex growth rate and the sector (TEST_*).
pub struct Cx<'a, const B: usize> {
    pub ctx: &'a Ctx<B>,
    pub lr: DD,
    pub li: DD,
    pub test: u8,
}

impl<const B: usize> Cx<'_, B> {
    /// the test field: drop the tangent of a coefficient of a (TEST_ON only) ...
    #[inline(always)]
    fn hold_a(&self, x: &mut Fl<Dc, B>) {
        if self.test == TEST_ON {
            hold(x);
        }
    }
    /// ... and of c (TEST_ON and its control TEST_MUT)
    #[inline(always)]
    fn hold_c(&self, x: &mut Fl<Dc, B>) {
        if self.test != TEST_OFF {
            hold(x);
        }
    }
}

/// the value alone: a coefficient of the background, held fixed under the perturbation
#[inline(always)]
fn hold<const B: usize>(x: &mut Fl<Dc, B>) {
    for b in 0..B {
        x.0[b].r = DV::zero();
        x.0[b].i = DV::zero();
    }
}

#[inline(always)]
fn recip_int(k: usize) -> DD {
    DD::ONE / DD::from_f64(k as f64)
}

/// d/dtau = omega D, plus lambda on the (complex) tangent. With li = 0 the same operations as model.rs's dtau.
#[inline(always)]
fn dtau_c<const B: usize>(cx: &Cx<B>, x: &Fl<Dc, B>, omega: Dc) -> Fl<Dc, B> {
    let mut r = cx.ctx.d.apply(x).mul_t(omega);
    if cx.lr.hi == 0.0 && cx.li.hi == 0.0 {
        return r;
    }
    let a = DV::splat(cx.lr);
    let b = DV::splat(cx.li);
    for k in 0..B {
        let t = x.0[k];
        r.0[k].r = r.0[k].r + (t.r * a - t.i * b);
        r.0[k].i = r.0[k].i + (t.i * a + t.r * b);
    }
    r
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (model.rs centre_series, in x / sigma, with dtau -> dtau_c)
// ---------------------------------------------------------------------------------------------------------------
fn centre_series_c<const B: usize>(cx: &Cx<B>, p1: &Fl<Dc, B>, c0: &Fl<Dc, B>, omega: Dc) -> [Vec<Fl<Dc, B>>; 4] {
    let ctx = cx.ctx;
    let sig = series_scale(ctx.nodes[0]);
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
    cx.hold_c(&mut cc[0]);
    pp[1] = p1.mul_pwr2(sig);
    mm[1] = -pp[1];
    a2[0] = aa[0] * aa[0];
    let invc = Fl::<Dc, B>::splat(DD::ONE) / cc[0];
    for k in 2..=kmax {
        let kf = k as f64;
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        a2[k] = cs(&aa, &aa, k); // aa[k] still 0
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        aa[k] = (s.mulf(0.5) - t3.muls(n1.mul_f(0.5))).muls(DD::ONE / (DD::from_f64(kf) + n1));
        cx.hold_a(&mut aa[k]);
        a2[k] = cs(&aa, &aa, k);
        cc[k] = cs_range(&cc, &a2, k, 0, k - 1).muls(n1 * recip_int(k));
        cx.hold_c(&mut cc[k]);
        let dp = dtau_c(cx, &pp[k - 1], omega).mul_pwr2(sig);
        let dm = dtau_c(cx, &mm[k - 1], omega).mul_pwr2(sig);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0).mul_pwr2(sig) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0).mul_pwr2(sig) + s1m;
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
    let x = DD::from_f64(cx.ctx.nodes[0]).mul_pwr2(1.0 / series_scale(cx.ctx.nodes[0]));
    [horner(&s[0], x), horner(&s[1], x), horner(&s[2], x), horner(&s[3], x)]
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor stepping in x (model.rs taylor_coeffs / taylor_step / taylor_path with dtau -> dtau_c)
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
    cx.hold_a(&mut w.a[0]);
    cx.hold_c(&mut w.c[0]);
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
        cx.hold_c(&mut w.c[k + 1]);
        cx.hold_a(&mut w.a[k + 1]);
        // W = x (x - c), V = x (x + c)
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
// horizon series (model.rs horizon_series, in u / sigma, with complex lambda in the tangent solves)
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

/// The horizon solves' factors: `lus` at the values; `lus_t` = A + lr (M x M, model::horizon_lus with shift lr; for
/// the tangents when li = 0, then empty when lr = 0 too: `lus` serves); `lus2` the 2M x 2M form when li != 0.
struct HLus {
    lus: Vec<DDLu>,
    lus_t: Vec<DDLu>,
    lus2: Vec<DDLu>,
}

impl HLus {
    fn new<const B: usize>(cx: &Cx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<HLus, String> {
        let ctx = cx.ctx;
        let l = ctx.len();
        let delta = DD { hi: zh[l - 1], lo: zl[l - 1] };
        let ah = ctx.fld(zh, zl, ctx.off_ah());
        let lus = hm::horizon_lus(ctx, &ah, delta, par, DD::ZERO)?;
        let (lus_t, lus2) = if cx.li.hi != 0.0 {
            (Vec::new(), horizon_lus_c(ctx, &ah, delta, cx.lr, cx.li, par)?)
        } else if cx.lr.hi != 0.0 {
            (hm::horizon_lus(ctx, &ah, delta, par, cx.lr)?, Vec::new())
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(HLus { lus, lus_t, lus2 })
    }
}

/// (omega D + diag kappa) f = r at the values (lu, real M x M); for the complex tangent
/// (A + lambda) f_d = r_d - kappa_d f_v (Delta fixed, so omega has no tangent): with li = 0 the M x M solves of A + lr
/// for re and im (model.rs psolve's operations), otherwise the 2M x 2M real form.
fn psolve_c<const B: usize>(h: &HLus, kk: usize, kappa: &Fl<Dc, B>, rhs: &Fl<Dc, B>) -> Fl<Dc, B> {
    let m = 8 * B;
    let fv = h.lus[kk].solve_lanes(&rhs.val());
    let rr = rhs.tan_re() - kappa.tan_re() * fv;
    let ri = rhs.tan_im() - kappa.tan_im() * fv;
    if h.lus2.is_empty() {
        let lu_t = if h.lus_t.is_empty() { &h.lus[kk] } else { &h.lus_t[kk] };
        return Fl::from_vri(&fv, &lu_t.solve_lanes(&rr), &lu_t.solve_lanes(&ri));
    }
    let mut hh = vec![0.0; m];
    let mut ll = vec![0.0; m];
    let mut b = vec![DD::ZERO; 2 * m];
    rr.store(&mut hh, &mut ll);
    for i in 0..m {
        b[i] = DD { hi: hh[i], lo: ll[i] };
    }
    ri.store(&mut hh, &mut ll);
    for i in 0..m {
        b[i + m] = DD { hi: hh[i], lo: ll[i] };
    }
    h.lus2[kk].solve(&mut b);
    for i in 0..m {
        hh[i] = b[i].hi;
        ll[i] = b[i].lo;
    }
    let fr = Fl::<DV, B>::load(&hh, &ll);
    for i in 0..m {
        hh[i] = b[i + m].hi;
        ll[i] = b[i + m].lo;
    }
    let fi = Fl::<DV, B>::load(&hh, &ll);
    Fl::from_vri(&fv, &fr, &fi)
}

fn horizon_series_c<const B: usize>(cx: &Cx<B>, m0: &Fl<Dc, B>, a0: &Fl<Dc, B>, omega: Dc, hl: &HLus) -> [Vec<Fl<Dc, B>>; 4] {
    let ctx = cx.ctx;
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
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
    cx.hold_a(&mut aa[0]);
    cc[0] = one;
    mm[0] = *m0;
    a2[0] = aa[0] * aa[0];
    dm[0] = dtau_c(cx, &mm[0], omega);
    let sone = one.mul_pwr2(sig);
    let wfun = |c: &Vec<Fl<Dc, B>>, i: usize| -> Fl<Dc, B> {
        match i {
            0 => one - c[0],
            1 => -c[1] - sone,
            _ => -c[i],
        }
    };
    let zfun = |c: &Vec<Fl<Dc, B>>, i: usize| -> Fl<Dc, B> {
        match i {
            0 => c[0] + one,
            1 => c[1] - sone,
            _ => c[i],
        }
    };
    ww[0] = wfun(&cc, 0);
    zc[0] = zfun(&cc, 0);
    let izc0 = one / zc[0];
    for kk in 0..=ks {
        let kf = kk as f64;
        let ik1 = recip_int(kk + 1);
        let ca1 = cs(&cc, &a2, kk) - cc[kk];
        cc[kk + 1] = (cc[kk].mulf(kf) - ca1.muls(n1)).muls(ik1).mul_pwr2(sig);
        cx.hold_c(&mut cc[kk + 1]);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk).mul_pwr2(sig);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64).mul_pwr2(isig) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk).mul_pwr2(sig);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let tau = if kk >= 1 { -dp[kk - 1].mul_pwr2(sig) } else { z };
        let r = (((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).muls(nh);
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).adds(-nh);
        pp[kk] = psolve_c(hl, kk, &kappa, &(-r));
        dp[kk] = dtau_c(cx, &pp[kk], omega);
        q2[kk] = cs(&pp, &pp, kk) + cs(&mm, &mm, kk);
        if kk == ks {
            break;
        }
        let ag = cs(&aa, &q2, kk) + (aa[kk] - cs(&aa, &a2, kk)).muls(n1);
        aa[kk + 1] = (aa[kk].mulf(kf) - ag.mulf(0.5)).muls(ik1).mul_pwr2(sig);
        cx.hold_a(&mut aa[kk + 1]);
        a2[kk + 1] = cs(&aa, &aa, kk + 1);
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1].mul_pwr2(sig) } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let rest = (((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).muls(nh);
        mm[kk + 1] = (rest * izc0).muls(ik1).mul_pwr2(sig);
        dm[kk + 1] = dtau_c(cx, &mm[kk + 1], omega);
    }
    pp.truncate(ks + 1);
    mm.truncate(ks + 1);
    aa.truncate(ks + 1);
    cc.truncate(ks + 1);
    [pp, mm, aa, cc]
}

fn block_state_c<const B: usize>(cx: &Cx<B>, b: usize, inp: &[Fl<Dc, B>], delta: Dc, hl: &HLus) -> State<Dc, B> {
    let ctx = cx.ctx;
    let omega = Dc::splat(DD::TWO_PI) / delta;
    let k = ctx.k();
    if b == 0 {
        centre_state_c(cx, &inp[0], &inp[1], omega)
    } else if b < k {
        let mut w = TWork::<B>::new(ctx.kt);
        taylor_path_c(cx, &[inp[0], inp[1], inp[2], inp[3]], &ctx.segs[b - 1], omega, &mut w)
    } else {
        let s = horizon_series_c(cx, &inp[0], &inp[1], omega, hl);
        let u = ctx.u0().mul_pwr2(1.0 / series_scale(ctx.u0().hi));
        [horner(&s[0], u), horner(&s[1], u), horner(&s[2], u), horner(&s[3], u)]
    }
}

/// whether input field f of block b is a test-field unknown (p1; a node's p, m; m_h)
fn psi_input<const B: usize>(ctx: &Ctx<B>, b: usize, f: usize) -> bool {
    if b == 0 || b == ctx.k() {
        f == 0
    } else {
        f < 2
    }
}

/// Complex mode matrix M(lambda), (L-1) x (L-1), column-major with leading dimension ld_for(L-1):
/// (re_hi, re_lo, im_hi, im_lo, ld). With cx.test != TEST_OFF only the test field's columns are computed (the others
/// hold only the -1 of their target), as test_matrix_c needs.
#[allow(clippy::type_complexity)]
pub fn mode_matrix_c<const B: usize>(cx: &Cx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize), String> {
    let ctx = cx.ctx;
    let l = ctx.len();
    if zh.len() != l || zl.len() != l {
        return Err(format!("z must have length {l}"));
    }
    if ctx.horizon_alpha != 0 {
        return Err("horizon_alpha must be 0 (the horizon series in a)".into());
    }
    let n = l - 1;
    let m = Ctx::<B>::M;
    let ld = ld_for(n);
    let delta = DD { hi: zh[l - 1], lo: zl[l - 1] };
    let hl = HLus::new(cx, zh, zl, par)?;
    let nb = ctx.k() + 1;
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for b in 0..nb {
        let ins = ctx.block_inputs(b);
        for f in 0..ins.len() {
            if cx.test != TEST_OFF && !psi_input(ctx, b, f) {
                continue;
            }
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
                Fl::from_vri(&v, &t, &Fl::<DV, B>::zero())
            })
            .collect();
        let d = Dc::cst(DV::splat(delta));
        let s = block_state_c(cx, b, &inp, d, &hl);
        let mut out = Vec::with_capacity(4 * m);
        for q in 0..4 {
            let (mut tr, mut ti) = (s[q].tan_re(), s[q].tan_im());
            if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
                tr = lo.apply(&tr);
                ti = lo.apply(&ti);
            }
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

/// The test field's unknowns (column indices of z in the mode matrix): p1, each node's p and m, m_h. The psi rows are
/// the p, m rows of every block, at the same positions within the layout (row b 4M + r, r < 2M).
pub fn psi_index<const B: usize>(ctx: &Ctx<B>) -> (Vec<usize>, Vec<usize>) {
    let m = Ctx::<B>::M;
    let mut cols: Vec<usize> = (0..m).collect();
    for i in 0..ctx.k() {
        cols.extend(ctx.off_node(i)..ctx.off_node(i) + 2 * m);
    }
    cols.extend(ctx.off_mh()..ctx.off_mh() + m);
    let mut rows = Vec::with_capacity(cols.len());
    for b in 0..ctx.k() + 1 {
        rows.extend(b * 4 * m..b * 4 * m + 2 * m);
    }
    (rows, cols)
}

/// The test field's mode matrix: the psi rows and columns of mode_matrix_c with cx.test set, (2M(K+1))^2,
/// column-major with leading dimension ld_for(2M(K+1)): (re_hi, re_lo, im_hi, im_lo, ld).
#[allow(clippy::type_complexity)]
pub fn test_matrix_c<const B: usize>(cx: &Cx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize), String> {
    if cx.test == TEST_OFF {
        return Err("test_matrix_c needs Cx::test set".into());
    }
    let (rh, rl, ih, il, ld) = mode_matrix_c(cx, zh, zl, par)?;
    let (rows, cols) = psi_index(cx.ctx);
    let n = cols.len();
    let ldp = ld_for(n);
    let mut out = [vec![0.0; ldp * n], vec![0.0; ldp * n], vec![0.0; ldp * n], vec![0.0; ldp * n]];
    for (jc, &c) in cols.iter().enumerate() {
        for (ir, &r) in rows.iter().enumerate() {
            for (o, src) in out.iter_mut().zip([&rh, &rl, &ih, &il]) {
                o[jc * ldp + ir] = src[c * ld + r];
            }
        }
    }
    let [a, b, c, d] = out;
    Ok((a, b, c, d, ldp))
}


// ---------------------------------------------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------------------------------------------
type CMat = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize);

trait CApi: Send + Sync {
    fn len(&self) -> usize;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn mode_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<CMat, String>;
    fn test_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, test: u8, par: bool) -> Result<CMat, String>;
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>);
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
        mode_matrix_c(&Cx { ctx: self, lr, li, test: TEST_OFF }, zh, zl, par)
    }
    fn test_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, test: u8, par: bool) -> Result<CMat, String> {
        test_matrix_c(&Cx { ctx: self, lr, li, test }, zh, zl, par)
    }
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>) {
        psi_index(self)
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
                mutate: 0,
                f64_dd: 0,
                horizon_alpha: 0,
            }) as Box<dyn CApi>, )*
            _ => return Err(PyValueError::new_err("M must be one of 16, 32, 48, 64")),
        }
    };
}

fn slice<'a>(a: &'a PyReadonlyArray1<f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

fn test_mode(mode: u8) -> PyResult<u8> {
    if mode == TEST_ON || mode == TEST_MUT {
        Ok(mode)
    } else {
        Err(PyValueError::new_err(format!("mode must be 1 (the test field) or 2 (its control), not {mode}")))
    }
}

/// (n, n) arrays (row index first) from a column-major matrix with leading dimension ld
fn to2(v: &[f64], n: usize, ld: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        for r in 0..n {
            a[[r, c]] = v[c * ld + r];
        }
    }
    a
}

/// Same constructor as hpkernel.Solver; segs must be given (take them from hpkernel's Solver.get_segs()). The layout
/// is the default one (Ctx::horizon_alpha 0, no deliberate breakage).
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
        Ok((to2(&rh, n, ld).into_pyarray(py), to2(&rl, n, ld).into_pyarray(py), to2(&ih, n, ld).into_pyarray(py), to2(&il, n, ld).into_pyarray(py)))
    }

    /// The test field's det M_psi(lam) (mode 1; 2 the control, a's tangent kept): (ln|det|, arg det, min |u_kk|,
    /// max |u_kk|).
    #[pyo3(signature = (zh, zl, lr_hi, li_hi, lr_lo=0.0, li_lo=0.0, mode=1))]
    #[allow(clippy::too_many_arguments)]
    fn test_det(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, lr_hi: f64, li_hi: f64, lr_lo: f64, li_lo: f64, mode: u8) -> PyResult<(f64, f64, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (lr, li) = (DD::new(lr_hi, lr_lo), DD::new(li_hi, li_lo));
        py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let (rh, rl, ih, il, ld) = api.test_matrix_c(h, l, lr, li, test, true)?;
                let n = rh.len() / ld;
                clu_logdet(n, ld, rh, rl, ih, il, true)
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// The test field's mode matrix M_psi(lam) as four (N, N) arrays (re_hi, re_lo, im_hi, im_lo), N = 2M(K+1).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (zh, zl, lr, li, mode=1))]
    fn test_matrix<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
        mode: u8,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.test_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), test, true)))
            .map_err(PyValueError::new_err)?;
        let n = rh.len() / ld;
        Ok((to2(&rh, n, ld).into_pyarray(py), to2(&rl, n, ld).into_pyarray(py), to2(&ih, n, ld).into_pyarray(py), to2(&il, n, ld).into_pyarray(py)))
    }

    /// M_psi(lam) v in double-double for a real double-double vector v (length N): (re_hi, re_lo, im_hi, im_lo).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    #[pyo3(signature = (zh, zl, vh, vl, lr, li, mode=1))]
    fn test_matvec<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        vh: PyReadonlyArray1<'py, f64>,
        vl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
        mode: u8,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (vh, vl) = (slice(&vh, "vh")?.to_vec(), slice(&vl, "vl")?.to_vec());
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.test_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), test, true)))
            .map_err(PyValueError::new_err)?;
        let n = rh.len() / ld;
        if vh.len() != n || vl.len() != n {
            return Err(PyValueError::new_err(format!("v must have length {n}")));
        }
        let mut re = vec![DD::ZERO; n];
        let mut im = vec![DD::ZERO; n];
        for j in 0..n {
            let v = DD { hi: vh[j], lo: vl[j] };
            for i in 0..n {
                let k = j * ld + i;
                re[i] = re[i] + DD { hi: rh[k], lo: rl[k] } * v;
                im[i] = im[i] + DD { hi: ih[k], lo: il[k] } * v;
            }
        }
        let hi = |x: &[DD]| x.iter().map(|d| d.hi).collect::<Vec<f64>>().into_pyarray(py);
        let lo = |x: &[DD]| x.iter().map(|d| d.lo).collect::<Vec<f64>>().into_pyarray(py);
        Ok((hi(&re), lo(&re), hi(&im), lo(&im)))
    }

    /// (rows, cols): the psi rows and columns of the mode matrix (indices into its L-1 rows and columns)
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>) {
        self.api.psi_index()
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
