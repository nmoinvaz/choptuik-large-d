//! The DSS critical-solution equations in double-double: centre series, Taylor stepping in x, horizon series, and
//! the multiple-shooting residual and its exact Jacobian.
//!
//! Fields p, m (odd), a, c (even) on the half tau grid; d/dtau = omega d/dtheta, omega = 2 pi / Delta. With x the
//! self-similar radius (horizon at x = 1, where c = 1):
//!
//!   x c_x         = (n-1)(c a^2 - c)                         =: G
//!   2 x a_x       = a (p^2 + m^2) + (n-1)(a - a a^2)          =: H
//!   x (x - c) p_x = -omega x p_th + G p - c p + (n/2) c (p + m)
//!   x (x + c) m_x = -omega x m_th - G m + c m - (n/2) c (p + m)
//!
//! Unknowns z (length L = 4M(K+1) + 1, M = 8B, K nodes): centre data p1, c0 (M each); for each node the state
//! p, m, a, c (4M); horizon data m_h, a_h (M each); Delta. Residual blocks: centre series at node 0 minus node 0;
//! Taylor integration node b-1 -> node b minus node b (b = 1..K-1); horizon series at u = 1 - x_{K-1} minus node
//! K-1; and a phase row (sum_j w_j p1_j). All in physical values on the grid.
use crate::dd::DD;
use crate::dv::DV;
use crate::field::{cs, cs_range, cs_w, horner, DMat, Fl, State};
use crate::lane::{Du, Lane};
use crate::linalg::{ld_for, DDLu};
use rayon::prelude::*;

pub struct Ctx<const B: usize> {
    pub n: DD,
    /// folded odd-sector d/dtheta (M x M)
    pub d: DMat<B>,
    /// phase-row weights on p1
    pub phase: Fl<DV, B>,
    pub nodes: Vec<f64>,
    /// segs[b-1]: x points of the Taylor steps from nodes[b-1] to nodes[b], both ends included (b = 1..K-1)
    pub segs: Vec<Vec<f64>>,
    /// Taylor order (coefficients 0..=kt)
    pub kt: usize,
    /// centre series order
    pub kc: usize,
    /// horizon series order
    pub ks: usize,
}

impl<const B: usize> Ctx<B> {
    pub const M: usize = 8 * B;
    pub fn k(&self) -> usize {
        self.nodes.len()
    }
    pub fn len(&self) -> usize {
        4 * Self::M * (self.k() + 1) + 1
    }
    pub fn off_node(&self, i: usize) -> usize {
        2 * Self::M + 4 * Self::M * i
    }
    pub fn off_mh(&self) -> usize {
        2 * Self::M + 4 * Self::M * self.k()
    }
    pub fn off_ah(&self) -> usize {
        3 * Self::M + 4 * Self::M * self.k()
    }
    /// z offsets of the fields block b depends on (besides Delta)
    pub fn block_inputs(&self, b: usize) -> Vec<usize> {
        let m = Self::M;
        let k = self.k();
        if b == 0 {
            vec![0, m]
        } else if b < k {
            let o = self.off_node(b - 1);
            vec![o, o + m, o + 2 * m, o + 3 * m]
        } else {
            vec![self.off_mh(), self.off_ah()]
        }
    }
    pub fn block_target(&self, b: usize) -> usize {
        if b < self.k() {
            self.off_node(b)
        } else {
            self.off_node(self.k() - 1)
        }
    }
    pub fn fld(&self, zh: &[f64], zl: &[f64], off: usize) -> Fl<DV, B> {
        Fl::load(&zh[off..off + Self::M], &zl[off..off + Self::M])
    }
    pub fn u0(&self) -> DD {
        DD::ONE - DD::from_f64(self.nodes[self.k() - 1])
    }
}

#[inline(always)]
fn recip_int(k: usize) -> DD {
    DD::ONE / DD::from_f64(k as f64)
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (regular centre), evaluated at nodes[0]
// ---------------------------------------------------------------------------------------------------------------
pub fn centre_series<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> [Vec<Fl<T, B>>; 4] {
    let n = ctx.n;
    let n1 = n - DD::ONE;
    let nh = n.mul_f(0.5);
    let kmax = ctx.kc;
    let z = Fl::<T, B>::zero();
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
    let invc = Fl::<T, B>::splat(DD::ONE) / cc[0];
    for k in 2..=kmax {
        let kf = k as f64;
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        a2[k] = cs(&aa, &aa, k); // aa[k] still 0
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        aa[k] = (s.mulf(0.5) - t3.muls(n1.mul_f(0.5))).muls(DD::ONE / (DD::from_f64(kf) + n1));
        a2[k] = cs(&aa, &aa, k);
        cc[k] = cs_range(&cc, &a2, k, 0, k - 1).muls(n1 * recip_int(k));
        let dp = ctx.d.apply(&pp[k - 1]).mul_t(omega);
        let dm = ctx.d.apply(&mm[k - 1]).mul_t(omega);
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
        // [m11 m12; m21 m22] = c [mu11 mu12; mu21 mu22]; det = c^2 delta, delta = -(k-1)(k-1+n)
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

pub fn centre_state<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> State<T, B> {
    let s = centre_series(ctx, p1, c0, omega);
    let x = DD::from_f64(ctx.nodes[0]);
    [horner(&s[0], x), horner(&s[1], x), horner(&s[2], x), horner(&s[3], x)]
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor stepping in x
// ---------------------------------------------------------------------------------------------------------------
pub struct TWork<T: Lane, const B: usize> {
    pub p: Vec<Fl<T, B>>,
    pub m: Vec<Fl<T, B>>,
    pub a: Vec<Fl<T, B>>,
    pub c: Vec<Fl<T, B>>,
    a2: Vec<Fl<T, B>>,
    q2: Vec<Fl<T, B>>,
    g: Vec<Fl<T, B>>,
    h: Vec<Fl<T, B>>,
    pth: Vec<Fl<T, B>>,
    mth: Vec<Fl<T, B>>,
    ww: Vec<Fl<T, B>>,
    vv: Vec<Fl<T, B>>,
}

impl<T: Lane, const B: usize> TWork<T, B> {
    pub fn new(kt: usize) -> Self {
        let v = || vec![Fl::<T, B>::zero(); kt + 2];
        TWork { p: v(), m: v(), a: v(), c: v(), a2: v(), q2: v(), g: v(), h: v(), pth: v(), mth: v(), ww: v(), vv: v() }
    }
}

/// Taylor coefficients 0..=kt of the state about x0 (in w.p, w.m, w.a, w.c)
pub fn taylor_coeffs<T: Lane, const B: usize>(ctx: &Ctx<B>, s: &State<T, B>, x0: f64, omega: T, w: &mut TWork<T, B>) {
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
    let mut iw0 = Fl::<T, B>::zero();
    let mut iv0 = Fl::<T, B>::zero();
    for k in 0..kt {
        let kf = k as f64;
        w.a2[k] = cs(&w.a, &w.a, k);
        w.q2[k] = cs(&w.p, &w.p, k) + cs(&w.m, &w.m, k);
        w.g[k] = (cs(&w.c, &w.a2, k) - w.c[k]).muls(n1);
        w.h[k] = cs(&w.a, &w.q2, k) + (w.a[k] - cs(&w.a, &w.a2, k)).muls(n1);
        w.pth[k] = ctx.d.apply(&w.p[k]).mul_t(omega);
        w.mth[k] = ctx.d.apply(&w.m[k]).mul_t(omega);
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
            iw0 = Fl::<T, B>::splat(DD::ONE) / w.ww[0];
            iv0 = Fl::<T, B>::splat(DD::ONE) / w.vv[0];
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

pub fn taylor_step<T: Lane, const B: usize>(ctx: &Ctx<B>, s: &State<T, B>, x0: f64, x1: f64, omega: T, w: &mut TWork<T, B>) -> State<T, B> {
    taylor_coeffs(ctx, s, x0, omega, w);
    let h = DD::from_f64(x1) - DD::from_f64(x0);
    let kt = ctx.kt;
    [horner(&w.p[..=kt], h), horner(&w.m[..=kt], h), horner(&w.a[..=kt], h), horner(&w.c[..=kt], h)]
}

pub fn taylor_path<T: Lane, const B: usize>(ctx: &Ctx<B>, s0: &State<T, B>, xs: &[f64], omega: T, w: &mut TWork<T, B>) -> State<T, B> {
    let mut s = *s0;
    for i in 0..xs.len() - 1 {
        s = taylor_step(ctx, &s, xs[i], xs[i + 1], omega, w);
    }
    s
}

/// max over fields and lanes of |coefficient k| (hi part, value)
fn coef_norm<T: Lane, const B: usize>(w: &TWork<T, B>, k: usize) -> f64 {
    [&w.p, &w.m, &w.a, &w.c].iter().map(|q| q[k].val().amax()).fold(0.0, f64::max)
}

/// Adaptive step points from xa to xb for a state s (values): h from the last two Taylor coefficients so that the
/// truncation term is ~ tol (absolute; the fields are O(1)), capped at hmax.
pub fn plan_path<const B: usize>(ctx: &Ctx<B>, s0: &State<DV, B>, xa: f64, xb: f64, omega: DV, tol: f64, hmax: f64) -> Vec<f64> {
    let mut w = TWork::<DV, B>::new(ctx.kt);
    let kt = ctx.kt;
    let mut xs = vec![xa];
    let mut x = xa;
    let mut s = *s0;
    let dir = if xb > xa { 1.0 } else { -1.0 };
    while (xb - x) * dir > 1e-15 {
        taylor_coeffs(ctx, &s, x, omega, &mut w);
        let e1 = coef_norm(&w, kt - 1).max(1e-300);
        let e2 = coef_norm(&w, kt).max(1e-300);
        let h1 = (tol / e1).powf(1.0 / (kt - 1) as f64);
        let h2 = (tol / e2).powf(1.0 / kt as f64);
        let mut h = h1.min(h2).min(hmax);
        let rest = (xb - x) * dir;
        if h >= rest {
            h = rest;
        } else if h > 0.6 * rest {
            h = 0.5 * rest; // two near-equal last steps instead of a sliver
        }
        let xn = if h == rest { xb } else { x + dir * h };
        s = taylor_step(ctx, &s, x, xn, omega, &mut w);
        x = xn;
        xs.push(x);
    }
    xs
}

// ---------------------------------------------------------------------------------------------------------------
// horizon series (regular horizon), in u = 1 - x
// ---------------------------------------------------------------------------------------------------------------

/// LU factors of (omega D + diag kappa_kk), kk = 0..=ks, at the given (a_h, Delta) values:
/// kappa_kk = (c1 + 1) kk + c1 + 1 - n/2, c1 = (n-1)(1 - a_h^2).
pub fn horizon_lus<const B: usize>(ctx: &Ctx<B>, a0: &Fl<DV, B>, delta: DD, par: bool) -> Result<Vec<DDLu>, String> {
    let m = Ctx::<B>::M;
    let ld = ld_for(m);
    let omega = DD::TWO_PI / delta;
    let n1 = ctx.n - DD::ONE;
    let c1 = (Fl::<DV, B>::splat(DD::ONE) - *a0 * *a0).muls(n1);
    let c1s: Vec<DD> = (0..m).map(|i| c1.get(i)).collect();
    // omega D (column-major, ld)
    let mut base_h = vec![0.0; ld * m];
    let mut base_l = vec![0.0; ld * m];
    for j in 0..m {
        let col = ctx.d.cols[j].muls(omega);
        col.store(&mut base_h[j * ld..j * ld + m], &mut base_l[j * ld..j * ld + m]);
    }
    let nh = ctx.n.mul_f(0.5);
    let one_kk = |kk: usize| -> Result<DDLu, String> {
        let mut h = base_h.clone();
        let mut l = base_l.clone();
        for i in 0..m {
            let kap = (c1s[i] + DD::ONE).mul_f(kk as f64) + c1s[i] + DD::ONE - nh;
            let v = DD { hi: h[i * ld + i], lo: l[i * ld + i] } + kap;
            h[i * ld + i] = v.hi;
            l[i * ld + i] = v.lo;
        }
        DDLu::factor(m, ld, h, l, false)
    };
    if par {
        (0..=ctx.ks).into_par_iter().map(one_kk).collect()
    } else {
        (0..=ctx.ks).map(one_kk).collect()
    }
}

/// (omega D + diag kappa) f = r with the LU at the values; for tangents the derivative of the system:
/// A f_d = r_d - omega_d D f_v - kappa_d f_v.
fn psolve<T: Lane, const B: usize>(ctx: &Ctx<B>, lu: &DDLu, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Fl<T, B> {
    let fv = lu.solve_lanes(&rhs.val());
    if !T::TANGENT {
        return Fl::from_vt(&fv, &Fl::zero());
    }
    let dfv = ctx.d.apply::<DV>(&fv);
    let od = Fl::<DV, B>([omega.tan(); B]);
    let r = (rhs.tan() - dfv * od) - kappa.tan() * fv;
    let fd = lu.solve_lanes(&r);
    Fl::from_vt(&fv, &fd)
}

pub fn horizon_series<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    omega: T,
    lus: &[DDLu],
) -> [Vec<Fl<T, B>>; 4] {
    let n1 = ctx.n - DD::ONE;
    let nh = ctx.n.mul_f(0.5);
    let ks = ctx.ks;
    let len = ks + 2;
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
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
    dm[0] = ctx.d.apply(&mm[0]).mul_t(omega);
    let wfun = |c: &Vec<Fl<T, B>>, i: usize| -> Fl<T, B> {
        match i {
            0 => one - c[0],
            1 => -c[1] - one,
            _ => -c[i],
        }
    };
    let zfun = |c: &Vec<Fl<T, B>>, i: usize| -> Fl<T, B> {
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
        pp[kk] = psolve(ctx, &lus[kk], &kappa, &(-r), omega);
        dp[kk] = ctx.d.apply(&pp[kk]).mul_t(omega);
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
        dm[kk + 1] = ctx.d.apply(&mm[kk + 1]).mul_t(omega);
    }
    pp.truncate(ks + 1);
    mm.truncate(ks + 1);
    aa.truncate(ks + 1);
    cc.truncate(ks + 1);
    [pp, mm, aa, cc]
}

pub fn horizon_state<T: Lane, const B: usize>(ctx: &Ctx<B>, m0: &Fl<T, B>, a0: &Fl<T, B>, omega: T, lus: &[DDLu]) -> State<T, B> {
    let s = horizon_series(ctx, m0, a0, omega, lus);
    let u = ctx.u0();
    [horner(&s[0], u), horner(&s[1], u), horner(&s[2], u), horner(&s[3], u)]
}

// ---------------------------------------------------------------------------------------------------------------
// blocks, residual, Jacobian
// ---------------------------------------------------------------------------------------------------------------
pub fn block_state<T: Lane, const B: usize>(ctx: &Ctx<B>, b: usize, inp: &[Fl<T, B>], delta: T, lus: &[DDLu]) -> State<T, B> {
    let omega = T::splat(DD::TWO_PI) / delta;
    let k = ctx.k();
    if b == 0 {
        centre_state(ctx, &inp[0], &inp[1], omega)
    } else if b < k {
        let mut w = TWork::<T, B>::new(ctx.kt);
        taylor_path(ctx, &[inp[0], inp[1], inp[2], inp[3]], &ctx.segs[b - 1], omega, &mut w)
    } else {
        horizon_state(ctx, &inp[0], &inp[1], omega, lus)
    }
}

fn delta_of(zh: &[f64], zl: &[f64]) -> DD {
    let l = zh.len();
    DD { hi: zh[l - 1], lo: zl[l - 1] }
}

pub fn phase_row<const B: usize>(ctx: &Ctx<B>, p1: &Fl<DV, B>) -> DD {
    let mut acc = DD::ZERO;
    for i in 0..Ctx::<B>::M {
        acc = acc + ctx.phase.get(i) * p1.get(i);
    }
    acc
}

/// Residual (hi, lo), length L.
pub fn residual<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
    let l = ctx.len();
    let m = Ctx::<B>::M;
    assert_eq!(zh.len(), l);
    let delta = delta_of(zh, zl);
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par)?;
    let one_block = |b: usize| -> State<DV, B> {
        let inp: Vec<Fl<DV, B>> = ctx.block_inputs(b).iter().map(|&o| ctx.fld(zh, zl, o)).collect();
        block_state::<DV, B>(ctx, b, &inp, DV::splat(delta), &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<DV, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let mut fh = vec![0.0; l];
    let mut fl = vec![0.0; l];
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let d = s[q] - ctx.fld(zh, zl, t + q * m);
            let o = b * 4 * m + q * m;
            d.store(&mut fh[o..o + m], &mut fl[o..o + m]);
        }
    }
    let ph = phase_row(ctx, &ctx.fld(zh, zl, 0));
    fh[l - 1] = ph.hi;
    fl[l - 1] = ph.lo;
    Ok((fh, fl))
}

/// Exact Jacobian dF/dz by forward-mode duals, column-major with leading dimension ld_for(L): (hi, lo, ld).
pub fn jacobian<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let delta = delta_of(zh, zl);
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par)?;
    let nb = ctx.k() + 1;
    // tasks: (block, input field index or usize::MAX for Delta, lane)
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for b in 0..nb {
        let ins = ctx.block_inputs(b);
        for f in 0..ins.len() {
            for j in 0..m {
                tasks.push((b, f, j));
            }
        }
        tasks.push((b, usize::MAX, 0));
    }
    let column = |&(b, f, j): &(usize, usize, usize)| -> Vec<DD> {
        let ins = ctx.block_inputs(b);
        let inp: Vec<Fl<Du, B>> = ins
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
                Fl::<Du, B>::from_vt(&v, &t)
            })
            .collect();
        let dt = if f == usize::MAX { DV::splat(DD::ONE) } else { DV::zero() };
        let d = Du { v: DV::splat(delta), d: dt };
        let s = block_state::<Du, B>(ctx, b, &inp, d, &lus);
        let mut out = Vec::with_capacity(4 * m);
        for q in 0..4 {
            let t = s[q].tan();
            for i in 0..m {
                out.push(t.get(i));
            }
        }
        out
    };
    let cols: Vec<Vec<DD>> = if par { tasks.par_iter().map(column).collect() } else { tasks.iter().map(column).collect() };
    let mut jh = vec![0.0; ld * l];
    let mut jl = vec![0.0; ld * l];
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let c = if f == usize::MAX { l - 1 } else { ctx.block_inputs(b)[f] + j };
        for (r, v) in col.iter().enumerate() {
            jh[c * ld + b * 4 * m + r] = v.hi;
            jl[c * ld + b * 4 * m + r] = v.lo;
        }
    }
    for b in 0..nb {
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            let idx = (t + r) * ld + b * 4 * m + r;
            let v = DD { hi: jh[idx], lo: jl[idx] } - DD::ONE;
            jh[idx] = v.hi;
            jl[idx] = v.lo;
        }
    }
    for i in 0..m {
        let w = ctx.phase.get(i);
        jh[i * ld + l - 1] = w.hi;
        jl[i * ld + l - 1] = w.lo;
    }
    Ok((jh, jl, ld))
}
