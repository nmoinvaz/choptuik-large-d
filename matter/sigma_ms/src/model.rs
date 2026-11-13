//! The sigma-model's DSS equations: centre series, Taylor stepping in x, horizon series, as multishoot's `Equations`
//! (`Sig`), and the entry points on multishoot's engine.
//!
//! Fields p, m, psi (odd under tau -> tau + Delta/2) and a, c (even) on the half tau grid, d/dtau = omega d/dtheta,
//! omega = 2 pi / Delta. With x the self-similar radius (horizon at x = 1, where c = 1), n = D - 2, kappa^2 = n eps
//! (eps = 1/eta), sigma = sin(kappa psi)/kappa, gamma = cos(kappa psi) (solver/matter/sigdss_solver.py, checked in
//! solver/matter/sigma_eqs.py):
//!
//!   x c_x         = (n-1)(c a^2 - c) - 2n c a^2 sigma^2                                =: G
//!   2 x a_x       = a (p^2 + m^2) + (n-1)(a - a^3) + 2n a^3 sigma^2                      =: H
//!   x psi_x       = (p + m)/2
//!   x (x - c) p_x = -omega x p_th + (x c_x) p - c p + (n/2) c (p + m) - n c a^2 sigma gamma
//!   x (x + c) m_x = -omega x m_th - (x c_x) m + c m - (n/2) c (p + m) + n c a^2 sigma gamma
//!
//! sigma and gamma enter the Taylor recursions through d sigma = gamma d psi, d gamma = -kappa^2 sigma d psi, order by
//! order; their order-0 values come from the series of sin(s)/s and cos(s) in w = kappa^2 psi^2 (`sin_cos_field`), so
//! that kappa = 0 (eta = infinity, the l = 1 multiplet) is the same code with sigma = psi and gamma = 1.
//!
//! Unknowns z (multishoot's layout, NF = 5, NL = 2, NR = 3, NG = 1; length 5M(K+1) + 1): centre data q (psi ~ q x,
//! p ~ m ~ q x) and c0; for each node the state p, m, a, c, psi; horizon data m_h, a_h, psi_h; Delta. Residual blocks:
//! centre series at node 0 minus node 0; Taylor path of each segment minus its target node; horizon series at
//! u0 = 1 - x_{K-1} minus node K-1; the phase row sum_j w_j q_j.
//!
//! Copied and adapted from hp's model.rs (the Choptuik scalar: four fields, centre datum p1 with m ~ -p1 x): its
//! Taylor recursion, centre and horizon series, horizon LUs and refined periodic solves, with the sigma-model terms,
//! the fifth field psi and the l = 1 centre structure (p ~ m ~ q x, the 2 x 2 order-k system of sigdss_solver.py's
//! centre_series). hp's f64, octo-double, jet and mode-matrix lanes are not carried over.
use crate::planted::Planted;
use multidouble::dd::DD;
use multidouble::field::{cs, cs_w, horner, FieldSolve, Fl};
use multidouble::linalg::{ld_for, DDLu};
use multidouble::native::{DMat, DV};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use multishoot::control::Control;
use multishoot::lane::Lane;
use multishoot::taylor::Plan;
use multishoot::{Equations, ExtraRow, Shooting, State, Work};
use rayon::prelude::*;

/// fields per point: p, m, a, c, psi
pub const NF: usize = 5;
/// the scratch series of the Taylor recursion (`Work::aux`): a2, q2, g, h, pth, mth, ww, vv, sn, co, ss, g2, sg, hs
const N_AUX: usize = 14;
/// terms of the series of sin(s)/s and cos(s) in w = s^2: k = 0..N_SIN-1, the first dropped term w^26/53! ~ 2e-70
/// for |w| <= 1 (|kappa psi| <= 1 rad; |u| <= 0.3 at eta >= 5, D = 4)
const N_SIN: usize = 26;

/// `Ctx::mutate`, the controls of the planted check (planted.py --mutate): gamma's recursion in the Taylor steps with
/// kappa^2 scaled by 1 + 2^-30
pub const MUT_GAMMA: u8 = 1;
/// the coefficient 2n of the a^3 sigma^2 term of 2 x a_x in the Taylor steps scaled by 1 + 2^-30
pub const MUT_SIGMA_H: u8 = 2;
/// the quad-double entry points: the state read in its double-double part only (multishoot's Control::HalfParts)
pub const MUT_QD_STATE_DD: u8 = 9;
/// the quad-double horizon solves without their refinement (to the double-double LU's accuracy)
pub const MUT_QD_NOREFINE: u8 = 7;

/// refinement steps of a horizon solve in the quad-double lanes (hp's QD_REFINE); a tangent lane takes one more
const QD_REFINE: usize = 2;

pub struct Ctx<const B: usize> {
    pub n: DD,
    /// kappa^2 = n eps
    pub k2: DD,
    /// folded odd-sector d/dtheta (M x M)
    pub d: DMat<B>,
    /// phase-row weights on q
    pub phase: Fl<DV, B>,
    pub nodes: Vec<f64>,
    /// segs[b-1]: x points of the Taylor steps of segment b, from its source node to its target node
    pub segs: Vec<Vec<f64>>,
    /// Taylor order of the steps
    pub kt: usize,
    /// centre series order
    pub kc: usize,
    /// horizon series order
    pub ks: usize,
    /// the last segment that runs outward
    pub match_node: usize,
    /// MUT_* (0 none)
    pub mutate: u8,
    /// the Jacobian column dropped by multishoot's Control::DropColumn (the planted check's second control)
    pub drop_col: Option<usize>,
    /// the source terms of a manufactured solution (planted check); None for the sigma-model itself
    pub planted: Option<Planted<B>>,
}

impl<const B: usize> Ctx<B> {
    pub const M: usize = 8 * B;
    pub fn k(&self) -> usize {
        self.nodes.len()
    }
    pub fn len(&self) -> usize {
        NF * Self::M * (self.k() + 1) + 1
    }
    pub fn off_node(&self, i: usize) -> usize {
        2 * Self::M + NF * Self::M * i
    }
    /// offset of the horizon data m_h, a_h, psi_h
    pub fn off_h(&self) -> usize {
        self.off_node(self.k())
    }
    pub fn fld(&self, zh: &[f64], zl: &[f64], off: usize) -> Fl<DV, B> {
        Fl::load(&zh[off..off + Self::M], &zl[off..off + Self::M])
    }
    pub fn u0(&self) -> DD {
        DD::ONE - DD::from_f64(self.nodes[self.k() - 1])
    }
    /// whether segment b (1 <= b < K) runs inward
    pub fn seg_inward(&self, b: usize) -> bool {
        b > self.match_node && b < self.k()
    }
    /// The engine's problem on this context, with the phase row as its one extra row and the control of `drop_col`
    pub fn shooting(&self) -> Shooting<Sig<'_, B>, B> {
        let mut s = Shooting::new(Sig(self), self.nodes.clone(), self.kt);
        s.segs = self.segs.clone();
        s.match_node = self.match_node;
        s.extra = vec![ExtraRow { terms: (0..Self::M).map(|i| (i, self.phase.get(i))).collect(), rhs: DD::ZERO }];
        if let Some(c) = self.drop_col {
            s.set_control(Control::DropColumn(c));
        }
        s
    }
    /// `shooting` for an evaluation in P, with the quad-double state control MUT_QD_STATE_DD
    pub fn shooting_in<P: Precision<Native>>(&self) -> Shooting<Sig<'_, B>, B> {
        let mut s = self.shooting();
        if P::PARTS == 4 && self.mutate == MUT_QD_STATE_DD {
            s.set_control(Control::HalfParts);
        }
        s
    }
}

/// the series scale (multidouble's pow2_near): the centre and horizon series are computed in x / sigma (u / sigma),
/// sigma the power of two nearest the evaluation radius, so that their coefficients c_k sigma^k stay near O(1)
pub use multidouble::dd::pow2_near as series_scale;

#[inline(always)]
fn lane_int<T: Lane>(k: usize) -> T {
    T::splat(DD::from_f64(k as f64))
}

/// omega = 2 pi / Delta in the lane
#[inline(always)]
fn omega_of<T: Lane>(delta: T) -> T {
    T::splat(DD::TWO_PI) / delta
}

/// d/dtau = omega d/dtheta on an odd field
#[inline(always)]
fn dtau<T: Lane, const B: usize>(ctx: &Ctx<B>, x: &Fl<T, B>, omega: T) -> Fl<T, B> {
    ctx.d.apply(x).mul_t(omega)
}

/// sigma(psi) = sin(kappa psi)/kappa = psi S(w) and gamma(psi) = cos(kappa psi) = C(w), w = kappa^2 psi^2, from the
/// series S(w) = sum (-w)^k/(2k+1)!, C(w) = sum (-w)^k/(2k)! by Horner's rule, each division by an exact integer in
/// the lane's precision
pub fn sin_cos_field<T: Lane, const B: usize>(psi: &Fl<T, B>, k2: DD) -> (Fl<T, B>, Fl<T, B>) {
    let w = (*psi * *psi).muls(k2);
    let one = Fl::<T, B>::splat(DD::ONE);
    let mut s = one;
    let mut c = one;
    for k in (1..N_SIN).rev() {
        let ds = Fl::<T, B>::splat(DD::from_f64(((2 * k) * (2 * k + 1)) as f64));
        let dc = Fl::<T, B>::splat(DD::from_f64(((2 * k - 1) * (2 * k)) as f64));
        s = one - (w * s) / ds;
        c = one - (w * c) / dc;
    }
    (*psi * s, c)
}

/// x c_x at order k (without a source): (n-1)((c a^2)_k - c_k) - 2n (c g2)_k, g2 = a^2 sigma^2
#[inline(always)]
fn g_coef<T: Lane, const B: usize>(ctx: &Ctx<B>, c: &[Fl<T, B>], a2: &[Fl<T, B>], g2: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    let n1 = ctx.n - DD::ONE;
    let n2 = ctx.n.mul_f(2.0);
    (cs(c, a2, k) - c[k]).muls(n1) - cs(c, g2, k).muls(n2)
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor stepping in x
// ---------------------------------------------------------------------------------------------------------------

/// Taylor coefficients 1..=w.kt of the state about x0, order 0 in w.c[0..5] (p, m, a, c, psi), at the given omega
pub fn taylor_coeffs<T: Lane, const B: usize>(ctx: &Ctx<B>, x0: f64, omega: T, w: &mut Work<T, B>) {
    let kt = w.kt;
    let [p, m, a, c, q] = &mut w.c[..] else { unreachable!("five fields") };
    let [a2, q2, g, h, pth, mth, ww, vv, sn, co, ss, g2, sg, hs] = &mut w.aux[..] else {
        unreachable!("fourteen scratch series")
    };
    let n = ctx.n;
    let n1 = n - DD::ONE;
    let nh = n.mul_f(0.5);
    let mut n2h = n.mul_f(2.0);
    let mut k2 = ctx.k2;
    if ctx.mutate == MUT_GAMMA {
        k2 = k2 + k2.mul_f(2f64.powi(-30));
    }
    if ctx.mutate == MUT_SIGMA_H {
        n2h = n2h + n2h.mul_f(2f64.powi(-30));
    }
    let x0d = DD::from_f64(x0);
    let x2 = [x0d * x0d, x0d.mul_f(2.0), DD::ONE];
    let src = ctx.planted.as_ref().map(|pl| pl.series(x0d, 1.0, kt, 1.0));
    let mut iw0 = Fl::<T, B>::zero();
    let mut iv0 = Fl::<T, B>::zero();
    for k in 0..kt {
        let kf = k as f64;
        if k == 0 {
            let (s0, c0) = sin_cos_field(&q[0], ctx.k2);
            sn[0] = s0;
            co[0] = c0;
        } else {
            let ik = lane_int::<T>(k);
            sn[k] = cs_w(q, co, k, 1, k, |j| j as f64) / Fl([ik; B]);
            co[k] = -(cs_w(q, sn, k, 1, k, |j| j as f64) / Fl([ik; B])).muls(k2);
        }
        a2[k] = cs(a, a, k);
        q2[k] = cs(p, p, k) + cs(m, m, k);
        ss[k] = cs(sn, sn, k);
        g2[k] = cs(a2, ss, k);
        sg[k] = cs(sn, co, k);
        hs[k] = cs(a2, sg, k);
        g[k] = g_coef(ctx, c, a2, g2, k);
        h[k] = cs(a, q2, k) + (a[k] - cs(a, a2, k)).muls(n1) + cs(a, g2, k).muls(n2h);
        if let Some(s) = &src {
            g[k] = g[k] + Fl::cst(&s[3][k]);
            h[k] = h[k] + Fl::cst(&s[2][k]);
        }
        pth[k] = dtau(ctx, &p[k], omega);
        mth[k] = dtau(ctx, &m[k], omega);
        let mut xp = pth[k].muls(x0d);
        let mut xm = mth[k].muls(x0d);
        if k > 0 {
            xp = xp + pth[k - 1];
            xm = xm + mth[k - 1];
        }
        let cp = cs(c, p, k);
        let cm = cs(c, m, k);
        let cpm = cp + cm;
        let chs = cs(c, hs, k).muls(n);
        let mut rp = (((cs(g, p, k) - xp) - cp) + cpm.muls(nh)) - chs;
        let mut rm = (((cm - xm) - cs(g, m, k)) - cpm.muls(nh)) + chs;
        let mut rq = (p[k] + m[k]).mulf(0.5);
        if let Some(s) = &src {
            rp = rp + Fl::cst(&s[0][k]);
            rm = rm + Fl::cst(&s[1][k]);
            rq = rq + Fl::cst(&s[4][k]);
        }
        // 1 / (x0 (k + 1)), the product exact in double-double
        let ik = T::splat(DD::ONE) / T::splat(x0d * DD::from_f64(kf + 1.0));
        c[k + 1] = (g[k] - c[k].mulf(kf)).mul_t(ik);
        a[k + 1] = (h[k].mulf(0.5) - a[k].mulf(kf)).mul_t(ik);
        q[k + 1] = (rq - q[k].mulf(kf)).mul_t(ik);
        // W = x (x - c), V = x (x + c)
        let mut xc = c[k].muls(x0d);
        if k > 0 {
            xc = xc + c[k - 1];
        }
        if k <= 2 {
            ww[k] = (-xc).adds(x2[k]);
            vv[k] = xc.adds(x2[k]);
        } else {
            ww[k] = -xc;
            vv[k] = xc;
        }
        if k == 0 {
            iw0 = Fl::<T, B>::splat(DD::ONE) / ww[0];
            iv0 = Fl::<T, B>::splat(DD::ONE) / vv[0];
        }
        let (sp, sm) = if k >= 1 {
            (cs_w(ww, p, k + 1, 1, k, |j| (k + 1 - j) as f64), cs_w(vv, m, k + 1, 1, k, |j| (k + 1 - j) as f64))
        } else {
            (Fl::zero(), Fl::zero())
        };
        let ik1 = Fl([lane_int::<T>(k + 1); B]);
        p[k + 1] = ((rp - sp) * iw0) / ik1;
        m[k + 1] = ((rm - sm) * iv0) / ik1;
    }
}

/// Work for one Taylor step of the five fields at order kt
pub fn work<T: Lane, const B: usize>(kt: usize) -> Work<T, B> {
    Work::new(NF, N_AUX, kt)
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (regular centre: psi ~ q x, p ~ m ~ q x), evaluated at nodes[0]
// ---------------------------------------------------------------------------------------------------------------

/// Centre-series coefficients of p, m, a, c, psi in the scaled variable x / sigma, sigma = series_scale(nodes[0]):
/// entry k is c_k sigma^k. The order-k system for (P_k, M_k) is sigdss_solver.py's centre_series: sigma's order-k
/// coefficient carries Q_k = (P_k + M_k)/(2k) (sigma_0 = 0, gamma_0 = 1), which enters the matrix through
/// al = n/2 - n/(2k).
pub fn centre_series<T: Lane, const B: usize>(ctx: &Ctx<B>, q1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> [Vec<Fl<T, B>>; 5] {
    let sig = series_scale(ctx.nodes[0]);
    let n = ctx.n;
    let n1 = n - DD::ONE;
    let nh = n.mul_f(0.5);
    let n2 = n.mul_f(2.0);
    let k2 = ctx.k2;
    let kmax = ctx.kc;
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
    let v = || vec![z; kmax + 1];
    let (mut pp, mut mm, mut aa, mut cc, mut qq) = (v(), v(), v(), v(), v());
    let (mut sn, mut co, mut a2, mut q2, mut ss, mut g2, mut sg, mut hs) = (v(), v(), v(), v(), v(), v(), v(), v());
    let src = ctx.planted.as_ref().map(|pl| pl.series(DD::ZERO, 1.0, kmax, sig));
    aa[0] = one;
    cc[0] = *c0;
    co[0] = one;
    a2[0] = one;
    pp[1] = q1.mul_pwr2(sig);
    mm[1] = pp[1];
    qq[1] = pp[1];
    sn[1] = pp[1];
    sg[1] = sn[1];
    hs[1] = sg[1];
    let invc = one / cc[0];
    for k in 1..=kmax {
        let kf = k as f64;
        let ik = Fl([lane_int::<T>(k); B]);
        if k >= 2 {
            // gamma_k is complete (the j = k term has sigma_0 = 0); sigma_k lacks Q_k gamma_0 = Q_k
            co[k] = -(cs_w(&qq, &sn, k, 1, k - 1, |j| j as f64) / ik).muls(k2);
            sn[k] = cs_w(&qq, &co, k, 1, k - 1, |j| j as f64) / ik;
        }
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        ss[k] = cs(&sn, &sn, k);
        a2[k] = cs(&aa, &aa, k); // aa[k] still 0
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        g2[k] = cs(&a2, &ss, k);
        let u = cs(&aa, &g2, k);
        // (k + n - 1) A_k = s/2 + n u - (n-1) t3/2 (+ S_a/2)
        let mut ra = (s.mulf(0.5) + u.muls(n)) - t3.muls(n1.mul_f(0.5));
        if let Some(sr) = &src {
            ra = ra + Fl::cst(&sr[2][k]).mulf(0.5);
        }
        aa[k] = ra / Fl::splat(DD::from_f64(kf) + n1);
        a2[k] = cs(&aa, &aa, k);
        // k C_k = (n-1)((C A2)_k - C_k) - 2n (C G2)_k (+ S_c), with cc[k] = 0 in the sums
        let mut rc = cs(&cc, &a2, k).muls(n1) - cs(&cc, &g2, k).muls(n2);
        if let Some(sr) = &src {
            rc = rc + Fl::cst(&sr[3][k]);
        }
        cc[k] = rc / ik;
        if k == 1 {
            continue;
        }
        // the p, m equations at order k (index sum k; the terms of order k - 1 alone gain one sigma)
        let dp = dtau(ctx, &pp[k - 1], omega).mul_pwr2(sig);
        let dm = dtau(ctx, &mm[k - 1], omega).mul_pwr2(sig);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0).mul_pwr2(sig) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0).mul_pwr2(sig) + s1m;
        let xcxp = cs_w(&cc, &pp, k, 1, k, |j| j as f64);
        let xcxm = cs_w(&cc, &mm, k, 1, k, |j| j as f64);
        let cp = cs(&cc, &pp, k);
        let cm = cs(&cc, &mm, k);
        let cpm = cp + cm;
        // sigma gamma and a^2 sigma gamma at order k without Q_k
        sg[k] = cs(&sn, &co, k);
        hs[k] = cs(&a2, &sg, k);
        let ch = cs(&cc, &hs, k).muls(n);
        let mut r3 = ((((dp + xpx) - xcxp) + cp) - cpm.muls(nh)) + ch;
        let mut r4 = ((((dm + xmx) + xcxm) - cm) + cpm.muls(nh)) - ch;
        if let Some(sr) = &src {
            r3 = r3 - Fl::cst(&sr[0][k]);
            r4 = r4 - Fl::cst(&sr[1][k]);
        }
        // c0 [mu11 mu12; mu21 mu22] (P_k, M_k) = -(r3, r4), al = n/2 - n/(2k), det = -(k-1)^2 (k + n)/k
        let al = nh - nh / DD::from_f64(kf);
        let mu11 = DD::from_f64(1.0 - kf) - al;
        let mu12 = -al;
        let mu21 = al;
        let mu22 = DD::from_f64(kf - 1.0) + al;
        let det = mu11 * mu22 - mu12 * mu21;
        let idel = DD::ONE / det;
        pp[k] = ((r4.muls(mu12) - r3.muls(mu22)) * invc).muls(idel);
        mm[k] = ((r3.muls(mu21) - r4.muls(mu11)) * invc).muls(idel);
        qq[k] = (pp[k] + mm[k]).mulf(0.5) / ik;
        sn[k] = sn[k] + qq[k];
        sg[k] = cs(&sn, &co, k);
        hs[k] = cs(&a2, &sg, k);
    }
    [pp, mm, aa, cc, qq]
}

pub fn centre_state<T: Lane, const B: usize>(ctx: &Ctx<B>, q1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> State<T, B> {
    let s = centre_series(ctx, q1, c0, omega);
    let x = DD::from_f64(ctx.nodes[0]).mul_pwr2(1.0 / series_scale(ctx.nodes[0]));
    s.iter().map(|f| horner(f, x)).collect()
}

// ---------------------------------------------------------------------------------------------------------------
// horizon series (regular horizon), in u = 1 - x
// ---------------------------------------------------------------------------------------------------------------

/// c1 = dc/du at the horizon, -(x c_x) there, from the horizon data at double-double values: the same operations as
/// the lanes' cc[1] in horizon_series (so the LUs are those of the lanes' value systems)
fn horizon_c1<T: Lane, const B: usize>(ctx: &Ctx<B>, ah: &Fl<T, B>, psih: &Fl<T, B>) -> Fl<T, B> {
    let (s0, _) = sin_cos_field(psih, ctx.k2);
    let a2 = [*ah * *ah];
    let sn = [s0];
    let ss = [cs(&sn, &sn, 0)];
    let g2 = [cs(&a2, &ss, 0)];
    let cc = [Fl::<T, B>::splat(DD::ONE)];
    let mut g0 = g_coef(ctx, &cc, &a2, &g2, 0);
    if let Some(pl) = &ctx.planted {
        g0 = g0 + Fl::cst(&pl.series(DD::ONE, -1.0, 0, 1.0)[3][0]);
    }
    -g0
}

/// LU factors of (omega D + diag kappa_kk), kk = 0..=ks, at the given (a_h, psi_h, Delta) values:
/// kappa_kk = (c1 + 1) kk + c1 + 1 - n/2
pub fn horizon_lus<const B: usize>(ctx: &Ctx<B>, ah: &Fl<DV, B>, psih: &Fl<DV, B>, delta: DD, par: bool) -> Result<Vec<DDLu>, String> {
    let m = Ctx::<B>::M;
    let ld = ld_for(m);
    let omega = DD::TWO_PI / delta;
    let c1 = horizon_c1(ctx, ah, psih);
    let c1s: Vec<DD> = (0..m).map(|i| c1.get(i)).collect();
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

/// The preparation of one evaluation (multishoot's `Equations::Prep`): the horizon LUs
#[derive(Default)]
pub struct HPrep {
    pub lus: Vec<DDLu>,
}

/// (omega D + diag kappa) f = r with the LU at the values; for tangents the derivative of the system,
/// A f_d = r_d - omega_d D f_v - kappa_d f_v; quad-double lanes refined to their precision (hp's psolve, its DV, Du,
/// QV and Qu branches)
fn psolve<T: Lane, const B: usize>(ctx: &Ctx<B>, prep: &HPrep, kk: usize, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Fl<T, B> {
    let lu = &prep.lus[kk];
    assert!(
        T::NJ == 0 && !T::VAL_OD && !T::VAL_F64 && !T::TAN_F64,
        "sigmams: the horizon solve supports double-double and quad-double lanes only"
    );
    if T::VAL_QD {
        // Moler's refinement with the residual in the lane's precision (multidouble's solve_lanes_refine_t)
        let steps = if ctx.mutate == MUT_QD_NOREFINE { 0 } else { QD_REFINE + T::TANGENT as usize };
        return lu.solve_lanes_refine_t(lu, rhs, steps, |f| *rhs - (ctx.d.apply(f).mul_t(omega) + *kappa * *f));
    }
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

/// Horizon-series coefficients of p, m, a, c, psi in the scaled variable u / sigma, sigma = series_scale(u0)
pub fn horizon_series<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    psi0: &Fl<T, B>,
    omega: T,
    prep: &HPrep,
) -> [Vec<Fl<T, B>>; 5] {
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
    let n = ctx.n;
    let n1 = n - DD::ONE;
    let nh = n.mul_f(0.5);
    let n2 = n.mul_f(2.0);
    let k2 = ctx.k2;
    let ks = ctx.ks;
    let len = ks + 2;
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
    let v = || vec![z; len];
    let (mut pp, mut mm, mut aa, mut cc, mut qq) = (v(), v(), v(), v(), v());
    let (mut sn, mut co, mut a2, mut q2, mut ss, mut g2, mut sg, mut hs) = (v(), v(), v(), v(), v(), v(), v(), v());
    let (mut dp, mut dm, mut pm, mut vv, mut ww, mut zc) = (v(), v(), v(), v(), v(), v());
    // the sources as series in u about x = 1, scaled
    let src = ctx.planted.as_ref().map(|pl| pl.series(DD::ONE, -1.0, ks + 1, sig));
    aa[0] = *a0;
    cc[0] = one;
    mm[0] = *m0;
    qq[0] = *psi0;
    a2[0] = aa[0] * aa[0];
    dm[0] = dtau(ctx, &mm[0], omega);
    let sone = one.mul_pwr2(sig);
    // the scaled coefficients of w = (1 - u) - c and z = c + (1 - u)
    let wfun = |c: &Vec<Fl<T, B>>, i: usize| -> Fl<T, B> {
        match i {
            0 => one - c[0],
            1 => -c[1] - sone,
            _ => -c[i],
        }
    };
    let zfun = |c: &Vec<Fl<T, B>>, i: usize| -> Fl<T, B> {
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
        let ik1 = Fl([lane_int::<T>(kk + 1); B]);
        // sigma, gamma and their products at order kk (psi_kk known from the previous order)
        if kk == 0 {
            let (s0, c0) = sin_cos_field(&qq[0], k2);
            sn[0] = s0;
            co[0] = c0;
        } else {
            let ik = Fl([lane_int::<T>(kk); B]);
            sn[kk] = cs_w(&qq, &co, kk, 1, kk, |j| j as f64) / ik;
            co[kk] = -(cs_w(&qq, &sn, kk, 1, kk, |j| j as f64) / ik).muls(k2);
        }
        ss[kk] = cs(&sn, &sn, kk);
        g2[kk] = cs(&a2, &ss, kk);
        sg[kk] = cs(&sn, &co, kk);
        hs[kk] = cs(&a2, &sg, kk);
        // c: (kk+1) C_{kk+1} = kk C_kk - (x c_x)_kk
        let mut gk = g_coef(ctx, &cc, &a2, &g2, kk);
        if let Some(s) = &src {
            gk = gk + Fl::cst(&s[3][kk]);
        }
        cc[kk + 1] = ((cc[kk].mulf(kf) - gk) / ik1).mul_pwr2(sig);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk).mul_pwr2(sig);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64).mul_pwr2(isig) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk).mul_pwr2(sig);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let tau = if kk >= 1 { -dp[kk - 1].mul_pwr2(sig) } else { z };
        let chs = cs(&cc, &hs, kk).muls(n);
        let mut r = ((((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).muls(nh)) + chs;
        if let Some(s) = &src {
            r = r - Fl::cst(&s[0][kk]);
        }
        // kappa (and the LUs of horizon_lus) take the unscaled first-order coefficient
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).adds(-nh);
        pp[kk] = psolve(ctx, prep, kk, &kappa, &(-r), omega);
        dp[kk] = dtau(ctx, &pp[kk], omega);
        q2[kk] = cs(&pp, &pp, kk) + cs(&mm, &mm, kk);
        if kk == ks {
            break;
        }
        // a: (kk+1) A_{kk+1} = kk A_kk - (2 x a_x)_kk / 2
        let mut hk = cs(&aa, &q2, kk) + (aa[kk] - cs(&aa, &a2, kk)).muls(n1) + cs(&aa, &g2, kk).muls(n2);
        if let Some(s) = &src {
            hk = hk + Fl::cst(&s[2][kk]);
        }
        aa[kk + 1] = ((aa[kk].mulf(kf) - hk.mulf(0.5)) / ik1).mul_pwr2(sig);
        a2[kk + 1] = cs(&aa, &aa, kk + 1);
        // m: (kk+1) z_0 M_{kk+1} = the rest
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1].mul_pwr2(sig) } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let mut rest = ((((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).muls(nh)) - chs;
        // psi: (kk+1) Q_{kk+1} = kk Q_kk - (x psi_x)_kk
        let mut rq = pm[kk].mulf(0.5);
        if let Some(s) = &src {
            rest = rest - Fl::cst(&s[1][kk]);
            rq = rq + Fl::cst(&s[4][kk]);
        }
        mm[kk + 1] = ((rest * izc0) / ik1).mul_pwr2(sig);
        dm[kk + 1] = dtau(ctx, &mm[kk + 1], omega);
        qq[kk + 1] = ((qq[kk].mulf(kf) - rq) / ik1).mul_pwr2(sig);
    }
    for f in [&mut pp, &mut mm, &mut aa, &mut cc, &mut qq] {
        f.truncate(ks + 1);
    }
    [pp, mm, aa, cc, qq]
}

pub fn horizon_state<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    psi0: &Fl<T, B>,
    omega: T,
    prep: &HPrep,
) -> State<T, B> {
    let s = horizon_series(ctx, m0, a0, psi0, omega, prep);
    let u = ctx.u0().mul_pwr2(1.0 / series_scale(ctx.u0().hi));
    s.iter().map(|f| horner(f, u)).collect()
}

// ---------------------------------------------------------------------------------------------------------------
// the equations for multishoot
// ---------------------------------------------------------------------------------------------------------------

/// The sigma-model's equations as multishoot's `Equations`: fields p, m, a, c, psi (NF = 5), centre data q, c0
/// (NL = 2), horizon data m_h, a_h, psi_h (NR = 3) and Delta (NG = 1)
pub struct Sig<'a, const B: usize>(pub &'a Ctx<B>);

pub(crate) fn delta_of(zh: &[f64], zl: &[f64]) -> DD {
    let l = zh.len();
    DD { hi: zh[l - 1], lo: zl[l - 1] }
}

impl<const B: usize> Equations<B> for Sig<'_, B> {
    const NF: usize = NF;
    const NL: usize = 2;
    const NR: usize = 3;
    const NG: usize = 1;
    const NAUX: usize = N_AUX;
    type Prep = HPrep;

    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<HPrep, String> {
        let ctx = self.0;
        let m = Ctx::<B>::M;
        let o = ctx.off_h();
        let ah = ctx.fld(zh, zl, o + m);
        let psih = ctx.fld(zh, zl, o + 2 * m);
        Ok(HPrep { lus: horizon_lus(ctx, &ah, &psih, delta_of(zh, zl), true)? })
    }

    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        taylor_coeffs(self.0, x0, omega_of(g[0]), w);
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], _prep: &HPrep) -> State<T, B> {
        centre_state(self.0, &data[0], &data[1], omega_of(g[0]))
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], prep: &HPrep) -> State<T, B> {
        horizon_state(self.0, &data[0], &data[1], &data[2], omega_of(g[0]), prep)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// entry points in any precision (multishoot's, at multidouble's Precision)
// ---------------------------------------------------------------------------------------------------------------

/// The residual in P, one scalar per row
pub fn residual_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
    ctx.shooting_in::<P>().residual::<P>(z, par)
}

/// The exact Jacobian in P by forward-mode duals: (PARTS column-major part arrays, ld)
pub fn jacobian_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
    ctx.shooting_in::<P>().jacobian::<P>(z, par)
}

/// The Newton step in P at z (multishoot's newton_step)
pub fn newton_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<multishoot::newton::Step<P>, String> {
    ctx.shooting_in::<P>().newton_step::<P>(z, par)
}

/// Step points of every segment planned from the node states of z (multishoot's plan_segments: each step's last
/// two Taylor terms below tol, at most hmax)
pub fn plan<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>> {
    let mut sh = ctx.shooting();
    sh.plan_segments(zh, zl, &Plan { kt: ctx.kt, tol, hmax });
    sh.segs
}

/// The state (five fields, double-double values) carried by Taylor steps through the points xs at Delta of z
pub fn integrate<const B: usize>(ctx: &Ctx<B>, delta: DD, s0: &State<DV, B>, xs: &[f64]) -> State<DV, B> {
    let mut w = work::<DV, B>(ctx.kt);
    let g = [DV::splat(delta)];
    multishoot::taylor::taylor_path(&Sig(ctx), s0, xs, &g, &mut w)
}
