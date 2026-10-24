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
use crate::lane::{BaseLane, Dtn, Du, F8, Fu, Fun, Lane};
use crate::field::FieldSolve;
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
    /// Optional (Pi_low, Pi_high): projectors onto the low / high tau-harmonics of an odd field on the half grid.
    /// With it, the horizon block's p rows become Pi_low (p mismatch) + Pi_high p1: the high harmonics of the p
    /// matching are dropped and the high harmonics of p1 pinned to zero instead. Those p1 harmonics reach the matching
    /// point only through exponentially small couplings (the near-null directions), and their effect on Delta is
    /// below 1e-12 per unit change.
    pub freeze: Option<(DMat<B>, DMat<B>)>,
    /// Index of the node where the outward integration from the centre meets the inward integration from the
    /// horizon side. Segments b <= match_node run outward (node b-1 -> node b); segments b > match_node run inward
    /// (node b -> node b-1). nodes.len() - 1 (the default) is the all-outward layout.
    pub match_node: usize,
    /// Growth rate for the mode matrix: with lam != 0, every tau-derivative of a tangent (perturbation) gains + lam
    /// (perturbations ~ exp(lam tau)); values are unaffected. Zero for the residual and the ordinary Jacobian.
    pub lam: DD,
    /// Deliberate breakage of the f64 lanes only (`jacobian_f64`), for the control of the Jacobian-agreement check
    /// (hp/mixed_check.py --mutate): 0 none, 1 no tau derivative, 2 omega scaled by 1 + 1e-9, 3 the horizon solves'
    /// kappa tangent term dropped; 6 the centre series' scale wrong at one order (twice sigma for the order-7 terms
    /// built from order 6), the control of the scaled series' bit-identity (hp/highm_check.py scaled). The
    /// double-double lanes never read it.
    pub mutate: u8,
    /// Parts of the f64 Jacobian (TAN_F64 lanes) computed in double-double instead, rounded to f64; bit flags HDD_*
    /// (notes/LARGE_D_MIXED.md). 0, the default, changes nothing. The double-double lanes never read it.
    pub f64_dd: u8,
}

/// `Ctx::f64_dd`: the horizon series' periodic solves (`psolve`) in double-double for the f64 lanes, value and
/// tangents, with the full double-double LU factors instead of their hi parts
pub const HDD_SOLVE: u8 = 1;
/// `Ctx::f64_dd`: no early end of the horizon series for f64 tangents (`Tail`); the series runs to ks
pub const HDD_NOTAIL: u8 = 2;
/// `Ctx::f64_dd`: the horizon block's column tasks (m_h, a_h, Delta) with `Du` lanes, i.e. the double-double
/// Jacobian's columns of that block, rounded to f64
pub const HDD_BLOCK: u8 = 4;
/// `Ctx::f64_dd`: the last segment's column tasks (block K - 1) with `Du` lanes
pub const HDD_LASTSEG: u8 = 8;
/// `Ctx::f64_dd`: the horizon block's column tasks with `Dtn<4>` lanes (a double-double base trajectory, f64
/// tangents); with HDD_SOLVE also the tangent solves in double-double
pub const HDD_DTN: u8 = 16;
/// `Ctx::f64_dd`, for the double-double tangent lanes (`Du`) of `jvp`: the centre and horizon series stop once their
/// tangent terms are below double-double resolution (`Tail` with TAIL_TOL_DD), instead of running to kc and ks. The
/// scan's kc = 420 at large D is far past that (the centre series' terms fall to 1e-134 at D = 50), and the centre
/// series is most of a product's cost there.
pub const HDD_DU_CUT: u8 = 32;

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
            let o = if self.seg_inward(b) { self.off_node(b) } else { self.off_node(b - 1) };
            vec![o, o + m, o + 2 * m, o + 3 * m]
        } else {
            vec![self.off_mh(), self.off_ah()]
        }
    }
    /// segment b (1 <= b < K) runs inward, from node b to node b-1
    pub fn seg_inward(&self, b: usize) -> bool {
        b > self.match_node && b < self.k()
    }
    pub fn block_target(&self, b: usize) -> usize {
        if b < self.k() {
            if self.seg_inward(b) {
                self.off_node(b - 1)
            } else {
                self.off_node(b)
            }
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

/// d/dtau = omega D, plus lam on the tangent part (perturbations ~ exp(lam tau); see Ctx::lam).
#[inline(always)]
fn dtau<T: Lane, const B: usize>(ctx: &Ctx<B>, x: &Fl<T, B>, omega: T) -> Fl<T, B> {
    if T::TAN_F64 && ctx.mutate == 1 {
        return Fl::zero();
    }
    let r = ctx.d.apply(x).mul_t(omega);
    if T::TANGENT && ctx.lam.hi != 0.0 {
        if T::TAN_F64 {
            // every f64 tangent gains lam (hi part) times itself
            let (mut r, l) = (r, crate::v8::V8::splat(ctx.lam.hi));
            for b in 0..B {
                for k in 0..T::NT {
                    let t = x.0[b].tan64(k).fma(l, r.0[b].tan64(k));
                    r.0[b].set_tan64(k, t);
                }
            }
            return r;
        }
        r + Fl::from_vt(&Fl::zero(), &x.tan().muls(ctx.lam))
    } else {
        r
    }
}

/// Early end of the centre and horizon series for the f64-tangent lanes (a Jacobian in f64 needs their terms only
/// to the f64 rounding level, and the series orders kc, ks are set for double-double values): the series stops after
/// order k once the tangent terms max |c_k| r^k (r the evaluation radius, over the four fields, all lanes and
/// tangents) of TAIL_RUN consecutive orders are all below TAIL_TOL times the largest tangent term so far, k >=
/// TAIL_MIN. The dropped tail is then below the rounding of an f64 sum (checked against the double-double Jacobian
/// by hp/mixed_check.py). The double-double lanes never stop early.
struct Tail {
    rk: f64,
    r: f64,
    big: f64,
    run: usize,
}

const TAIL_TOL: f64 = 8.673617379884035e-19; // 2^-60
/// the cut of HDD_DU_CUT (double-double tangents)
const TAIL_TOL_DD: f64 = 7.703719777548943e-34; // 2^-110
const TAIL_RUN: usize = 4;
const TAIL_MIN: usize = 8;

impl Tail {
    fn new(r: f64) -> Tail {
        Tail { rk: 1.0, r, big: 0.0, run: 0 }
    }
    /// whether the series of lane type T stops early under ctx: f64 tangents always (TAIL_TOL), double-double
    /// tangents with HDD_DU_CUT (TAIL_TOL_DD)
    fn active<T: Lane, const B: usize>(ctx: &Ctx<B>) -> bool {
        T::TAN_F64 || (T::TANGENT && ctx.f64_dd & HDD_DU_CUT != 0)
    }
    /// record order k as `done`, for the lanes `active` allows
    fn done_any<T: Lane, const B: usize>(&mut self, k: usize, f: [&Fl<T, B>; 4]) -> bool {
        if T::TAN_F64 {
            return self.done(k, f);
        }
        let mut a = 0.0f64;
        for x in f {
            for b in 0..B {
                a = a.max(x.0[b].tan().hi.abs().hmax());
            }
        }
        let t = a * self.rk;
        self.rk *= self.r;
        self.big = self.big.max(t);
        self.run = if t <= TAIL_TOL_DD * self.big { self.run + 1 } else { 0 };
        self.big > 0.0 && k >= TAIL_MIN && self.run >= TAIL_RUN
    }
    /// record order k (called for k = 0, 1, 2, ... in turn); true when the series may stop after it
    fn done<T: Lane, const B: usize>(&mut self, k: usize, f: [&Fl<T, B>; 4]) -> bool {
        let mut a = 0.0f64;
        for x in f {
            for b in 0..B {
                for t in 0..T::NT {
                    a = a.max(x.0[b].tan64(t).abs().hmax());
                }
            }
        }
        let t = a * self.rk;
        self.rk *= self.r;
        self.big = self.big.max(t);
        self.run = if t <= TAIL_TOL * self.big { self.run + 1 } else { 0 };
        self.big > 0.0 && k >= TAIL_MIN && self.run >= TAIL_RUN
    }
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (regular centre), evaluated at nodes[0]
// ---------------------------------------------------------------------------------------------------------------
/// The power of two nearest to x (x > 0): the series below are computed in the scaled variable x / series_scale(x0)
/// (u / series_scale(u0) at the horizon), so that their coefficients c_k sigma^k stay near the evaluation radius's
/// scale instead of growing like r^-k, which overflows the f64 exponent range (and the f64 Jacobian's tangents first)
/// when r is small and the order high (r ~ 1e-3 at 100 orders: 1e300). Scaling by a power of two is exact in binary
/// floating point, so the values agree bit for bit with the unscaled recursion wherever that one neither overflows nor
/// underflows (notes/HIGH_M.md 9.5).
pub fn series_scale(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    2f64.powi(x.log2().round() as i32)
}

/// Centre-series coefficients of p, m, a, c in the scaled variable x / sigma, sigma = series_scale(nodes[0]): entry k
/// is c_k sigma^k (multiply by sigma^-k for the coefficients in x).
pub fn centre_series<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> [Vec<Fl<T, B>>; 4] {
    let sig = series_scale(ctx.nodes[0]);
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
    pp[1] = p1.mul_pwr2(sig);
    mm[1] = -pp[1];
    a2[0] = aa[0] * aa[0];
    let invc = Fl::<T, B>::splat(DD::ONE) / cc[0];
    let mut tail = Tail::new(ctx.nodes[0] / sig);
    let cut = Tail::active::<T, B>(ctx);
    if cut {
        tail.done_any(0, [&pp[0], &mm[0], &aa[0], &cc[0]]);
        tail.done_any(1, [&pp[1], &mm[1], &aa[1], &cc[1]]);
    }
    for k in 2..=kmax {
        let kf = k as f64;
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        a2[k] = cs(&aa, &aa, k); // aa[k] still 0
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        aa[k] = (s.mulf(0.5) - t3.muls(n1.mul_f(0.5))).muls(DD::ONE / (DD::from_f64(kf) + n1));
        a2[k] = cs(&aa, &aa, k);
        cc[k] = cs_range(&cc, &a2, k, 0, k - 1).muls(n1 * recip_int(k));
        // in the scaled variable every term of the x^(k-1) equation carries sigma^k: those built from order k - 1
        // alone (the d/dtau and x p' terms) gain one factor sigma, the Cauchy sums of index sum k none
        let sk = if T::TAN_F64 && ctx.mutate == 6 && k == 7 { 2.0 * sig } else { sig };
        let dp = dtau(ctx, &pp[k - 1], omega).mul_pwr2(sk);
        let dm = dtau(ctx, &mm[k - 1], omega).mul_pwr2(sk);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0).mul_pwr2(sk) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0).mul_pwr2(sk) + s1m;
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
        if cut && tail.done_any(k, [&pp[k], &mm[k], &aa[k], &cc[k]]) {
            for v in [&mut pp, &mut mm, &mut aa, &mut cc] {
                v.truncate(k + 1);
            }
            break;
        }
    }
    [pp, mm, aa, cc]
}

pub fn centre_state<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> State<T, B> {
    let s = centre_series(ctx, p1, c0, omega);
    let x = DD::from_f64(ctx.nodes[0]).mul_pwr2(1.0 / series_scale(ctx.nodes[0]));
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
        w.pth[k] = dtau(ctx, &w.p[k], omega);
        w.mth[k] = dtau(ctx, &w.m[k], omega);
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
pub fn horizon_lus<const B: usize>(ctx: &Ctx<B>, a0: &Fl<DV, B>, delta: DD, par: bool, shift: DD) -> Result<Vec<DDLu>, String> {
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
            let kap = (c1s[i] + DD::ONE).mul_f(kk as f64) + c1s[i] + DD::ONE - nh + shift;
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

/// f64 tangent k of a field (TAN_F64 lanes) as an f64 field
#[inline(always)]
fn tan_f8<T: Lane, const B: usize>(x: &Fl<T, B>, k: usize) -> Fl<F8, B> {
    let mut r = [multidouble::dual::F8(crate::v8::V8::zero()); B];
    for b in 0..B {
        r[b] = multidouble::dual::F8(x.0[b].tan64(k));
    }
    Fl(r)
}

/// (omega D + diag kappa) f = r with the LU at the values; for tangents the derivative of the system:
/// A_t f_d = r_d - omega_d D f_v - kappa_d f_v, with A_t = A + lam (lu_t; the same LU when lam = 0).
fn psolve<T: Lane, const B: usize>(ctx: &Ctx<B>, lu: &DDLu, lu_t: &DDLu, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Fl<T, B> {
    if T::VAL_F64 || T::TAN_F64 {
        // f64 values: the value solve in plain f64 with the hi parts of the double-double LU factors; f64 tangents:
        // each tangent's solve likewise. With HDD_SOLVE both in double-double (the f64 right-hand side solved exactly
        // by the double-double factors, then rounded to f64).
        let dds = ctx.f64_dd & HDD_SOLVE != 0;
        let fv = if T::VAL_F64 && !dds { lu.solve_lanes_hi(&rhs.val()) } else { lu.solve_lanes(&rhs.val()) };
        let z = Fl::<DV, B>::zero();
        let mut out = Fl::<T, B>::from_vt(&fv, &z);
        if !T::TANGENT {
            return out;
        }
        let fv8 = Fl::<F8, B>::from_vt(&fv, &z);
        let dfv = ctx.d.apply(&fv8);
        for k in 0..T::NT {
            let od = multidouble::dual::F8(omega.tan64(k));
            let mut r = tan_f8(rhs, k) - dfv.mul_t(od);
            if ctx.mutate != 3 {
                r = r - tan_f8(kappa, k) * fv8;
            }
            let fd = if dds { lu_t.solve_lanes(&r.val()) } else { lu_t.solve_lanes_hi(&r.val()) };
            for b in 0..B {
                out.0[b].set_tan64(k, fd.0[b].hi);
            }
        }
        return out;
    }
    let fv = lu.solve_lanes(&rhs.val());
    if !T::TANGENT {
        return Fl::from_vt(&fv, &Fl::zero());
    }
    let dfv = ctx.d.apply::<DV>(&fv);
    let od = Fl::<DV, B>([omega.tan(); B]);
    let r = (rhs.tan() - dfv * od) - kappa.tan() * fv;
    let fd = lu_t.solve_lanes(&r);
    Fl::from_vt(&fv, &fd)
}

/// Horizon-series coefficients of p, m, a, c in the scaled variable u / sigma, sigma = series_scale(u0) (u = 1 - x):
/// entry kk is c_kk sigma^kk, as for centre_series.
pub fn horizon_series<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    omega: T,
    lus: &[DDLu],
    lus_t: &[DDLu],
) -> [Vec<Fl<T, B>>; 4] {
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
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
    dm[0] = dtau(ctx, &mm[0], omega);
    // the scaled coefficients (sigma^i times) of w = (1 - u) - c and z = c + (1 - u); `sone` is the u term's sigma
    let sone = one.mul_pwr2(sig);
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
    let mut tail = Tail::new(ctx.u0().hi * isig);
    let cut = Tail::active::<T, B>(ctx) && ctx.f64_dd & HDD_NOTAIL == 0;
    let mut last = ks;
    // In the scaled variable each term of the order-kk equations carries sigma^kk: a coefficient of order kk + 1 made
    // from order-kk terms gains sigma, a term of index sum kk - 1 gains sigma, one of index sum kk + 1 loses it.
    for kk in 0..=ks {
        let kf = kk as f64;
        let ik1 = recip_int(kk + 1);
        let t = cs(&cc, &a2, kk);
        cc[kk + 1] = (cc[kk].mulf(kf) - (t - cc[kk]).muls(n1)).muls(ik1).mul_pwr2(sig);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk).mul_pwr2(sig);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64).mul_pwr2(isig) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk).mul_pwr2(sig);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let tau = if kk >= 1 { -dp[kk - 1].mul_pwr2(sig) } else { z };
        let r = (((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).muls(nh);
        // kappa (and the LUs of horizon_lus) take the unscaled first-order coefficient
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).adds(-nh);
        pp[kk] = psolve(ctx, &lus[kk], &lus_t[kk], &kappa, &(-r), omega);
        dp[kk] = dtau(ctx, &pp[kk], omega);
        q2[kk] = cs(&pp, &pp, kk) + cs(&mm, &mm, kk);
        if kk == ks || (cut && tail.done_any(kk, [&pp[kk], &mm[kk], &aa[kk], &cc[kk]])) {
            last = kk;
            break;
        }
        let a3 = cs(&aa, &a2, kk);
        let ag = cs(&aa, &q2, kk) + (aa[kk] - a3).muls(n1);
        aa[kk + 1] = (aa[kk].mulf(kf) - ag.mulf(0.5)).muls(ik1).mul_pwr2(sig);
        a2[kk + 1] = cs(&aa, &aa, kk + 1);
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1].mul_pwr2(sig) } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let rest = (((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).muls(nh);
        mm[kk + 1] = (rest * izc0).muls(ik1).mul_pwr2(sig);
        dm[kk + 1] = dtau(ctx, &mm[kk + 1], omega);
    }
    pp.truncate(last + 1);
    mm.truncate(last + 1);
    aa.truncate(last + 1);
    cc.truncate(last + 1);
    [pp, mm, aa, cc]
}

pub fn horizon_state<T: Lane, const B: usize>(ctx: &Ctx<B>, m0: &Fl<T, B>, a0: &Fl<T, B>, omega: T, lus: &[DDLu], lus_t: &[DDLu]) -> State<T, B> {
    let s = horizon_series(ctx, m0, a0, omega, lus, lus_t);
    let u = ctx.u0().mul_pwr2(1.0 / series_scale(ctx.u0().hi));
    [horner(&s[0], u), horner(&s[1], u), horner(&s[2], u), horner(&s[3], u)]
}

// ---------------------------------------------------------------------------------------------------------------
// blocks, residual, Jacobian
// ---------------------------------------------------------------------------------------------------------------
pub fn block_state<T: Lane, const B: usize>(ctx: &Ctx<B>, b: usize, inp: &[Fl<T, B>], delta: T, lus: &[DDLu], lus_t: &[DDLu]) -> State<T, B> {
    let mut omega = T::splat(DD::TWO_PI) / delta;
    if T::TAN_F64 && ctx.mutate == 2 {
        omega = omega.mulf(1.0 + 1e-9);
    }
    let k = ctx.k();
    if b == 0 {
        centre_state(ctx, &inp[0], &inp[1], omega)
    } else if b < k {
        let mut w = TWork::<T, B>::new(ctx.kt);
        taylor_path(ctx, &[inp[0], inp[1], inp[2], inp[3]], &ctx.segs[b - 1], omega, &mut w)
    } else {
        horizon_state(ctx, &inp[0], &inp[1], omega, lus, lus_t)
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
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par, DD::ZERO)?;
    let one_block = |b: usize| -> State<DV, B> {
        let inp: Vec<Fl<DV, B>> = ctx.block_inputs(b).iter().map(|&o| ctx.fld(zh, zl, o)).collect();
        block_state::<DV, B>(ctx, b, &inp, DV::splat(delta), &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<DV, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let mut fh = vec![0.0; l];
    let mut fl = vec![0.0; l];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let mut d = s[q] - ctx.fld(zh, zl, t + q * m);
            if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                d = lo.apply(&d) + hi.apply(&ctx.fld(zh, zl, 0));
            }
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
    jacobian_lanes::<Du, B>(ctx, zh, zl, par)
}

/// The Jacobian in plain f64 (the same columns with `Fu` lanes: constants and the base state rounded to f64, the
/// horizon solves with the hi parts of the double-double LU factors), column-major with leading dimension
/// ld_for(L): (J, ld). For the mixed-precision Newton step (double-double residual, f64 Jacobian).
pub fn jacobian_f64<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    let (jh, _, ld) = jacobian_lanes::<Fu, B>(ctx, zh, zl, par)?;
    Ok((jh, ld))
}

/// The f64 Jacobian with `Dtn<N>` lanes: a double-double base trajectory and N f64 tangents per evaluation (N
/// columns of one block at a time), column-major with leading dimension ld_for(L): (J, ld).
pub fn jacobian_dtn<const N: usize, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    let (jh, _, ld) = jacobian_lanes::<Dtn<N>, B>(ctx, zh, zl, par)?;
    Ok((jh, ld))
}

/// The f64 Jacobian with `Fun<N>` lanes: an f64 base trajectory and N f64 tangents per evaluation: (J, ld).
pub fn jacobian_fun<const N: usize, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    let (jh, _, ld) = jacobian_lanes::<Fun<N>, B>(ctx, zh, zl, par)?;
    Ok((jh, ld))
}

/// Several column tasks of one block from one evaluation with T::NT f64 tangents (TAN_F64 lanes): tangent k carries
/// task g[k]; unused tangents stay zero. Returns the columns in the order of g.
pub fn jac_columns_n<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    zh: &[f64],
    zl: &[f64],
    delta: DD,
    lus: &[DDLu],
    lus_t: &[DDLu],
    g: &[(usize, usize, usize)],
) -> Vec<Vec<DD>> {
    assert!(T::TAN_F64 && !g.is_empty() && g.len() <= T::NT && g.iter().all(|t| t.0 == g[0].0));
    let m = Ctx::<B>::M;
    let b = g[0].0;
    let z = Fl::<DV, B>::zero();
    let inp: Vec<Fl<T, B>> = ctx
        .block_inputs(b)
        .iter()
        .enumerate()
        .map(|(fi, &o)| {
            let mut x = Fl::<T, B>::from_vt(&ctx.fld(zh, zl, o), &z);
            for (k, &(_, f, j)) in g.iter().enumerate() {
                if f == fi {
                    let mut e = [0.0; 8];
                    e[j % 8] = 1.0;
                    x.0[j / 8].set_tan64(k, crate::v8::V8::from_array(e));
                }
            }
            x
        })
        .collect();
    let mut d = T::from_vt(DV::splat(delta), DV::zero());
    for (k, t) in g.iter().enumerate() {
        if t.1 == usize::MAX {
            d.set_tan64(k, crate::v8::V8::splat(1.0));
        }
    }
    let s = block_state::<T, B>(ctx, b, &inp, d, lus, lus_t);
    (0..g.len())
        .map(|k| {
            let mut out = Vec::with_capacity(4 * m);
            for q in 0..4 {
                let mut t = Fl::<DV, B>::zero();
                for bb in 0..B {
                    t.0[bb].hi = s[q].0[bb].tan64(k);
                }
                if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
                    t = lo.apply(&t);
                }
                for i in 0..m {
                    out.push(t.get(i));
                }
            }
            out
        })
        .collect()
}

/// a[idx] = f(a[idx]) on a (hi, lo) matrix; an empty lo means lo parts zero and not kept
#[inline(always)]
fn upd(jh: &mut [f64], jl: &mut [f64], idx: usize, f: impl Fn(DD) -> DD) {
    let keep = !jl.is_empty();
    let v = f(DD { hi: jh[idx], lo: if keep { jl[idx] } else { 0.0 } });
    jh[idx] = v.hi;
    if keep {
        jl[idx] = v.lo;
    }
}

/// The Jacobian's column tasks: (block, input field index or usize::MAX for Delta, lane).
pub fn jac_tasks<const B: usize>(ctx: &Ctx<B>) -> Vec<(usize, usize, usize)> {
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for b in 0..ctx.k() + 1 {
        let ins = ctx.block_inputs(b);
        for f in 0..ins.len() {
            for j in 0..Ctx::<B>::M {
                tasks.push((b, f, j));
            }
        }
        tasks.push((b, usize::MAX, 0));
    }
    tasks
}

/// One column task: block b's output state (4M values, before the -1 of the target node) differentiated along a
/// unit tangent on lane j of input field f (Delta for f = usize::MAX), with dual lanes T.
pub fn jac_column<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    zh: &[f64],
    zl: &[f64],
    delta: DD,
    lus: &[DDLu],
    lus_t: &[DDLu],
    (b, f, j): (usize, usize, usize),
) -> Vec<DD> {
    let m = Ctx::<B>::M;
    let ins = ctx.block_inputs(b);
    let inp: Vec<Fl<T, B>> = ins
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
            Fl::<T, B>::from_vt(&v, &t)
        })
        .collect();
    let dt = if f == usize::MAX { DV::splat(DD::ONE) } else { DV::zero() };
    let d = T::from_vt(DV::splat(delta), dt);
    let s = block_state::<T, B>(ctx, b, &inp, d, lus, lus_t);
    let mut out = Vec::with_capacity(4 * m);
    for q in 0..4 {
        let mut t = s[q].tan();
        if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
            t = lo.apply(&t);
        }
        for i in 0..m {
            out.push(t.get(i));
        }
    }
    out
}

/// Jacobian columns by forward-mode duals of the dual lane type T (one evaluation per column, a unit tangent on one
/// unknown), assembled column-major with leading dimension ld_for(L): (hi, lo, ld). Du: the exact double-double
/// Jacobian. Fu: f64; lo is then empty.
fn jacobian_lanes<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let delta = delta_of(zh, zl);
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par, DD::ZERO)?;
    // the shifted LUs only for a mode matrix (lam != 0); otherwise the same factors, not a copy (at M = 384 and
    // ks = 250 one set is 0.6 GB)
    let lus_shift = if ctx.lam.hi != 0.0 { Some(horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par, ctx.lam)?) } else { None };
    let lus_t: &[DDLu] = lus_shift.as_deref().unwrap_or(&lus);
    let nb = ctx.k() + 1;
    let tasks = jac_tasks(ctx);
    let cols: Vec<Vec<DD>> = if T::TAN_F64 && ctx.f64_dd & (HDD_BLOCK | HDD_LASTSEG | HDD_DTN) != 0 {
        // the f64 Jacobian with some blocks' tasks on other lanes (Ctx::f64_dd): Du (double-double, rounded to f64
        // when stored below) or Dtn<4>; the other tasks as in the branches below
        let k = ctx.k();
        let du = |b: usize| (ctx.f64_dd & HDD_BLOCK != 0 && b == k) || (ctx.f64_dd & HDD_LASTSEG != 0 && k >= 2 && b == k - 1);
        let dtn = |b: usize| ctx.f64_dd & HDD_DTN != 0 && b == k && !du(b);
        // (lane kind, tasks): 0 T in groups of NT (one task per unit when NT = 1), 1 Du, 2 Dtn<4>
        let mut units: Vec<(u8, Vec<(usize, usize, usize)>)> = Vec::new();
        for &t in &tasks {
            let (kind, cap) = if du(t.0) { (1u8, 1) } else if dtn(t.0) { (2, 4) } else { (0, T::NT.max(1)) };
            match units.last_mut() {
                Some((kd, g)) if *kd == kind && g.len() < cap && g[0].0 == t.0 => g.push(t),
                _ => units.push((kind, vec![t])),
            }
        }
        let unit = |u: &(u8, Vec<(usize, usize, usize)>)| -> Vec<Vec<DD>> {
            match u.0 {
                1 => vec![jac_column::<Du, B>(ctx, zh, zl, delta, &lus, lus_t, u.1[0])],
                2 => jac_columns_n::<Dtn<4>, B>(ctx, zh, zl, delta, &lus, lus_t, &u.1),
                _ if T::NT > 1 => jac_columns_n::<T, B>(ctx, zh, zl, delta, &lus, lus_t, &u.1),
                _ => vec![jac_column::<T, B>(ctx, zh, zl, delta, &lus, lus_t, u.1[0])],
            }
        };
        let gc: Vec<Vec<Vec<DD>>> = if par { units.par_iter().map(unit).collect() } else { units.iter().map(unit).collect() };
        gc.into_iter().flatten().collect()
    } else if T::TAN_F64 && T::NT > 1 {
        // several tangents per evaluation: the tasks of each block in groups of NT, in task order
        let mut groups: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        for &t in &tasks {
            match groups.last_mut() {
                Some(g) if g.len() < T::NT && g[0].0 == t.0 => g.push(t),
                _ => groups.push(vec![t]),
            }
        }
        let group = |g: &Vec<(usize, usize, usize)>| jac_columns_n::<T, B>(ctx, zh, zl, delta, &lus, lus_t, g);
        let gc: Vec<Vec<Vec<DD>>> = if par { groups.par_iter().map(group).collect() } else { groups.iter().map(group).collect() };
        gc.into_iter().flatten().collect()
    } else {
        let column = |t: &(usize, usize, usize)| -> Vec<DD> { jac_column::<T, B>(ctx, zh, zl, delta, &lus, lus_t, *t) };
        if par {
            tasks.par_iter().map(column).collect()
        } else {
            tasks.iter().map(column).collect()
        }
    };
    let keep_lo = !T::TAN_F64;
    let mut jh = vec![0.0; ld * l];
    let mut jl = if keep_lo { vec![0.0; ld * l] } else { Vec::new() };
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let c = if f == usize::MAX { l - 1 } else { ctx.block_inputs(b)[f] + j };
        for (r, v) in col.iter().enumerate() {
            jh[c * ld + b * 4 * m + r] = v.hi;
            if keep_lo {
                jl[c * ld + b * 4 * m + r] = v.lo;
            }
        }
    }
    for b in 0..nb {
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            if let (true, Some((lo, hi))) = (b == ctx.k() && r < m, &ctx.freeze) {
                // p rows of the horizon block: -Pi_low on the target node's p, +Pi_high on p1
                for j in 0..m {
                    upd(&mut jh, &mut jl, (t + j) * ld + b * 4 * m + r, |v| v - lo.cols[j].get(r));
                    upd(&mut jh, &mut jl, j * ld + b * 4 * m + r, |v| v + hi.cols[j].get(r));
                }
                continue;
            }
            upd(&mut jh, &mut jl, (t + r) * ld + b * 4 * m + r, |v| v - DD::ONE);
        }
    }
    for i in 0..m {
        let w = ctx.phase.get(i);
        jh[i * ld + l - 1] = w.hi;
        if keep_lo {
            jl[i * ld + l - 1] = w.lo;
        }
    }
    Ok((jh, jl, ld))
}

/// Deliberate breakage of `jvp`, for the controls of hp/largeD_mixed_check.py: the Delta tangent scaled by
/// 1 + 1e-20 (a subtle error in one column's contribution)
pub const JVP_MUT_DELTA: u8 = 1;
/// `jvp` control: the horizon block's state tangent dropped
pub const JVP_MUT_HORIZON: u8 = 2;
/// `jvp` control: the phase row's product dropped
pub const JVP_MUT_PHASE: u8 = 3;

/// Exact Jacobian-vector product J v in double-double: the tangent of the residual along v, by one forward-mode
/// pass per block (`Du` lanes, the lanes of the double-double Jacobian) with the block's inputs and Delta carrying
/// v's entries as their tangent. It equals the double-double Jacobian times v to double-double rounding, for a cost of
/// about K + 1 Jacobian columns. Returns (Jv hi, Jv lo, F hi, F lo): the residual comes with it. mutate: JVP_MUT_*
/// (controls; 0 none).
#[allow(clippy::type_complexity)]
pub fn jvp<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], vh: &[f64], vl: &[f64], par: bool, mutate: u8) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let l = ctx.len();
    let m = Ctx::<B>::M;
    if zh.len() != l || zl.len() != l || vh.len() != l || vl.len() != l {
        return Err(format!("z and v must have length {l}"));
    }
    if ctx.lam.hi != 0.0 {
        return Err("jvp: no mode-matrix shift (lam) supported".into());
    }
    let delta = delta_of(zh, zl);
    let mut dv = delta_of(vh, vl);
    if mutate == JVP_MUT_DELTA {
        dv = dv + dv * DD::from_f64(1e-20);
    }
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par, DD::ZERO)?;
    let one_block = |b: usize| -> State<Du, B> {
        let inp: Vec<Fl<Du, B>> = ctx.block_inputs(b).iter().map(|&o| Fl::from_vt(&ctx.fld(zh, zl, o), &ctx.fld(vh, vl, o))).collect();
        block_state::<Du, B>(ctx, b, &inp, Du::from_vt(DV::splat(delta), DV::splat(dv)), &lus, &lus)
    };
    let nb = ctx.k() + 1;
    // the blocks are few (K + 1) and unequal; with par, each runs on its own task
    let states: Vec<State<Du, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let (mut jh, mut jl) = (vec![0.0; l], vec![0.0; l]);
    let (mut fh, mut fl) = (vec![0.0; l], vec![0.0; l]);
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let mut d = s[q].tan() - ctx.fld(vh, vl, t + q * m);
            let mut f = s[q].val() - ctx.fld(zh, zl, t + q * m);
            if mutate == JVP_MUT_HORIZON && b == kb {
                d = -ctx.fld(vh, vl, t + q * m);
            }
            if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                d = lo.apply(&d) + hi.apply(&ctx.fld(vh, vl, 0));
                f = lo.apply(&f) + hi.apply(&ctx.fld(zh, zl, 0));
            }
            let o = b * 4 * m + q * m;
            d.store(&mut jh[o..o + m], &mut jl[o..o + m]);
            f.store(&mut fh[o..o + m], &mut fl[o..o + m]);
        }
    }
    let ph = if mutate == JVP_MUT_PHASE { DD::ZERO } else { phase_row(ctx, &ctx.fld(vh, vl, 0)) };
    jh[l - 1] = ph.hi;
    jl[l - 1] = ph.lo;
    let pf = phase_row(ctx, &ctx.fld(zh, zl, 0));
    fh[l - 1] = pf.hi;
    fl[l - 1] = pf.lo;
    Ok((jh, jl, fh, fl))
}

/// The Newton system of the mixed-precision step in block storage: the f64 Jacobian (TAN_F64 lanes T: `Fu`,
/// `Fun<N>` or `Dtn<N>`), built block by block and factored by the staircase LU (`block::BlockLu`) without forming
/// the dense matrix. Each residual block's columns are computed (in parallel when par) by the same functions as in
/// `jacobian_lanes`, so its entries are those of the dense f64 Jacobian; then the block's step is factored. Not for
/// the frozen-harmonics layout (Ctx::freeze couples the horizon block to p1) or a mode matrix (lam != 0).
/// mutate: block::MUT_* (controls).
pub fn jacobian_block<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool, mutate: u8) -> Result<crate::block::BlockLu, String> {
    use crate::block::{BlockLu, Group};
    if ctx.freeze.is_some() {
        return Err("block storage does not support the frozen-harmonics layout (kfreeze)".into());
    }
    if ctx.lam.hi != 0.0 || !T::TAN_F64 {
        return Err("block storage is for the f64 Jacobian only".into());
    }
    let m = Ctx::<B>::M;
    let k = ctx.k();
    let l = ctx.len();
    let delta = delta_of(zh, zl);
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, par, DD::ZERO)?;
    let mut groups = vec![Group { delta: false, zoff: 0, zlen: 2 * m }];
    for t in 0..k {
        groups.push(Group { delta: true, zoff: ctx.off_node(t), zlen: 4 * m });
    }
    groups.push(Group { delta: true, zoff: ctx.off_mh(), zlen: 2 * m });
    let rblocks: Vec<(usize, usize)> = (0..=k).map(|b| (b * 4 * m, 4 * m)).collect();
    let tasks = jac_tasks(ctx);
    let gs = groups.clone();
    let fill = |b: usize, w: &mut [f64], ld: usize, r0: usize| -> Result<(), String> {
        let (p, q) = (gs[b], gs[b + 1]);
        let np = p.len();
        // column of z index zi within this step (panel group p, right group q; Delta -> q's copy)
        let pos = |zi: usize| -> Result<usize, String> {
            if zi == l - 1 {
                Ok(np)
            } else if zi >= p.zoff && zi < p.zoff + p.zlen {
                Ok(p.delta as usize + zi - p.zoff)
            } else if zi >= q.zoff && zi < q.zoff + q.zlen {
                Ok(np + q.delta as usize + zi - q.zoff)
            } else {
                Err(format!("block {b}: unknown {zi} outside its two groups"))
            }
        };
        let bt: Vec<(usize, usize, usize)> = tasks.iter().filter(|t| t.0 == b).cloned().collect();
        let cols: Vec<Vec<DD>> = if T::NT > 1 {
            let groups_b: Vec<&[(usize, usize, usize)]> = bt.chunks(T::NT).collect();
            let one = |g: &&[(usize, usize, usize)]| jac_columns_n::<T, B>(ctx, zh, zl, delta, &lus, &lus, g);
            let gc: Vec<Vec<Vec<DD>>> = if par { groups_b.par_iter().map(one).collect() } else { groups_b.iter().map(one).collect() };
            gc.into_iter().flatten().collect()
        } else {
            let one = |t: &(usize, usize, usize)| jac_column::<T, B>(ctx, zh, zl, delta, &lus, &lus, *t);
            if par {
                bt.par_iter().map(one).collect()
            } else {
                bt.iter().map(one).collect()
            }
        };
        let ins = ctx.block_inputs(b);
        for (&(_, f, j), col) in bt.iter().zip(&cols) {
            let c = if f == usize::MAX { np } else { pos(ins[f] + j)? };
            for (r, v) in col.iter().enumerate() {
                w[c * ld + r0 + r] = v.hi;
            }
        }
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            let c = pos(t + r)?;
            let o = c * ld + r0 + r;
            w[o] = (DD { hi: w[o], lo: 0.0 } - DD::ONE).hi;
        }
        if b == 0 {
            for i in 0..m {
                w[pos(i)? * ld + r0 - 1] = ctx.phase.get(i).hi;
            }
        }
        Ok(())
    };
    BlockLu::factor(l, groups, &rblocks, l - 1, l - 1, fill, par, mutate)
}

/// A solver context read from a file written by hp/dump_ctx.py (Rust benches and data-driven tests).
#[cfg(test)]
pub(crate) mod ctxfile {
    use super::*;

    pub fn read(path: &str) -> Vec<f64> {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }

    pub fn m_of(v: &[f64]) -> usize {
        v[0] as usize
    }

    /// (ctx, zh, zl)
    pub fn load<const B: usize>(v: &[f64]) -> (Ctx<B>, Vec<f64>, Vec<f64>) {
        let m = v[0] as usize;
        assert_eq!(m, 8 * B);
        let n = DD::new(v[1], v[2]);
        let (kt, kc, ks, k) = (v[3] as usize, v[4] as usize, v[5] as usize, v[6] as usize);
        let mut i = 7;
        let nodes = v[i..i + k].to_vec();
        i += k;
        let mut segs = Vec::new();
        for _ in 0..k - 1 {
            let len = v[i] as usize;
            segs.push(v[i + 1..i + 1 + len].to_vec());
            i += 1 + len;
        }
        let d = DMat::<B>::from_rows(&v[i..i + m * m], &v[i + m * m..i + 2 * m * m]);
        i += 2 * m * m;
        let phase = Fl::<DV, B>::load(&v[i..i + m], &v[i + m..i + 2 * m]);
        i += 2 * m;
        let l = v[i] as usize;
        let zh = v[i + 1..i + 1 + l].to_vec();
        let zl = v[i + 1 + l..i + 1 + 2 * l].to_vec();
        let ctx = Ctx::<B> { n, d, phase, nodes, segs, kt, kc, ks, freeze: None, match_node: k - 1, lam: DD::ZERO, mutate: 0, f64_dd: 0 };
        assert_eq!(ctx.len(), l);
        (ctx, zh, zl)
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::field::cs;
    use std::time::Instant;

    /// seconds per column for the sample tasks (one block), one evaluation per column (T::NT <= 1) or groups of
    /// T::NT columns per evaluation (the sample is then extended to whole groups of consecutive lanes)
    fn time_cols<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], lus: &[DDLu], tasks: &[(usize, usize, usize)]) -> f64 {
        let delta = delta_of(zh, zl);
        let mut s = 0.0;
        let t0 = Instant::now();
        let n = if T::TAN_F64 && T::NT > 1 {
            let mut n = 0;
            for g in tasks.chunks(T::NT) {
                s += jac_columns_n::<T, B>(ctx, zh, zl, delta, lus, lus, g)[0][3].hi;
                n += g.len();
            }
            n
        } else {
            for &t in tasks {
                s += jac_column::<T, B>(ctx, zh, zl, delta, lus, lus, t)[3].hi;
            }
            tasks.len()
        };
        assert!(s.is_finite());
        t0.elapsed().as_secs_f64() / n as f64
    }

    fn prim<T: Lane, const B: usize>(ctx: &Ctx<B>, name: &str) {
        let mut f = Fl::<T, B>::zero();
        for b in 0..B {
            f.0[b] = T::splat(DD::from_f64(0.1 + b as f64 * 0.01));
        }
        let reps = 20000;
        let t0 = Instant::now();
        for _ in 0..reps {
            f = ctx.d.apply(&f).mulf(0.01);
        }
        let t_apply = t0.elapsed().as_secs_f64() / reps as f64;
        let xs: Vec<Fl<T, B>> = (0..40).map(|k| f.mulf(1.0 / (k + 1) as f64)).collect();
        let t0 = Instant::now();
        let mut acc = Fl::<T, B>::zero();
        for _ in 0..reps / 10 {
            for k in 0..36 {
                acc = acc + cs(&xs, &xs, k);
            }
        }
        let t_cs = t0.elapsed().as_secs_f64() / (reps / 10 * 36) as f64;
        let per = T::NT.max(1) as f64;
        println!(
            "  {name:6}: DMat::apply {:.2} us, cs (k = 0..35, mean) {:.3} us; per tangent {:.2} / {:.3} us   [{:.1e}]",
            t_apply * 1e6,
            t_cs * 1e6,
            t_apply * 1e6 / per,
            t_cs * 1e6 / per,
            acc.val().amax()
        );
    }

    fn run<const B: usize>(v: &[f64]) {
        let (ctx, zh, zl) = ctxfile::load::<B>(v);
        let delta = delta_of(&zh, &zl);
        let lus = horizon_lus(&ctx, &ctx.fld(&zh, &zl, ctx.off_ah()), delta, false, DD::ZERO).unwrap();
        println!("M = {}, L = {}, single thread; seconds for all columns of a block (from a sample)", 8 * B, ctx.len());
        prim::<Du, B>(&ctx, "Du");
        prim::<Fu, B>(&ctx, "Fu");
        prim::<Fun<4>, B>(&ctx, "Fun<4>");
        prim::<Dtn<1>, B>(&ctx, "Dtn<1>");
        prim::<Dtn<4>, B>(&ctx, "Dtn<4>");
        prim::<Dtn<8>, B>(&ctx, "Dtn<8>");
        let all = jac_tasks(&ctx);
        let nb = ctx.k() + 1;
        let names = ["Du", "Fu", "Fun<2>", "Fun<4>", "Fun<8>", "Dtn<1>", "Dtn<4>", "Dtn<8>"];
        let mut tot = [0.0; 8];
        for b in 0..nb {
            let ncol = all.iter().filter(|t| t.0 == b).count() as f64;
            // 16 consecutive lanes of the second input field (each evaluation group within one field)
            let sample: Vec<(usize, usize, usize)> = all.iter().filter(|t| t.0 == b && t.1 == 1 && t.2 < 16).cloned().collect();
            let ts = [
                time_cols::<Du, B>(&ctx, &zh, &zl, &lus, &sample[..4]),
                time_cols::<Fu, B>(&ctx, &zh, &zl, &lus, &sample[..8]),
                time_cols::<Fun<2>, B>(&ctx, &zh, &zl, &lus, &sample[..8]),
                time_cols::<Fun<4>, B>(&ctx, &zh, &zl, &lus, &sample[..8]),
                time_cols::<Fun<8>, B>(&ctx, &zh, &zl, &lus, &sample),
                time_cols::<Dtn<1>, B>(&ctx, &zh, &zl, &lus, &sample[..4]),
                time_cols::<Dtn<4>, B>(&ctx, &zh, &zl, &lus, &sample[..8]),
                time_cols::<Dtn<8>, B>(&ctx, &zh, &zl, &lus, &sample),
            ];
            let steps = if b > 0 && b < ctx.k() { ctx.segs[b - 1].len() - 1 } else { 0 };
            let mut line = format!("  block {b:2} ({ncol} columns, {steps:2} steps):");
            for i in 0..8 {
                tot[i] += ts[i] * ncol;
                line += &format!(" {} {:6.3}", names[i], ts[i] * ncol);
            }
            println!("{line}");
        }
        let mut line = String::from("  total (one thread):");
        for i in 0..8 {
            line += &format!(" {} {:.2} s ({:.1}x)", names[i], tot[i], tot[0] / tot[i]);
        }
        println!("{line}");
    }

    /// cargo test --release --lib bench_jac -- --ignored --nocapture   (env FILE from hp/dump_ctx.py)
    #[test]
    #[ignore]
    fn bench_jac() {
        let v = ctxfile::read(&std::env::var("FILE").expect("FILE"));
        match ctxfile::m_of(&v) / 8 {
            6 => run::<6>(&v),
            8 => run::<8>(&v),
            10 => run::<10>(&v),
            b => panic!("B = {b} not built into the bench"),
        }
    }
}
