//! The DSS critical-solution equations in double-double: centre series, Taylor stepping in x, horizon series, as
//! multishoot's `Equations` (`Dss`), and the kernel's entry points on multishoot's engine (`Shooting`).
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
//!
//! This is multishoot's layout with NF = 4 fields, NL = 2 (p1, c0) and NR = 2 (m_h, a_h) endpoint fields, one global
//! (Delta) and one extra row (the phase row): `Ctx::shooting` gives the engine's problem, whose residual, exact
//! Jacobians (double-double, quad-, octo-double, f64 lanes, staircase storage), jets, Jacobian-vector product and mode
//! matrix serve the kernel. `Dss` supplies the equations: the Taylor recursion, the centre and horizon series as the
//! endpoint maps, and the horizon LUs as the preparation of an evaluation.
use crate::dd::DD;
use crate::dv::DV;
use crate::field::{cs, cs_range, cs_w, horner, DMat, Fl, State, TangentTail};
use crate::lane::{Dc, Fu, Fun, Lane, F8};
use crate::linalg::{ld_for, DDLu, StaircaseLu};
use crate::qv::QV;
use multidouble::dual::Param;
use multidouble::native::{Dtn, OV};
use multidouble::precision::{Double, Precision};
use multidouble::simd::Native;
use multishoot::jacobian::Seed;
use multishoot::control::Control;
use multishoot::periodic::Parity;
use multishoot::{Equations, ExtraRow, ProjectedRows, Rate, Shooting, Work};
use crate::field::FieldSolve;
use rayon::prelude::*;
use std::any::Any;

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
    /// below 1e-12 per unit change. The rows are multishoot's projected rows (`Shooting::project_rows`, the cutoff
    /// read from Pi_low by `kfreeze_of`), whose projector is formed in octo-double and read in each solve's
    /// precision; these double-double matrices give the cutoff and serve hp/modecount_cx.
    pub freeze: Option<(DMat<B>, DMat<B>)>,
    /// Index of the node where the outward integration from the centre meets the inward integration from the
    /// horizon side. Segments b <= match_node run outward (node b-1 -> node b); segments b > match_node run inward
    /// (node b -> node b-1). nodes.len() - 1 (the default) is the all-outward layout.
    pub match_node: usize,
    /// Growth rate of a mode matrix in the layout of hp/modecount_cx, which reads it; hp's own mode matrix passes its
    /// rate to the engine (`mode_matrix`, multishoot's `Rate`), and hp's equations take the rate from the engine
    /// (`Work::rate`, `HPrep::rate`), so this stays zero here.
    pub lam: DD,
    /// Deliberate breakage of the f64 lanes only (`jacobian_f64`), for the control of the Jacobian-agreement check
    /// (hp/mixed_check.py --mutate): 0 none, 1 no tau derivative, 2 omega scaled by 1 + 1e-9, 3 the horizon solves'
    /// kappa tangent term dropped; 6 the centre series' scale wrong at one order (twice sigma for the order-7 terms
    /// built from order 6), the control of the scaled series' bit-identity (hp/highm_check.py scaled). The
    /// double-double lanes never read it. The quad-double lanes read QD_MUT_NOREFINE and QD_MUT_REFINE4 only (and
    /// `residual_in` QD_MUT_STATE_DD), and the octo-double lanes the same two (and `residual_in` OD_MUT_STATE_QD).
    pub mutate: u8,
    /// Parts of the f64 Jacobian (TAN_F64 lanes) computed in double-double instead, rounded to f64; bit flags HDD_*
    /// (notes/LARGE_D_MIXED.md). 0, the default, changes nothing. The double-double lanes never read it.
    pub f64_dd: u8,
    /// How the horizon series and `horizon_lus` handle a = 1 + alpha (HALPHA_*): 0, the default, as before (in a);
    /// HALPHA_ON carries alpha = a - 1 and writes a^2 - 1 and a - a^3 in alpha, where they no longer cancel;
    /// HALPHA_MUT, the control, carries alpha but forms them from 1 + alpha as before (the cancellation put back).
    /// All lanes read it.
    pub horizon_alpha: u8,
}

/// `Ctx::f64_dd`: the horizon series' periodic solves (`psolve`) in double-double for the f64 lanes, value and
/// tangents, with the full double-double LU factors instead of their hi parts
pub const HDD_SOLVE: u8 = 1;
/// `Ctx::mutate` for the quad-double and octo-double lanes (the control of their precision checks): the horizon
/// solves without their refinement, so only to the double-double LU's accuracy
pub const QD_MUT_NOREFINE: u8 = 7;
/// `Ctx::mutate` for the quad-double lanes: four refinement steps, the reference of the refinement's convergence check;
/// for the octo-double lanes eight steps, twice that, as their refinement takes twice the steps
pub const QD_MUT_REFINE4: u8 = 8;
/// refinement steps of a horizon solve in the quad-double lanes (VAL_QD): the double-double LU solves to about eps_dd times the
/// condition number (pivots 1e-3..1e2), so two steps reach quad-double; a tangent lane takes one more, as its first
/// solve ignores the derivative of the system
const QD_REFINE: usize = 2;
/// refinement steps of a horizon solve in the octo-double lanes (VAL_OD): twice QD_REFINE, as each step gains about
/// cond 2^-104 and octo-double needs twice quad-double's bits (multidouble's `Lane::VAL_OD`); a tangent lane (`Ou`)
/// takes one more, as for QD_REFINE
const OD_REFINE: usize = 4;
/// `Ctx::f64_dd`: no early end of the horizon series for f64 tangents (`TangentTail`); the series runs to ks
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
/// tangent terms are below double-double resolution (`TangentTail` with TAIL_TOL_DD), instead of running to kc and ks. The
/// scan's kc = 420 at large D is far past that (the centre series' terms fall to 1e-134 at D = 50), and the centre
/// series is most of a product's cost there.
pub const HDD_DU_CUT: u8 = 32;

/// `Ctx::horizon_alpha`: the horizon series carries alpha = a - 1 (a(1) = a_h = 1 + O(1e-3) at large D) and forms
/// a^2 - 1 = 2 alpha + alpha^2 and a - a^3 = -(2 alpha + 3 alpha^2 + alpha^3) from it; `horizon_lus` likewise
/// c1 = (n - 1)(1 - a_h^2) = -(n - 1)(2 alpha_h + alpha_h^2). In a, both differences lose about -log10(alpha_h) digits
/// to cancellation, and the horizon series' rounding errors excite the mass mode, which grows by G = x_m^-(n-1)
/// between the horizon and the last node x_m (7e8 at D = 60): the large-D residual floor (notes/LARGE_D_MIXED.md
/// section 10). alpha_h = a_h - 1 is exact, so the unknowns and the solution are unchanged; only the rounding in the
/// series differs.
pub const HALPHA_ON: u8 = 1;
/// `Ctx::horizon_alpha`: the control of HALPHA_ON. alpha is carried as with HALPHA_ON, but a^2 - 1 and a - a^3 are
/// formed from a = 1 + alpha as without it (the cancellation put back); the results are those of 0 bit for bit.
pub const HALPHA_MUT: u8 = 2;

/// the scratch series of the Taylor recursion (`Work::aux`): a2, q2, g, h, pth, mth, ww, vv
const N_AUX: usize = 8;

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
    /// The engine's problem on this context: `Dss` on the nodes, segments, Taylor order and matching node, with the
    /// phase row as its one extra row. Its layout is this context's (multishoot's `Shooting` with NL = 2, NF = 4,
    /// NR = 2 and the global Delta last).
    pub fn shooting(&self) -> Shooting<Dss<'_, B>, B> {
        let mut s = Shooting::new(Dss(self), self.nodes.clone(), self.kt);
        s.segs = self.segs.clone();
        s.match_node = self.match_node;
        s.extra = vec![ExtraRow { terms: (0..Self::M).map(|i| (i, self.phase.get(i))).collect(), rhs: DD::ZERO }];
        if let Some((lo, _)) = &self.freeze {
            // the horizon block's p rows, p1 pinned to the high harmonics
            s.project_rows(self.k(), 0, 0, Parity::Odd, kfreeze_of(lo)).expect("the horizon block's p rows");
        }
        s
    }
}

/// The cutoff k_max of Pi_low, the projector onto the odd harmonics k <= k_max of a field on the M points of the half
/// grid: each harmonic adds 2 / M to every diagonal entry, so the trace is twice their number and
/// k_max = 2 round(tr / 2) - 1.
pub fn kfreeze_of<const B: usize>(lo: &DMat<B>) -> usize {
    let tr: f64 = (0..Ctx::<B>::M).map(|i| lo.cols[i].get(i).hi).sum();
    (2 * (tr / 2.0).round() as usize).max(1) - 1
}

/// The cutoff of Pi_low after checking that it is multishoot's projector onto the odd harmonics k <= k_max
/// (`ProjectedRows`, formed in octo-double) to double-double rounding, 1e-30 per entry: the frozen rows use that
/// projector, not this matrix.
pub fn check_freeze<const B: usize>(lo: &DMat<B>) -> Result<usize, String> {
    let m = Ctx::<B>::M;
    let kf = kfreeze_of(lo);
    let p = ProjectedRows::new(m, 0, 0, 0, Parity::Odd, kf);
    let worst = (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| (lo.cols[j].get(i) - p.entry::<Double>(true, i, j, false)).hi.abs())
        .fold(0.0, f64::max);
    if worst > 1e-30 || worst.is_nan() {
        return Err(format!("plow is not the projector onto the odd harmonics k <= {kf} (largest difference {worst:.1e})"));
    }
    Ok(kf)
}

#[inline(always)]
fn recip_int(k: usize) -> DD {
    DD::ONE / DD::from_f64(k as f64)
}

/// d/dtau = omega D, its tangents shifted by the growth rate of a mode-matrix evaluation (multishoot's `Rate::shift`:
/// perturbations ~ exp(lambda tau) gain lambda times themselves; values are unaffected; zero rate: unchanged)
#[inline(always)]
fn dtau<T: Lane, const B: usize>(ctx: &Ctx<B>, x: &Fl<T, B>, omega: T, rate: &Rate) -> Fl<T, B> {
    if T::TAN_F64 && ctx.mutate == 1 {
        return Fl::zero();
    }
    rate.shift(x, ctx.d.apply(x).mul_t(omega))
}

/// omega = 2 pi / Delta in the lane, from the global Delta (with the f64 lanes' control ctx.mutate 2, scaled by
/// 1 + 1e-9)
#[inline(always)]
fn omega_of<T: Lane, const B: usize>(ctx: &Ctx<B>, delta: T) -> T {
    let omega = T::splat(DD::TWO_PI) / delta;
    if T::TAN_F64 && ctx.mutate == 2 {
        omega.mulf(1.0 + 1e-9)
    } else {
        omega
    }
}

/// Early end of the centre and horizon series for the f64-tangent lanes (a Jacobian in f64 needs their terms only
/// to the f64 rounding level, and the series orders kc, ks are set for double-double values): multidouble's
/// `TangentTail`, which stops a series after order k once the tangent terms max |c_k| r^k (r the evaluation radius,
/// over the four fields, all lanes and tangents) of TAIL_RUN consecutive orders are all below TAIL_TOL times the
/// largest tangent term so far, k >= TAIL_MIN. The dropped tail is then below the rounding of an f64 sum (checked
/// against the double-double Jacobian by hp/mixed_check.py). The double-double lanes never stop early, unless
/// HDD_DU_CUT asks (TAIL_TOL_DD).
const TAIL_TOL: f64 = 8.673617379884035e-19; // 2^-60
/// the cut of HDD_DU_CUT (double-double tangents)
const TAIL_TOL_DD: f64 = 7.703719777548943e-34; // 2^-110
const TAIL_RUN: usize = 4;
const TAIL_MIN: usize = 8;

/// whether the series of lane type T stop early under ctx: f64 tangents always (TAIL_TOL), double-double tangents
/// with HDD_DU_CUT (TAIL_TOL_DD)
fn tail_active<T: Lane, const B: usize>(ctx: &Ctx<B>) -> bool {
    T::TAN_F64 || (T::TANGENT && ctx.f64_dd & HDD_DU_CUT != 0)
}

/// the tail monitor of lane type T for a series evaluated at radius r (in the scaled variable)
fn tail_of<T: Lane>(r: f64) -> TangentTail {
    TangentTail::new(r, if T::TAN_F64 { TAIL_TOL } else { TAIL_TOL_DD }, TAIL_RUN, TAIL_MIN)
}

// ---------------------------------------------------------------------------------------------------------------
// centre series (regular centre), evaluated at nodes[0]
// ---------------------------------------------------------------------------------------------------------------
/// The power of two nearest to x (x > 0; multidouble's `pow2_near`): the series below are computed in the scaled
/// variable x / series_scale(x0)
/// (u / series_scale(u0) at the horizon), so that their coefficients c_k sigma^k stay near the evaluation radius's
/// scale instead of growing like r^-k, which overflows the f64 exponent range (and the f64 Jacobian's tangents first)
/// when r is small and the order high (r ~ 1e-3 at 100 orders: 1e300). Scaling by a power of two is exact in binary
/// floating point, so the values agree bit for bit with the unscaled recursion wherever that one neither overflows nor
/// underflows (notes/HIGH_M.md 9.5).
pub use multidouble::dd::pow2_near as series_scale;

/// Centre-series coefficients of p, m, a, c in the scaled variable x / sigma, sigma = series_scale(nodes[0]): entry k
/// is c_k sigma^k (multiply by sigma^-k for the coefficients in x). rate: the growth rate of a mode-matrix
/// evaluation (`Rate::ZERO` otherwise).
pub fn centre_series<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T, rate: &Rate) -> [Vec<Fl<T, B>>; 4] {
    let sig = series_scale(ctx.nodes[0]);
    let n = dim::<T, B>(ctx);
    let n1 = n - T::P::one();
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
    let mut tail = tail_of::<T>(ctx.nodes[0] / sig);
    let cut = tail_active::<T, B>(ctx);
    if cut {
        tail.done(0, &[&pp[0], &mm[0], &aa[0], &cc[0]]);
        tail.done(1, &[&pp[1], &mm[1], &aa[1], &cc[1]]);
    }
    for k in 2..=kmax {
        let kf = k as f64;
        q2[k] = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        a2[k] = cs(&aa, &aa, k); // aa[k] still 0
        let s = cs(&aa, &q2, k);
        let t3 = cs(&aa, &a2, k);
        aa[k] = (s.mulf(0.5) - t3.mulp(n1.mul_f(0.5))).mulp(T::P::one() / (T::P::from_f64(kf) + n1));
        a2[k] = cs(&aa, &aa, k);
        cc[k] = cs_range(&cc, &a2, k, 0, k - 1).mulp(n1 * T::P::from_dd(recip_int(k)));
        // in the scaled variable every term of the x^(k-1) equation carries sigma^k: those built from order k - 1
        // alone (the d/dtau and x p' terms) gain one factor sigma, the Cauchy sums of index sum k none
        let sk = if T::TAN_F64 && ctx.mutate == 6 && k == 7 { 2.0 * sig } else { sig };
        let dp = dtau(ctx, &pp[k - 1], omega, rate).mul_pwr2(sk);
        let dm = dtau(ctx, &mm[k - 1], omega, rate).mul_pwr2(sk);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0).mul_pwr2(sk) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0).mul_pwr2(sk) + s1m;
        let xcxp = cs_w(&cc, &pp, k, 1, k, |j| j as f64);
        let xcxm = cs_w(&cc, &mm, k, 1, k, |j| j as f64);
        let cp = cs(&cc, &pp, k);
        let cm = cs(&cc, &mm, k);
        let cpm = cp + cm;
        let r3 = (((dp + xpx) - xcxp) + cp) - cpm.mulp(nh);
        let r4 = (((dm + xmx) + xcxm) - cm) + cpm.mulp(nh);
        // [m11 m12; m21 m22] = c [mu11 mu12; mu21 mu22]; det = c^2 delta, delta = -(k-1)(k-1+n)
        let mu11 = T::P::from_f64(1.0 - kf) - nh;
        let mu12 = -nh;
        let mu21 = nh;
        let mu22 = T::P::from_f64(kf - 1.0) + nh;
        let idel = T::P::one() / (-(T::P::from_f64(kf - 1.0) * (T::P::from_f64(kf - 1.0) + n)));
        pp[k] = ((r4.mulp(mu12) - r3.mulp(mu22)) * invc).mulp(idel);
        mm[k] = ((r3.mulp(mu21) - r4.mulp(mu11)) * invc).mulp(idel);
        if cut && tail.done(k, &[&pp[k], &mm[k], &aa[k], &cc[k]]) {
            for v in [&mut pp, &mut mm, &mut aa, &mut cc] {
                v.truncate(k + 1);
            }
            break;
        }
    }
    [pp, mm, aa, cc]
}

pub fn centre_state<T: Lane, const B: usize>(ctx: &Ctx<B>, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T, rate: &Rate) -> State<T, B> {
    let s = centre_series(ctx, p1, c0, omega, rate);
    let x = DD::from_f64(ctx.nodes[0]).mul_pwr2(1.0 / series_scale(ctx.nodes[0]));
    [horner(&s[0], x), horner(&s[1], x), horner(&s[2], x), horner(&s[3], x)]
}

// ---------------------------------------------------------------------------------------------------------------
// Taylor stepping in x
// ---------------------------------------------------------------------------------------------------------------

/// Taylor coefficients 1..=w.kt of the state about x0, order 0 in w.c[0..4] (p, m, a, c): the recursion of
/// `Dss::taylor_coeffs` at the given omega, its tau derivatives shifted by `w.rate`
pub fn taylor_coeffs<T: Lane, const B: usize>(ctx: &Ctx<B>, x0: f64, omega: T, w: &mut Work<T, B>) {
    let kt = w.kt;
    let rate = w.rate;
    let [p, m, a, c] = &mut w.c[..] else { unreachable!("four fields") };
    let [a2, q2, g, h, pth, mth, ww, vv] = &mut w.aux[..] else { unreachable!("eight scratch series") };
    let n1 = dim::<T, B>(ctx) - T::P::one();
    let nh = dim::<T, B>(ctx).mul_f(0.5);
    let x0d = DD::from_f64(x0);
    let inv_x0 = DD::ONE / x0d;
    let x2 = [x0d * x0d, x0d.mul_f(2.0), DD::ONE];
    let mut iw0 = Fl::<T, B>::zero();
    let mut iv0 = Fl::<T, B>::zero();
    for k in 0..kt {
        let kf = k as f64;
        a2[k] = cs(a, a, k);
        q2[k] = cs(p, p, k) + cs(m, m, k);
        g[k] = (cs(c, a2, k) - c[k]).mulp(n1);
        h[k] = cs(a, q2, k) + (a[k] - cs(a, a2, k)).mulp(n1);
        pth[k] = dtau(ctx, &p[k], omega, &rate);
        mth[k] = dtau(ctx, &m[k], omega, &rate);
        let mut xp = pth[k].muls(x0d);
        let mut xm = mth[k].muls(x0d);
        if k > 0 {
            xp = xp + pth[k - 1];
            xm = xm + mth[k - 1];
        }
        let cp = cs(c, p, k);
        let cm = cs(c, m, k);
        let cpm = cp + cm;
        let rp = ((cs(g, p, k) - xp) - cp) + cpm.mulp(nh);
        let rm = ((cm - xm) - cs(g, m, k)) - cpm.mulp(nh);
        let ik = inv_x0 * recip_int(k + 1);
        c[k + 1] = (g[k] - c[k].mulf(kf)).muls(ik);
        a[k + 1] = (h[k].mulf(0.5) - a[k].mulf(kf)).muls(ik);
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
        let ik1 = recip_int(k + 1);
        let (sp, sm) = if k >= 1 {
            (cs_w(ww, p, k + 1, 1, k, |j| (k + 1 - j) as f64), cs_w(vv, m, k + 1, 1, k, |j| (k + 1 - j) as f64))
        } else {
            (Fl::zero(), Fl::zero())
        };
        p[k + 1] = ((rp - sp) * iw0).muls(ik1);
        m[k + 1] = ((rm - sm) * iv0).muls(ik1);
    }
}

/// Work for one Taylor step of the four fields at order kt
pub fn work<T: Lane, const B: usize>(kt: usize) -> Work<T, B> {
    Work::new(4, N_AUX, kt)
}

/// One Taylor step of the state s from x0 to x1 at the given omega (multishoot's `taylor::taylor_step` with the
/// Delta of the evaluation replaced by its omega)
pub fn taylor_step<T: Lane, const B: usize>(ctx: &Ctx<B>, s: &State<T, B>, x0: f64, x1: f64, omega: T, w: &mut Work<T, B>) -> State<T, B> {
    for (f, sf) in s.iter().enumerate() {
        w.c[f][0] = *sf;
    }
    taylor_coeffs(ctx, x0, omega, w);
    let h = DD::from_f64(x1) - DD::from_f64(x0);
    let kt = w.kt;
    std::array::from_fn(|f| horner(&w.c[f][..=kt], h))
}

/// Taylor steps through the points xs (the first the start) at the given omega
pub fn taylor_path<T: Lane, const B: usize>(ctx: &Ctx<B>, s0: &State<T, B>, xs: &[f64], omega: T, w: &mut Work<T, B>) -> State<T, B> {
    let mut s = *s0;
    for i in 0..xs.len() - 1 {
        s = taylor_step(ctx, &s, xs[i], xs[i + 1], omega, w);
    }
    s
}

/// max over fields and lanes of |coefficient k| (hi part, value)
fn coef_norm<T: Lane, const B: usize>(w: &Work<T, B>, k: usize) -> f64 {
    w.c.iter().map(|q| q[k].val().amax()).fold(0.0, f64::max)
}

/// Adaptive step points from xa to xb for a state s (values): h from the last two Taylor coefficients so that the
/// truncation term is ~ tol (absolute; the fields are O(1)), capped at hmax. hp's planner, kept for its plans' bits:
/// multishoot's `plan_path` takes its roots from `kth_root` instead of `powf` and stops at the end exactly, so its
/// plans differ from these in the last bits.
pub fn plan_path<const B: usize>(ctx: &Ctx<B>, s0: &State<DV, B>, xa: f64, xb: f64, omega: DV, tol: f64, hmax: f64) -> Vec<f64> {
    let mut w = work::<DV, B>(ctx.kt);
    let kt = ctx.kt;
    // HP_PLAN_TRACE=N prints the position, step and time every N steps and at the end (diagnosis only)
    let trace = std::env::var("HP_PLAN_TRACE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    let t0 = std::time::Instant::now();
    let mut xs = vec![xa];
    let mut x = xa;
    let mut s = *s0;
    let dir = if xb > xa { 1.0 } else { -1.0 };
    while (xb - x) * dir > 1e-15 {
        for (f, sf) in s.iter().enumerate() {
            w.c[f][0] = *sf;
        }
        taylor_coeffs(ctx, x, omega, &mut w);
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
        // the step's sum from the coefficients just formed (they depend on s and x only)
        let hd = DD::from_f64(xn) - DD::from_f64(x);
        s = std::array::from_fn(|f| horner(&w.c[f][..=kt], hd));
        x = xn;
        xs.push(x);
        if trace > 0 && (xs.len() - 1) % trace == 0 {
            eprintln!("plan_path {xa}->{xb}: step {}, x {x:.12}, h {h:.3e}, {:.1} s", xs.len() - 1, t0.elapsed().as_secs_f64());
        }
    }
    if trace > 0 {
        eprintln!("plan_path {xa}->{xb}: done, {} steps, {:.1} s", xs.len() - 1, t0.elapsed().as_secs_f64());
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
    let c1 = c1_of(ctx, a0, n1);
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

/// c1 = (n - 1)(1 - a_h^2), in alpha = a_h - 1 with HALPHA_ON
fn c1_of<const B: usize>(ctx: &Ctx<B>, a0: &Fl<DV, B>, n1: DD) -> Fl<DV, B> {
    if ctx.horizon_alpha == HALPHA_ON {
        let al = *a0 - Fl::<DV, B>::splat(DD::ONE);
        -(al.mulf(2.0) + al * al).muls(n1)
    } else {
        (Fl::<DV, B>::splat(DD::ONE) - *a0 * *a0).muls(n1)
    }
}

/// LU factors of the real 2M x 2M form of (omega D + diag kappa_kk + lambda), kk = 0..=ks, at a complex growth rate
/// lambda = lr + i li: [[A + lr, -li], [li, A + lr]] acting on (f_re, f_im) (hp/modecount_cx's horizon_lus_c)
pub fn horizon_lus_c<const B: usize>(ctx: &Ctx<B>, a0: &Fl<DV, B>, delta: DD, lr: DD, li: DD, par: bool) -> Result<Vec<DDLu>, String> {
    let m = Ctx::<B>::M;
    let m2 = 2 * m;
    let ld = ld_for(m2);
    let omega = DD::TWO_PI / delta;
    let n1 = ctx.n - DD::ONE;
    let c1 = c1_of(ctx, a0, n1);
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

/// The preparation of one evaluation (multishoot's `Equations::Prep`): the horizon LUs at the state's double-double
/// a_h and Delta, and for a mode-matrix evaluation its growth rate with the factors it changes.
#[derive(Default)]
pub struct HPrep {
    /// (omega D + diag kappa_kk), kk = 0..=ks
    pub lus: Vec<DDLu>,
    /// the same plus lr, for the tangents at a real rate lr (empty: `lus` serve)
    pub lus_t: Vec<DDLu>,
    /// the 2M x 2M real form at a complex rate (`horizon_lus_c`; empty at a real rate)
    pub lus2: Vec<DDLu>,
    pub rate: Rate,
}

impl HPrep {
    /// the preparation of the LUs alone (rate zero)
    pub fn of(lus: Vec<DDLu>) -> Self {
        HPrep { lus, ..Default::default() }
    }
}

/// a preparation that failed (a horizon LU with a zero or non-finite pivot) panics with this prefix; `guard` returns
/// it as the entry point's error, as before the engine
const PREP_PANIC: &str = "hp horizon LUs: ";

/// The result of f, with a failed preparation's panic (PREP_PANIC) as its error; any other panic goes on unwinding
fn guard<R>(f: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(e) => match e.downcast_ref::<String>().and_then(|s| s.strip_prefix(PREP_PANIC)) {
            Some(msg) => Err(msg.to_string()),
            None => std::panic::resume_unwind(e),
        },
    }
}

fn prep_ok(r: Result<Vec<DDLu>, String>) -> Vec<DDLu> {
    r.unwrap_or_else(|e| std::panic::panic_any(format!("{PREP_PANIC}{e}")))
}

/// (omega D + diag kappa) f = r with the LU at the values; for tangents the derivative of the system:
/// A_t f_d = r_d - omega_d D f_v - kappa_d f_v, with A_t = A + lambda (`HPrep::lus_t`; the same LU at rate zero).
fn psolve<T: Lane, const B: usize>(ctx: &Ctx<B>, prep: &HPrep, kk: usize, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Fl<T, B> {
    let lu = &prep.lus[kk];
    let lu_t = if prep.lus_t.is_empty() { lu } else { &prep.lus_t[kk] };
    if T::NJ > 0 && T::VAL_OD {
        // an octo-double jet (multidouble 4.0.0's OJet): as the quad-double jet below, the right-hand side of order k
        // in octo-double and each order's solve refined to octo-double, with twice the steps
        debug_assert!(prep.rate.is_zero());
        let steps = match ctx.mutate {
            QD_MUT_NOREFINE => 0,
            QD_MUT_REFINE4 => 8,
            _ => OD_REFINE,
        };
        let om: Vec<OV> = (0..T::NJ).map(|j| omega.coef_od(j)).collect();
        let k0 = kappa.coef_od(0);
        return lu.solve_lanes_jet_refine_od(
            rhs,
            steps,
            |j, x, r| (r - ctx.d.apply(x).mul_t(om[j])) - kappa.coef_od(j) * *x,
            |r, x| *r - (ctx.d.apply(x).mul_t(om[0]) + k0 * *x),
        );
    }
    if T::VAL_OD {
        // octo-double values (OV) or value and tangent (Ou): refined as the quad-double lanes below (Moler's
        // refinement with the residual in the lane's precision), to octo-double
        let steps = match ctx.mutate {
            QD_MUT_NOREFINE => 0,
            QD_MUT_REFINE4 => 8,
            _ => OD_REFINE + T::TANGENT as usize,
        };
        return lu.solve_lanes_refine_t(lu_t, rhs, steps, |f| *rhs - (ctx.d.apply(f).mul_t(omega) + *kappa * *f));
    }
    if T::NJ > 0 && T::VAL_QD {
        // a quad-double jet: order by order as below, the right-hand side of order k in quad-double and each order's
        // solve refined to quad-double with the lane's omega_0 and kappa_0 (as the branch after this one)
        debug_assert!(prep.rate.is_zero());
        let steps = match ctx.mutate {
            QD_MUT_NOREFINE => 0,
            QD_MUT_REFINE4 => 4,
            _ => QD_REFINE,
        };
        let om: Vec<QV> = (0..T::NJ).map(|j| omega.coef_qd(j)).collect();
        let k0 = kappa.coef_qd(0);
        return lu.solve_lanes_jet_refine(
            rhs,
            steps,
            |j, x, r| (r - ctx.d.apply(x).mul_t(om[j])) - kappa.coef_qd(j) * *x,
            |r, x| *r - (ctx.d.apply(x).mul_t(om[0]) + k0 * *x),
        );
    }
    if T::VAL_QD {
        // the LU was factored at the double-double a_h and Delta; the refinement solves the lane's own system (omega
        // and kappa from the quad-double a_h and Delta), its residual formed in the lane's precision (iterative
        // refinement with the residual in higher precision: Moler, J. ACM 14 (1967) 316, doi:10.1145/321386.321394),
        // value and tangent together (multidouble's solve_lanes_refine_t: a tangent lane's residual is that of the
        // differentiated system, solved with lu_t).
        let steps = match ctx.mutate {
            QD_MUT_NOREFINE => 0,
            QD_MUT_REFINE4 => 4,
            _ => QD_REFINE + T::TANGENT as usize,
        };
        return lu.solve_lanes_refine_t(lu_t, rhs, steps, |f| *rhs - (ctx.d.apply(f).mul_t(omega) + *kappa * *f));
    }
    if T::NJ > 0 {
        // a jet: order by order with the LU at the values, the tangent branch below continued past order 1,
        // A_0 f_k = r_k - sum_{j=1..k} (omega_j D + kappa_j) f_{k-j}; no mode-matrix shift with jets
        debug_assert!(prep.rate.is_zero());
        return lu.solve_lanes_jet(rhs, |j, x, r| {
            let dfv = ctx.d.apply::<DV>(x);
            let od = Fl::<DV, B>([omega.coef(j); B]);
            (r - dfv * od) - kappa.coef(j) * *x
        });
    }
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
            let mut r = rhs.tan64(k) - dfv.mul_t(od);
            if ctx.mutate != 3 {
                r = r - kappa.tan64(k) * fv8;
            }
            let fd = if dds { lu_t.solve_lanes(&r.val()) } else { lu_t.solve_lanes_hi(&r.val()) };
            for b in 0..B {
                out.0[b].set_tan64(k, fd.0[b].hi);
            }
        }
        return out;
    }
    if T::TANGENT {
        if let Some(f) = psolve_dc(ctx, prep, kk, kappa, rhs, omega) {
            return f;
        }
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

/// psolve on complex tangents (`Dc`, the lanes of multishoot's mode matrix; None for any other lane): the value as
/// psolve's, each tangent part with psolve's operations on the real tangent, A_t f_d = r_d - omega_d D f_v -
/// kappa_d f_v; at a real rate the M x M solves of A + lr for the real and imaginary parts, at a complex one the
/// 2M x 2M real form (`HPrep::lus2`)
fn psolve_dc<T: Lane, const B: usize>(ctx: &Ctx<B>, prep: &HPrep, kk: usize, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Option<Fl<T, B>> {
    let kappa = (kappa as &dyn Any).downcast_ref::<Fl<Dc, B>>()?;
    let rhs = (rhs as &dyn Any).downcast_ref::<Fl<Dc, B>>()?;
    let omega = (&omega as &dyn Any).downcast_ref::<Dc>()?;
    let m = Ctx::<B>::M;
    let lu = &prep.lus[kk];
    let lu_t = if prep.lus_t.is_empty() { lu } else { &prep.lus_t[kk] };
    let fv = lu.solve_lanes(&rhs.val());
    let dfv = ctx.d.apply::<DV>(&fv);
    let (or, oi) = (Fl::<DV, B>([omega.r; B]), Fl::<DV, B>([omega.i; B]));
    let rr = (rhs.tan_re() - dfv * or) - kappa.tan_re() * fv;
    let ri = (rhs.tan_im() - dfv * oi) - kappa.tan_im() * fv;
    let out = if prep.lus2.is_empty() || ctx.mutate == MODE_MUT_REAL_SOLVE {
        Fl::<Dc, B>::from_vri(&fv, &lu_t.solve_lanes(&rr), &lu_t.solve_lanes(&ri))
    } else {
        let mut hh = vec![0.0; m];
        let mut ll = vec![0.0; m];
        let mut b = vec![DD::ZERO; 2 * m];
        for (part, f) in [&rr, &ri].into_iter().enumerate() {
            f.store(&mut hh, &mut ll);
            for i in 0..m {
                b[part * m + i] = DD { hi: hh[i], lo: ll[i] };
            }
        }
        prep.lus2[kk].solve(&mut b);
        let part = |o: usize| {
            let (h, l): (Vec<f64>, Vec<f64>) = b[o..o + m].iter().map(|v| (v.hi, v.lo)).unzip();
            Fl::<DV, B>::load(&h, &l)
        };
        Fl::<Dc, B>::from_vri(&fv, &part(0), &part(m))
    };
    (&out as &dyn Any).downcast_ref::<Fl<T, B>>().copied()
}

/// Horizon-series coefficients of p, m, a, c in the scaled variable u / sigma, sigma = series_scale(u0) (u = 1 - x):
/// entry kk is c_kk sigma^kk, as for centre_series. prep: the horizon LUs (and the rate of a mode-matrix evaluation).
pub fn horizon_series<T: Lane, const B: usize>(ctx: &Ctx<B>, m0: &Fl<T, B>, a0: &Fl<T, B>, omega: T, prep: &HPrep) -> [Vec<Fl<T, B>>; 4] {
    let rate = &prep.rate;
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
    let n1 = dim::<T, B>(ctx) - T::P::one();
    let nh = dim::<T, B>(ctx).mul_f(0.5);
    // kappa's n: the control JET_MUT_KAPPA freezes it (the exponents of the horizon block at fixed n)
    let nh_kappa = if ctx.mutate == JET_MUT_KAPPA { T::P::from_dd(ctx.n.mul_f(0.5)) } else { nh };
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
    // Ctx::horizon_alpha: aa holds alpha = a - 1 (its order 0 only differs) and a2 alpha^2; am1 is a^2 - 1. The
    // control (HALPHA_MUT) rebuilds a = 1 + alpha (af, af2 = a^2) and forms everything from it as without alpha.
    let alpha = ctx.horizon_alpha != 0;
    let ctl = ctx.horizon_alpha == HALPHA_MUT;
    let mut am1 = if alpha { vec![z; len] } else { Vec::new() };
    let mut af = if ctl { vec![z; len] } else { Vec::new() };
    let mut af2 = if ctl { vec![z; len] } else { Vec::new() };
    aa[0] = if alpha { *a0 - one } else { *a0 };
    cc[0] = one;
    mm[0] = *m0;
    a2[0] = aa[0] * aa[0];
    if ctl {
        af[0] = aa[0] + one;
        af2[0] = af[0] * af[0];
    }
    dm[0] = dtau(ctx, &mm[0], omega, rate);
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
    let mut tail = tail_of::<T>(ctx.u0().hi * isig);
    let cut = tail_active::<T, B>(ctx) && ctx.f64_dd & HDD_NOTAIL == 0;
    let mut last = ks;
    // In the scaled variable each term of the order-kk equations carries sigma^kk: a coefficient of order kk + 1 made
    // from order-kk terms gains sigma, a term of index sum kk - 1 gains sigma, one of index sum kk + 1 loses it.
    for kk in 0..=ks {
        let kf = kk as f64;
        let ik1 = recip_int(kk + 1);
        // (c (a^2 - 1))_kk
        let ca1 = if ctl {
            cs(&cc, &af2, kk) - cc[kk]
        } else if alpha {
            am1[kk] = aa[kk].mulf(2.0) + a2[kk];
            cs(&cc, &am1, kk)
        } else {
            cs(&cc, &a2, kk) - cc[kk]
        };
        cc[kk + 1] = (cc[kk].mulf(kf) - ca1.mulp(n1)).muls(ik1).mul_pwr2(sig);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk).mul_pwr2(sig);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64).mul_pwr2(isig) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk).mul_pwr2(sig);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let tau = if kk >= 1 { -dp[kk - 1].mul_pwr2(sig) } else { z };
        let r = (((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).mulp(nh);
        // kappa (and the LUs of horizon_lus) take the unscaled first-order coefficient
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).addp(-nh_kappa);
        pp[kk] = psolve(ctx, prep, kk, &kappa, &(-r), omega);
        dp[kk] = dtau(ctx, &pp[kk], omega, rate);
        q2[kk] = cs(&pp, &pp, kk) + cs(&mm, &mm, kk);
        if kk == ks || (cut && tail.done(kk, &[&pp[kk], &mm[kk], &aa[kk], &cc[kk]])) {
            last = kk;
            break;
        }
        // (a q2)_kk + (n - 1) (a - a^3)_kk
        let ag = if ctl {
            cs(&af, &q2, kk) + (af[kk] - cs(&af, &af2, kk)).mulp(n1)
        } else if alpha {
            let a_a3 = -((aa[kk].mulf(2.0) + a2[kk].mulf(3.0)) + cs(&aa, &a2, kk));
            (q2[kk] + cs(&aa, &q2, kk)) + a_a3.mulp(n1)
        } else {
            cs(&aa, &q2, kk) + (aa[kk] - cs(&aa, &a2, kk)).mulp(n1)
        };
        aa[kk + 1] = (aa[kk].mulf(kf) - ag.mulf(0.5)).muls(ik1).mul_pwr2(sig);
        a2[kk + 1] = cs(&aa, &aa, kk + 1);
        if ctl {
            af[kk + 1] = aa[kk + 1];
            af2[kk + 1] = cs(&af, &af, kk + 1);
        }
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1].mul_pwr2(sig) } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let rest = (((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).mulp(nh);
        mm[kk + 1] = (rest * izc0).muls(ik1).mul_pwr2(sig);
        dm[kk + 1] = dtau(ctx, &mm[kk + 1], omega, rate);
    }
    if alpha {
        aa[0] = *a0;
    }
    pp.truncate(last + 1);
    mm.truncate(last + 1);
    aa.truncate(last + 1);
    cc.truncate(last + 1);
    [pp, mm, aa, cc]
}

pub fn horizon_state<T: Lane, const B: usize>(ctx: &Ctx<B>, m0: &Fl<T, B>, a0: &Fl<T, B>, omega: T, prep: &HPrep) -> State<T, B> {
    let s = horizon_series(ctx, m0, a0, omega, prep);
    let u = ctx.u0().mul_pwr2(1.0 / series_scale(ctx.u0().hi));
    [horner(&s[0], u), horner(&s[1], u), horner(&s[2], u), horner(&s[3], u)]
}

// ---------------------------------------------------------------------------------------------------------------
// the equations for multishoot
// ---------------------------------------------------------------------------------------------------------------

/// hp's equations as multishoot's `Equations`: the fields p, m, a, c (NF = 4), the centre data p1, c0 (NL = 2), the
/// horizon data m_h, a_h (NR = 2) and Delta (NG = 1), on the context's constants and controls
pub struct Dss<'a, const B: usize>(pub &'a Ctx<B>);

impl<const B: usize> Equations<B> for Dss<'_, B> {
    const NF: usize = 4;
    const NL: usize = 2;
    const NR: usize = 2;
    const NG: usize = 1;
    const NAUX: usize = N_AUX;
    type Prep = HPrep;

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> HPrep {
        self.prepare_rate(zh, zl, Rate::ZERO)
    }

    /// the horizon LUs at the state's double-double a_h and Delta (the preconditioners of the refined solves in quad-
    /// and octo-double), and at a nonzero rate the factors of the shifted solves (hp/modecount_cx's HLus)
    fn prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> HPrep {
        let ctx = self.0;
        let ah = ctx.fld(zh, zl, ctx.off_ah());
        let delta = delta_of(zh, zl);
        let lus = prep_ok(horizon_lus(ctx, &ah, delta, true, DD::ZERO));
        let (lus_t, lus2) = if rate.im.hi != 0.0 {
            (Vec::new(), prep_ok(horizon_lus_c(ctx, &ah, delta, rate.re, rate.im, true)))
        } else if rate.re.hi != 0.0 {
            (prep_ok(horizon_lus(ctx, &ah, delta, true, rate.re)), Vec::new())
        } else {
            (Vec::new(), Vec::new())
        };
        HPrep { lus, lus_t, lus2, rate }
    }

    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        taylor_coeffs(self.0, x0, omega_of(self.0, g[0]), w);
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], prep: &HPrep) -> multishoot::State<T, B> {
        centre_state(self.0, &data[0], &data[1], omega_of(self.0, g[0]), &prep.rate).to_vec()
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], prep: &HPrep) -> multishoot::State<T, B> {
        horizon_state(self.0, &data[0], &data[1], omega_of(self.0, g[0]), prep).to_vec()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// residual, Jacobian (multishoot's)
// ---------------------------------------------------------------------------------------------------------------

pub(crate) fn delta_of(zh: &[f64], zl: &[f64]) -> DD {
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

fn split(f: &[DD]) -> (Vec<f64>, Vec<f64>) {
    f.iter().map(|v| (v.hi, v.lo)).unzip()
}

/// Residual (hi, lo), length L.
pub fn residual<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
    let sh = ctx.shooting();
    let z = [zh, zl];
    let f = guard(|| sh.residual::<Double>(&z, par))?;
    Ok(split(&f))
}

/// The residual on a jet state z(h) = sum_k z_k h^k with the dimension n + h as the jet variable: zs are the
/// coefficient vectors z_k (hi, lo), at most N of them, and the result the residual's coefficients of orders
/// 0..nout-1 (nout <= N), each (hi, lo) of length L. With z_k absent (zero) the order-k coefficient is the right-hand
/// side of order k of the Taylor continuation in D (notes/QUAD_DOUBLE.md 13.15). The horizon LUs are those of the
/// values (z_0); psolve continues them order by order.
pub fn residual_jet<const N: usize, const B: usize>(ctx: &Ctx<B>, zs: &[(&[f64], &[f64])], par: bool, nout: usize) -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    let zp: Vec<Vec<&[f64]>> = zs.iter().map(|&(h, lo)| vec![h, lo]).collect();
    let f = residual_jet_in::<Double, N, B>(ctx, &zp, par, nout)?;
    Ok(f.iter().map(|v| split(v)).collect())
}

/// Exact Jacobian dF/dz by forward-mode duals, column-major with leading dimension ld_for(L): (hi, lo, ld).
pub fn jacobian<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    let (mut c, ld) = jacobian_in::<Double, B>(ctx, &[zh, zl], par)?;
    let jl = c.pop().expect("two parts");
    let jh = c.pop().expect("two parts");
    Ok((jh, jl, ld))
}

// ---------------------------------------------------------------------------------------------------------------
// the residual, its jet, the exact Jacobian and the Newton step in any precision (quad-double: notes/QUAD_DOUBLE.md
// section 12; octo-double: notes/OCTO_DOUBLE.md): multishoot's at multidouble's `Precision` (`Double`, `Quad`, `Octo`),
// the state in PARTS parts, the horizon solves refined to the lanes' precision (psolve, `Lane::VAL_QD`,
// `Lane::VAL_OD`), the model's constants double-double, so that the discrete system is the double-double solver's
// ---------------------------------------------------------------------------------------------------------------

/// `Ctx::mutate` for the quad-double entry points (the control of their precision checks): the state read in its
/// double-double part only (multishoot's `Control::HalfParts`)
pub const QD_MUT_STATE_DD: u8 = 9;
/// `Ctx::mutate` for the octo-double entry points: the state read in its quad-double part only (its first four parts)
pub const OD_MUT_STATE_QD: u8 = 10;

impl<const B: usize> Ctx<B> {
    /// `shooting` for an evaluation in P: the state read in its lower half of parts (multishoot's
    /// `Control::HalfParts`) under P's state mutation, QD_MUT_STATE_DD or OD_MUT_STATE_QD
    pub fn shooting_in<P: Precision<Native>>(&self) -> Shooting<Dss<'_, B>, B> {
        let mut s = self.shooting();
        let code = match P::PARTS {
            4 => QD_MUT_STATE_DD,
            8 => OD_MUT_STATE_QD,
            _ => 0,
        };
        if code != 0 && self.mutate == code {
            s.set_control(Control::HalfParts);
        }
        s
    }
}

/// The residual with every block in the precision P, one scalar per row: each block reads its inputs, Delta and its
/// target in all parts, and the projected rows and the phase row are in P too
pub fn residual_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
    guard(|| ctx.shooting_in::<P>().residual::<P>(z, par))
}

/// The residual on a jet state z(h) = sum_k z_k h^k in P (`P::J<N>` lanes), with the dimension n + h as the jet
/// variable: at most N coefficient vectors, each in PARTS parts, and F's coefficients of orders 0..nout-1 (nout <= N).
/// The horizon LUs are those of z_0's double-double part, the preconditioners of the refined solves; the D-dependent
/// constants are formed in the jet's parameter arithmetic, so order 0 is residual_in's residual up to that rounding.
pub fn residual_jet_in<P: Precision<Native>, const N: usize, const B: usize>(
    ctx: &Ctx<B>,
    zs: &[Vec<&[f64]>],
    par: bool,
    nout: usize,
) -> Result<Vec<Vec<P::Scalar>>, String> {
    guard(|| ctx.shooting_in::<P>().residual_jet::<P, N>(zs, par, nout))
}

/// The exact Jacobian in P by forward-mode duals (`P::U` lanes, one evaluation per column): (PARTS column-major part
/// arrays with leading dimension ld_for(L), ld), part q of entry (i, j) at c[q][j ld + i]
pub fn jacobian_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
    guard(|| ctx.shooting_in::<P>().jacobian::<P>(z, par))
}

/// The Newton step in P at z: dz = -J^-1 F in PARTS parts, F, the dense LU in P and the seconds of the residual, the
/// Jacobian, the factorization and the solve (multishoot's `newton_step`)
pub fn newton_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<multishoot::newton::Step<P>, String> {
    guard(|| ctx.shooting_in::<P>().newton_step::<P>(z, par))
}

/// The Jacobian in plain f64 (the same columns with `Fu` lanes: constants and the base state rounded to f64, the
/// horizon solves with the hi parts of the double-double LU factors), column-major with leading dimension
/// ld_for(L): (J, ld). For the mixed-precision Newton step (double-double residual, f64 Jacobian).
pub fn jacobian_f64<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    jacobian_f64_lanes::<Fu, B>(ctx, zh, zl, par)
}

/// The f64 Jacobian with `Dtn<N>` lanes: a double-double base trajectory and N f64 tangents per evaluation (N
/// columns of one block at a time), column-major with leading dimension ld_for(L): (J, ld).
pub fn jacobian_dtn<const N: usize, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    jacobian_f64_lanes::<Dtn<N>, B>(ctx, zh, zl, par)
}

/// The f64 Jacobian with `Fun<N>` lanes: an f64 base trajectory and N f64 tangents per evaluation: (J, ld).
pub fn jacobian_fun<const N: usize, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    jacobian_f64_lanes::<Fun<N>, B>(ctx, zh, zl, par)
}

/// The f64 Jacobian with T::NT tangents per evaluation (TAN_F64 lanes): multishoot's `jacobian_f64`; with
/// Ctx::f64_dd's HDD_BLOCK, HDD_LASTSEG or HDD_DTN some blocks' columns on other lanes
/// (`jacobian_f64_hdd`).
fn jacobian_f64_lanes<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    if ctx.f64_dd & (HDD_BLOCK | HDD_LASTSEG | HDD_DTN) != 0 {
        return jacobian_f64_hdd::<T, B>(ctx, zh, zl, par);
    }
    let sh = ctx.shooting();
    guard(|| sh.jacobian_f64::<T>(&[zh, zl], par))
}

/// The f64 Jacobian with some blocks' column tasks on other lanes (Ctx::f64_dd): `Du` (double-double, rounded to f64
/// when stored) for the horizon block (HDD_BLOCK) or the last segment (HDD_LASTSEG), `Dtn<4>` for the horizon block
/// (HDD_DTN), the other tasks with T in groups of T::NT; every column from multishoot's column evaluations, assembled
/// as `jacobian_f64` does, with its projected rows (multishoot's crate-private `project_f64_block`, here through
/// `ProjectedRows`) formed on the double-double columns of `Du` lanes before their rounding.
fn jacobian_f64_hdd<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool) -> Result<(Vec<f64>, usize), String> {
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let sh = ctx.shooting();
    let z = [zh, zl];
    let g = sh.globals::<Double>(&z);
    let prep = guard(|| Ok(sh.eq.prepare(zh, zl)))?;
    let tasks = sh.jac_tasks();
    let k = ctx.k();
    let du = |b: usize| (ctx.f64_dd & HDD_BLOCK != 0 && b == k) || (ctx.f64_dd & HDD_LASTSEG != 0 && k >= 2 && b == k - 1);
    let dtn = |b: usize| ctx.f64_dd & HDD_DTN != 0 && b == k && !du(b);
    // (lane kind, tasks): 0 T in groups of NT, 1 Du, 2 Dtn<4>
    let mut units: Vec<(u8, Vec<(usize, Seed)>)> = Vec::new();
    for &t in &tasks {
        let (kind, cap) = if du(t.0) { (1u8, 1) } else if dtn(t.0) { (2, 4) } else { (0, T::NT.max(1)) };
        match units.last_mut() {
            Some((kd, gr)) if *kd == kind && gr.len() < cap && gr[0].0 == t.0 => gr.push(t),
            _ => units.push((kind, vec![t])),
        }
    }
    let f64_cols = |c: Vec<Vec<f64>>| -> Vec<Vec<DD>> { c.into_iter().map(|v| v.into_iter().map(DD::from_f64).collect()).collect() };
    let unit = |u: &(u8, Vec<(usize, Seed)>)| -> Vec<Vec<DD>> {
        match u.0 {
            1 => vec![sh.jac_column::<Double>(&z, &g, &prep, u.1[0])],
            2 => f64_cols(sh.jac_columns_f64::<Dtn<4>>(&z, &g, &prep, &u.1)),
            _ => f64_cols(sh.jac_columns_f64::<T>(&z, &g, &prep, &u.1)),
        }
    };
    let gc: Vec<Vec<Vec<DD>>> = guard(|| Ok(if par { units.par_iter().map(unit).collect() } else { units.iter().map(unit).collect() }))?;
    let cols: Vec<Vec<DD>> = gc.into_iter().flatten().collect();
    let mut jh = vec![0.0; ld * l];
    // the double-double columns of the projected blocks' tasks, by unknown
    let mut dd_col: std::collections::HashMap<usize, &Vec<DD>> = std::collections::HashMap::new();
    for (&(b, seed), col) in tasks.iter().zip(&cols) {
        let c = sh.seed_columns(b, seed)[0];
        for (r, v) in col.iter().enumerate() {
            jh[c * ld + b * 4 * m + r] = v.hi;
        }
        if sh.projected.iter().any(|p| p.block == b) {
            dd_col.insert(c, col);
        }
    }
    for b in 0..=k {
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            let idx = (t + r) * ld + b * 4 * m + r;
            jh[idx] = (DD { hi: jh[idx], lo: 0.0 } - DD::ONE).hi;
        }
    }
    // projected rows as multishoot's jacobian_f64 forms them: each column's rows times Pi in double-double, rounded
    // (a Du task's from its double-double column), and Pi' added on the pinned unknowns
    for p in &sh.projected {
        let r0 = sh.row_block(p.block) + p.field * m;
        for j in 0..l {
            if (0..m).all(|i| jh[j * ld + r0 + i] == 0.0) {
                continue;
            }
            let x: Vec<DD> = match dd_col.get(&j) {
                Some(col) => col[p.field * m..(p.field + 1) * m].to_vec(),
                None => (0..m).map(|i| DD::from_f64(jh[j * ld + r0 + i])).collect(),
            };
            for (i, v) in p.apply::<Double>(&x, None, false).into_iter().enumerate() {
                jh[j * ld + r0 + i] = v.hi;
            }
        }
        for jj in 0..m {
            for i in 0..m {
                let a = (p.pin + jj) * ld + r0 + i;
                jh[a] = (DD::from_f64(jh[a]) + p.entry::<Double>(false, i, jj, false)).hi;
            }
        }
    }
    for i in 0..m {
        jh[i * ld + l - 1] = ctx.phase.get(i).hi;
    }
    Ok((jh, ld))
}

/// Deliberate breakage of `jvp`, for the controls of hp/largeD_mixed_check.py: the Delta tangent scaled by
/// 1 + 1e-20 (a subtle error in one column's contribution)
pub const JVP_MUT_DELTA: u8 = 1;
/// residual_jet's controls (Ctx::mutate): the dimension frozen (the jet variable a constant, so every coefficient
/// past the value is that of a fixed n: the Taylor continuation then predicts no change, and must fail its branch
/// test), and kappa's dependence on n frozen (the horizon exponents at fixed n: must fail the residual test)
pub const JET_MUT_FREEZE_N: u8 = 21;
pub const JET_MUT_KAPPA: u8 = 22;

/// the dimension in the lane's parameter arithmetic (multidouble's Param): the double-double scalar on every lane
/// but a jet, bit for bit as before; on a jet the series n + h, so the coefficients carry their own derivatives in
/// n (the Taylor continuation in D, notes/QUAD_DOUBLE.md 13.15); JET_MUT_FREEZE_N keeps it a constant
fn dim<T: Lane, const B: usize>(ctx: &Ctx<B>) -> T::P {
    if ctx.mutate == JET_MUT_FREEZE_N {
        T::P::from_dd(ctx.n)
    } else {
        <T::P as Param>::var(ctx.n)
    }
}
/// the control of the complex-rate mode matrix's check (mode_tests): the horizon solves of complex tangents with the
/// M x M factors of the values (lus_t, here lus) at every rate, the 2M x 2M form ignored
pub const MODE_MUT_REAL_SOLVE: u8 = 23;
/// `jvp` control: the horizon block's state tangent dropped
pub const JVP_MUT_HORIZON: u8 = 2;
/// `jvp` control: the phase row's product dropped
pub const JVP_MUT_PHASE: u8 = 3;

/// Exact Jacobian-vector product J v in double-double: the tangent of the residual along v, by one forward-mode
/// pass per block (`Du` lanes, the lanes of the double-double Jacobian) with the block's inputs and Delta carrying
/// v's entries as their tangent (multishoot's `jvp`). It equals the double-double Jacobian times v to double-double
/// rounding, for a cost of about K + 1 Jacobian columns. Returns (Jv hi, Jv lo, F hi, F lo): the residual comes with
/// it. mutate: JVP_MUT_* (controls; 0 none).
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
    // JVP_MUT_DELTA: the Delta tangent dv + dv 1e-20
    let mut vd = (vh.to_vec(), vl.to_vec());
    if mutate == JVP_MUT_DELTA {
        let dv = delta_of(vh, vl);
        let dv = dv + dv * DD::from_f64(1e-20);
        (vd.0[l - 1], vd.1[l - 1]) = (dv.hi, dv.lo);
    }
    let sh = ctx.shooting();
    let (mut jv, f) = guard(|| sh.jvp::<Double>(&[zh, zl], &[vd.0.as_slice(), vd.1.as_slice()], par))?;
    let kb = ctx.k();
    if mutate == JVP_MUT_HORIZON {
        let (t, r0) = (ctx.block_target(kb), kb * 4 * m);
        for q in 0..4 {
            let d = -ctx.fld(vh, vl, t + q * m);
            for i in 0..m {
                jv[r0 + q * m + i] = d.get(i);
            }
        }
        // its projected rows: Pi on the rows, Pi' on v's pinned unknowns, as the engine forms them
        for p in &sh.projected {
            let r0 = sh.row_block(p.block) + p.field * m;
            let y: Vec<DD> = (0..m).map(|j| DD::new(vh[p.pin + j], vl[p.pin + j])).collect();
            let rows = p.apply::<Double>(&jv[r0..r0 + m], Some(&y), false);
            jv[r0..r0 + m].copy_from_slice(&rows);
        }
    }
    if mutate == JVP_MUT_PHASE {
        jv[l - 1] = DD::ZERO;
    }
    let (jh, jl) = split(&jv);
    let (fh, fl) = split(&f);
    Ok((jh, jl, fh, fl))
}

/// The Newton system of the mixed-precision step in block storage: the f64 Jacobian (TAN_F64 lanes T: `Fu`,
/// `Fun<N>` or `Dtn<N>`) built block by block and factored by the staircase LU (multishoot's `jacobian_staircase`,
/// multidouble's `StaircaseLu`) without forming the dense matrix; its entries are those of the dense f64 Jacobian.
/// Not for the frozen-harmonics layout (Ctx::freeze couples the horizon block to p1) or a mode matrix (lam != 0).
/// mutate: MUT_PIVOT, MUT_LINK (controls).
pub fn jacobian_block<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], par: bool, mutate: u8) -> Result<StaircaseLu, String> {
    if ctx.freeze.is_some() {
        return Err("block storage does not support the frozen-harmonics layout (kfreeze)".into());
    }
    if ctx.lam.hi != 0.0 || !T::TAN_F64 {
        return Err("block storage is for the f64 Jacobian only".into());
    }
    let mut sh = ctx.shooting();
    sh.set_control(match mutate {
        MUT_PIVOT => Control::StaircasePivot,
        MUT_LINK => Control::StaircaseLink,
        _ => Control::None,
    });
    guard(|| sh.jacobian_staircase::<T>(&[zh, zl], par))
}

/// `jacobian_block`'s deliberate breakage for the controls (hp/highm_check.py): one pivot of the staircase LU's middle
/// step scaled by 1 + 1e-6 (multishoot's `Control::StaircasePivot`)
pub const MUT_PIVOT: u8 = 1;
/// the link rows replaced by delta_s = 0, the copies of Delta no longer tied to it (`Control::StaircaseLink`)
pub const MUT_LINK: u8 = 2;

/// The mode matrix M(lam) at a real growth rate lam: the linearization of the residual for perturbations
/// ~ exp(lam tau), Delta held fixed, over the block rows and the unknowns before Delta (multishoot's `mode_matrix`,
/// its real part). Column-major with leading dimension ld_for(L-1): (hi, lo, ld).
pub fn mode_matrix<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], lam: DD, par: bool) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    let sh = ctx.shooting();
    let mm = guard(|| sh.mode_matrix(&[zh, zl], Rate::new(lam, DD::ZERO), par))?;
    let ld = mm.ld;
    let mut re = mm.re;
    let l = re.pop().expect("two parts");
    let h = re.pop().expect("two parts");
    Ok((h, l, ld))
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
        let ctx = Ctx::<B> { n, d, phase, nodes, segs, kt, kc, ks, freeze: None, match_node: k - 1, lam: DD::ZERO, mutate: 0, f64_dd: 0, horizon_alpha: 0 };
        assert_eq!(ctx.len(), l);
        (ctx, zh, zl)
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::field::cs;
    use crate::lane::Du;
    use std::time::Instant;

    /// seconds per column for the sample tasks (one block), one evaluation per column (`Du`) or groups of T::NT
    /// columns per evaluation (f64 lanes; the sample is then extended to whole groups of consecutive lanes)
    fn time_cols<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], prep: &HPrep, tasks: &[(usize, Seed)]) -> f64 {
        let sh = ctx.shooting();
        let z = [zh, zl];
        let g = sh.globals::<Double>(&z);
        let mut s = 0.0;
        let t0 = Instant::now();
        let n = if T::TAN_F64 {
            let mut n = 0;
            for gr in tasks.chunks(T::NT) {
                s += sh.jac_columns_f64::<T>(&z, &g, prep, gr)[0][3];
                n += gr.len();
            }
            n
        } else {
            for &t in tasks {
                s += sh.jac_column::<Double>(&z, &g, prep, t)[3].hi;
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
        let prep = HPrep::of(horizon_lus(&ctx, &ctx.fld(&zh, &zl, ctx.off_ah()), delta, false, DD::ZERO).unwrap());
        println!("M = {}, L = {}, single thread; seconds for all columns of a block (from a sample)", 8 * B, ctx.len());
        prim::<Du, B>(&ctx, "Du");
        prim::<Fu, B>(&ctx, "Fu");
        prim::<Fun<4>, B>(&ctx, "Fun<4>");
        prim::<Dtn<1>, B>(&ctx, "Dtn<1>");
        prim::<Dtn<4>, B>(&ctx, "Dtn<4>");
        prim::<Dtn<8>, B>(&ctx, "Dtn<8>");
        let all = ctx.shooting().jac_tasks();
        let nb = ctx.k() + 1;
        let names = ["Du", "Fu", "Fun<2>", "Fun<4>", "Fun<8>", "Dtn<1>", "Dtn<4>", "Dtn<8>"];
        let mut tot = [0.0; 8];
        for b in 0..nb {
            let ncol = all.iter().filter(|t| t.0 == b).count() as f64;
            // 16 consecutive lanes of the second input field (each evaluation group within one field)
            let sample: Vec<(usize, Seed)> = all.iter().filter(|t| t.0 == b && matches!(t.1, Seed::Field(1, j) if j < 16)).cloned().collect();
            let ts = [
                time_cols::<Du, B>(&ctx, &zh, &zl, &prep, &sample[..4]),
                time_cols::<Fu, B>(&ctx, &zh, &zl, &prep, &sample[..8]),
                time_cols::<Fun<2>, B>(&ctx, &zh, &zl, &prep, &sample[..8]),
                time_cols::<Fun<4>, B>(&ctx, &zh, &zl, &prep, &sample[..8]),
                time_cols::<Fun<8>, B>(&ctx, &zh, &zl, &prep, &sample),
                time_cols::<Dtn<1>, B>(&ctx, &zh, &zl, &prep, &sample[..4]),
                time_cols::<Dtn<4>, B>(&ctx, &zh, &zl, &prep, &sample[..8]),
                time_cols::<Dtn<8>, B>(&ctx, &zh, &zl, &prep, &sample),
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
