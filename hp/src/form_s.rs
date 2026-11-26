//! Form S of the DSS equations: B = 1/a^2 slaved to p, m and c by the momentum constraint with a_x taken from the
//! Hamiltonian one, at every x (Gundlach, Phys. Rev. D 55, 695 (1997), gr-qc/9604019, Eq. (19) and App. C; in D
//! dimensions notes/LIT_GAPS_DSS.md 2):
//!
//!   omega dB/dtheta = [(n - 1) + q] B - (n - 1),      q = ((x - c) p^2 + (x + c) m^2) / x,
//!
//! a periodic linear solve in theta at each x. a is not integrated in x, so form A's mode a - 1 ~ x^-(n-1) (the
//! momentum-constraint violation, notes/LIT_GAPS_DSS.md 2) does not exist. The fields are p, m (odd) and c (even), with
//!
//!   x c_x         = (n-1) c (a^2 - 1)                    =: G,   a^2 = 1/B
//!   x (x - c) p_x = -omega x p_th + G p - c p + (n/2) c (p + m)
//!   x (x + c) m_x = -omega x m_th - G m + c m - (n/2) c (p + m)
//!
//! as in `model` (form A). Unknowns z (L = 3M(K+1) + 1): p1, c0 | node i: p, m, c | m_h | Delta. Residual blocks: the
//! centre series at node 0 minus node 0; the Taylor steps between nodes; the horizon series at u0 minus the last node,
//! in the rows p, m, c; and the phase row. B's Taylor and series coefficients come order by order from
//!
//!   (omega D_e - (n - 1) - q_0) B_k = sum_{j=1..k} q_j B_{k-j}   (k >= 1),     (omega D_e - (n - 1) - q_0) B_0 = -(n - 1),
//!
//! D_e the folded even-sector d/dtheta (`Ctx::de`), one LU per expansion point (each Taylor step, the centre, the
//! horizon), solved in each lane as `model::psolve` solves the horizon's periodic systems: the LU at the values in
//! double-double, refined in the lane's precision, and a tangent's system differentiated.
use crate::dd::DD;
use crate::dv::DV;
use crate::field::FieldSolve;
use crate::field::{cs, cs_range, cs_w, horner, DMat, Fl};
use crate::lane::Lane;
use crate::linalg::{ld_for, DDLu};
use crate::model::{delta_of, dim, horizon_lus, kfreeze_of, mul_recip, omega_of, psolve, recip_int, recip_lane, recip_lane_parts, series_scale, Ctx, HPrep};
use crate::qd::QD;
use multidouble::od::OD;
use crate::qv::QV;
use multidouble::dual::Param;
use multidouble::native::OV;
use multidouble::simd::Native;
use multidouble::precision::Precision;
use multidouble::series::{Relaxed, RELAXED_MUT_DROP};
use multishoot::periodic::Parity;
use multishoot::{Equations, ExtraRow, Rate, Shooting, Work};

/// scratch series of the Taylor recursion: B, a^2, q, p^2 + m^2, p^2 - m^2, G, p_th, m_th, x (x - c), x (x + c)
const NAUX_S: usize = 10;

/// `Ctx::mutate` for form S's entry points (the control of hp/form_s.py check): q's (x - c) and (x + c) swapped in the
/// slaving of the Taylor steps and the centre series, q = ((x + c) p^2 + (x - c) m^2) / x (mc_states.py's
/// --mutate=flux, notes/LIT_GAPS_DSS.md 2); the horizon series is left as it is
pub const FS_MUT_FLUX: u8 = 51;

/// `Ctx::mutate`, the control of the Moebius horizon series (`horizon_mob`): z(t)'s coefficients from a wrong map,
/// z = (1 - t)/(1 - a t), the sign of the denominator's a flipped while h and e keep the right one
pub const FS_MUT_MOBIUS: u8 = 52;

/// `Ctx::mutate`, the control of the Moebius series' products with z (`horizon_mob`): the two-term recurrence's
/// coefficient a t_l multiplied by 1 + 2^-60
pub const FS_MUT_ZREC: u8 = 53;

/// `Ctx::mutate`: the Moebius series' products with z as full Cauchy sums with z's coefficients, O(ks^2) (the reference
/// the two-term recurrence is checked against), instead of the recurrence
pub const FS_ZCAUCHY: u8 = 54;

/// `Ctx::mutate`, the control of the relaxed products of the Moebius series (`horizon_mob` with `Ctx::hrelax`): the
/// first square of the product c U left out (multidouble's `series::RELAXED_MUT_DROP`)
pub const FS_MUT_RELAX_DROP: u8 = 55;

/// One Cauchy product x y of the Moebius series: its order-k sums by multidouble's relaxed product (`Relaxed`, when
/// `Ctx::hrelax` is set and the lane has fast squares) or term by term (`cs_range`, bitwise the sums before)
struct Prod<T: Lane, const B: usize>(Option<Relaxed<T, B>>);

impl<T: Lane, const B: usize> Prod<T, B> {
    fn new<const BB: usize>(ctx: &Ctx<BB>, len: usize, drop: bool) -> Self {
        if ctx.hrelax == 0 || T::BLOCK_MIN == 0 {
            return Prod(None);
        }
        let mut r = Relaxed::with_min(len, T::BLOCK_MIN);
        if drop && ctx.mutate == FS_MUT_RELAX_DROP {
            r.mutate = RELAXED_MUT_DROP;
        }
        Prod(Some(r))
    }
    /// sum_(i=lo..=hi) x_i y_(k-i)
    #[inline(always)]
    fn sum(&self, x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize, lo: usize, hi: usize) -> Fl<T, B> {
        match &self.0 {
            Some(r) => r.sum(x, y, k, lo, hi),
            None => cs_range(x, y, k, lo, hi),
        }
    }
    /// whether the end of step e forms squares
    fn forms(&self, e: usize) -> bool {
        self.0.as_ref().is_some_and(|r| r.squares_at(e))
    }
    /// the end of step e: x and y final to order e - 1
    fn step(&mut self, x: &[Fl<T, B>], y: &[Fl<T, B>], e: usize) {
        if let Some(r) = &mut self.0 {
            r.step(x, y, e);
        }
    }
}

/// refinement steps of a slaving solve: as model's QD_REFINE (2) and OD_REFINE (4), one more on a tangent lane
const QD_REFINE_S: usize = 2;
const OD_REFINE_S: usize = 4;

impl<const B: usize> Ctx<B> {
    /// form S's length 3M(K + 1) + 1
    pub fn len_s(&self) -> usize {
        3 * Self::M * (self.k() + 1) + 1
    }
    /// form S's offset of node i
    pub fn off_node_s(&self, i: usize) -> usize {
        2 * Self::M + 3 * Self::M * i
    }
    /// form S's offset of m_h
    pub fn off_mh_s(&self) -> usize {
        2 * Self::M + 3 * Self::M * self.k()
    }
    /// The engine's problem for form S (`DssS`): `shooting`'s nodes, segments, Taylor order, matching node, phase row
    /// and kfreeze projection, on three fields with m_h the one right endpoint field.
    pub fn shooting_s(&self) -> Shooting<DssS<'_, B>, B> {
        let mut s = Shooting::new(DssS(self), self.nodes.clone(), self.kt);
        s.segs = self.segs.clone();
        s.match_node = self.match_node;
        s.extra = vec![ExtraRow { terms: (0..Self::M).map(|i| (i, self.phase.get(i))).collect(), rhs: DD::ZERO }];
        if let Some((lo, _)) = &self.freeze {
            s.project_rows(self.k(), 0, 0, Parity::Odd, kfreeze_of(lo)).expect("the horizon block's p rows");
        }
        s
    }
}

/// the double-double LU of omega D_e + diag kappa (M x M, column-major)
pub fn slave_lu<const B: usize>(de: &DMat<B>, omega: DD, kappa: &[DD]) -> Result<DDLu, String> {
    let m = 8 * B;
    let ld = ld_for(m);
    let mut h = vec![0.0; ld * m];
    let mut l = vec![0.0; ld * m];
    for j in 0..m {
        let col = de.cols[j].muls(omega);
        col.store(&mut h[j * ld..j * ld + m], &mut l[j * ld..j * ld + m]);
    }
    for i in 0..m {
        let v = DD { hi: h[i * ld + i], lo: l[i * ld + i] } + kappa[i];
        h[i * ld + i] = v.hi;
        l[i * ld + i] = v.lo;
    }
    DDLu::factor(m, ld, h, l, false)
}

/// (omega D_e + diag kappa) f = rhs in the lane, with the LU at the values (`model::psolve`'s branches, on D_e and at
/// rate zero): refined to the lane's precision on quad- and octo-double lanes, a tangent's system A f_d = r_d -
/// omega_d D_e f_v - kappa_d f_v, a jet's order by order
fn ssolve<T: Lane, const B: usize>(de: &DMat<B>, lu: &DDLu, kappa: &Fl<T, B>, rhs: &Fl<T, B>, omega: T) -> Fl<T, B> {
    if T::NJ > 0 && T::VAL_OD {
        let om: Vec<OV> = (0..T::NJ).map(|j| omega.coef_od(j)).collect();
        let k0 = kappa.coef_od(0);
        return lu.solve_lanes_jet_refine_od(
            rhs,
            OD_REFINE_S,
            |j, x, r| (r - de.apply(x).mul_t(om[j])) - kappa.coef_od(j) * *x,
            |r, x| *r - (de.apply(x).mul_t(om[0]) + k0 * *x),
        );
    }
    if T::VAL_OD {
        return lu.solve_lanes_refine_t(lu, rhs, OD_REFINE_S + T::TANGENT as usize, |f| *rhs - (de.apply(f).mul_t(omega) + *kappa * *f));
    }
    if T::NJ > 0 && T::VAL_QD {
        let om: Vec<QV> = (0..T::NJ).map(|j| omega.coef_qd(j)).collect();
        let k0 = kappa.coef_qd(0);
        return lu.solve_lanes_jet_refine(
            rhs,
            QD_REFINE_S,
            |j, x, r| (r - de.apply(x).mul_t(om[j])) - kappa.coef_qd(j) * *x,
            |r, x| *r - (de.apply(x).mul_t(om[0]) + k0 * *x),
        );
    }
    if T::VAL_QD {
        return lu.solve_lanes_refine_t(lu, rhs, QD_REFINE_S + T::TANGENT as usize, |f| *rhs - (de.apply(f).mul_t(omega) + *kappa * *f));
    }
    if T::NJ > 0 {
        return lu.solve_lanes_jet(rhs, |j, x, r| {
            let dfv = de.apply::<DV>(x);
            let od = Fl::<DV, B>([omega.coef(j); B]);
            (r - dfv * od) - kappa.coef(j) * *x
        });
    }
    if T::VAL_F64 || T::TAN_F64 {
        let fv = if T::VAL_F64 { lu.solve_lanes_hi(&rhs.val()) } else { lu.solve_lanes(&rhs.val()) };
        let z = Fl::<DV, B>::zero();
        let mut out = Fl::<T, B>::from_vt(&fv, &z);
        if !T::TANGENT {
            return out;
        }
        let fv8 = Fl::<crate::lane::F8, B>::from_vt(&fv, &z);
        let dfv = de.apply(&fv8);
        for k in 0..T::NT {
            let od = multidouble::dual::F8(omega.tan64(k));
            let r = (rhs.tan64(k) - dfv.mul_t(od)) - kappa.tan64(k) * fv8;
            let fd = lu.solve_lanes_hi(&r.val());
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
    let dfv = de.apply::<DV>(&fv);
    let od = Fl::<DV, B>([omega.tan(); B]);
    let r = (rhs.tan() - dfv * od) - kappa.tan() * fv;
    let fd = lu.solve_lanes(&r);
    Fl::from_vt(&fv, &fd)
}

/// the double-double values of a lane field, one DD per point
fn vals<T: Lane, const B: usize>(x: &Fl<T, B>) -> Vec<DD> {
    let v = x.val();
    (0..8 * B).map(|i| v.get(i)).collect()
}

/// a^2 = 1/B as a series, coefficient k from B's coefficients 0..=k and a^2's below k
#[inline(always)]
fn inv_at<T: Lane, const B: usize>(bb: &[Fl<T, B>], ia: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    if k == 0 {
        Fl::<T, B>::splat(DD::ONE) / bb[0]
    } else {
        -(cs_range(bb, ia, k, 1, k) * ia[0])
    }
}

/// Taylor coefficients 1..=w.kt of p, m, c about x0 (w.c[0..3]), with B and a^2 = 1/B in w.aux, from order 0
pub fn taylor_coeffs_s<T: Lane, const B: usize>(ctx: &Ctx<B>, de: &DMat<B>, x0: f64, omega: T, w: &mut Work<T, B>) {
    let kt = w.kt;
    let rate = w.rate;
    debug_assert!(rate.is_zero(), "form S: no mode matrix");
    let [p, m, c] = &mut w.c[..] else { unreachable!("three fields") };
    let [bb, ia, q, q2, d2, g, pth, mth, ww, vv] = &mut w.aux[..NAUX_S] else { unreachable!("ten scratch series") };
    let n1 = dim::<T, B>(ctx) - T::P::one();
    let nh = dim::<T, B>(ctx).mul_f(0.5);
    let x0d = DD::from_f64(x0);
    let inv_x0 = DD::ONE / x0d;
    let x2 = [x0d * x0d, x0d.mul_f(2.0), DD::ONE];
    let one = Fl::<T, B>::splat(DD::ONE);
    let mut iw0 = Fl::<T, B>::zero();
    let mut iv0 = Fl::<T, B>::zero();
    let mut kap0 = Fl::<T, B>::zero();
    let mut lu: Option<DDLu> = None;
    for k in 0..kt {
        let kf = k as f64;
        let pp2 = cs(p, p, k);
        let mm2 = cs(m, m, k);
        q2[k] = pp2 + mm2;
        d2[k] = pp2 - mm2;
        // x q = x (p^2 + m^2) - c (p^2 - m^2), x = x0 + h
        let cd = cs(c, d2, k);
        let mut s = if ctx.mutate == FS_MUT_FLUX { q2[k].muls(x0d) + cd } else { q2[k].muls(x0d) - cd };
        if k > 0 {
            s = s + q2[k - 1];
            q[k] = (s - q[k - 1]).muls(inv_x0);
        } else {
            q[k] = s.muls(inv_x0);
        }
        if k == 0 {
            kap0 = (-q[0]).addp(-n1);
            let lu0 = slave_lu(de, omega.val().lane(0), &vals(&kap0)).expect("form S: the slaving LU of a Taylor step");
            bb[0] = ssolve(de, &lu0, &kap0, &(-(one.mulp(n1))), omega);
            lu = Some(lu0);
        } else {
            let rhs = cs_range(q, bb, k, 1, k);
            bb[k] = ssolve(de, lu.as_ref().expect("the step's LU"), &kap0, &rhs, omega);
        }
        ia[k] = inv_at(bb, ia, k);
        g[k] = (cs(c, ia, k) - c[k]).mulp(n1);
        pth[k] = crate::model::dtau(ctx, &p[k], omega, &rate);
        mth[k] = crate::model::dtau(ctx, &m[k], omega, &rate);
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

/// Centre-series coefficients of p, m, c in the scaled variable x / sigma (model::centre_series's, a's recursion
/// replaced by the slaving of B with q_0 = 0, lu_c the LU of omega D_e - (n - 1))
pub fn centre_series_s<T: Lane, const B: usize>(ctx: &Ctx<B>, de: &DMat<B>, lu_c: &DDLu, p1: &Fl<T, B>, c0: &Fl<T, B>, omega: T) -> [Vec<Fl<T, B>>; 3] {
    let rate = Rate::ZERO;
    let sig = series_scale(ctx.nodes[0]);
    let isig = 1.0 / sig;
    let n = dim::<T, B>(ctx);
    let n1 = n - T::P::one();
    let nh = n.mul_f(0.5);
    let kmax = ctx.kc;
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
    let mut pp = vec![z; kmax + 2];
    let mut mm = vec![z; kmax + 2];
    let mut cc = vec![z; kmax + 2];
    let mut bb = vec![z; kmax + 2];
    let mut ia = vec![z; kmax + 2];
    let mut q = vec![z; kmax + 2];
    let mut dmx = vec![z; kmax + 2];
    bb[0] = one;
    ia[0] = one;
    cc[0] = *c0;
    pp[1] = p1.mul_pwr2(sig);
    mm[1] = -pp[1];
    let invc = one / cc[0];
    let kap = -(one.mulp(n1));
    for k in 2..=kmax {
        let kf = k as f64;
        // p_k, m_k: model::centre_series's (they read c to order k - 1 only, as p_0 = m_0 = 0)
        let dp = crate::model::dtau(ctx, &pp[k - 1], omega, &rate).mul_pwr2(sig);
        let dm = crate::model::dtau(ctx, &mm[k - 1], omega, &rate).mul_pwr2(sig);
        let s1p = cs_w(&cc, &pp, k, 1, k - 1, |i| (k - i) as f64);
        let s1m = cs_w(&cc, &mm, k, 1, k - 1, |i| (k - i) as f64);
        let xpx = pp[k - 1].mulf(kf - 1.0).mul_pwr2(sig) - s1p;
        let xmx = mm[k - 1].mulf(kf - 1.0).mul_pwr2(sig) + s1m;
        let xcxp = cs_w(&cc, &pp, k, 1, k, |j| j as f64);
        let xcxm = cs_w(&cc, &mm, k, 1, k, |j| j as f64);
        let cp = cs(&cc, &pp, k);
        let cm = cs(&cc, &mm, k);
        let cpm = cp + cm;
        let r3 = (((dp + xpx) - xcxp) + cp) - cpm.mulp(nh);
        let r4 = (((dm + xmx) + xcxm) - cm) + cpm.mulp(nh);
        let mu11 = T::P::from_f64(1.0 - kf) - nh;
        let mu12 = -nh;
        let mu21 = nh;
        let mu22 = T::P::from_f64(kf - 1.0) + nh;
        let idel = T::P::one() / (-(T::P::from_f64(kf - 1.0) * (T::P::from_f64(kf - 1.0) + n)));
        pp[k] = ((r4.mulp(mu12) - r3.mulp(mu22)) * invc).mulp(idel);
        mm[k] = ((r3.mulp(mu21) - r4.mulp(mu11)) * invc).mulp(idel);
        // q = p^2 + m^2 - c (p^2 - m^2) / x; (p^2 - m^2) / x's scaled coefficient k is that of p^2 - m^2 at k + 1 over sigma
        dmx[k] = (cs(&pp, &pp, k + 1) - cs(&mm, &mm, k + 1)).mul_pwr2(isig);
        let cdx = cs(&cc, &dmx, k);
        let q2k = cs(&pp, &pp, k) + cs(&mm, &mm, k);
        q[k] = if ctx.mutate == FS_MUT_FLUX { q2k + cdx } else { q2k - cdx };
        bb[k] = ssolve(de, lu_c, &kap, &cs_range(&q, &bb, k, 1, k), omega);
        ia[k] = inv_at(&bb, &ia, k);
        cc[k] = cs_range(&cc, &ia, k, 0, k - 1).mulp(n1 * T::P::from_dd(recip_int(k)));
    }
    pp.truncate(kmax + 1);
    mm.truncate(kmax + 1);
    cc.truncate(kmax + 1);
    [pp, mm, cc]
}

/// Horizon-series coefficients of p, m, c in the scaled variable u / sigma (model::horizon_series's, a's recursion
/// replaced by the slaving of B: q_0 = 2 m_h^2, lu_h the LU of omega D_e - (n - 1) - q_0 at the values)
pub fn horizon_series_s<T: Lane, const B: usize>(ctx: &Ctx<B>, de: &DMat<B>, prep: &SPrep, m0: &Fl<T, B>, omega: T) -> [Vec<Fl<T, B>>; 3] {
    let rate = Rate::ZERO;
    let lu_h = prep.lu_h.as_ref().expect("form S: the horizon's slaving LU");
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
    let n1 = dim::<T, B>(ctx) - T::P::one();
    let nh = dim::<T, B>(ctx).mul_f(0.5);
    let ks = ctx.ks;
    let len = ks + 2;
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
    let mut pp = vec![z; len];
    let mut mm = vec![z; len];
    let mut cc = vec![z; len];
    let mut bb = vec![z; len];
    let mut ia = vec![z; len];
    let mut qs = vec![z; len];
    let mut p2 = vec![z; len];
    let mut m2 = vec![z; len];
    let mut dp = vec![z; len];
    let mut dm = vec![z; len];
    let mut pm = vec![z; len];
    let mut vv = vec![z; len];
    let mut ww = vec![z; len];
    let mut zc = vec![z; len];
    cc[0] = one;
    mm[0] = *m0;
    m2[0] = mm[0] * mm[0];
    dm[0] = crate::model::dtau(ctx, &mm[0], omega, &rate);
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
    // ww and zc are the scaled coefficients of x (x - c) and x (x + c) (model::horizon_series's); wr and zr those of
    // x - c and x + c alone, for q
    let mut wr = vec![z; len];
    let mut zr = vec![z; len];
    ww[0] = wfun(&cc, 0);
    zc[0] = zfun(&cc, 0);
    let izc0 = one / zc[0];
    let mut kap = z;
    let mut last = ks;
    for kk in 0..=ks {
        let kf = kk as f64;
        // x q = (x - c) p^2 + (x + c) m^2, x = 1 - u: q_kk = s_kk + sigma q_(kk-1) in the scaled variable; (x - c)_0 = 0,
        // so p^2 enters to order kk - 1
        wr[kk] = wfun(&cc, kk);
        zr[kk] = zfun(&cc, kk);
        let s = cs(&wr, &p2, kk) + cs(&zr, &m2, kk);
        qs[kk] = if kk > 0 { s + qs[kk - 1].mul_pwr2(sig) } else { s };
        if kk == 0 {
            kap = (-qs[0]).addp(-n1);
            bb[0] = ssolve(de, lu_h, &kap, &(-(one.mulp(n1))), omega);
        } else {
            bb[kk] = ssolve(de, lu_h, &kap, &cs_range(&qs, &bb, kk, 1, kk), omega);
        }
        ia[kk] = inv_at(&bb, &ia, kk);
        let ca1 = cs(&cc, &ia, kk) - cc[kk];
        cc[kk + 1] = mul_recip(ctx, cc[kk].mulf(kf) - ca1.mulp(n1), kk + 1).mul_pwr2(sig);
        ww[kk + 1] = wfun(&cc, kk + 1) - wfun(&cc, kk).mul_pwr2(sig);
        vv[kk] = cc[kk + 1].mulf((kk + 1) as f64).mul_pwr2(isig) - cc[kk].mulf(kf);
        zc[kk + 1] = zfun(&cc, kk + 1) - zfun(&cc, kk).mul_pwr2(sig);
        pm[kk] = pp[kk] + mm[kk];
        let wpu = cs_w(&pp, &ww, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let tau = if kk >= 1 { -dp[kk - 1].mul_pwr2(sig) } else { z };
        let r = (((tau - wpu) + cs(&vv, &pp, kk)) + cs(&cc, &pp, kk)) - cs(&cc, &pm, kk).mulp(nh);
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).addp(-nh);
        pp[kk] = psolve(ctx, &prep.h, kk, &kappa, &(-r), omega);
        dp[kk] = crate::model::dtau(ctx, &pp[kk], omega, &rate);
        p2[kk] = cs(&pp, &pp, kk);
        if kk == ks {
            last = kk;
            break;
        }
        let tau = if kk >= 1 { dm[kk] - dm[kk - 1].mul_pwr2(sig) } else { dm[kk] };
        pm[kk] = pp[kk] + mm[kk];
        let zmu = cs_w(&mm, &zc, kk + 1, 0, kk, |i| i as f64).mul_pwr2(isig);
        let rest = (((tau - zmu) - cs(&vv, &mm, kk)) - cs(&cc, &mm, kk)) + cs(&cc, &pm, kk).mulp(nh);
        mm[kk + 1] = mul_recip(ctx, rest * izc0, kk + 1).mul_pwr2(sig);
        dm[kk + 1] = crate::model::dtau(ctx, &mm[kk + 1], omega, &rate);
        m2[kk + 1] = cs(&mm, &mm, kk + 1);
    }
    pp.truncate(last + 1);
    mm.truncate(last + 1);
    cc.truncate(last + 1);
    [pp, mm, cc]
}

/// A constant of the Moebius series in the lane's own precision: its octo-double parts, applied as `mul_recip` applies
/// 1/k (the f64 and double-double lanes the leading pair, the quad- and octo-double lanes their own rounding, a jet
/// coefficient by coefficient). The series' constants (t_l, 1/(1 + a), h, e, z's coefficients) are not exact in
/// double-double, so a lane's double-double parameters (`Lane::P`) would cap them at 1e-32.
#[derive(Clone, Copy)]
struct Cst([f64; 8]);

impl Cst {
    fn of(v: OD) -> Self {
        Cst(v.x)
    }
    fn mul<T: Lane, const B: usize>(&self, x: Fl<T, B>) -> Fl<T, B> {
        let r = &self.0;
        let np = recip_lane_parts::<T>();
        if np <= 2 {
            return x.muls(DD { hi: r[0], lo: r[1] });
        }
        if T::NJ == 0 {
            return x.mul_t(recip_lane::<T>(r));
        }
        let mut y = x;
        if T::VAL_OD {
            let c = OV::splat(OD { x: *r });
            for (yb, xb) in y.0.iter_mut().zip(&x.0) {
                for i in 0..T::NJ {
                    yb.set_coef_od(i, xb.coef_od(i) * c);
                }
            }
        } else {
            let c = QV::splat(QD { x: [r[0], r[1], r[2], r[3]] });
            for (yb, xb) in y.0.iter_mut().zip(&x.0) {
                for i in 0..T::NJ {
                    yb.set_coef_qd(i, xb.coef_qd(i) * c);
                }
            }
        }
        y
    }
    /// x plus the constant
    fn add<T: Lane, const B: usize>(&self, x: Fl<T, B>) -> Fl<T, B> {
        x + Fl::<T, B>([recip_lane::<T>(&self.0); B])
    }
}

/// sum_{j=0..k} s_j y_{k-j}, s a series of constants
fn cvs<T: Lane, const B: usize>(s: &[Cst], y: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    let mut acc = Fl::<T, B>::zero();
    for j in (0..=k).rev() {
        acc = acc + s[j].mul(y[k - j]);
    }
    acc
}

/// Form S's horizon series in the Moebius variable t = (1 - z)/(1 + a z), z = x^2, a = ctx.hmob, scaled as tau =
/// t/t_l so that the last node x_l is at tau = 1 (#1090; solver/inner_large_d/mobius_s.py's FormS.horizon, whose
/// variable keeps the centre far from x_l, so the critical solution's series is no longer a minimal solution of its
/// recurrence). Fields U = p + m, V = (p - m)/x, X = B - 1, G = 2z c_z, with z d/dz = -h d/dt, h = (1 - t)(1 + at)/(1 + a),
/// z^2 d/dz = -e d/dt, e = (1 - t)^2/(1 + a):
///
///   A + B = 2h(c - 1) U_t + 2(hc - e) V_t + omega (U + zV)_th - G (U + V) + zV - (n - 1) c U = 0
///   A     = -2h (U_t - c V_t) + omega U_th - G V = 0,      c_t = -G/(2h),      G (1 + X) = -(n - 1) c X,
///   (omega D_e - (n - 1) - q) X = q,      q = (U^2 + z V^2)/2 - c U V.
///
/// Order k of A + B fixes P_k = U_k + V_k through omega D + (k + 1)(1 - G_0) - n/2, the u series' operator at order k
/// (c1 = -G_0), so `prep.h`'s LUs and `psolve` serve; order k of A fixes Q_(k+1) = U_(k+1) - V_(k+1); q_k needs only
/// Q_k; X's solves are `prep.lu_h`'s. Data Q_0 = 2 m_h, c_0 = 1. Returns p = (U + x_l V)/2, m = (U - x_l V)/2 and c at
/// x_l, the sums of orders 0..=ks. The series' constants are formed in octo-double and applied in the lane's precision
/// (`Cst`).
pub fn horizon_mob<T: Lane, const B: usize>(ctx: &Ctx<B>, de: &DMat<B>, prep: &SPrep, m0: &Fl<T, B>, omega: T) -> [Fl<T, B>; 3] {
    let rate = Rate::ZERO;
    let lu_h = prep.lu_h.as_ref().expect("form S: the horizon's slaving LU");
    let one_p = T::P::one();
    let n1 = dim::<T, B>(ctx) - one_p;
    let nh1 = dim::<T, B>(ctx).mul_f(0.5) - one_p;
    let xl = *ctx.nodes.last().expect("form S: nodes");
    let o1 = OD::ONE;
    let xo = OD::from_f64(xl);
    let zl = xo * xo;
    let ao = OD::from_f64(ctx.hmob);
    // t at x_l, the unit of tau
    let tso = (o1 - zl) / (o1 + ao * zl);
    let ioo = o1 / tso;
    let ia1o = o1 / (o1 + ao);
    let ts = Cst::of(tso);
    let hh = [Cst::of(ia1o), Cst::of((ao - o1) * ia1o * tso), Cst::of(-(ao * ia1o * tso * tso))];
    let hneg = [Cst::of(-ia1o), Cst::of(-((ao - o1) * ia1o * tso)), Cst::of(ao * ia1o * tso * tso)];
    let eneg = [Cst::of(-ia1o), Cst::of(OD::from_f64(2.0) * ia1o * tso), Cst::of(-(ia1o * tso * tso))];
    let two_h0 = Cst::of(OD::from_f64(2.0) * ia1o);
    let mts2 = Cst::of(-(tso * OD::from_f64(0.5)));
    let io = Cst::of(ioo);
    let two_io = Cst::of(OD::from_f64(2.0) * ioo);
    let m2io = Cst::of(-(OD::from_f64(2.0) * ioo));
    // 1/(2 io h_0) = (1 + a) t_l / 2
    let iq = Cst::of((o1 + ao) * tso * OD::from_f64(0.5));
    let ap = T::P::from_f64(ctx.hmob);
    let ks = ctx.ks;
    let len = ks + 3;
    // z(tau): 1, -(1 + a) t_l, then times -a t_l per order (FS_MUT_MOBIUS: a's sign flipped here only). A product zW
    // comes from (1 + a t_l tau) zW = (1 - t_l tau) W, zW_k = W_k - t_l W_(k-1) - a t_l zW_(k-1) (t2v's, #1090): O(1)
    // per order where the Cauchy sum with z's coefficients (FS_ZCAUCHY) is O(k)
    let amo = if ctx.mutate == FS_MUT_MOBIUS { -ao } else { ao };
    let zcauchy = ctx.mutate == FS_ZCAUCHY;
    let rzo = if ctx.mutate == FS_MUT_ZREC { (amo * tso) * (o1 + OD::from_f64(2f64.powi(-60))) } else { amo * tso };
    let rz = Cst::of(rzo);
    // the order-k coefficient of z W from W_k, W_(k-1) and z W's order k - 1
    let zrec = |w: &Vec<Fl<T, B>>, zw: &Vec<Fl<T, B>>, k: usize| -> Fl<T, B> {
        if k == 0 {
            w[0]
        } else {
            (w[k] - ts.mul(w[k - 1])) - rz.mul(zw[k - 1])
        }
    };
    let zs: Vec<Cst> = if zcauchy {
        let mut v = vec![Cst::of(o1); len];
        if len > 1 {
            let mut zj = -((o1 + amo) * tso);
            v[1] = Cst::of(zj);
            let r = -(amo * tso);
            for item in v.iter_mut().take(len).skip(2) {
                zj = zj * r;
                *item = Cst::of(zj);
            }
        }
        v
    } else {
        Vec::new()
    };
    let z = Fl::<T, B>::zero();
    let one = Fl::<T, B>::splat(DD::ONE);
    let mut uu = vec![z; len];
    let mut vv = vec![z; len];
    let mut xx = vec![z; len];
    let mut gg = vec![z; len];
    let mut cc = vec![z; len];
    let mut q = vec![z; len];
    let mut du = vec![z; len];
    let mut dv = vec![z; len];
    let mut qn = vec![z; len];
    let mut vvc = vec![z; len];
    let mut uvc = vec![z; len];
    let mut hcm = vec![z; len];
    let mut hce = vec![z; len];
    // z VV and z V, each order's final value (after P_k)
    let mut zvvc = vec![z; len];
    let mut zv = vec![z; len];
    // the Cauchy products (`Prod`): V V, U V, U U, c (U V), q X, c X, G X, hcm U', hce V', G U, G V, c U, c V'
    let mut pvv = Prod::<T, B>::new(ctx, len, false);
    let mut puv = Prod::<T, B>::new(ctx, len, false);
    let mut puu = Prod::<T, B>::new(ctx, len, false);
    let mut pcuv = Prod::<T, B>::new(ctx, len, false);
    let mut pqx = Prod::<T, B>::new(ctx, len, false);
    let mut pcx = Prod::<T, B>::new(ctx, len, false);
    let mut pgx = Prod::<T, B>::new(ctx, len, false);
    let mut phu = Prod::<T, B>::new(ctx, len, false);
    let mut phv = Prod::<T, B>::new(ctx, len, false);
    let mut pgu = Prod::<T, B>::new(ctx, len, false);
    let mut pgv = Prod::<T, B>::new(ctx, len, false);
    let mut pcu = Prod::<T, B>::new(ctx, len, true);
    let mut pcv = Prod::<T, B>::new(ctx, len, false);
    cc[0] = one;
    qn[0] = m0.mulf(2.0);
    let mut kap = z;
    let mut ib0 = z;
    for k in 0..=ks {
        let kf = k as f64;
        uu[k] = qn[k].mul_pwr2(0.5);
        vv[k] = -uu[k];
        vvc[k] = pvv.sum(&vv, &vv, k, 0, k);
        uvc[k] = puv.sum(&uu, &vv, k, 0, k);
        let zvv = if zcauchy { cvs(&zs, &vvc, k) } else { zrec(&vvc, &zvvc, k) };
        q[k] = (puu.sum(&uu, &uu, k, 0, k) + zvv).mul_pwr2(0.5) - pcuv.sum(&cc, &uvc, k, 0, k);
        if k == 0 {
            kap = (-q[0]).addp(-n1);
            xx[0] = ssolve(de, lu_h, &kap, &q[0], omega);
            ib0 = one / (one + xx[0]);
        } else {
            let rhs = q[k] + pqx.sum(&q, &xx, k, 1, k);
            xx[k] = ssolve(de, lu_h, &kap, &rhs, omega);
        }
        let gs = if k >= 1 {
            pcx.sum(&cc, &xx, k, 0, k).mulp(-n1) - pgx.sum(&gg, &xx, k, 0, k - 1)
        } else {
            pcx.sum(&cc, &xx, k, 0, k).mulp(-n1)
        };
        gg[k] = gs * ib0;
        let mut s = mts2.mul(gg[k]);
        for i in 1..=k.min(2) {
            s = s - hh[i].mul(cc[k - i + 1]).mulf((k - i + 1) as f64);
        }
        cc[k + 1] = mul_recip(ctx, s.mulp(one_p + ap), k + 1);
        let mut hc = hh[0].mul(cc[k]);
        if k >= 1 {
            hc = hc + hh[1].mul(cc[k - 1]);
        }
        if k >= 2 {
            hc = hc + hh[2].mul(cc[k - 2]);
        }
        hcm[k] = if k <= 2 { hneg[k].add(hc) } else { hc };
        hce[k] = if k <= 2 { eneg[k].add(hc) } else { hc };
        let deriv = if k >= 1 {
            du[k - 1] = uu[k].mulf(kf);
            dv[k - 1] = vv[k].mulf(kf);
            two_io.mul(phu.sum(&hcm, &du, k, 1, k) + phv.sum(&hce, &dv, k, 1, k))
        } else {
            z
        };
        let zvk = if zcauchy { cvs(&zs, &vv, k) } else { zrec(&vv, &zv, k) };
        let r = ((((deriv + crate::model::dtau(ctx, &(uu[k] + zvk), omega, &rate)) - pgu.sum(&gg, &uu, k, 0, k))
            - pgv.sum(&gg, &vv, k, 0, k))
            + zvk)
            - pcu.sum(&cc, &uu, k, 0, k).mulp(n1);
        // (k + 1)(1 - G_0) - n/2 as io k (2 h_0 c_1 + h_1 - e_1) - G_0 - (n/2 - 1), h_1 - e_1 = t_l
        let kappa = (io.mul(ts.add(two_h0.mul(cc[1]))).mulf(kf) - gg[0]).addp(-nh1);
        let pk = psolve(ctx, &prep.h, k, &kappa, &(-r), omega).mul_pwr2(0.5);
        uu[k] = uu[k] + pk;
        vv[k] = vv[k] + pk;
        if k >= 1 {
            du[k - 1] = uu[k].mulf(kf);
            dv[k - 1] = vv[k].mulf(kf);
        }
        vvc[k] = pvv.sum(&vv, &vv, k, 0, k);
        uvc[k] = puv.sum(&uu, &vv, k, 0, k);
        if !zcauchy {
            zvvc[k] = zrec(&vvc, &zvvc, k);
            zv[k] = zrec(&vv, &zv, k);
        }
        du[k] = z;
        dv[k] = z;
        let mut ak = z;
        for i in 0..=k.min(2) {
            let mm = k - i;
            ak = ak + hh[i].mul(du[mm] - pcv.sum(&cc, &dv, mm, 0, mm));
        }
        let ak = (m2io.mul(ak) + crate::model::dtau(ctx, &uu[k], omega, &rate)) - pgv.sum(&gg, &vv, k, 0, k);
        qn[k + 1] = mul_recip(ctx, iq.mul(ak), k + 1);
        // the end of step k: every series final to order k - 1 (U' and V' too); the products' squares on threads
        if pvv.forms(k) {
            rayon::scope(|sc| {
                sc.spawn(|_| pvv.step(&vv, &vv, k));
                sc.spawn(|_| puv.step(&uu, &vv, k));
                sc.spawn(|_| puu.step(&uu, &uu, k));
                sc.spawn(|_| pcuv.step(&cc, &uvc, k));
                sc.spawn(|_| pqx.step(&q, &xx, k));
                sc.spawn(|_| pcx.step(&cc, &xx, k));
                sc.spawn(|_| pgx.step(&gg, &xx, k));
                sc.spawn(|_| phu.step(&hcm, &du, k));
                sc.spawn(|_| phv.step(&hce, &dv, k));
                sc.spawn(|_| pgu.step(&gg, &uu, k));
                sc.spawn(|_| pgv.step(&gg, &vv, k));
                sc.spawn(|_| pcu.step(&cc, &uu, k));
                sc.spawn(|_| pcv.step(&cc, &dv, k));
            });
        } else {
            for (p, x, y) in [
                (&mut pvv, &vv, &vv),
                (&mut puv, &uu, &vv),
                (&mut puu, &uu, &uu),
                (&mut pcuv, &cc, &uvc),
                (&mut pqx, &q, &xx),
                (&mut pcx, &cc, &xx),
                (&mut pgx, &gg, &xx),
                (&mut phu, &hcm, &du),
                (&mut phv, &hce, &dv),
                (&mut pgu, &gg, &uu),
                (&mut pgv, &gg, &vv),
                (&mut pcu, &cc, &uu),
                (&mut pcv, &cc, &dv),
            ] {
                p.step(x, y, k);
            }
        }
    }
    let (mut us, mut vs, mut csum) = (z, z, z);
    for k in (0..=ks).rev() {
        us = us + uu[k];
        vs = vs + vv[k];
        csum = csum + cc[k];
    }
    let xv = vs.mulf(xl);
    [(us + xv).mul_pwr2(0.5), (us - xv).mul_pwr2(0.5), csum]
}

/// The preparation of one form-S evaluation: the horizon LUs of `model::horizon_lus` at a_h = B_0^-1/2, B_0 slaved
/// from the state's double-double m_h, and the slaving LUs at the horizon and the centre
#[derive(Default)]
pub struct SPrep {
    pub h: HPrep,
    pub lu_h: Option<DDLu>,
    pub lu_c: Option<DDLu>,
}

/// B_0 = 1/a_h^2 at the horizon from the double-double m_h and Delta: (omega D_e - (n - 1) - 2 m_h^2) B_0 = -(n - 1),
/// and that operator's LU
pub fn horizon_b0<const B: usize>(ctx: &Ctx<B>, mh: &Fl<DV, B>, delta: DD) -> Result<(Vec<DD>, DDLu), String> {
    let de = ctx.de.as_ref().ok_or("form S needs the even-sector d/dtheta (Solver.set_dmat_even)")?;
    let m = 8 * B;
    let omega = DD::TWO_PI / delta;
    let n1 = ctx.n - DD::ONE;
    let kap: Vec<DD> = (0..m)
        .map(|i| {
            let v = mh.get(i);
            -(n1 + (v * v).mul_f(2.0))
        })
        .collect();
    let lu = slave_lu(de, omega, &kap)?;
    let mut b = vec![-n1; m];
    lu.solve(&mut b);
    Ok((b, lu))
}

pub struct DssS<'a, const B: usize>(pub &'a Ctx<B>);

impl<'a, const B: usize> DssS<'a, B> {
    fn de(&self) -> &'a DMat<B> {
        self.0.de.as_ref().expect("form S needs the even-sector d/dtheta (Solver.set_dmat_even)")
    }
}

impl<const B: usize> Equations<B> for DssS<'_, B> {
    const NF: usize = 3;
    const NL: usize = 2;
    const NR: usize = 1;
    const NG: usize = 1;
    const NAUX: usize = NAUX_S;
    type Prep = SPrep;

    fn prepare(&self, zh: &[f64], zl: &[f64]) -> SPrep {
        self.try_prepare(zh, zl).expect("form S: the preparation")
    }

    fn try_prepare(&self, zh: &[f64], zl: &[f64]) -> Result<SPrep, String> {
        let ctx = self.0;
        let de = ctx.de.as_ref().ok_or("form S needs the even-sector d/dtheta (Solver.set_dmat_even)")?;
        let delta = delta_of(zh, zl);
        let mh = ctx.fld(zh, zl, ctx.off_mh_s());
        let (b0, lu_h) = horizon_b0(ctx, &mh, delta)?;
        let m = 8 * B;
        let (ah_h, ah_l): (Vec<f64>, Vec<f64>) = b0
            .iter()
            .map(|v| {
                let a = DD::ONE / v.sqrt();
                (a.hi, a.lo)
            })
            .unzip();
        let ah = Fl::<DV, B>::load(&ah_h, &ah_l);
        let lus = horizon_lus(ctx, &ah, delta, true, DD::ZERO)?;
        let n1 = ctx.n - DD::ONE;
        let lu_c = slave_lu(de, DD::TWO_PI / delta, &vec![-n1; m])?;
        Ok(SPrep { h: HPrep::of(lus), lu_h: Some(lu_h), lu_c: Some(lu_c) })
    }

    fn try_prepare_rate(&self, zh: &[f64], zl: &[f64], rate: Rate) -> Result<SPrep, String> {
        if !rate.is_zero() {
            return Err("form S: no mode matrix".to_string());
        }
        self.try_prepare(zh, zl)
    }

    fn taylor_coeffs<T: Lane>(&self, x0: f64, g: &[T], w: &mut Work<T, B>) {
        taylor_coeffs_s(self.0, self.de(), x0, omega_of(self.0, g[0]), w);
    }

    fn left_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], prep: &SPrep) -> multishoot::State<T, B> {
        let ctx = self.0;
        let s = centre_series_s(ctx, self.de(), prep.lu_c.as_ref().expect("form S: the centre's LU"), &data[0], &data[1], omega_of(ctx, g[0]));
        let x = DD::from_f64(ctx.nodes[0]).mul_pwr2(1.0 / series_scale(ctx.nodes[0]));
        vec![horner(&s[0], x), horner(&s[1], x), horner(&s[2], x)]
    }

    fn right_state<T: Lane>(&self, _x: f64, data: &[Fl<T, B>], g: &[T], prep: &SPrep) -> multishoot::State<T, B> {
        let ctx = self.0;
        if ctx.hmob > 0.0 {
            let [p, m, c] = horizon_mob(ctx, self.de(), prep, &data[0], omega_of(ctx, g[0]));
            return vec![p, m, c];
        }
        let s = horizon_series_s(ctx, self.de(), prep, &data[0], omega_of(ctx, g[0]));
        let u = ctx.u0().mul_pwr2(1.0 / series_scale(ctx.u0().hi));
        vec![horner(&s[0], u), horner(&s[1], u), horner(&s[2], u)]
    }
}

/// form S's state carried by Taylor steps from the last node through the points xs (xs[0] the last node's x), in P:
/// (p, m, c) at the points whose index is in rec, as P::PARTS part arrays of rec.len() x 3M values (record-major). The
/// globals (Delta) and the last node's fields are z's; the steps are xs's consecutive points (the caller's plan).
pub fn path_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], xs: &[f64], rec: &[usize]) -> Result<Vec<Vec<f64>>, String> {
    let sh = ctx.shooting_s();
    sh.check_z::<P>(z)?;
    let m = Ctx::<B>::M;
    let o = ctx.off_node_s(ctx.k() - 1);
    let s0: Vec<Fl<P::V, B>> = (0..3).map(|f| sh.load::<P>(z, o + f * m)).collect();
    let g: Vec<P::V> = sh.globals::<P>(z).into_iter().map(P::splat).collect();
    let eq = DssS(ctx);
    let mut w = Work::<P::V, B>::new(3, NAUX_S, ctx.kt);
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| Vec::with_capacity(rec.len() * 3 * m)).collect();
    let mut s = s0;
    let push = |s: &Vec<Fl<P::V, B>>, out: &mut Vec<Vec<f64>>| {
        for f in s.iter() {
            for i in 0..m {
                let v = P::field_get(f, i);
                for (q, o) in out.iter_mut().enumerate() {
                    o.push(P::part(v, q));
                }
            }
        }
    };
    let mut next = 0;
    if rec.first() == Some(&0) {
        push(&s, &mut out);
        next = 1;
    }
    for i in 0..xs.len().saturating_sub(1) {
        s = multishoot::taylor::taylor_step(&eq, &s, xs[i], xs[i + 1], &g, &mut w);
        while next < rec.len() && rec[next] == i + 1 {
            push(&s, &mut out);
            next += 1;
        }
    }
    Ok(out)
}

/// form S's horizon series at the last node in P (`DssS::right_state`, p, m and c at x_l from z's m_h and Delta), as
/// P::PARTS part arrays of 3M values (field-major): the horizon block of the residual without the Taylor steps, for
/// hp/form_s_relaxed.py's checks and timings
pub fn horizon_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]]) -> Result<Vec<Vec<f64>>, String> {
    let sh = ctx.shooting_s();
    sh.check_z::<P>(z)?;
    let m = Ctx::<B>::M;
    let mh = sh.load::<P>(z, ctx.off_mh_s());
    let g: Vec<P::V> = sh.globals::<P>(z).into_iter().map(P::splat).collect();
    let eq = DssS(ctx);
    let prep = eq.try_prepare(z[0], z[1])?;
    let x = *ctx.nodes.last().ok_or("form S: nodes")?;
    let s = eq.right_state(x, &[mh], &g, &prep);
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| Vec::with_capacity(3 * m)).collect();
    for f in s.iter() {
        for i in 0..m {
            let v = P::field_get(f, i);
            for (q, o) in out.iter_mut().enumerate() {
                o.push(P::part(v, q));
            }
        }
    }
    Ok(out)
}

/// form S's J[rows, cols] in P (multishoot's jacobian_sub: only the column tasks whose block meets rows; projected
/// rows supported), P::PARTS part arrays of rows.len() x cols.len(), row-major
pub fn jacobian_sub_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], rows: &[usize], cols: &[usize], par: bool) -> Result<Vec<Vec<f64>>, String> {
    ctx.shooting_s().jacobian_sub::<P>(z, rows, cols, par)
}

/// form S's residual in P, one scalar per row
pub fn residual_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
    ctx.shooting_s().residual::<P>(z, par)
}

/// form S's exact Jacobian in P (PARTS column-major part arrays with leading dimension ld, ld)
pub fn jacobian_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
    ctx.shooting_s().jacobian::<P>(z, par)
}

/// form S's Newton step in P
pub fn newton_s_in<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<multisolver::newton::Step<P>, String> {
    ctx.shooting_s().newton_step::<P>(z, par)
}

/// form S's octo-double exact Jacobian with n tangents per evaluation (1 or 2; model::jacobian_o_tangents's)
pub fn jacobian_s_o<const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool, n: usize) -> Result<(Vec<Vec<f64>>, usize), String> {
    use multidouble::precision::Octo;
    let sh = ctx.shooting_s();
    match n {
        1 => sh.jacobian::<Octo>(z, par),
        2 => sh.jacobian_tangents::<Octo, multidouble::native::Oun<2>>(z, par),
        _ => Err(format!("form S: tangents must be 1 or 2, not {n}")),
    }
}

/// form S's layout as a problem whose octo-double Jacobian takes n tangents per evaluation
struct TangentJacS<'a, const B: usize> {
    sh: &'a Shooting<DssS<'a, B>, B>,
    ctx: &'a Ctx<B>,
    n: usize,
}

impl<const B: usize> multisolver::Problem<multidouble::precision::Octo> for TangentJacS<'_, B> {
    fn len(&self) -> usize {
        self.sh.len()
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<multidouble::od::OD>, String> {
        self.sh.residual::<multidouble::precision::Octo>(z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        jacobian_s_o(self.ctx, z, par, self.n)
    }
}

/// form S's octo-double Newton step with the Jacobian of `jacobian_s_o`
pub fn newton_s_o<const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool, n: usize) -> Result<multisolver::newton::Step<multidouble::precision::Octo>, String> {
    let sh = ctx.shooting_s();
    if n == 1 {
        return sh.newton_step::<multidouble::precision::Octo>(z, par);
    }
    multisolver::newton::newton_step::<multidouble::precision::Octo, _>(&TangentJacS { sh: &sh, ctx, n }, z, par)
}
