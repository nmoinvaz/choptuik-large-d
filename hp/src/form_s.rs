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
use crate::model::{delta_of, dim, horizon_lus, kfreeze_of, mul_recip, omega_of, psolve, recip_int, series_scale, Ctx, HPrep};
use crate::qv::QV;
use multidouble::dual::Param;
use multidouble::native::OV;
use multidouble::simd::Native;
use multidouble::precision::Precision;
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

/// sum_{j=0..k} s_j y_{k-j}, s a scalar series
fn cvs<T: Lane, const B: usize>(s: &[T::P], y: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    let mut acc = Fl::<T, B>::zero();
    for j in (0..=k).rev() {
        acc = acc + y[k - j].mulp(s[j]);
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
/// x_l, the sums of orders 0..=ks.
pub fn horizon_mob<T: Lane, const B: usize>(ctx: &Ctx<B>, de: &DMat<B>, prep: &SPrep, m0: &Fl<T, B>, omega: T) -> [Fl<T, B>; 3] {
    let rate = Rate::ZERO;
    let lu_h = prep.lu_h.as_ref().expect("form S: the horizon's slaving LU");
    let one_p = T::P::one();
    let n1 = dim::<T, B>(ctx) - one_p;
    let nh1 = dim::<T, B>(ctx).mul_f(0.5) - one_p;
    let xl = *ctx.nodes.last().expect("form S: nodes");
    let zl = T::P::from_f64(xl) * T::P::from_f64(xl);
    let ap = T::P::from_f64(ctx.hmob);
    // t at x_l, the unit of tau
    let ts = (one_p - zl) / (one_p + ap * zl);
    let io = one_p / ts;
    let ia1 = one_p / (one_p + ap);
    let hh = [ia1, (ap - one_p) * ia1 * ts, -(ap * ia1 * ts * ts)];
    let ee = [ia1, (ia1 * ts).mul_f(-2.0), ia1 * ts * ts];
    let ks = ctx.ks;
    let len = ks + 3;
    // z(tau): 1, -(1 + a) t_l, then times -a t_l per order (FS_MUT_MOBIUS: a's sign flipped here only)
    let am = if ctx.mutate == FS_MUT_MOBIUS { -ap } else { ap };
    let mut zs = vec![one_p; len];
    if len > 1 {
        zs[1] = -((one_p + am) * ts);
        let r = -(am * ts);
        for j in 2..len {
            zs[j] = zs[j - 1] * r;
        }
    }
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
    cc[0] = one;
    qn[0] = m0.mulf(2.0);
    let mut kap = z;
    let mut ib0 = z;
    let two_io = io.mul_f(2.0);
    // 1/(2 io h_0) = (1 + a) t_l / 2
    let iq = (one_p + ap) * ts.mul_f(0.5);
    for k in 0..=ks {
        let kf = k as f64;
        uu[k] = qn[k].mul_pwr2(0.5);
        vv[k] = -uu[k];
        vvc[k] = cs(&vv, &vv, k);
        uvc[k] = cs(&uu, &vv, k);
        q[k] = (cs(&uu, &uu, k) + cvs(&zs, &vvc, k)).mul_pwr2(0.5) - cs(&cc, &uvc, k);
        if k == 0 {
            kap = (-q[0]).addp(-n1);
            xx[0] = ssolve(de, lu_h, &kap, &q[0], omega);
            ib0 = one / (one + xx[0]);
        } else {
            let rhs = q[k] + cs_range(&q, &xx, k, 1, k);
            xx[k] = ssolve(de, lu_h, &kap, &rhs, omega);
        }
        let gs = if k >= 1 { cs(&cc, &xx, k).mulp(-n1) - cs_range(&gg, &xx, k, 0, k - 1) } else { cs(&cc, &xx, k).mulp(-n1) };
        gg[k] = gs * ib0;
        let mut s = gg[k].mulp(ts.mul_f(-0.5));
        for i in 1..=k.min(2) {
            s = s - cc[k - i + 1].mulp(hh[i].mul_f((k - i + 1) as f64));
        }
        cc[k + 1] = mul_recip(ctx, s.mulp(one_p + ap), k + 1);
        let mut hc = cc[k].mulp(hh[0]);
        if k >= 1 {
            hc = hc + cc[k - 1].mulp(hh[1]);
        }
        if k >= 2 {
            hc = hc + cc[k - 2].mulp(hh[2]);
        }
        hcm[k] = if k <= 2 { hc.addp(-hh[k]) } else { hc };
        hce[k] = if k <= 2 { hc.addp(-ee[k]) } else { hc };
        let deriv = if k >= 1 {
            du[k - 1] = uu[k].mulf(kf);
            dv[k - 1] = vv[k].mulf(kf);
            (cs_range(&hcm, &du, k, 1, k) + cs_range(&hce, &dv, k, 1, k)).mulp(two_io)
        } else {
            z
        };
        let zvk = cvs(&zs, &vv, k);
        let r = ((((deriv + crate::model::dtau(ctx, &(uu[k] + zvk), omega, &rate)) - cs(&gg, &uu, k)) - cs(&gg, &vv, k)) + zvk)
            - cs(&cc, &uu, k).mulp(n1);
        let kappa = ((cc[1].mulp(hh[0].mul_f(2.0)).addp(hh[1] - ee[1])).mulp(io.mul_f(kf)) - gg[0]).addp(-nh1);
        let pk = psolve(ctx, &prep.h, k, &kappa, &(-r), omega).mul_pwr2(0.5);
        uu[k] = uu[k] + pk;
        vv[k] = vv[k] + pk;
        if k >= 1 {
            du[k - 1] = uu[k].mulf(kf);
            dv[k - 1] = vv[k].mulf(kf);
        }
        vvc[k] = cs(&vv, &vv, k);
        uvc[k] = cs(&uu, &vv, k);
        du[k] = z;
        dv[k] = z;
        let mut ak = z;
        for i in 0..=k.min(2) {
            let mm = k - i;
            ak = ak + (du[mm] - cs(&cc, &dv, mm)).mulp(hh[i]);
        }
        let ak = (ak.mulp(-two_io) + crate::model::dtau(ctx, &uu[k], omega, &rate)) - cs(&gg, &vv, k);
        qn[k + 1] = mul_recip(ctx, ak.mulp(iq), k + 1);
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
