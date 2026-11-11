//! The Newton step in quad-double (notes/QUAD_DOUBLE.md, section 12): the residual with every block in quad-double,
//! its exact Jacobian by forward-mode duals over quad-double lanes (multidouble's `Qu`), the dense quad-double LU
//! (`QDLu`), and the step -J^-1 F in quad-double. The state comes in four parts (z[0] + z[1] + z[2] + z[3], as
//! multidouble's QD). Every operation on the state is quad-double, the horizon solves refined to quad-double
//! (model::psolve, `Lane::VAL_QD`); the model's constants stay double-double, so the discrete system is the one the
//! double-double solver solves. Near and above D = 60 the double-double Jacobian's rounding, u sigma_max with
//! u = 2^-104, exceeds its smallest singular value (README, "Why the solver stalls"); here u is 2^-209.
//!
//! ```text
//! cargo test --release --lib qsolve_    on the synthetic context of qd_residual::synth: the quad-double residual and
//!     Jacobian against the double-double ones; the Jacobian against central differences of the quad-double residual
//!     (control: the horizon solves unrefined); the refinement converged; a planted root found to quad-double; and a
//!     perturbation of 2^-180, below double-double resolution, seen by the residual and undone by Newton (control: the
//!     state read in double-double)
//! ```
use crate::field::Fl;
use crate::linalg::QDLu;
use crate::model::Ctx;
use crate::qd::QD;
use crate::qv::QV;
use crate::xsolve;
use multidouble::precision::Quad;
#[cfg(test)]
use crate::{dd::DD, qd::QdAcc};

/// `Ctx::mutate` for qsolve (the control of its precision checks): the state read in its double-double part only
pub const QD_MUT_STATE_DD: u8 = 9;

/// A field from the four parts of a quad-double state (only the first two with QD_MUT_STATE_DD)
pub fn fld_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], off: usize) -> Fl<QV, B> {
    xsolve::fld::<Quad, B>(ctx, &z, off)
}

/// The residual with every block in quad-double, one QD per row. Each block reads its inputs, Delta and its target
/// in all four parts; the freeze projection and the phase row are quad-double too.
pub fn residual_parts<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<Vec<QD>, String> {
    xsolve::residual_parts::<Quad, B>(ctx, &z, par)
}

/// `residual_parts` with each row rounded to double-double (F is small, so the rounding is relative to F): (F hi,
/// F lo)
pub fn residual_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
    let f = residual_parts(ctx, z, par)?;
    Ok(f.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip())
}

/// The residual on a jet state z(h) = sum_k z_k h^k in quad-double (`QJet<N>` lanes), with the dimension n + h as
/// the jet variable (xsolve::residual_jet at `Quad`): each z_k in four parts, at most N of them, F's coefficients of
/// orders 0..nout-1 (nout <= N), one QD per row. The D-dependent constants are formed in the jet's parameter
/// arithmetic (`QSeries<N>`), so order 0 is residual_parts' residual up to that rounding.
pub fn residual_jet_q<const N: usize, const B: usize>(ctx: &Ctx<B>, zs: &[[&[f64]; 4]], par: bool, nout: usize) -> Result<Vec<Vec<QD>>, String> {
    let zs: Vec<Vec<&[f64]>> = zs.iter().map(|z| z.to_vec()).collect();
    xsolve::residual_jet::<Quad, N, B>(ctx, &zs, par, nout, "residual_jet_q")
}

/// The exact Jacobian dF/dz in quad-double by forward-mode duals (`Qu` lanes, one evaluation per column), assembled
/// as model::jacobian does: column-major with leading dimension ld_for(L), part q of entry (i, j) at c[q][j ld + i].
pub fn jacobian_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<([Vec<f64>; 4], usize), String> {
    let (c, ld) = xsolve::jacobian::<Quad, B>(ctx, &z, par)?;
    Ok((c.try_into().expect("four parts"), ld))
}

/// A quad-double Newton step: dz = -J^-1 F (four parts), F (hi, lo), the LU, and the seconds of the residual, the
/// Jacobian, the factorization and the solve.
pub struct QStep {
    pub dz: [Vec<f64>; 4],
    pub fh: Vec<f64>,
    pub fl: Vec<f64>,
    pub lu: QDLu,
    pub t: [f64; 4],
}

/// The Newton step in quad-double at z
pub fn newton_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<QStep, String> {
    let s = xsolve::newton::<Quad, B>(ctx, &z, par)?;
    Ok(QStep { dz: s.dz.try_into().expect("four parts"), fh: s.fh, fl: s.fl, lu: s.lu, t: s.t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{jacobian, residual, residual_jet, JET_MUT_FREEZE_N, QD_MUT_NOREFINE, QD_MUT_REFINE4};
    use crate::qv::axpy_parts;
    use crate::qd_residual::synth;

    fn parts(zh: &[f64], zl: &[f64]) -> [Vec<f64>; 4] {
        [zh.to_vec(), zl.to_vec(), vec![0.0; zh.len()], vec![0.0; zh.len()]]
    }

    fn view(z: &[Vec<f64>; 4]) -> [&[f64]; 4] {
        [&z[0], &z[1], &z[2], &z[3]]
    }

    fn qd_at(c: &[Vec<f64>; 4], k: usize) -> QD {
        QD { x: std::array::from_fn(|q| c[q][k]) }
    }

    /// z + s v in quad-double, v = e_j (j < L) or v_i = ((i mod 7) - 3) / 3 (j = usize::MAX), Delta excluded
    fn shifted(z: &[Vec<f64>; 4], j: usize, s: f64) -> [Vec<f64>; 4] {
        let l = z[0].len();
        let mut d: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; l]);
        if j == usize::MAX {
            for i in 0..l - 1 {
                d[0][i] = s * ((i % 7) as f64 - 3.0) / 3.0;
            }
        } else {
            d[0][j] = s;
        }
        axpy_parts(1.0, view(&d), view(z))
    }

    /// the quad-double residual and Jacobian at a double-double state agree with the double-double ones to about
    /// their rounding: max |difference| / max(1, |entry|)
    #[test]
    fn qsolve_agrees_with_double_double() {
        let (ctx, zh, zl) = synth();
        let z = parts(&zh, &zl);
        let (fh, fl) = residual(&ctx, &zh, &zl, false).unwrap();
        let (gh, gl) = residual_q(&ctx, view(&z), false).unwrap();
        let fs = fh.iter().map(|v| v.abs()).fold(1.0, f64::max);
        let df = (0..fh.len()).map(|i| ((DD::new(fh[i], fl[i]) - DD::new(gh[i], gl[i])).to_f64()).abs()).fold(0.0, f64::max) / fs;
        let (jh, jl, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
        let (c, ldq) = jacobian_q(&ctx, view(&z), false).unwrap();
        assert_eq!(ld, ldq);
        let js = jh.iter().map(|v| v.abs()).fold(1.0, f64::max);
        let dj = (0..jh.len()).map(|k| (qd_at(&c, k) - QD::from_dd(DD::new(jh[k], jl[k]))).to_f64().abs()).fold(0.0, f64::max) / js;
        println!("quad-double vs double-double: residual {df:.1e}, Jacobian {dj:.1e} (relative to the largest entry)");
        assert!(df < 1e-28 && dj < 1e-28, "residual {df:e}, Jacobian {dj:e}");
    }

    /// max over columns j of |J_q e_j - (F(z + h e_j) - F(z - h e_j)) / 2h| / max|J|, F the quad-double residual
    /// (residual_parts), h = 2^-70: the differences' rounding, about 2^-209 G / h, and their truncation, h^2 times
    /// F's third derivative, are both near 1e-42, far below a double-double Jacobian's error (about 1e-32)
    fn jac_vs_difference(mutate: u8) -> f64 {
        let (mut ctx, zh, zl) = synth();
        ctx.mutate = mutate;
        let l = ctx.len();
        let z = parts(&zh, &zl);
        let (c, ld) = jacobian_q(&ctx, view(&z), false).unwrap();
        ctx.mutate = 0;
        let js = (0..l).map(|j| (0..l).map(|i| c[0][j * ld + i].abs()).fold(0.0, f64::max)).fold(1.0, f64::max);
        let h = 2f64.powi(-70);
        let mut worst = 0.0f64;
        for j in (0..l).step_by(7).chain([l - 1]) {
            let fp = residual_parts(&ctx, view(&shifted(&z, j, h)), false).unwrap();
            let fm = residual_parts(&ctx, view(&shifted(&z, j, -h)), false).unwrap();
            for i in 0..l {
                let fd = (fp[i] - fm[i]).mul_pwr2(1.0 / (2.0 * h));
                worst = worst.max((qd_at(&c, j * ld + i) - fd).to_f64().abs() / js);
            }
        }
        worst
    }

    #[test]
    fn qsolve_jacobian_matches_differences() {
        let e = jac_vs_difference(0);
        let ec = jac_vs_difference(QD_MUT_NOREFINE);
        println!("Jacobian vs central differences of the quad-double residual: {e:.1e}; horizon solves unrefined: {ec:.1e}");
        assert!(e < 1e-38, "{e:e}");
        assert!(ec > 1e3 * e, "control: unrefined horizon solves not seen ({ec:e} vs {e:e})");
    }

    /// two refinement steps (three for tangents) give the four-step Jacobian to quad-double rounding
    #[test]
    fn qsolve_refinement_converged() {
        let (mut ctx, zh, zl) = synth();
        let z = parts(&zh, &zl);
        let (a, _) = jacobian_q(&ctx, view(&z), false).unwrap();
        ctx.mutate = QD_MUT_REFINE4;
        let (b, _) = jacobian_q(&ctx, view(&z), false).unwrap();
        let js = a[0].iter().map(|v| v.abs()).fold(1.0, f64::max);
        let d = (0..a[0].len()).map(|k| (qd_at(&a, k) - qd_at(&b, k)).to_f64().abs()).fold(0.0, f64::max) / js;
        println!("Jacobian, default refinement vs four steps: {d:.1e}");
        assert!(d < 1e-58, "{d:e}");
    }

    /// Newton in quad-double on the planted system G(z) = F(z) - F(z*), whose root is the synthetic state z*, from
    /// z* + s v: the errors max |z - z*| after each step
    fn planted_newton(mutate: u8, s: f64, steps: usize) -> Vec<f64> {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let zs = parts(&zh, &zl);
        let f0 = residual_parts(&ctx, view(&zs), false).unwrap();
        ctx.mutate = mutate;
        let mut z = shifted(&zs, usize::MAX, s);
        let mut errs = Vec::new();
        for _ in 0..steps {
            errs.push((0..l).map(|i| (qd_at(&z, i) - qd_at(&zs, i)).to_f64().abs()).fold(0.0, f64::max));
            let f = residual_parts(&ctx, view(&z), false).unwrap();
            let (c, ld) = jacobian_q(&ctx, view(&z), false).unwrap();
            let lu = QDLu::factor(l, ld, c, false).unwrap();
            let mut b: Vec<QD> = (0..l).map(|i| f0[i] - f[i]).collect();
            lu.solve(&mut b);
            let dz: [Vec<f64>; 4] = std::array::from_fn(|q| b.iter().map(|v| v.x[q]).collect());
            z = axpy_parts(1.0, view(&dz), view(&z));
        }
        errs.push((0..l).map(|i| (qd_at(&z, i) - qd_at(&zs, i)).to_f64().abs()).fold(0.0, f64::max));
        errs
    }

    fn show(v: &[f64]) -> String {
        v.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(", ")
    }

    /// from 1e-6 away, quadratic convergence to the planted root within quad-double (errs[0] is the start's error)
    #[test]
    fn qsolve_newton_converges() {
        let errs = planted_newton(0, 1e-6, 5);
        println!("planted root from 1e-6, |z - z*| per step: {}", show(&errs));
        assert!(errs[4] < 1e-50 && errs[5] < 1e-58, "{errs:?}");
    }

    /// Precision: z* moved by 2^-180 v, a change double-double cannot represent (its resolution is 2^-106 of the
    /// entries). The quad-double residual sees it, F(z* + d) - F(z*) = J d to within 1e-6 relative, and one Newton
    /// step undoes it to below 2^-200. Control: the state read in its double-double part (QD_MUT_STATE_DD) sees no
    /// change at all, and the check fails.
    fn sub_dd_response(mutate: u8) -> (f64, f64) {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let zs = parts(&zh, &zl);
        ctx.mutate = mutate;
        let s = 2f64.powi(-180);
        let zp = shifted(&zs, usize::MAX, s);
        let f0 = residual_parts(&ctx, view(&zs), false).unwrap();
        let f1 = residual_parts(&ctx, view(&zp), false).unwrap();
        let (c, ld) = jacobian_q(&ctx, view(&zs), false).unwrap();
        let v: Vec<QD> = (0..l).map(|i| if i < l - 1 { QD::from_f64(((i % 7) as f64 - 3.0) / 3.0) } else { QD::ZERO }).collect();
        let mut worst = 0.0f64;
        let mut big = 0.0f64;
        for i in 0..l {
            let jv = (0..l).fold(QdAcc::zero(), |a, j| a.add_prod(qd_at(&c, j * ld + i), v[j])).finish().mul_pwr2(s);
            big = big.max(jv.to_f64().abs());
            worst = worst.max((f1[i] - f0[i] - jv).to_f64().abs());
        }
        let errs = planted_newton(mutate, s, 1);
        (worst / big, errs[1] / s)
    }

    #[test]
    fn qsolve_resolves_sub_double_double_changes() {
        let (resp, undo) = sub_dd_response(0);
        let (resp_c, undo_c) = sub_dd_response(QD_MUT_STATE_DD);
        println!("change of 2^-180: response error {resp:.1e} of J d, Newton leaves {undo:.1e} of it; state read in double-double: {resp_c:.1e}, {undo_c:.1e}");
        assert!(resp < 1e-6 && undo < 2f64.powi(-20), "response {resp:e}, left {undo:e}");
        assert!(resp_c > 0.5 && undo_c > 0.5, "control: the double-double state saw the change ({resp_c:e}, {undo_c:e})");
    }

    /// the direction of `shifted` (j = usize::MAX) times s, in four parts
    fn direction(l: usize, s: f64) -> [Vec<f64>; 4] {
        shifted(&std::array::from_fn(|_| vec![0.0; l]), usize::MAX, s)
    }

    /// F(h) = sum_k F_k h^k of the quad-double jet (N = 24) on the state jet z_0 + h v, v = direction(1), against
    /// residual_parts at (z_0 + h v, n + h) for h = 2^-6, 2^-7, 2^-8 (n + h exact in f64): max |difference| /
    /// max(1, |F|). The difference is the truncation, which falls as h^24 while it exceeds the rounding, and
    /// residual_parts' D-dependent constants rounded to double-double.
    fn jet_q_vs_residual(mutate: u8) -> [f64; 3] {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let n0 = ctx.n.hi;
        let z0 = parts(&zh, &zl);
        let v = direction(l, 1.0);
        ctx.mutate = mutate;
        let fj = residual_jet_q::<24, 1>(&ctx, &[view(&z0), view(&v)], false, 24).unwrap();
        ctx.mutate = 0;
        let mut worst = [0.0f64; 3];
        for (w, e) in [-6, -7, -8].into_iter().enumerate() {
            let h = 2f64.powi(e);
            ctx.n = DD::from_f64(n0 + h);
            let f = residual_parts(&ctx, view(&shifted(&z0, usize::MAX, h)), false).unwrap();
            let fs = f.iter().map(|v| v.to_f64().abs()).fold(1.0, f64::max);
            for i in 0..l {
                let p = (0..24).rev().fold(QD::ZERO, |a, k| a.mul_pwr2(h) + fj[k][i]);
                worst[w] = worst[w].max((p - f[i]).to_f64().abs() / fs);
            }
        }
        worst
    }

    #[test]
    fn qsolve_jet_sums_to_the_residual() {
        let [e6, e7, e8] = jet_q_vs_residual(0);
        let [c6, c7, c8] = jet_q_vs_residual(JET_MUT_FREEZE_N);
        println!("quad-double jet summed vs the quad-double residual at n + h: h = 2^-6 {e6:.1e}, 2^-7 {e7:.1e}, 2^-8 {e8:.1e}; dimension frozen: {c6:.1e}, {c7:.1e}, {c8:.1e}");
        // the truncation's order: orders 0..23 right leave the order-24 term, a factor 2^24 per halving of h
        let p = (e6 / e7).log2();
        assert!((23.0..25.0).contains(&p), "truncation order {p:.2}, not 24");
        let (e, ec) = (e7.max(e8), c7.min(c8));
        assert!(e < 1e-28, "{e:e}");
        assert!(ec > 1e3 * e, "control: the frozen dimension not seen ({ec:e} vs {e:e})");
    }

    /// the quad-double jet's coefficients against the double-double jet's (model::residual_jet) on the same state jet
    /// z_0 + h v: max over orders k < 24 of max_i |difference| / max(1, max_i |F_k|)
    #[test]
    fn qsolve_jet_agrees_with_double_double_jet() {
        let (ctx, zh, zl) = synth();
        let l = ctx.len();
        let z0 = parts(&zh, &zl);
        let v = direction(l, 1.0);
        let fq = residual_jet_q::<24, 1>(&ctx, &[view(&z0), view(&v)], false, 24).unwrap();
        let fd = residual_jet::<24, 1>(&ctx, &[(&zh, &zl), (&v[0], &v[1])], false, 24).unwrap();
        let mut worst = 0.0f64;
        for k in 0..24 {
            let fs = fq[k].iter().map(|v| v.to_f64().abs()).fold(1.0, f64::max);
            for i in 0..l {
                worst = worst.max((fq[k][i] - QD::from_dd(DD::new(fd[k].0[i], fd[k].1[i]))).to_f64().abs() / fs);
            }
        }
        println!("quad-double jet vs double-double jet, 24 orders: {worst:.1e} (relative to max(1, |F_k|))");
        assert!(worst < 1e-28, "{worst:e}");
    }

    /// F_1 = J v + dF/dn is linear in v: a change of v by 2^-180 w (w_i = ((i mod 5) - 2) / 2, Delta excluded),
    /// below double-double resolution, must change F_1 by J(z_0) 2^-180 w (jacobian_q): the error of the response
    /// relative to its size. v_i = ((i mod 7) - 3) / 3 in quad-double (all four parts), so that the change lands in
    /// its last parts. Controls: v read in its double-double part (QD_MUT_STATE_DD) sees no change at all; with the
    /// order-1 solves unrefined (QD_MUT_NOREFINE) the rows that pass through a horizon solve lose it.
    fn jet_sub_dd_response(mutate: u8) -> f64 {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let z0 = parts(&zh, &zl);
        let mut v: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; l]);
        for i in 0..l - 1 {
            let x = QD::from_f64((i % 7) as f64 - 3.0) / QD::from_f64(3.0);
            for q in 0..4 {
                v[q][i] = x.x[q];
            }
        }
        let s = 2f64.powi(-180);
        let mut d: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; l]);
        for i in 0..l - 1 {
            d[0][i] = s * ((i % 5) as f64 - 2.0) / 2.0;
        }
        let vp = axpy_parts(1.0, view(&d), view(&v));
        ctx.mutate = mutate;
        let f0 = residual_jet_q::<8, 1>(&ctx, &[view(&z0), view(&v)], false, 2).unwrap();
        let f1 = residual_jet_q::<8, 1>(&ctx, &[view(&z0), view(&vp)], false, 2).unwrap();
        ctx.mutate = 0;
        let (c, ld) = jacobian_q(&ctx, view(&z0), false).unwrap();
        let (mut worst, mut big) = (0.0f64, 0.0f64);
        for i in 0..l {
            let jv = (0..l).fold(QdAcc::zero(), |a, j| a.add_prod(qd_at(&c, j * ld + i), QD::from_f64(d[0][j]))).finish();
            big = big.max(jv.to_f64().abs());
            worst = worst.max((f1[1][i] - f0[1][i] - jv).to_f64().abs());
        }
        worst / big
    }

    #[test]
    fn qsolve_jet_resolves_sub_double_double_changes() {
        let r = jet_sub_dd_response(0);
        let rd = jet_sub_dd_response(QD_MUT_STATE_DD);
        let rn = jet_sub_dd_response(QD_MUT_NOREFINE);
        println!("jet order 1, change of 2^-180 in v: response error {r:.1e} of J d; v read in double-double: {rd:.1e}; unrefined: {rn:.1e}");
        assert!(r < 1e-6, "{r:e}");
        assert!(rd > 0.5, "control: the double-double v saw the change ({rd:e})");
        assert!(rn > 1e3 * r, "control: the unrefined solves kept the change ({rn:e} vs {r:e})");
    }
}
