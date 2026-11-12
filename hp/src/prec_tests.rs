//! Tests of the residual, the exact Jacobian, the Newton step and the jet in quad-double and octo-double (model::
//! residual_in, jacobian_in, newton_in and residual_jet_in at `Quad` and `Octo`, multishoot's at those precisions) on
//! the synthetic context of qd_residual::synth.
//!
//! Quad-double:
//!
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
//!
//! Octo-double:
//!
//! The Newton step and the Taylor jet in octo-double (notes/OCTO_DOUBLE.md): the quad-double step one level up, on multidouble
//! 4.0.0's octo-double lanes. The residual with every block in octo-double (`OV` lanes), its jet in the dimension
//! (`OJet<N>` lanes, the D-dependent constants in the jet's parameter arithmetic `OSeries<N>`), its exact Jacobian by
//! forward-mode duals over octo-double lanes (`Ou`), the dense octo-double LU (`ODLu`), and the step -J^-1 F in
//! octo-double. The state comes in eight parts (z[0] + ... + z[7], as multidouble's OD). The horizon solves are refined
//! to octo-double (model::psolve, `Lane::VAL_OD`, OD_REFINE steps); the model's constants stay double-double, as in
//! quad-double, so the discrete system is the one the double-double solver solves. The octo-double jet is for the large-D
//! chain past the quad-double jet's reach: its coefficients err by about the quad-double unit times G^2 (G the mass
//! mode's gain, about 0.7^-(D-3)), which reaches the chain's tolerance near D 170-185.
//!
//! ```text
//! cargo test --release --lib osolve_    on the synthetic context of qd_residual::synth, criteria fixed before the
//!     first run (the doc of each test): the octo-double residual, Jacobian and 24-order jet against the quad-double
//!     ones; the Jacobian against central differences of the octo-double residual (controls: the horizon solves
//!     unrefined, the quad-double Jacobian); the refinement converged; a planted root found to octo-double; a
//!     perturbation of 2^-300, below quad-double resolution, seen by the residual and undone by Newton (control: the
//!     state read in quad-double); the jet summed against the residual at n + h (control: the dimension frozen); the
//!     jet's order 1 resolving a change of 2^-300 (controls: the state read in quad-double, the solves unrefined)
//! ```

mod qd {
    use crate::dd::DD;
    use crate::linalg::QDLu;
    use crate::model::{jacobian, residual, residual_jet, Ctx, JET_MUT_FREEZE_N, QD_MUT_NOREFINE, QD_MUT_REFINE4, QD_MUT_STATE_DD};
    use crate::qd::{QdAcc, QD};
    use crate::qd_residual::synth;
    use crate::qv::axpy_parts;
    use multidouble::precision::Quad;

    /// the residual in quad-double (model::residual_in), one QD per row
    pub(super) fn residual_parts<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<Vec<QD>, String> {
        crate::model::residual_in::<Quad, B>(ctx, &z, par)
    }

    /// `residual_parts` rounded to double-double: (F hi, F lo)
    fn residual_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
        Ok(residual_parts(ctx, z, par)?.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip())
    }

    /// the jet in quad-double (model::residual_jet_in)
    pub(super) fn residual_jet_q<const N: usize, const B: usize>(ctx: &Ctx<B>, zs: &[[&[f64]; 4]], par: bool, nout: usize) -> Result<Vec<Vec<QD>>, String> {
        let zs: Vec<Vec<&[f64]>> = zs.iter().map(|z| z.to_vec()).collect();
        crate::model::residual_jet_in::<Quad, N, B>(ctx, &zs, par, nout)
    }

    /// the exact Jacobian in quad-double (model::jacobian_in): (four parts, ld)
    pub(super) fn jacobian_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<([Vec<f64>; 4], usize), String> {
        let (c, ld) = crate::model::jacobian_in::<Quad, B>(ctx, &z, par)?;
        Ok((c.try_into().expect("four parts"), ld))
    }

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

mod od {
    use super::qd::{jacobian_q, residual_jet_q, residual_parts};
    use crate::dd::DD;
    use crate::linalg::{ld_for, ODLu};
    use crate::model::{Ctx, JET_MUT_FREEZE_N, OD_MUT_STATE_QD, QD_MUT_NOREFINE, QD_MUT_REFINE4};
    use crate::qd::QD;
    use crate::qd_residual::synth;
    use multidouble::od::{OdAcc, OD};
    use multidouble::precision::Octo;

    /// the residual in octo-double (model::residual_in), one OD per row
    fn residual_parts_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<Vec<OD>, String> {
        crate::model::residual_in::<Octo, B>(ctx, &z, par)
    }

    /// the jet in octo-double (model::residual_jet_in)
    fn residual_jet_o<const N: usize, const B: usize>(ctx: &Ctx<B>, zs: &[[&[f64]; 8]], par: bool, nout: usize) -> Result<Vec<Vec<OD>>, String> {
        let zs: Vec<Vec<&[f64]>> = zs.iter().map(|z| z.to_vec()).collect();
        crate::model::residual_jet_in::<Octo, N, B>(ctx, &zs, par, nout)
    }

    /// the exact Jacobian in octo-double (model::jacobian_in): (eight parts, ld)
    fn jacobian_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<([Vec<f64>; 8], usize), String> {
        let (c, ld) = crate::model::jacobian_in::<Octo, B>(ctx, &z, par)?;
        Ok((c.try_into().expect("eight parts"), ld))
    }

    /// the eight parts of a double-double state (the lower six zero)
    fn parts8(zh: &[f64], zl: &[f64]) -> [Vec<f64>; 8] {
        std::array::from_fn(|q| match q {
            0 => zh.to_vec(),
            1 => zl.to_vec(),
            _ => vec![0.0; zh.len()],
        })
    }

    fn view(z: &[Vec<f64>; 8]) -> [&[f64]; 8] {
        std::array::from_fn(|q| &z[q][..])
    }

    /// the first four parts, the quad-double view of a state that is exact in quad-double
    fn view4(z: &[Vec<f64>; 8]) -> [&[f64]; 4] {
        std::array::from_fn(|q| &z[q][..])
    }

    fn od_at(c: &[Vec<f64>; 8], k: usize) -> OD {
        OD { x: std::array::from_fn(|q| c[q][k]) }
    }

    /// x + s y in octo-double, elementwise (s a power of two or exact in f64)
    fn axpy(x: &[Vec<f64>; 8], s: f64, y: &[Vec<f64>; 8]) -> [Vec<f64>; 8] {
        let l = x[0].len();
        let mut out: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; l]);
        for i in 0..l {
            let v = od_at(x, i) + od_at(y, i).mul_f(s);
            for q in 0..8 {
                out[q][i] = v.x[q];
            }
        }
        out
    }

    /// v = e_j (j < L) or v_i = ((i mod 7) - 3) / 3 in octo-double, all eight parts (j = usize::MAX), Delta excluded
    fn direction(l: usize, j: usize) -> [Vec<f64>; 8] {
        let mut d: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; l]);
        if j == usize::MAX {
            for i in 0..l - 1 {
                let x = OD::from_f64((i % 7) as f64 - 3.0) / OD::from_f64(3.0);
                for q in 0..8 {
                    d[q][i] = x.x[q];
                }
            }
        } else {
            d[0][j] = 1.0;
        }
        d
    }

    /// the same direction rounded to quad-double (its first four parts after QD's division), exact in quad-double
    fn direction_q(l: usize) -> [Vec<f64>; 8] {
        let mut d: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; l]);
        for i in 0..l - 1 {
            let x = QD::from_f64((i % 7) as f64 - 3.0) / QD::from_f64(3.0);
            for q in 0..4 {
                d[q][i] = x.x[q];
            }
        }
        d
    }

    fn max_abs(v: &[OD]) -> f64 {
        v.iter().map(|x| x.to_f64().abs()).fold(1.0, f64::max)
    }

    /// Pass (fixed before the first run): the octo-double residual, its 24-order jet on (z_0, v) and its Jacobian, at a
    /// state and direction exact in quad-double, each within 1e-58 of the quad-double ones (qsolve), relative to max(1,
    /// |entries|): the quad-double ones' rounding, about 1e-62, with room for the condition of the blocks
    #[test]
    fn osolve_agrees_with_quad_double() {
        let (ctx, zh, zl) = synth();
        let l = ctx.len();
        let z = parts8(&zh, &zl);
        let v = direction_q(l);
        let rq = residual_parts(&ctx, view4(&z), false).unwrap();
        let ro = residual_parts_o(&ctx, view(&z), false).unwrap();
        let fs = rq.iter().map(|x| x.to_f64().abs()).fold(1.0, f64::max);
        let dr = (0..l).map(|i| (ro[i] - OD::from_qd(rq[i])).to_f64().abs()).fold(0.0, f64::max) / fs;
        let jq = residual_jet_q::<24, 1>(&ctx, &[view4(&z), view4(&v)], false, 24).unwrap();
        let jo = residual_jet_o::<24, 1>(&ctx, &[view(&z), view(&v)], false, 24).unwrap();
        let mut dj = 0.0f64;
        for k in 0..24 {
            let fs = jq[k].iter().map(|x| x.to_f64().abs()).fold(1.0, f64::max);
            for i in 0..l {
                dj = dj.max((jo[k][i] - OD::from_qd(jq[k][i])).to_f64().abs() / fs);
            }
        }
        let (cq, ld) = jacobian_q(&ctx, view4(&z), false).unwrap();
        let (co, ldo) = jacobian_o(&ctx, view(&z), false).unwrap();
        assert_eq!(ld, ldo);
        let js = cq[0].iter().map(|x| x.abs()).fold(1.0, f64::max);
        let dc = (0..ld * l).map(|k| (od_at(&co, k) - OD::from_qd(QD { x: std::array::from_fn(|q| cq[q][k]) })).to_f64().abs()).fold(0.0, f64::max) / js;
        println!("octo-double vs quad-double: residual {dr:.1e}, 24-order jet {dj:.1e}, Jacobian {dc:.1e}");
        assert!(dr < 1e-58 && dj < 1e-58 && dc < 1e-58, "{dr:e} {dj:e} {dc:e}");
    }

    /// max over columns j of |J e_j - (F(z + h e_j) - F(z - h e_j)) / 2h| / max|J|, F the octo-double residual
    /// (residual_parts_o) and J `jac` (the octo-double Jacobian, or the quad-double one for the control), h = 2^-140:
    /// the differences' rounding, about 2^-418 G / h, and their truncation, h^2 times F's third derivative, are both
    /// near 1e-84 (G and F''' of order one on the synthetic context)
    fn jac_vs_difference(mutate: u8, quad: bool) -> f64 {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let z = parts8(&zh, &zl);
        ctx.mutate = mutate;
        let c: [Vec<f64>; 8] = if quad {
            let (cq, _) = jacobian_q(&ctx, view4(&z), false).unwrap();
            std::array::from_fn(|q| if q < 4 { cq[q].clone() } else { vec![0.0; cq[0].len()] })
        } else {
            jacobian_o(&ctx, view(&z), false).unwrap().0
        };
        let ld = ld_for(l);
        ctx.mutate = 0;
        let js = (0..l).map(|j| (0..l).map(|i| c[0][j * ld + i].abs()).fold(0.0, f64::max)).fold(1.0, f64::max);
        let h = 2f64.powi(-140);
        let mut worst = 0.0f64;
        for j in (0..l).step_by(7).chain([l - 1]) {
            let fp = residual_parts_o(&ctx, view(&axpy(&z, h, &direction(l, j))), false).unwrap();
            let fm = residual_parts_o(&ctx, view(&axpy(&z, -h, &direction(l, j))), false).unwrap();
            for i in 0..l {
                let fd = (fp[i] - fm[i]).mul_pwr2(1.0 / (2.0 * h));
                worst = worst.max((od_at(&c, j * ld + i) - fd).to_f64().abs() / js);
            }
        }
        worst
    }

    /// Pass (fixed before the first run): the octo-double Jacobian within 1e-78 of the central differences. Controls,
    /// each must exceed 1e3 times the error: the horizon solves unrefined (QD_MUT_NOREFINE), and the quad-double
    /// Jacobian (qsolve::jacobian_q) on the same differences, about 1e-62
    #[test]
    fn osolve_jacobian_matches_differences() {
        let e = jac_vs_difference(0, false);
        let ec = jac_vs_difference(QD_MUT_NOREFINE, false);
        let eq = jac_vs_difference(0, true);
        println!("Jacobian vs central differences of the octo-double residual: {e:.1e}; horizon solves unrefined: {ec:.1e}; quad-double Jacobian: {eq:.1e}");
        assert!(e < 1e-78, "{e:e}");
        assert!(ec > 1e3 * e, "control: unrefined horizon solves not seen ({ec:e} vs {e:e})");
        assert!(eq > 1e3 * e, "control: the quad-double Jacobian passes ({eq:e} vs {e:e})");
    }

    /// Pass (fixed before the first run): OD_REFINE steps (five for the Ou tangents) give the eight-step Jacobian
    /// (QD_MUT_REFINE4 on the octo-double lanes) to within 1e-118 of its largest entry
    #[test]
    fn osolve_refinement_converged() {
        let (mut ctx, zh, zl) = synth();
        let z = parts8(&zh, &zl);
        let (a, _) = jacobian_o(&ctx, view(&z), false).unwrap();
        ctx.mutate = QD_MUT_REFINE4;
        let (b, _) = jacobian_o(&ctx, view(&z), false).unwrap();
        let js = a[0].iter().map(|v| v.abs()).fold(1.0, f64::max);
        let d = (0..a[0].len()).map(|k| (od_at(&a, k) - od_at(&b, k)).to_f64().abs()).fold(0.0, f64::max) / js;
        println!("Jacobian, default refinement vs eight steps: {d:.1e}");
        assert!(d < 1e-118, "{d:e}");
    }

    /// the synthetic state moved by 2^-60 v (v of `direction`, all eight parts; Delta unmoved), so that every entry
    /// but Delta uses all eight parts: a change far below the quad-double unit then falls below its fourth part
    fn full_state() -> [Vec<f64>; 8] {
        let (ctx, zh, zl) = synth();
        axpy(&parts8(&zh, &zl), 2f64.powi(-60), &direction(ctx.len(), usize::MAX))
    }

    /// Newton in octo-double on the planted system G(z) = F(z) - F(z*), whose root z* is the synthetic state (or
    /// `full_state` with full), from z* + s v: the errors max |z - z*| after each step
    fn planted_newton(mutate: u8, s: f64, steps: usize, full: bool) -> Vec<f64> {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let zs = if full { full_state() } else { parts8(&zh, &zl) };
        let f0 = residual_parts_o(&ctx, view(&zs), false).unwrap();
        ctx.mutate = mutate;
        let mut z = axpy(&zs, s, &direction(l, usize::MAX));
        let mut errs = Vec::new();
        for _ in 0..steps {
            errs.push((0..l).map(|i| (od_at(&z, i) - od_at(&zs, i)).to_f64().abs()).fold(0.0, f64::max));
            let f = residual_parts_o(&ctx, view(&z), false).unwrap();
            let (c, ld) = jacobian_o(&ctx, view(&z), false).unwrap();
            let lu = ODLu::factor(l, ld, c, false).unwrap();
            let mut b: Vec<OD> = (0..l).map(|i| f0[i] - f[i]).collect();
            lu.solve(&mut b);
            for i in 0..l {
                let v = od_at(&z, i) + b[i];
                for q in 0..8 {
                    z[q][i] = v.x[q];
                }
            }
        }
        errs.push((0..l).map(|i| (od_at(&z, i) - od_at(&zs, i)).to_f64().abs()).fold(0.0, f64::max));
        errs
    }

    fn show(v: &[f64]) -> String {
        v.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(", ")
    }

    /// Pass (fixed before the first run): from 1e-6 away, quadratic convergence to the planted root within
    /// octo-double: after five steps below 1e-100, after six below 1e-118 (errs[0] is the start's error)
    #[test]
    fn osolve_newton_converges() {
        let errs = planted_newton(0, 1e-6, 6, false);
        println!("planted root from 1e-6, |z - z*| per step: {}", show(&errs));
        assert!(errs[5] < 1e-100 && errs[6] < 1e-118, "{errs:?}");
    }

    /// z* (`full_state`, every entry in eight parts) moved by 2^-300 v (v of `direction`), a change quad-double cannot
    /// represent (its resolution is 2^-212 of the entries, and the change lies below z*'s fourth part): the response
    /// F(z* + d) - F(z*) - J d relative to |J d|, and what one Newton step leaves of the change, relative to it
    fn sub_qd_response(mutate: u8) -> (f64, f64) {
        let (mut ctx, _, _) = synth();
        let l = ctx.len();
        let zs = full_state();
        ctx.mutate = mutate;
        let s = 2f64.powi(-300);
        let v = direction(l, usize::MAX);
        let zp = axpy(&zs, s, &v);
        let f0 = residual_parts_o(&ctx, view(&zs), false).unwrap();
        let f1 = residual_parts_o(&ctx, view(&zp), false).unwrap();
        let (c, ld) = jacobian_o(&ctx, view(&zs), false).unwrap();
        let (mut worst, mut big) = (0.0f64, 0.0f64);
        for i in 0..l {
            let jv = (0..l).fold(OdAcc::zero(), |a, j| a.add_prod(od_at(&c, j * ld + i), od_at(&v, j))).finish().mul_pwr2(s);
            big = big.max(jv.to_f64().abs());
            worst = worst.max((f1[i] - f0[i] - jv).to_f64().abs());
        }
        let errs = planted_newton(mutate, s, 1, true);
        (worst / big, errs[1] / s)
    }

    /// Pass (fixed before the first run): the octo-double residual sees the change, F(z* + d) - F(z*) = J d to within
    /// 1e-6 relative, and one Newton step leaves less than 2^-20 of it. Control: the state read in its quad-double
    /// part (OD_MUT_STATE_QD) sees no change at all, both ratios above 0.5. (On the first run z* was the synthetic
    /// state itself, two parts per entry, and the control did not fire: quad-double held the change in its third part.
    /// z* now uses all eight parts, notes/OCTO_DOUBLE.md.)
    #[test]
    fn osolve_resolves_sub_quad_double_changes() {
        let (resp, undo) = sub_qd_response(0);
        let (resp_c, undo_c) = sub_qd_response(OD_MUT_STATE_QD);
        println!("change of 2^-300: response error {resp:.1e} of J d, Newton leaves {undo:.1e} of it; state read in quad-double: {resp_c:.1e}, {undo_c:.1e}");
        assert!(resp < 1e-6 && undo < 2f64.powi(-20), "response {resp:e}, left {undo:e}");
        assert!(resp_c > 0.5 && undo_c > 0.5, "control: the quad-double state saw the change ({resp_c:e}, {undo_c:e})");
    }

    /// F(h) = sum_k F_k h^k of the octo-double jet (N = 24) on the state jet z_0 + h v, v = direction, against
    /// residual_parts_o at (z_0 + h v, n + h) for h = 2^-6, 2^-7, 2^-8 (n + h exact in f64): max |difference| /
    /// max(1, |F|). The difference is the truncation, which falls as h^24 while it exceeds the rounding, and
    /// residual_parts_o's D-dependent constants rounded to double-double (as qsolve's check)
    fn jet_o_vs_residual(mutate: u8) -> [f64; 3] {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let n0 = ctx.n.hi;
        let z0 = parts8(&zh, &zl);
        let v = direction(l, usize::MAX);
        ctx.mutate = mutate;
        let fj = residual_jet_o::<24, 1>(&ctx, &[view(&z0), view(&v)], false, 24).unwrap();
        ctx.mutate = 0;
        let mut worst = [0.0f64; 3];
        for (w, e) in [-6, -7, -8].into_iter().enumerate() {
            let h = 2f64.powi(e);
            ctx.n = DD::from_f64(n0 + h);
            let f = residual_parts_o(&ctx, view(&axpy(&z0, h, &v)), false).unwrap();
            let fs = max_abs(&f);
            for i in 0..l {
                let p = (0..24).rev().fold(OD::ZERO, |a, k| a.mul_pwr2(h) + fj[k][i]);
                worst[w] = worst[w].max((p - f[i]).to_f64().abs() / fs);
            }
        }
        worst
    }

    /// Pass (fixed before the first run): the truncation's order, log2 of the errors' ratio at 2^-6 and 2^-7, in
    /// [23, 25]; the error at 2^-7 and 2^-8 below 1e-28 (residual_parts_o's double-double constants); control: the
    /// dimension frozen (JET_MUT_FREEZE_N) above 1e3 times it
    #[test]
    fn osolve_jet_sums_to_the_residual() {
        let [e6, e7, e8] = jet_o_vs_residual(0);
        let [c6, c7, c8] = jet_o_vs_residual(JET_MUT_FREEZE_N);
        println!("octo-double jet summed vs the octo-double residual at n + h: h = 2^-6 {e6:.1e}, 2^-7 {e7:.1e}, 2^-8 {e8:.1e}; dimension frozen: {c6:.1e}, {c7:.1e}, {c8:.1e}");
        let p = (e6 / e7).log2();
        assert!((23.0..25.0).contains(&p), "truncation order {p:.2}, not 24");
        let (e, ec) = (e7.max(e8), c7.min(c8));
        assert!(e < 1e-28, "{e:e}");
        assert!(ec > 1e3 * e, "control: the frozen dimension not seen ({ec:e} vs {e:e})");
    }

    /// F_1 = J v + dF/dn is linear in v: a change of v by 2^-300 w (w_i = ((i mod 5) - 2) / 2, Delta excluded), below
    /// quad-double resolution, must change F_1 by J(z_0) 2^-300 w (jacobian_o): the error of the response relative to
    /// its size. v is `direction`, all eight parts in use, so that the change lands in its last parts.
    fn jet_sub_qd_response(mutate: u8) -> f64 {
        let (mut ctx, zh, zl) = synth();
        let l = ctx.len();
        let z0 = parts8(&zh, &zl);
        let v = direction(l, usize::MAX);
        let s = 2f64.powi(-300);
        let mut w: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; l]);
        for i in 0..l - 1 {
            w[0][i] = ((i % 5) as f64 - 2.0) / 2.0;
        }
        let vp = axpy(&v, s, &w);
        ctx.mutate = mutate;
        let f0 = residual_jet_o::<8, 1>(&ctx, &[view(&z0), view(&v)], false, 2).unwrap();
        let f1 = residual_jet_o::<8, 1>(&ctx, &[view(&z0), view(&vp)], false, 2).unwrap();
        ctx.mutate = 0;
        let (c, ld) = jacobian_o(&ctx, view(&z0), false).unwrap();
        let (mut worst, mut big) = (0.0f64, 0.0f64);
        for i in 0..l {
            let jv = (0..l).fold(OdAcc::zero(), |a, j| a.add_prod(od_at(&c, j * ld + i), OD::from_f64(w[0][j]))).finish().mul_pwr2(s);
            big = big.max(jv.to_f64().abs());
            worst = worst.max((f1[1][i] - f0[1][i] - jv).to_f64().abs());
        }
        worst / big
    }

    /// Pass (fixed before the first run): the response within 1e-6 of J d. Controls: v read in its quad-double part
    /// (OD_MUT_STATE_QD) sees no change at all (above 0.5); with the order-1 solves unrefined (QD_MUT_NOREFINE) the
    /// rows that pass through a horizon solve lose it (above 1e3 times the error)
    #[test]
    fn osolve_jet_resolves_sub_quad_double_changes() {
        let r = jet_sub_qd_response(0);
        let rq = jet_sub_qd_response(OD_MUT_STATE_QD);
        let rn = jet_sub_qd_response(QD_MUT_NOREFINE);
        println!("jet order 1, change of 2^-300 in v: response error {r:.1e} of J d; v read in quad-double: {rq:.1e}; unrefined: {rn:.1e}");
        assert!(r < 1e-6, "{r:e}");
        assert!(rq > 0.5, "control: the quad-double v saw the change ({rq:e})");
        assert!(rn > 1e3 * r, "control: the unrefined solves kept the change ({rn:e} vs {r:e})");
    }
}
