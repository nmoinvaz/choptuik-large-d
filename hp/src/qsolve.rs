//! The Newton step in quad-double (notes/QUAD_DOUBLE.md, section 12): the residual with every block in quad-double,
//! its exact Jacobian by forward-mode duals over quad-double lanes (multidouble's `Qu`), the dense quad-double LU
//! (`QDLu`), and the step -J^-1 F in quad-double. The state comes in four parts (z[0] + z[1] + z[2] + z[3], as
//! multidouble's QD). Every operation on the state is quad-double, the horizon solves refined to quad-double
//! (model::psolve with QD_SOLVE); the model's constants stay double-double, so the discrete system is the one the
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
use crate::dd::DD;
use crate::field::{Fl, State};
use crate::lane::{BaseLane, Qu};
use crate::linalg::{ld_for, DDLu, QDLu};
use crate::model::{block_state, delta_of, horizon_lus, jac_tasks, Ctx};
use crate::qd::{QdAcc, QD};
use crate::qv::QV;
use rayon::prelude::*;

/// `Ctx::mutate` for qsolve (the control of its precision checks): the state read in its double-double part only
pub const QD_MUT_STATE_DD: u8 = 9;

/// A field from the four parts of a quad-double state (only the first two with QD_MUT_STATE_DD)
pub fn fld_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], off: usize) -> Fl<QV, B> {
    let r = off..off + Ctx::<B>::M;
    if ctx.mutate == QD_MUT_STATE_DD {
        let zero = [0.0; 512];
        let n = Ctx::<B>::M;
        return Fl::load_parts(&z[0][r.clone()], &z[1][r], &zero[..n], &zero[..n]);
    }
    Fl::load_parts(&z[0][r.clone()], &z[1][r.clone()], &z[2][r.clone()], &z[3][r])
}

fn delta_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4]) -> QD {
    let l = z[0].len() - 1;
    if ctx.mutate == QD_MUT_STATE_DD {
        return QD::new(z[0][l], z[1][l], 0.0, 0.0);
    }
    QD::new(z[0][l], z[1][l], z[2][l], z[3][l])
}

/// the phase row sum_j w_j p1_j in quad-double
fn phase_q<const B: usize>(ctx: &Ctx<B>, p1: &Fl<QV, B>) -> QD {
    let mut acc = QdAcc::zero();
    for i in 0..Ctx::<B>::M {
        acc = acc.add_prod(QD::from_dd(ctx.phase.get(i)), p1.get(i));
    }
    acc.finish()
}

fn check<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4]) -> Result<(), String> {
    let l = ctx.len();
    if z.iter().any(|p| p.len() != l) {
        return Err(format!("z0..z3 must have length {l}"));
    }
    if ctx.lam.hi != 0.0 {
        return Err("qsolve: no mode-matrix shift (lam) supported".into());
    }
    Ok(())
}

/// the horizon LUs at the state's double-double part: the preconditioners of the refined horizon solves
fn lus_of<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<Vec<DDLu>, String> {
    horizon_lus(ctx, &ctx.fld(z[0], z[1], ctx.off_ah()), delta_of(z[0], z[1]), par, DD::ZERO)
}

/// The residual with every block in quad-double, one QD per row. Each block reads its inputs, Delta and its target
/// in all four parts; the freeze projection and the phase row are quad-double too.
pub fn residual_parts<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<Vec<QD>, String> {
    check(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let lus = lus_of(ctx, z, par)?;
    let dq = QV::splat(delta_q(ctx, z));
    let one_block = |b: usize| -> State<QV, B> {
        let inq: Vec<Fl<QV, B>> = ctx.block_inputs(b).iter().map(|&o| fld_q(ctx, z, o)).collect();
        block_state::<QV, B>(ctx, b, &inq, dq, &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<QV, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let p1 = fld_q(ctx, z, 0);
    let mut f = vec![QD::ZERO; l];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let mut d = s[q] - fld_q(ctx, z, t + q * m);
            if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                d = lo.apply(&d) + hi.apply(&p1);
            }
            let o = b * 4 * m + q * m;
            for i in 0..m {
                f[o + i] = d.get(i);
            }
        }
    }
    f[l - 1] = phase_q(ctx, &p1);
    Ok(f)
}

/// `residual_parts` with each row rounded to double-double (F is small, so the rounding is relative to F): (F hi,
/// F lo)
pub fn residual_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
    let f = residual_parts(ctx, z, par)?;
    Ok(f.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip())
}

/// One column task (model::jac_tasks): block b's output state differentiated along a unit tangent on lane j of input
/// field f (Delta for f = usize::MAX), with `Qu` lanes, in quad-double; the freeze projection applied as in
/// model::jac_column.
fn jac_column_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], dq: QD, lus: &[DDLu], (b, f, j): (usize, usize, usize)) -> Vec<QD> {
    let m = Ctx::<B>::M;
    let inp: Vec<Fl<Qu, B>> = ctx
        .block_inputs(b)
        .iter()
        .enumerate()
        .map(|(fi, &o)| {
            let v = fld_q(ctx, z, o);
            let mut r = [<Qu as BaseLane>::zero(); B];
            for bb in 0..B {
                r[bb].v = v.0[bb];
            }
            if fi == f {
                let mut e = [QD::ZERO; 8];
                e[j % 8] = QD::ONE;
                r[j / 8].d = QV::from_lanes(&e);
            }
            Fl(r)
        })
        .collect();
    let d = Qu { v: QV::splat(dq), d: if f == usize::MAX { QV::splat(QD::ONE) } else { QV::zero() } };
    let s = block_state::<Qu, B>(ctx, b, &inp, d, lus, lus);
    let mut out = Vec::with_capacity(4 * m);
    for q in 0..4 {
        let mut t = Fl::<QV, B>(s[q].0.map(|x| x.d));
        if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
            t = lo.apply(&t);
        }
        for i in 0..m {
            out.push(t.get(i));
        }
    }
    out
}

/// The exact Jacobian dF/dz in quad-double by forward-mode duals (`Qu` lanes, one evaluation per column), assembled
/// as model::jacobian does: column-major with leading dimension ld_for(L), part q of entry (i, j) at c[q][j ld + i].
pub fn jacobian_q<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], par: bool) -> Result<([Vec<f64>; 4], usize), String> {
    check(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let lus = lus_of(ctx, z, par)?;
    let dq = delta_q(ctx, z);
    let tasks = jac_tasks(ctx);
    let column = |t: &(usize, usize, usize)| jac_column_q(ctx, z, dq, &lus, *t);
    let cols: Vec<Vec<QD>> = if par { tasks.par_iter().map(column).collect() } else { tasks.iter().map(column).collect() };
    let mut c: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; ld * l]);
    let add = |c: &mut [Vec<f64>; 4], idx: usize, v: QD| {
        let s = QD { x: std::array::from_fn(|q| c[q][idx]) } + v;
        for q in 0..4 {
            c[q][idx] = s.x[q];
        }
    };
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let cj = if f == usize::MAX { l - 1 } else { ctx.block_inputs(b)[f] + j };
        for (r, v) in col.iter().enumerate() {
            for q in 0..4 {
                c[q][cj * ld + b * 4 * m + r] = v.x[q];
            }
        }
    }
    for b in 0..ctx.k() + 1 {
        let t = ctx.block_target(b);
        for r in 0..4 * m {
            let row = b * 4 * m + r;
            if let (true, Some((lo, hi))) = (b == ctx.k() && r < m, &ctx.freeze) {
                // p rows of the horizon block: -Pi_low on the target node's p, +Pi_high on p1
                for jj in 0..m {
                    add(&mut c, (t + jj) * ld + row, -QD::from_dd(lo.cols[jj].get(r)));
                    add(&mut c, jj * ld + row, QD::from_dd(hi.cols[jj].get(r)));
                }
                continue;
            }
            add(&mut c, (t + r) * ld + row, -QD::ONE);
        }
    }
    for i in 0..m {
        let w = ctx.phase.get(i);
        let o = i * ld + l - 1;
        c[0][o] = w.hi;
        c[1][o] = w.lo;
    }
    Ok((c, ld))
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
    let l = ctx.len();
    let mut t = [0.0f64; 4];
    let t0 = std::time::Instant::now();
    let f = residual_parts(ctx, z, par)?;
    t[0] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let (c, ld) = jacobian_q(ctx, z, par)?;
    t[1] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let lu = QDLu::factor(l, ld, c, par)?;
    t[2] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut b: Vec<QD> = f.iter().map(|&v| -v).collect();
    lu.solve(&mut b);
    t[3] = t0.elapsed().as_secs_f64();
    let dz = std::array::from_fn(|q| b.iter().map(|v| v.x[q]).collect());
    let (fh, fl) = f.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip();
    Ok(QStep { dz, fh, fl, lu, t })
}

/// y + a x elementwise in quad-double (four parts each)
pub fn qd_axpy(a: f64, x: [&[f64]; 4], y: [&[f64]; 4]) -> [Vec<f64>; 4] {
    let n = y[0].len();
    let mut out: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; n]);
    for i in 0..n {
        let s = QD { x: std::array::from_fn(|q| y[q][i]) } + QD { x: std::array::from_fn(|q| x[q][i]) }.mul_f(a);
        for q in 0..4 {
            out[q][i] = s.x[q];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{jacobian, residual, QD_MUT_NOREFINE, QD_MUT_REFINE4};
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
        qd_axpy(1.0, view(&d), view(z))
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
            z = qd_axpy(1.0, view(&dz), view(&z));
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
}
