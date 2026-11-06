//! The Newton step and the Taylor jet in octo-double (notes/OCTO_DOUBLE.md): qsolve.rs one level up, on multidouble
//! 4.0.0's octo-double lanes. The residual with every block in octo-double (`OV` lanes), its jet in the dimension
//! (`OJet<N>` lanes, the D-dependent constants in the jet's parameter arithmetic `OSeries<N>`), its exact Jacobian by
//! forward-mode duals over octo-double lanes (`Ou`), the dense octo-double LU (`ODLu`), and the step -J^-1 F in
//! octo-double. The state comes in eight parts (z[0] + ... + z[7], as multidouble's OD). The horizon solves are refined
//! to octo-double (model::psolve, `Lane::VAL_OD`, OD_REFINE steps); the model's constants stay double-double, as in
//! qsolve, so the discrete system is the one the double-double solver solves. The octo-double jet is for the large-D
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
use crate::dd::DD;
use crate::field::{Fl, State};
use crate::lane::{BaseLane, OJet, Ou};
use crate::linalg::{ld_for, DDLu, ODLu};
use crate::model::{block_state, delta_of, horizon_lus, jac_tasks, Ctx};
use multidouble::native::OV;
use multidouble::od::{OdAcc, OD};
use rayon::prelude::*;

/// `Ctx::mutate` for osolve (the control of its precision checks): the state read in its quad-double part only (its
/// first four parts)
pub const OD_MUT_STATE_QD: u8 = 10;

/// A field from the eight parts of an octo-double state (only the first four with OD_MUT_STATE_QD)
pub fn fld_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], off: usize) -> Fl<OV, B> {
    let m = Ctx::<B>::M;
    let zero = [0.0; 512];
    let parts: [&[f64]; 8] = std::array::from_fn(|q| if ctx.mutate == OD_MUT_STATE_QD && q >= 4 { &zero[..m] } else { &z[q][off..off + m] });
    Fl::load_parts_od(parts)
}

/// Delta, the state's last entry, renormalized from its eight parts (the first four with OD_MUT_STATE_QD)
fn delta_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8]) -> OD {
    let l = z[0].len() - 1;
    OD::new(std::array::from_fn(|q| if ctx.mutate == OD_MUT_STATE_QD && q >= 4 { 0.0 } else { z[q][l] }))
}

/// the phase row sum_j w_j p1_j in octo-double
fn phase_o<const B: usize>(ctx: &Ctx<B>, p1: &Fl<OV, B>) -> OD {
    let mut acc = OdAcc::zero();
    for i in 0..Ctx::<B>::M {
        acc = acc.add_prod(OD::from_dd(ctx.phase.get(i)), p1.get(i));
    }
    acc.finish()
}

fn check<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8]) -> Result<(), String> {
    let l = ctx.len();
    if z.iter().any(|p| p.len() != l) {
        return Err(format!("z0..z7 must have length {l}"));
    }
    if ctx.lam.hi != 0.0 {
        return Err("osolve: no mode-matrix shift (lam) supported".into());
    }
    Ok(())
}

/// the horizon LUs at the state's double-double part: the preconditioners of the refined horizon solves
fn lus_of<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<Vec<DDLu>, String> {
    horizon_lus(ctx, &ctx.fld(z[0], z[1], ctx.off_ah()), delta_of(z[0], z[1]), par, DD::ZERO)
}

/// The residual with every block in octo-double, one OD per row. Each block reads its inputs, Delta and its target in
/// all eight parts; the freeze projection and the phase row are octo-double too.
pub fn residual_parts_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<Vec<OD>, String> {
    check(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let lus = lus_of(ctx, z, par)?;
    let dq = OV::splat(delta_o(ctx, z));
    let one_block = |b: usize| -> State<OV, B> {
        let inq: Vec<Fl<OV, B>> = ctx.block_inputs(b).iter().map(|&o| fld_o(ctx, z, o)).collect();
        block_state::<OV, B>(ctx, b, &inq, dq, &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<OV, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let p1 = fld_o(ctx, z, 0);
    let mut f = vec![OD::ZERO; l];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let mut d = s[q] - fld_o(ctx, z, t + q * m);
            if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                d = lo.apply(&d) + hi.apply(&p1);
            }
            let o = b * 4 * m + q * m;
            for i in 0..m {
                f[o + i] = d.get(i);
            }
        }
    }
    f[l - 1] = phase_o(ctx, &p1);
    Ok(f)
}

/// `residual_parts_o` with each row rounded to double-double (F is small, so the rounding is relative to F): (F hi,
/// F lo)
pub fn residual_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
    let f = residual_parts_o(ctx, z, par)?;
    Ok(f.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip())
}

/// The residual on a jet state z(h) = sum_k z_k h^k in octo-double (`OJet<N>` lanes), with the dimension n + h as the
/// jet variable: qsolve::residual_jet_q with every block, the freeze projection and the phase row in octo-double and
/// each z_k in eight parts. zs holds at most N coefficient vectors; the result is F's coefficients of orders 0..nout-1
/// (nout <= N), one OD per row. The horizon LUs are those of z_0's double-double part, the preconditioners of the
/// refined solves (model::psolve). The D-dependent constants are formed in the jet's parameter arithmetic
/// (`OSeries<N>`, octo-double), where residual_parts_o forms them in double-double, so order 0 is residual_parts_o's
/// residual up to that rounding.
pub fn residual_jet_o<const N: usize, const B: usize>(ctx: &Ctx<B>, zs: &[[&[f64]; 8]], par: bool, nout: usize) -> Result<Vec<Vec<OD>>, String> {
    if zs.is_empty() || zs.len() > N || nout == 0 || nout > N {
        return Err(format!("residual_jet_o: 1..={N} coefficient vectors and 1..={N} outputs, not {} and {nout}", zs.len()));
    }
    for z in zs {
        check(ctx, *z)?;
    }
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let lus = lus_of(ctx, zs[0], par)?;
    // the coefficient-k field at offset o: z_k's, zero past the given vectors
    let fld_k = |k: usize, o: usize| -> Fl<OV, B> { if k < zs.len() { fld_o(ctx, zs[k], o) } else { Fl::zero() } };
    let fld_jet = |o: usize| -> Fl<OJet<N>, B> {
        let mut f = Fl::<OJet<N>, B>::zero();
        for k in 0..zs.len() {
            f.set_coef_od(k, &fld_k(k, o));
        }
        f
    };
    let mut dj = <OJet<N> as BaseLane>::zero();
    for (k, z) in zs.iter().enumerate() {
        dj.set_coef_od(k, OV::splat(delta_o(ctx, *z)));
    }
    let one_block = |b: usize| -> State<OJet<N>, B> {
        let inp: Vec<Fl<OJet<N>, B>> = ctx.block_inputs(b).iter().map(|&o| fld_jet(o)).collect();
        block_state::<OJet<N>, B>(ctx, b, &inp, dj, &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<OJet<N>, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let mut out = vec![vec![OD::ZERO; l]; nout];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            for (k, o_k) in out.iter_mut().enumerate() {
                let mut d = s[q].coef_od(k) - fld_k(k, t + q * m);
                if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                    d = lo.apply(&d) + hi.apply(&fld_k(k, 0));
                }
                let o = b * 4 * m + q * m;
                for i in 0..m {
                    o_k[o + i] = d.get(i);
                }
            }
        }
    }
    for (k, o_k) in out.iter_mut().enumerate() {
        o_k[l - 1] = phase_o(ctx, &fld_k(k, 0));
    }
    Ok(out)
}

/// One column task (model::jac_tasks): block b's output state differentiated along a unit tangent on lane j of input
/// field f (Delta for f = usize::MAX), with `Ou` lanes, in octo-double; the freeze projection applied as in
/// model::jac_column. The seed is multidouble's `Fl::from_vt_od` with the tangent `Fl::unit(j)`.
fn jac_column_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], dq: OD, lus: &[DDLu], (b, f, j): (usize, usize, usize)) -> Vec<OD> {
    let m = Ctx::<B>::M;
    let inp: Vec<Fl<Ou, B>> = ctx
        .block_inputs(b)
        .iter()
        .enumerate()
        .map(|(fi, &o)| {
            let t = if fi == f { Fl::<OV, B>::unit(j) } else { Fl::zero() };
            Fl::<Ou, B>::from_vt_od(&fld_o(ctx, z, o), &t)
        })
        .collect();
    let d = Ou { v: OV::splat(dq), d: if f == usize::MAX { OV::splat(OD::ONE) } else { OV::zero() } };
    let s = block_state::<Ou, B>(ctx, b, &inp, d, lus, lus);
    let mut out = Vec::with_capacity(4 * m);
    for q in 0..4 {
        let mut t = s[q].coef_od(1);
        if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
            t = lo.apply(&t);
        }
        for i in 0..m {
            out.push(t.get(i));
        }
    }
    out
}

/// The exact Jacobian dF/dz in octo-double by forward-mode duals (`Ou` lanes, one evaluation per column), assembled as
/// qsolve::jacobian_q does: column-major with leading dimension ld_for(L), part q of entry (i, j) at c[q][j ld + i].
pub fn jacobian_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<([Vec<f64>; 8], usize), String> {
    check(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let lus = lus_of(ctx, z, par)?;
    let dq = delta_o(ctx, z);
    let tasks = jac_tasks(ctx);
    let column = |t: &(usize, usize, usize)| jac_column_o(ctx, z, dq, &lus, *t);
    let cols: Vec<Vec<OD>> = if par { tasks.par_iter().map(column).collect() } else { tasks.iter().map(column).collect() };
    let mut c: [Vec<f64>; 8] = std::array::from_fn(|_| vec![0.0; ld * l]);
    let add = |c: &mut [Vec<f64>; 8], idx: usize, v: OD| {
        let s = OD { x: std::array::from_fn(|q| c[q][idx]) } + v;
        for q in 0..8 {
            c[q][idx] = s.x[q];
        }
    };
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let cj = if f == usize::MAX { l - 1 } else { ctx.block_inputs(b)[f] + j };
        for (r, v) in col.iter().enumerate() {
            for q in 0..8 {
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
                    add(&mut c, (t + jj) * ld + row, -OD::from_dd(lo.cols[jj].get(r)));
                    add(&mut c, jj * ld + row, OD::from_dd(hi.cols[jj].get(r)));
                }
                continue;
            }
            add(&mut c, (t + r) * ld + row, -OD::ONE);
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

/// An octo-double Newton step: dz = -J^-1 F (eight parts), F (hi, lo), the LU (kept by the Python binding for
/// olu_solve), and the seconds of the residual, the Jacobian, the factorization and the solve.
pub struct OStep {
    pub dz: [Vec<f64>; 8],
    pub fh: Vec<f64>,
    pub fl: Vec<f64>,
    pub lu: ODLu,
    pub t: [f64; 4],
}

/// The Newton step in octo-double at z
pub fn newton_o<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 8], par: bool) -> Result<OStep, String> {
    let l = ctx.len();
    let mut t = [0.0f64; 4];
    let t0 = std::time::Instant::now();
    let f = residual_parts_o(ctx, z, par)?;
    t[0] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let (c, ld) = jacobian_o(ctx, z, par)?;
    t[1] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let lu = ODLu::factor(l, ld, c, par)?;
    t[2] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut b: Vec<OD> = f.iter().map(|&v| -v).collect();
    lu.solve(&mut b);
    t[3] = t0.elapsed().as_secs_f64();
    let dz = std::array::from_fn(|q| b.iter().map(|v| v.x[q]).collect());
    let (fh, fl) = f.iter().map(|v| v.to_dd()).map(|d| (d.hi, d.lo)).unzip();
    Ok(OStep { dz, fh, fl, lu, t })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{JET_MUT_FREEZE_N, QD_MUT_NOREFINE, QD_MUT_REFINE4};
    use crate::qd::QD;
    use crate::qd_residual::synth;
    use crate::qsolve::{jacobian_q, residual_jet_q, residual_parts};

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
