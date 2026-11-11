//! The Newton step and the Taylor jet in quad-double or octo-double, written once over multidouble's `Precision`
//! (`Quad`, `Octo`): the residual with every block in that precision, its jet in the dimension, its exact Jacobian by
//! forward-mode duals, the dense LU and the step -J^-1 F. qsolve.rs and osolve.rs are this module at `Quad` and
//! `Octo`, under their own names and with their own tests (notes/QUAD_DOUBLE.md section 12, notes/OCTO_DOUBLE.md).
//! The state comes in PARTS parts. The horizon solves are refined to the lanes' precision (model::psolve,
//! `Lane::VAL_QD`, `Lane::VAL_OD`); the model's constants stay double-double, so the discrete system is the one the
//! double-double solver solves.
use crate::dd::DD;
use crate::field::{Fl, State};
use crate::lane::BaseLane;
use crate::linalg::{ld_for, DDLu};
use crate::model::{block_state, delta_of, horizon_lus, jac_tasks, Ctx};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use rayon::prelude::*;

/// The precision's name in messages, and its `Ctx::mutate` code that reads the state in its lower half of parts (the
/// control of the precision checks: qsolve::QD_MUT_STATE_DD, osolve::OD_MUT_STATE_QD)
fn name<P: Precision<Native>>() -> (&'static str, u8) {
    if P::PARTS == 4 {
        ("qsolve", crate::qsolve::QD_MUT_STATE_DD)
    } else {
        ("osolve", crate::osolve::OD_MUT_STATE_QD)
    }
}

/// the number of parts read: all, or the lower half under the precision's state mutation
fn kept<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>) -> usize {
    if ctx.mutate == name::<P>().1 {
        P::PARTS / 2
    } else {
        P::PARTS
    }
}

/// A field from the PARTS parts of a state at offset off
pub fn fld<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], off: usize) -> Fl<P::V, B> {
    let m = Ctx::<B>::M;
    let zero = [0.0; 512];
    let keep = kept::<P, B>(ctx);
    let parts: Vec<&[f64]> = (0..P::PARTS).map(|q| if q < keep { &z[q][off..off + m] } else { &zero[..m] }).collect();
    P::load_field(&parts)
}

/// Delta, the state's last entry, renormalized from its parts
fn delta<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]]) -> P::Scalar {
    let l = z[0].len() - 1;
    let parts: Vec<f64> = (0..kept::<P, B>(ctx)).map(|q| z[q][l]).collect();
    P::new(&parts)
}

/// the phase row sum_j w_j p1_j
fn phase<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, p1: &Fl<P::V, B>) -> P::Scalar {
    P::dot((0..Ctx::<B>::M).map(|i| (P::from_dd(ctx.phase.get(i)), P::field_get(p1, i))))
}

fn check<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]]) -> Result<(), String> {
    let l = ctx.len();
    if z.len() != P::PARTS || z.iter().any(|p| p.len() != l) {
        return Err(format!("z0..z{} must have length {l}", P::PARTS - 1));
    }
    if ctx.lam.hi != 0.0 {
        return Err(format!("{}: no mode-matrix shift (lam) supported", name::<P>().0));
    }
    Ok(())
}

/// the horizon LUs at the state's double-double part: the preconditioners of the refined horizon solves
fn lus_of<const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<DDLu>, String> {
    horizon_lus(ctx, &ctx.fld(z[0], z[1], ctx.off_ah()), delta_of(z[0], z[1]), par, DD::ZERO)
}

/// The residual with every block in the precision, one scalar per row. Each block reads its inputs, Delta and its
/// target in all parts; the freeze projection and the phase row are in the precision too.
pub fn residual_parts<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
    check::<P, B>(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let lus = lus_of(ctx, z, par)?;
    let dq = P::splat(delta::<P, B>(ctx, z));
    let one_block = |b: usize| -> State<P::V, B> {
        let inq: Vec<Fl<P::V, B>> = ctx.block_inputs(b).iter().map(|&o| fld::<P, B>(ctx, z, o)).collect();
        block_state::<P::V, B>(ctx, b, &inq, dq, &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<P::V, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let p1 = fld::<P, B>(ctx, z, 0);
    let mut f = vec![P::ZERO; l];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            let mut d = s[q] - fld::<P, B>(ctx, z, t + q * m);
            if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                d = lo.apply(&d) + hi.apply(&p1);
            }
            let o = b * 4 * m + q * m;
            for i in 0..m {
                f[o + i] = P::field_get(&d, i);
            }
        }
    }
    f[l - 1] = phase::<P, B>(ctx, &p1);
    Ok(f)
}

/// The residual on a jet state z(h) = sum_k z_k h^k (`P::J<N>` lanes), with the dimension n + h as the jet variable:
/// model::residual_jet with every block, the freeze projection and the phase row in the precision and each z_k in all
/// its parts. zs holds at most N coefficient vectors; the result is F's coefficients of orders 0..nout-1 (nout <= N),
/// one scalar per row. The horizon LUs are those of z_0's double-double part, the preconditioners of the refined
/// solves (model::psolve). The D-dependent constants are formed in the jet's parameter arithmetic, where
/// residual_parts forms them in double-double, so order 0 is residual_parts' residual up to that rounding.
pub fn residual_jet<P: Precision<Native>, const N: usize, const B: usize>(
    ctx: &Ctx<B>,
    zs: &[Vec<&[f64]>],
    par: bool,
    nout: usize,
    fname: &str,
) -> Result<Vec<Vec<P::Scalar>>, String> {
    if zs.is_empty() || zs.len() > N || nout == 0 || nout > N {
        return Err(format!("{fname}: 1..={N} coefficient vectors and 1..={N} outputs, not {} and {nout}", zs.len()));
    }
    for z in zs {
        check::<P, B>(ctx, z)?;
    }
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let lus = lus_of(ctx, &zs[0], par)?;
    // the coefficient-k field at offset o: z_k's, zero past the given vectors
    let fld_k = |k: usize, o: usize| -> Fl<P::V, B> { if k < zs.len() { fld::<P, B>(ctx, &zs[k], o) } else { Fl::zero() } };
    let fld_jet = |o: usize| -> Fl<P::J<N>, B> {
        let mut f = Fl::<P::J<N>, B>::zero();
        for k in 0..zs.len() {
            P::field_set_coef(&mut f, k, &fld_k(k, o));
        }
        f
    };
    let mut dj = <P::J<N> as BaseLane>::zero();
    for (k, z) in zs.iter().enumerate() {
        P::set_coef(&mut dj, k, P::splat(delta::<P, B>(ctx, z)));
    }
    let one_block = |b: usize| -> State<P::J<N>, B> {
        let inp: Vec<Fl<P::J<N>, B>> = ctx.block_inputs(b).iter().map(|&o| fld_jet(o)).collect();
        block_state::<P::J<N>, B>(ctx, b, &inp, dj, &lus, &lus)
    };
    let nb = ctx.k() + 1;
    let states: Vec<State<P::J<N>, B>> = if par { (0..nb).into_par_iter().map(one_block).collect() } else { (0..nb).map(one_block).collect() };
    let mut out = vec![vec![P::ZERO; l]; nout];
    let kb = ctx.k();
    for (b, s) in states.iter().enumerate() {
        let t = ctx.block_target(b);
        for q in 0..4 {
            for (k, o_k) in out.iter_mut().enumerate() {
                let mut d = P::field_coef(&s[q], k) - fld_k(k, t + q * m);
                if let (true, 0, Some((lo, hi))) = (b == kb, q, &ctx.freeze) {
                    d = lo.apply(&d) + hi.apply(&fld_k(k, 0));
                }
                let o = b * 4 * m + q * m;
                for i in 0..m {
                    o_k[o + i] = P::field_get(&d, i);
                }
            }
        }
    }
    for (k, o_k) in out.iter_mut().enumerate() {
        o_k[l - 1] = phase::<P, B>(ctx, &fld_k(k, 0));
    }
    Ok(out)
}

/// One column task (model::jac_tasks): block b's output state differentiated along a unit tangent on lane j of input
/// field f (Delta for f = usize::MAX), with `P::U` lanes seeded by `P::field_dual` with the tangent `Fl::unit(j)`; the
/// freeze projection applied as in model::jac_column.
fn jac_column<P: Precision<Native>, const B: usize>(
    ctx: &Ctx<B>,
    z: &[&[f64]],
    dq: P::Scalar,
    lus: &[DDLu],
    (b, f, j): (usize, usize, usize),
) -> Vec<P::Scalar> {
    let m = Ctx::<B>::M;
    let inp: Vec<Fl<P::U, B>> = ctx
        .block_inputs(b)
        .iter()
        .enumerate()
        .map(|(fi, &o)| {
            let t = if fi == f { Fl::<P::V, B>::unit(j) } else { Fl::zero() };
            P::field_dual(&fld::<P, B>(ctx, z, o), &t)
        })
        .collect();
    let d = P::dual(P::splat(dq), if f == usize::MAX { P::splat(P::ONE) } else { <P::V as BaseLane>::zero() });
    let s = block_state::<P::U, B>(ctx, b, &inp, d, lus, lus);
    let mut out = Vec::with_capacity(4 * m);
    for q in 0..4 {
        let mut t = P::field_coef(&s[q], 1);
        if let (true, 0, Some((lo, _))) = (b == ctx.k(), q, &ctx.freeze) {
            t = lo.apply(&t);
        }
        for i in 0..m {
            out.push(P::field_get(&t, i));
        }
    }
    out
}

/// The exact Jacobian dF/dz by forward-mode duals (`P::U` lanes, one evaluation per column), assembled as
/// model::jacobian does: column-major with leading dimension ld_for(L), part q of entry (i, j) at c[q][j ld + i].
pub fn jacobian<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
    check::<P, B>(ctx, z)?;
    let l = ctx.len();
    let m = Ctx::<B>::M;
    let ld = ld_for(l);
    let lus = lus_of(ctx, z, par)?;
    let dq = delta::<P, B>(ctx, z);
    let tasks = jac_tasks(ctx);
    let column = |t: &(usize, usize, usize)| jac_column::<P, B>(ctx, z, dq, &lus, *t);
    let cols: Vec<Vec<P::Scalar>> = if par { tasks.par_iter().map(column).collect() } else { tasks.iter().map(column).collect() };
    let mut c: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; ld * l]).collect();
    let add = |c: &mut [Vec<f64>], idx: usize, v: P::Scalar| {
        let cur: Vec<f64> = (0..P::PARTS).map(|q| c[q][idx]).collect();
        let s = P::from_raw(&cur) + v;
        for q in 0..P::PARTS {
            c[q][idx] = P::part(s, q);
        }
    };
    for (&(b, f, j), col) in tasks.iter().zip(&cols) {
        let cj = if f == usize::MAX { l - 1 } else { ctx.block_inputs(b)[f] + j };
        for (r, v) in col.iter().enumerate() {
            for q in 0..P::PARTS {
                c[q][cj * ld + b * 4 * m + r] = P::part(*v, q);
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
                    add(&mut c, (t + jj) * ld + row, -P::from_dd(lo.cols[jj].get(r)));
                    add(&mut c, jj * ld + row, P::from_dd(hi.cols[jj].get(r)));
                }
                continue;
            }
            add(&mut c, (t + r) * ld + row, -P::ONE);
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

/// A Newton step: dz = -J^-1 F (PARTS parts), F (hi, lo), the LU, and the seconds of the residual, the Jacobian, the
/// factorization and the solve.
pub struct Step<P: Precision<Native>> {
    pub dz: Vec<Vec<f64>>,
    pub fh: Vec<f64>,
    pub fl: Vec<f64>,
    pub lu: P::Lu,
    pub t: [f64; 4],
}

/// The Newton step at z
pub fn newton<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Step<P>, String> {
    let l = ctx.len();
    let mut t = [0.0f64; 4];
    let t0 = std::time::Instant::now();
    let f = residual_parts::<P, B>(ctx, z, par)?;
    t[0] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let (c, ld) = jacobian::<P, B>(ctx, z, par)?;
    t[1] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let lu = P::lu_factor(l, ld, c, par)?;
    t[2] = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut b: Vec<P::Scalar> = f.iter().map(|&v| -v).collect();
    P::lu_solve(&lu, &mut b);
    t[3] = t0.elapsed().as_secs_f64();
    let dz = (0..P::PARTS).map(|q| b.iter().map(|v| P::part(*v, q)).collect()).collect();
    let (fh, fl) = f.iter().map(|v| P::to_dd(*v)).map(|d| (d.hi, d.lo)).unzip();
    Ok(Step { dz, fh, fl, lu, t })
}
