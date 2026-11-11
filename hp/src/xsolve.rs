//! The Newton step and the Taylor jet in quad-double or octo-double, written once over multidouble's `Precision`
//! (`Quad`, `Octo`): the residual with every block in that precision, its jet in the dimension, its exact Jacobian by
//! forward-mode duals, the dense LU and the step -J^-1 F, each multishoot's at that precision on `Ctx::shooting`
//! (with hp's freeze projection, model::residual_p). qsolve.rs and osolve.rs are this module at `Quad` and `Octo`,
//! under their own names and with their own tests (notes/QUAD_DOUBLE.md section 12, notes/OCTO_DOUBLE.md). The state
//! comes in PARTS parts. The horizon solves are refined to the lanes' precision (model::psolve, `Lane::VAL_QD`,
//! `Lane::VAL_OD`); the model's constants stay double-double, so the discrete system is the one the double-double
//! solver solves.
use crate::field::Fl;
use crate::model::{jacobian_p, residual_jet_p, residual_p, Ctx, Dss};
use multidouble::precision::Precision;
use multidouble::simd::Native;
use multishoot::control::Control;
use multishoot::Shooting;

/// The precision's name in messages, and its `Ctx::mutate` code that reads the state in its lower half of parts (the
/// control of the precision checks: qsolve::QD_MUT_STATE_DD, osolve::OD_MUT_STATE_QD)
fn name<P: Precision<Native>>() -> (&'static str, u8) {
    if P::PARTS == 4 {
        ("qsolve", crate::qsolve::QD_MUT_STATE_DD)
    } else {
        ("osolve", crate::osolve::OD_MUT_STATE_QD)
    }
}

/// The engine's problem on ctx, the state read in its lower half of parts (multishoot's `Control::HalfParts`) under
/// the precision's state mutation
fn shooting<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>) -> Shooting<Dss<'_, B>, B> {
    let mut sh = ctx.shooting();
    if ctx.mutate == name::<P>().1 {
        sh.set_control(Control::HalfParts);
    }
    sh
}

/// A field from the PARTS parts of a state at offset off
pub fn fld<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], off: usize) -> Fl<P::V, B> {
    shooting::<P, B>(ctx).load::<P>(z, off)
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

/// The residual with every block in the precision, one scalar per row. Each block reads its inputs, Delta and its
/// target in all parts; the freeze projection and the phase row are in the precision too.
pub fn residual_parts<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
    check::<P, B>(ctx, z)?;
    residual_p::<P, B>(ctx, &shooting::<P, B>(ctx), z, par)
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
    residual_jet_p::<P, N, B>(ctx, &shooting::<P, B>(ctx), zs, par, nout)
}

/// The exact Jacobian dF/dz by forward-mode duals (`P::U` lanes, one evaluation per column), assembled as
/// model::jacobian does: column-major with leading dimension ld_for(L), part q of entry (i, j) at c[q][j ld + i].
pub fn jacobian<P: Precision<Native>, const B: usize>(ctx: &Ctx<B>, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
    check::<P, B>(ctx, z)?;
    jacobian_p::<P, B>(ctx, &shooting::<P, B>(ctx), z, par)
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
