//! Tests of the mode matrix on multishoot's engine (`Shooting::mode_matrix`, complex tangents `Dc`, the horizon
//! solves of model::psolve_dc) on the small synthetic context of `qd_residual::synth` (M = 8, five nodes):
//!
//! ```text
//! cargo test --release --lib mode_    at rate zero the mode matrix is the Jacobian's leading block bit for bit; at a
//!                                     complex rate lr + i h it is analytic in the rate: Im M(lr + i h) / h equals the
//!                                     central difference of Re M along lr (Cauchy-Riemann), which tests the 2M x 2M
//!                                     horizon solves (control MODE_MUT_REAL_SOLVE, those solves at the real rate's
//!                                     factors, must fail)
//! ```
use crate::dd::DD;
use crate::model::{jacobian, MODE_MUT_REAL_SOLVE};
use crate::qd_residual::synth;
use multishoot::Rate;

#[test]
fn mode_matrix_at_rate_zero_is_the_jacobian_block() {
    let (ctx, zh, zl) = synth();
    let (jh, jl, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let mm = ctx.shooting().mode_matrix(&[&zh, &zl], Rate::ZERO, false).unwrap();
    let n = ctx.len() - 1;
    assert_eq!(mm.n, n);
    let mut same = true;
    let mut imag0 = true;
    for c in 0..n {
        for r in 0..n {
            let k = c * mm.ld + r;
            same &= mm.re[0][k].to_bits() == jh[c * ld + r].to_bits() && mm.re[1][k].to_bits() == jl[c * ld + r].to_bits();
            imag0 &= mm.im[0][k] == 0.0 && mm.im[1][k] == 0.0;
        }
    }
    assert!(same, "the mode matrix at rate zero differs from the Jacobian's leading block");
    assert!(imag0, "the mode matrix at rate zero has an imaginary part");
}

/// max over entries of |Im M(lr + i h) / h - (Re M(lr + h) - Re M(lr - h)) / 2h| / max |dM/dlr|, with ctx.mutate
fn cauchy_riemann(mutate: u8) -> f64 {
    let (mut ctx, zh, zl) = synth();
    ctx.mutate = mutate;
    let sh = ctx.shooting();
    let z = [&zh[..], &zl[..]];
    let (lr, h) = (DD::from_f64(1.3), 2f64.powi(-20));
    let mc = sh.mode_matrix(&z, Rate::new(lr, DD::from_f64(h)), false).unwrap();
    let mp = sh.mode_matrix(&z, Rate::new(lr + DD::from_f64(h), DD::ZERO), false).unwrap();
    let mm = sh.mode_matrix(&z, Rate::new(lr - DD::from_f64(h), DD::ZERO), false).unwrap();
    let n = mc.n;
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for c in 0..n {
        for r in 0..n {
            let k = c * mc.ld + r;
            let im = DD { hi: mc.im[0][k], lo: mc.im[1][k] }.mul_pwr2(1.0 / h);
            let d = (DD { hi: mp.re[0][k], lo: mp.re[1][k] } - DD { hi: mm.re[0][k], lo: mm.re[1][k] }).mul_pwr2(0.5 / h);
            worst = worst.max((im - d).hi.abs());
            scale = scale.max(d.hi.abs());
        }
    }
    worst / scale
}

#[test]
fn mode_matrix_complex_rate_is_analytic() {
    let e = cauchy_riemann(0);
    let ec = cauchy_riemann(MODE_MUT_REAL_SOLVE);
    println!("mode matrix at 1.3 + i 2^-20: Im M / h against d Re M / d lr, relative {e:.2e}; control (horizon solves at the real factors) {ec:.2e}");
    // the central difference's truncation, h^2 M''' / 6, and Im M's own h^2 term are near 1e-12 relative
    assert!(e < 1e-9, "Cauchy-Riemann: {e:e}");
    assert!(ec > 1e-6, "control: the real-factor horizon solves are not seen ({ec:e})");
}
