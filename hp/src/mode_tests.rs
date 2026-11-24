//! Tests of the mode matrix on multishoot's engine (`Shooting::mode_matrix`, complex tangents `Dc`, the horizon
//! solves of model::psolve_dc) on the small synthetic context of `qd_residual::synth` (M = 8, five nodes):
//!
//! ```text
//! cargo test --release --lib mode_    at rate zero the mode matrix is the Jacobian's leading block bit for bit; at a
//!                                     complex rate lr + i h it is analytic in the rate: Im M(lr + i h) / h equals the
//!                                     central difference of Re M along lr (Cauchy-Riemann), which tests the 2M x 2M
//!                                     horizon solves (control MODE_MUT_REAL_SOLVE, those solves at the real rate's
//!                                     factors, must fail)
//! cargo test --release --lib mode_q_  the mode matrix at a real rate in quad-double (model::mode_matrix_in, #1082),
//!                                     criteria fixed before the first run in each test's doc: at Double it is
//!                                     mode_matrix's real part bit for bit; at Quad it agrees with Double to
//!                                     double-double rounding (control MODE_MUT_QD_UNSHIFTED, the refined horizon
//!                                     solves without the rate, must fail) and with Octo to quad-double rounding
//!                                     (control QD_MUT_NOREFINE); its refinement has converged; at rate zero it is the
//!                                     quad-double Jacobian's leading block bit for bit
//! ```
use crate::dd::DD;
use crate::model::{jacobian, mode_matrix_in, MODE_MUT_QD_UNSHIFTED, MODE_MUT_REAL_SOLVE, QD_MUT_NOREFINE, QD_MUT_REFINE4};
use crate::qd_residual::synth;
use multidouble::precision::{Double, Octo, Precision, Quad};
use multidouble::simd::Native;
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

// ---------------------------------------------------------------------------------------------------------------
// the mode matrix at a real rate in quad-double (model::mode_matrix_in, #1082)
// ---------------------------------------------------------------------------------------------------------------

/// the real rates of the checks: zero, and both signs
const RATES_Q: [f64; 3] = [0.0, 1.3, -0.7];

/// model::mode_matrix_in::<P> on the synthetic context at the real rate lam, its double-double state in P's parts and
/// Ctx::mutate set to mutate: (parts, ld, n)
fn mode_in<P: Precision<Native>>(lam: DD, mutate: u8) -> (Vec<Vec<f64>>, usize, usize) {
    let (mut ctx, zh, zl) = synth();
    ctx.mutate = mutate;
    let l = zh.len();
    let z: Vec<Vec<f64>> = (0..P::PARTS).map(|q| match q {
        0 => zh.clone(),
        1 => zl.clone(),
        _ => vec![0.0; l],
    }).collect();
    let zr: Vec<&[f64]> = z.iter().map(|p| p.as_slice()).collect();
    let (c, ld) = mode_matrix_in::<P, 1>(&ctx, &zr, lam, false).unwrap();
    (c, ld, ctx.len() - 1)
}

/// max |a - b| / max |b| over the n x n entries, each read in P from its first PARTS parts at most (P::new), the
/// difference formed in P
fn rel_gap<P: Precision<Native>>(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, ld: usize) -> f64 {
    let at = |c: &[Vec<f64>], k: usize| P::new(&c.iter().take(P::PARTS).map(|q| q[k]).collect::<Vec<_>>());
    let (mut d, mut s) = (0.0f64, 0.0f64);
    for j in 0..n {
        for i in 0..n {
            let k = j * ld + i;
            d = d.max(P::to_dd(at(a, k) - at(b, k)).hi.abs());
            s = s.max(P::to_dd(at(b, k)).hi.abs());
        }
    }
    d / s
}

/// The real-rate mode matrix at Double (`Du` lanes, psolve's double-double tangent branch) against the real part of
/// `mode_matrix` (`Dc` lanes, psolve_dc) at the rates RATES_Q. Pass: every entry equal in the bits of both parts, the
/// imaginary part zero. The control: the Double matrix at 1.3 moved by one ulp must differ.
#[test]
fn mode_q_double_is_mode_matrix() {
    let differs = |lr: f64, lr_real: f64| -> (usize, usize) {
        let (ctx, zh, zl) = synth();
        let mm = ctx.shooting().mode_matrix(&[&zh, &zl], Rate::new(DD::from_f64(lr), DD::ZERO), false).unwrap();
        let (c, ld, n) = mode_in::<Double>(DD::from_f64(lr_real), 0);
        assert_eq!((ld, n), (mm.ld, mm.n));
        let (mut differ, mut nonzero) = (0, 0);
        for j in 0..n {
            for i in 0..n {
                let k = j * ld + i;
                differ += (0..2).any(|q| c[q][k].to_bits() != mm.re[q][k].to_bits()) as usize;
                nonzero += (mm.im[0][k] != 0.0 || mm.im[1][k] != 0.0) as usize;
            }
        }
        (differ, nonzero)
    };
    for lr in RATES_Q {
        let (d, nz) = differs(lr, lr);
        println!("rate {lr}: Double against mode_matrix, {d} entries differ, {nz} imaginary parts nonzero");
        assert_eq!((d, nz), (0, 0), "rate {lr}");
    }
    let (dc, _) = differs(1.3, f64::from_bits(1.3f64.to_bits() + 1));
    println!("control, rate one ulp up: {dc} entries differ");
    assert!(dc > 0, "control did not fire");
}

/// The quad-double mode matrix against the double-double one: the same steps, truncation and inputs, so they differ
/// by double-double's rounding of the entries. Pass: max |M_qd - M_dd| / max |M_dd| at most 1e-28 at the rates
/// RATES_Q. The control: at 1.3 with MODE_MUT_QD_UNSHIFTED (the refined horizon solves of the tangents with the
/// residual of the unshifted system) the gap must exceed 1e-20.
#[test]
fn mode_q_agrees_with_double_double() {
    for lr in RATES_Q {
        let (a, ld, n) = mode_in::<Quad>(DD::from_f64(lr), 0);
        let (b, _, _) = mode_in::<Double>(DD::from_f64(lr), 0);
        let g = rel_gap::<Quad>(&a, &b, n, ld);
        println!("rate {lr}: quad-double against double-double, relative gap {g:.2e}");
        assert!(g <= 1e-28, "rate {lr}: {g:e}");
    }
    let (a, ld, n) = mode_in::<Quad>(DD::from_f64(1.3), MODE_MUT_QD_UNSHIFTED);
    let (b, _, _) = mode_in::<Double>(DD::from_f64(1.3), 0);
    let g = rel_gap::<Quad>(&a, &b, n, ld);
    println!("control, the refined horizon solves unshifted: relative gap {g:.2e}");
    assert!(g > 1e-20, "control did not fire");
}

/// The quad-double mode matrix against the octo-double one (both refine the horizon solves, the octo-double one to
/// octo-double): they differ by quad-double's rounding, so the quad-double entries carry no double-double rounding.
/// Pass: max |M_od - M_qd| / max |M_qd| at most 1e-58 at the rates RATES_Q. The control: the quad-double matrix with
/// its horizon solves unrefined (QD_MUT_NOREFINE) must exceed 1e-40 at 1.3.
#[test]
fn mode_q_agrees_with_octo_double() {
    for lr in RATES_Q {
        let (a, ld, n) = mode_in::<Octo>(DD::from_f64(lr), 0);
        let (b, _, _) = mode_in::<Quad>(DD::from_f64(lr), 0);
        let g = rel_gap::<Quad>(&a, &b, n, ld);
        println!("rate {lr}: octo-double against quad-double, relative gap {g:.2e}");
        assert!(g <= 1e-58, "rate {lr}: {g:e}");
    }
    let (a, ld, n) = mode_in::<Octo>(DD::from_f64(1.3), 0);
    let (b, _, _) = mode_in::<Quad>(DD::from_f64(1.3), QD_MUT_NOREFINE);
    let g = rel_gap::<Quad>(&a, &b, n, ld);
    println!("control, quad-double horizon solves unrefined: relative gap {g:.2e}");
    assert!(g > 1e-40, "control did not fire");
}

/// The refined horizon solves at a real rate have converged: the default steps (two, three for tangents) against four
/// (QD_MUT_REFINE4). Pass: max relative gap at most 1e-58 at 1.3 and -0.7.
#[test]
fn mode_q_refinement_converged() {
    for lr in [1.3, -0.7] {
        let (a, ld, n) = mode_in::<Quad>(DD::from_f64(lr), 0);
        let (b, _, _) = mode_in::<Quad>(DD::from_f64(lr), QD_MUT_REFINE4);
        let g = rel_gap::<Quad>(&a, &b, n, ld);
        println!("rate {lr}: default refinement against four steps, relative gap {g:.2e}");
        assert!(g <= 1e-58, "rate {lr}: {g:e}");
    }
}

/// At rate zero the quad-double mode matrix is the quad-double Jacobian's leading block (model::jacobian_in): pass,
/// every entry equal in the bits of all four parts. The control: the mode matrix at 2^-60 must differ.
#[test]
fn mode_q_rate_zero_is_the_jacobian_block() {
    let (ctx, zh, zl) = synth();
    let l = zh.len();
    let z: [&[f64]; 4] = [&zh, &zl, &vec![0.0; l], &vec![0.0; l]];
    let (j, ldj) = crate::model::jacobian_in::<Quad, 1>(&ctx, &z, false).unwrap();
    let differs = |lam: f64| -> usize {
        let (c, ld, n) = mode_in::<Quad>(DD::from_f64(lam), 0);
        (0..n)
            .flat_map(|jj| (0..n).map(move |i| (i, jj)))
            .filter(|&(i, jj)| (0..4).any(|q| c[q][jj * ld + i].to_bits() != j[q][jj * ldj + i].to_bits()))
            .count()
    };
    let (d, dc) = (differs(0.0), differs(2f64.powi(-60)));
    println!("rate zero against the quad-double Jacobian: {d} entries differ; control (rate 2^-60) {dc}");
    assert_eq!(d, 0);
    assert!(dc > 0, "control did not fire");
}
