//! Tests of the large-D mixed-precision additions (notes/LARGE_D_MIXED.md) on the small synthetic context of
//! `qd_residual::synth` (M = 8, five nodes):
//!
//! ```text
//! cargo test --release --lib jvp_     the exact Jacobian-vector product against the double-double Jacobian times v
//!                                     (control: the Delta tangent scaled by 1 + 1e-20 must fail)
//! cargo test --release --lib hdd_     the f64 Jacobian with Ctx::f64_dd = HDD_BLOCK: the horizon block's rows are the
//!                                     double-double Jacobian's hi parts and every other row the default f64 Jacobian's,
//!                                     bitwise (control: HDD_LASTSEG gives the last segment's rows, not the horizon's)
//! cargo test --release --lib halpha_  Ctx::horizon_alpha: the control HALPHA_MUT gives the default's residual and
//!                                     Jacobians bit for bit; HALPHA_ON changes them, by rounding only
//! ```
use crate::dd::DD;
use crate::model::{jacobian, jacobian_f64, jvp, residual, HALPHA_MUT, HALPHA_ON, HDD_BLOCK, HDD_LASTSEG, JVP_MUT_DELTA};
use crate::qd_residual::synth;

/// a deterministic pseudo-random vector in (-1, 1), with lo parts of relative size ~1e-17
fn vec_dd(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut h = vec![0.0; n];
    let mut l = vec![0.0; n];
    for i in 0..n {
        let v = DD::from_f64(next()) + DD::from_f64(next() * 1e-17);
        h[i] = v.hi;
        l[i] = v.lo;
    }
    (h, l)
}

/// max over entries of |jvp(v) - J v| / (|J| |v|)_i, J the double-double Jacobian, J v summed in double-double
fn jvp_vs_matrix(mutate: u8) -> f64 {
    let (ctx, zh, zl) = synth();
    let n = ctx.len();
    let (jh, jl, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let (vh, vl) = vec_dd(n, 7);
    let (yh, yl, _, _) = jvp(&ctx, &zh, &zl, &vh, &vl, false, mutate).unwrap();
    let mut worst = 0.0f64;
    for i in 0..n {
        let mut acc = DD::ZERO;
        let mut scale = 0.0;
        for j in 0..n {
            let a = DD { hi: jh[j * ld + i], lo: jl[j * ld + i] };
            acc = acc + a * DD { hi: vh[j], lo: vl[j] };
            scale += (a.hi * vh[j]).abs();
        }
        let d = (DD { hi: yh[i], lo: yl[i] } - acc).hi.abs();
        if scale > 0.0 {
            worst = worst.max(d / scale);
        }
    }
    worst
}

#[test]
fn jvp_matches_dd_jacobian() {
    let r = jvp_vs_matrix(0);
    let c = jvp_vs_matrix(JVP_MUT_DELTA);
    println!("jvp vs double-double Jacobian times v: {r:.2e}; with the Delta tangent x (1 + 1e-20): {c:.2e}");
    assert!(r < 1e-28, "jvp and J v differ by {r:e}");
    assert!(c > 1e-28, "control: the mutated product agrees ({c:e})");
}

/// (rows of block `blk` of the f64 Jacobian with Ctx::f64_dd = hdd equal to the dd Jacobian's hi parts bitwise,
///  every row outside block `blk` equal to the default f64 Jacobian's bitwise)
fn hdd_rows(hdd: u8, blk: usize) -> (bool, bool) {
    let (mut ctx, zh, zl) = synth();
    let n = ctx.len();
    let m = 8;
    let (jh, _, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let (j0, ld0) = jacobian_f64(&ctx, &zh, &zl, false).unwrap();
    ctx.f64_dd = hdd;
    let (j1, ld1) = jacobian_f64(&ctx, &zh, &zl, false).unwrap();
    assert!(ld == ld0 && ld == ld1);
    let rows = blk * 4 * m..(blk + 1) * 4 * m;
    let (mut in_blk, mut outside) = (true, true);
    for c in 0..n {
        for r in 0..n {
            let v = j1[c * ld + r].to_bits();
            if rows.contains(&r) {
                in_blk &= v == jh[c * ld + r].to_bits();
            } else {
                outside &= v == j0[c * ld + r].to_bits();
            }
        }
    }
    (in_blk, outside)
}

#[test]
fn hdd_block_rows_are_dd() {
    let (ctx, _, _) = synth();
    let k = ctx.k();
    let (a, b) = hdd_rows(HDD_BLOCK, k);
    let (c, d) = hdd_rows(HDD_LASTSEG, k);
    let (e, f) = hdd_rows(HDD_LASTSEG, k - 1);
    println!("HDD_BLOCK: horizon rows = dd hi {a}, other rows = f64 {b}; HDD_LASTSEG: horizon rows = dd hi {c}, rows outside the horizon block = f64 {d}; last segment rows = dd hi {e}, others = f64 {f}");
    assert!(a && b, "HDD_BLOCK rows");
    assert!(!c, "control: HDD_LASTSEG must not give the horizon rows in double-double");
    assert!(e && f, "HDD_LASTSEG rows");
}

/// (residual hi, lo; dd Jacobian hi, lo; f64 Jacobian) of the synthetic context with Ctx::horizon_alpha = mode
#[allow(clippy::type_complexity)]
fn with_alpha(mode: u8) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut ctx, zh, zl) = synth();
    ctx.horizon_alpha = mode;
    let (fh, fl) = residual(&ctx, &zh, &zl, false).unwrap();
    let (jh, jl, _) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let (j64, _) = jacobian_f64(&ctx, &zh, &zl, false).unwrap();
    (fh, fl, jh, jl, j64)
}

/// max |(ah + al) - (bh + bl)| (in double-double), and the largest |ah|
fn dd_diff(ah: &[f64], al: &[f64], bh: &[f64], bl: &[f64]) -> (f64, f64) {
    let mut d = 0.0f64;
    let mut s = 0.0f64;
    for i in 0..ah.len() {
        d = d.max((DD { hi: ah[i], lo: al[i] } - DD { hi: bh[i], lo: bl[i] }).hi.abs());
        s = s.max(ah[i].abs());
    }
    (d, s)
}

#[test]
fn halpha_control_bitwise_on_rounding_only() {
    let base = with_alpha(0);
    let ctl = with_alpha(HALPHA_MUT);
    assert!(ctl.0 == base.0 && ctl.1 == base.1, "control: the residual differs from the default");
    assert!(ctl.2 == base.2 && ctl.3 == base.3, "control: the double-double Jacobian differs from the default");
    assert!(ctl.4 == base.4, "control: the f64 Jacobian differs from the default");
    let on = with_alpha(HALPHA_ON);
    let (df, sf) = dd_diff(&on.0, &on.1, &base.0, &base.1);
    let (dj, sj) = dd_diff(&on.2, &on.3, &base.2, &base.3);
    let d64 = on.4.iter().zip(&base.4).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    let s64 = base.4.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    println!("HALPHA_ON against the default: residual {df:.2e} (largest entry {sf:.2e}), dd Jacobian {dj:.2e} ({sj:.2e}), f64 Jacobian {d64:.2e} ({s64:.2e})");
    // alpha = a - 1 is exact, so the two forms differ by rounding only; that they differ at all shows the alpha path ran
    // (a planted error, 3 alpha^2 as 3.000001 alpha^2, gives 9e-13 and 6e-9 and fails the bound)
    assert!(df > 0.0 || dj > 0.0, "HALPHA_ON gives the default's results bit for bit: the alpha path did not run");
    assert!(df <= 1e-28 * sf.max(1.0) && dj <= 1e-28 * sj.max(1.0), "HALPHA_ON differs by more than rounding");
    assert!(d64 <= 1e-12 * s64, "HALPHA_ON: the f64 Jacobian differs by more than rounding ({d64:e})");
}
