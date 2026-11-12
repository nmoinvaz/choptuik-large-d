//! Tests of the frozen rows (Ctx::freeze) on multishoot's projected rows, on the synthetic context of
//! `qd_residual::synth` (M = 8) with kfreeze 3:
//!
//! ```text
//! cargo test --release --lib freeze_   check_freeze takes the projector rounded to double-double and refuses one entry
//!                                      moved by 1e-25 (the control); the frozen jvp equals the frozen double-double
//!                                      Jacobian times v; the f64 Jacobian with HDD_BLOCK has the frozen dd Jacobian's
//!                                      hi parts in the horizon rows bitwise and the default f64 Jacobian's elsewhere
//!                                      (control: HDD_LASTSEG does not give the horizon rows)
//! ```
use crate::dd::DD;
use crate::field::DMat;
use crate::model::{check_freeze, jacobian, jacobian_f64, jvp, kfreeze_of, Ctx, HDD_BLOCK, HDD_LASTSEG};
use crate::qd_residual::synth;
use multidouble::precision::Double;
use multishoot::periodic::Parity;
use multishoot::ProjectedRows;

/// (Pi_low, Pi_high) of the odd harmonics k <= kf on M = 8 points, multishoot's projector rounded to double-double,
/// one entry of Pi_low moved by `bump`
fn pair(kf: usize, bump: f64) -> (DMat<1>, DMat<1>) {
    let m = 8;
    let p = ProjectedRows::new(m, 0, 0, 0, Parity::Odd, kf);
    let (mut h, mut l, mut hh, mut hl) = (vec![0.0; m * m], vec![0.0; m * m], vec![0.0; m * m], vec![0.0; m * m]);
    for i in 0..m {
        for j in 0..m {
            let v = p.entry::<Double>(true, i, j, false) + DD::from_f64(if (i, j) == (1, 2) { bump } else { 0.0 });
            let w = DD::from_f64(if i == j { 1.0 } else { 0.0 }) - v;
            (h[i * m + j], l[i * m + j], hh[i * m + j], hl[i * m + j]) = (v.hi, v.lo, w.hi, w.lo);
        }
    }
    (DMat::<1>::from_rows(&h, &l), DMat::<1>::from_rows(&hh, &hl))
}

fn frozen() -> (Ctx<1>, Vec<f64>, Vec<f64>) {
    let (mut ctx, zh, zl) = synth();
    ctx.freeze = Some(pair(3, 0.0));
    (ctx, zh, zl)
}

#[test]
fn freeze_projector_is_recognized() {
    for kf in [1, 3, 5, 7] {
        let (lo, _) = pair(kf, 0.0);
        assert_eq!(kfreeze_of(&lo), kf);
        assert_eq!(check_freeze(&lo), Ok(kf));
    }
    let (lo, _) = pair(3, 1e-25);
    assert!(check_freeze(&lo).is_err(), "control: a projector moved by 1e-25 is accepted");
}

/// max over rows of |jvp(v) - J v| / (|J| |v|)_i on the frozen context
#[test]
fn freeze_jvp_matches_dd_jacobian() {
    let (ctx, zh, zl) = frozen();
    let n = ctx.len();
    let (jh, jl, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let vh: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
    let vl = vec![0.0; n];
    let (yh, yl, _, _) = jvp(&ctx, &zh, &zl, &vh, &vl, false, 0).unwrap();
    let mut worst = 0.0f64;
    for i in 0..n {
        let (mut acc, mut scale) = (DD::ZERO, 0.0f64);
        for j in 0..n {
            let a = DD { hi: jh[j * ld + i], lo: jl[j * ld + i] };
            acc = acc + a * DD::from_f64(vh[j]);
            scale += (a.hi * vh[j]).abs();
        }
        if scale > 0.0 {
            worst = worst.max((DD { hi: yh[i], lo: yl[i] } - acc).hi.abs() / scale);
        }
    }
    println!("frozen jvp vs the frozen double-double Jacobian times v: {worst:.2e}");
    assert!(worst < 1e-28, "{worst:e}");
}

/// (rows of block `blk` of the f64 Jacobian with Ctx::f64_dd = hdd equal to the frozen dd Jacobian's hi parts
/// bitwise, every other row equal to the frozen default f64 Jacobian's bitwise)
fn hdd_rows(hdd: u8, blk: usize) -> (bool, bool) {
    let (mut ctx, zh, zl) = frozen();
    let n = ctx.len();
    let (jh, _, ld) = jacobian(&ctx, &zh, &zl, false).unwrap();
    let (j0, _) = jacobian_f64(&ctx, &zh, &zl, false).unwrap();
    ctx.f64_dd = hdd;
    let (j1, _) = jacobian_f64(&ctx, &zh, &zl, false).unwrap();
    let rows = blk * 32..(blk + 1) * 32;
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
fn freeze_hdd_block_rows_are_dd() {
    let k = frozen().0.k();
    let (a, b) = hdd_rows(HDD_BLOCK, k);
    let (c, _) = hdd_rows(HDD_LASTSEG, k);
    println!("frozen, HDD_BLOCK: horizon rows = dd hi {a}, other rows = f64 {b}; HDD_LASTSEG: horizon rows = dd hi {c}");
    assert!(a && b, "HDD_BLOCK rows with frozen rows");
    assert!(!c, "control: HDD_LASTSEG must not give the horizon rows in double-double");
}
