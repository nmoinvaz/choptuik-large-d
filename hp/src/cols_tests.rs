//! The lane-vector form of the Taylor recursion (`taylor_coeffs_cols`, the f64 lanes' default) against the field-wide
//! form (`taylor_coeffs_fields`) bit for bit: the Taylor coefficients on random and edge states, and the f64 Jacobian
//! of the synthetic context (qd_residual::synth_b), with 1, 4 and 8 f64 tangents, the double-double base and values
//! alone. Each check's control (Ctx::mutate = COLS_MUT_ULP, one ulp of p_1 in one lane of the lane-vector form) must
//! fail it.
//!
//! ```text
//! cargo test --profile release-dev --lib cols_
//! ```
use crate::dd::DD;
use crate::dv::DV;
use crate::field::Fl;
use crate::lane::{Dtn, Fu, Fun, Lane, F8};
use crate::model::{taylor_coeffs, work, Ctx, BY_FIELDS, COLS_MUT_ULP};
use crate::qd_residual::synth_b;
use crate::v8::V8;
use multishoot::Work;

/// xorshift64*: a fixed stream of uniform [0, 1) doubles
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The inputs of the coefficient checks: 0 random, 1 signed zeros (a third of the entries +0 or -0), 2 zero tangents,
/// 3 extreme magnitudes (entries times 2^+-500, x0 = 2^-600: overflow, underflow and their infinities and NaNs), 4
/// NaN and infinite entries among random ones
const EDGES: [&str; 5] = ["random", "signed zeros", "zero tangents", "extreme magnitudes", "NaN and inf"];

/// one entry of the kind `edge`
fn entry(r: &mut Rng, edge: usize) -> f64 {
    let x = 2.0 * r.next() - 1.0;
    match edge {
        1 if r.next() < 1.0 / 3.0 => {
            if r.next() < 0.5 {
                0.0
            } else {
                -0.0
            }
        }
        3 => x * if r.next() < 0.5 { 2f64.powi(500) } else { 2f64.powi(-500) },
        4 => match (r.next() * 8.0) as usize {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            _ => x,
        },
        _ => x,
    }
}

/// a field of entries of the kind `edge` (values in double-double: hi and a lo below it)
fn dv_field<const B: usize>(r: &mut Rng, edge: usize) -> Fl<DV, B> {
    let m = 8 * B;
    let hi: Vec<f64> = (0..m).map(|_| entry(r, edge)).collect();
    let lo: Vec<f64> = hi.iter().map(|h| if h.is_finite() { h * 2f64.powi(-60) * (r.next() - 0.5) } else { 0.0 }).collect();
    Fl::load(&hi, &lo)
}

/// a lane field: values and every f64 tangent drawn by `edge` (tangents zero for edge 2)
fn lane_field<T: Lane, const B: usize>(r: &mut Rng, edge: usize) -> Fl<T, B> {
    let v = dv_field::<B>(r, edge);
    let mut x = Fl::<T, B>::from_vt(&v, &Fl::zero());
    if T::TAN_F64 && edge != 2 {
        for k in 0..T::NT {
            let t = Fl::<T, B>::from_vt(&v, &dv_field::<B>(r, edge)).tan64(0);
            x.set_tan64(k, &t);
        }
    }
    x
}

/// whether two V8 agree bit for bit, a NaN matching any NaN (the payload of a NaN result depends on the operand order
/// the code generator chose)
fn same(a: V8, b: V8) -> bool {
    a.to_array().iter().zip(b.to_array()).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// entries (lane vectors' values, hi and lo, and f64 tangents) of two works that differ, and the number compared
fn work_diff<T: Lane, const B: usize>(a: &Work<T, B>, b: &Work<T, B>) -> (usize, usize) {
    let (mut nd, mut nt) = (0, 0);
    for (sa, sb) in a.c.iter().chain(&a.aux).zip(b.c.iter().chain(&b.aux)) {
        for (fa, fb) in sa.iter().zip(sb) {
            for (xa, xb) in fa.0.iter().zip(&fb.0) {
                let (va, vb) = (xa.val(), xb.val());
                let mut d = !same(va.hi, vb.hi) || !same(va.lo, vb.lo);
                if T::TAN_F64 {
                    d |= (0..T::NT).any(|k| !same(xa.tan64(k), xb.tan64(k)));
                }
                nd += d as usize;
                nt += 1;
            }
        }
    }
    (nd, nt)
}

/// the Taylor coefficients of a random state of kind `edge` by both forms at order kt: (differing entries, entries)
fn coeffs_diff<T: Lane, const B: usize>(ctx: &Ctx<B>, seed: u64, kt: usize, edge: usize) -> (usize, usize) {
    let mut r = Rng(seed);
    let x0 = if edge == 3 { 2f64.powi(-600) } else { 0.01 + 0.9 * r.next() };
    let mut omega = T::from_vt(DV::splat(DD::from_f64(1.0 + r.next())), DV::splat(DD::from_f64(entry(&mut r, edge))));
    if T::TAN_F64 && edge != 2 {
        for k in 1..T::NT {
            omega.set_tan64(k, V8::splat(entry(&mut r, edge)));
        }
    }
    let s: Vec<Fl<T, B>> = (0..4).map(|_| lane_field::<T, B>(&mut r, edge)).collect();
    let mut wf = work::<T, B>(kt);
    let mut wc = work::<T, B>(kt);
    for f in 0..4 {
        wf.c[f][0] = s[f];
        wc.c[f][0] = s[f];
    }
    BY_FIELDS.with(|c| c.set(true));
    taylor_coeffs(ctx, x0, omega, &mut wf);
    BY_FIELDS.with(|c| c.set(false));
    taylor_coeffs(ctx, x0, omega, &mut wc);
    work_diff(&wf, &wc)
}

/// every edge kind, orders kt 1, 2, 5, 20, 36 and three seeds each: (differing, compared), and the differing count of
/// the random inputs alone
fn coeffs_all<T: Lane, const B: usize>(mutate: u8) -> (usize, usize, usize) {
    let (mut ctx, _, _) = synth_b::<B>();
    ctx.mutate = mutate;
    let (mut nd, mut nt, mut nr) = (0, 0, 0);
    for (e, name) in EDGES.iter().enumerate() {
        let mut ne = 0;
        for kt in [1, 2, 5, 20, 36] {
            for s in 0..3u64 {
                let (d, t) = coeffs_diff::<T, B>(&ctx, 0x9e37_79b9_7f4a_7c15 ^ (s * 977 + kt as u64 * 31 + e as u64), kt, e);
                ne += d;
                nt += t;
            }
        }
        if e == 0 {
            nr = ne;
        }
        if mutate == 0 && ne > 0 {
            println!("  B {B} {}: {name}: {ne} entries differ", std::any::type_name::<T>());
        }
        nd += ne;
    }
    (nd, nt, nr)
}

/// Pass: on every lane type with f64 parts (values alone, 1, 4 and 8 f64 tangents, the double-double base with 4) and
/// B = 1, 2, 3, every Taylor coefficient and scratch series of the lane-vector form equals the field-wide form's bit
/// for bit, on random states, signed zeros, zero tangents, extreme magnitudes and NaN or infinite entries. Control:
/// with p_1 one ulp off (COLS_MUT_ULP) the coefficients of the random inputs must differ.
#[test]
fn cols_coeffs_bitwise() {
    fn one<T: Lane, const B: usize>() -> (usize, usize) {
        let (d, t, _) = coeffs_all::<T, B>(0);
        let (_, _, dm) = coeffs_all::<T, B>(COLS_MUT_ULP);
        println!("  B {B} {}: {d} of {t} entries differ; control: {dm} differ", std::any::type_name::<T>());
        assert_eq!(d, 0, "the lane-vector form differs from the field-wide one");
        assert!(dm > 0, "control: one ulp of p_1 not seen");
        (d, t)
    }
    let mut tot = 0;
    tot += one::<F8, 1>().1 + one::<F8, 3>().1;
    tot += one::<Fu, 1>().1 + one::<Fu, 2>().1 + one::<Fu, 3>().1;
    tot += one::<Fun<1>, 2>().1;
    tot += one::<Fun<4>, 1>().1 + one::<Fun<4>, 2>().1 + one::<Fun<4>, 3>().1;
    tot += one::<Fun<8>, 1>().1 + one::<Fun<8>, 3>().1;
    tot += one::<Dtn<4>, 1>().1 + one::<Dtn<4>, 2>().1;
    println!("Taylor coefficients, lane vectors against fields: {tot} entries compared, none differ");
}

/// the f64 Jacobian of the synthetic context at M = 8 B by lanes T: field-wide serial, lane-vector serial and lane-vector
/// parallel; (differing entries of the first two, of the last two, entries)
fn jac_diff<T: Lane, const B: usize>(mutate: u8) -> (usize, usize, usize) {
    let (mut ctx, zh, zl) = synth_b::<B>();
    ctx.mutate = mutate;
    let sh = ctx.shooting();
    let z = [&zh[..], &zl[..]];
    BY_FIELDS.with(|c| c.set(true));
    let (jf, ld) = sh.jacobian_f64::<T>(&z, false).unwrap();
    BY_FIELDS.with(|c| c.set(false));
    let (jc, _) = sh.jacobian_f64::<T>(&z, false).unwrap();
    let (jp, _) = sh.jacobian_f64::<T>(&z, true).unwrap();
    let l = ctx.len();
    let (mut d1, mut d2) = (0, 0);
    for j in 0..l {
        for i in 0..l {
            let q = j * ld + i;
            d1 += (jf[q].to_bits() != jc[q].to_bits()) as usize;
            d2 += (jc[q].to_bits() != jp[q].to_bits()) as usize;
        }
    }
    (d1, d2, l * l)
}

/// Pass: the f64 Jacobians of the synthetic context (M = 8, 16, 24) with 1, 4 and 8 f64 tangents and with the
/// double-double base agree bit for bit between the field-wide and the lane-vector forms, serial and parallel. Control:
/// with p_1 one ulp off (COLS_MUT_ULP) the lane-vector Jacobian must differ from the field-wide one.
#[test]
fn cols_jacobian_bitwise() {
    fn one<T: Lane, const B: usize>() {
        let (d1, d2, n) = jac_diff::<T, B>(0);
        let (m1, _, _) = jac_diff::<T, B>(COLS_MUT_ULP);
        println!("  M {} {}: {d1} of {n} entries differ (parallel: {d2}); control: {m1} differ", 8 * B, std::any::type_name::<T>());
        assert_eq!((d1, d2), (0, 0), "the lane-vector Jacobian differs from the field-wide one");
        assert!(m1 > 0, "control: one ulp of p_1 not seen in the Jacobian");
    }
    one::<Fu, 1>();
    one::<Fun<4>, 1>();
    one::<Fun<8>, 1>();
    one::<Dtn<4>, 1>();
    one::<Fu, 2>();
    one::<Fun<4>, 2>();
    one::<Fun<8>, 2>();
    one::<Dtn<4>, 2>();
    one::<Fun<4>, 3>();
}
