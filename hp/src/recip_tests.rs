//! Tests of the horizon series' 1/k (Ctx::exact_recip, #1043): with RECIP_EXACT each lane multiplies by 1/k rounded
//! once to its own precision, and with the flag off by `recip_int(k)` as before. Criteria fixed before the first run
//! (the doc of each test); each check's control must fail it.
//!
//! ```text
//! cargo test --profile release-dev --lib recip_
//! ```
use crate::dd::DD;
use crate::dv::DV;
use crate::field::Fl;
use crate::lane::{Dc, Dtn, Du, Fu, Fun, Jet, Lane, OJet, Ou, QJet, Qu, F8};
use crate::model::{mul_recip, recip_int, recip_lane, recip_lane_parts, recip_parts, recip_value, Ctx, RECIP_EXACT, RECIP_MUT_ULP, RECIP_PLANT, SPLIT_MIDDLE};
use crate::qd_residual::synth;
use crate::qv::QV;
use crate::split_tests::{parts8, planted, rel, rows_differ, session, LANES};
use multidouble::exact::Dy;
use multidouble::native::{Oun, OV};
use multidouble::od::OD;
use multidouble::precision::{Octo, Quad};
use std::cmp::Ordering;

/// the largest k checked: past the chain's planned horizon orders (ks 626 at D 400, so k <= 627; I966-dh370's
/// ks_chain3b plan)
const KMAX: usize = 4096;

/// Whether `parts` is the greedy round-to-nearest expansion of the exact value v / w (v, w exact dyadics, w > 0):
/// each part is nearer to v / w less the parts before it than both its f64 neighbours are, which is checked exactly
/// as |v - w (s + c)| for c the part and each neighbour (s the sum of the parts before it). None, or the index of the
/// first part that fails.
fn first_not_greedy(parts: &[f64], v: &Dy, w: &Dy) -> Option<usize> {
    let mut s = Dy::zero();
    for (i, &r) in parts.iter().enumerate() {
        let dist = |c: f64| v.sub(&w.mul(&s.add(&Dy::from_f64(c)))).abs();
        let d0 = dist(r);
        if [r.next_up(), r.next_down()].iter().any(|&c| dist(c).cmp_abs(&d0) != Ordering::Greater) {
            return Some(i);
        }
        s = s.add(&Dy::from_f64(r));
    }
    None
}

/// `first_not_greedy` for 1/k
fn greedy_recip(parts: &[f64], k: usize) -> Option<usize> {
    first_not_greedy(parts, &Dy::from_f64(1.0), &Dy::from_f64(k as f64))
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The value's parts of every lane of x (`recip_lane_parts::<T>()` of them, read without rounding: an octo-double
/// lane's eight, a quad-double's four, a double-double's hi and lo, an f64 value's hi), and whether all its other
/// coefficients (tangents, a jet's orders past 0, an f64 value's lo) are zero
fn lane_parts<T: Lane>(x: &T) -> ([Vec<f64>; 8], bool) {
    let np = recip_lane_parts::<T>();
    let higher = T::NJ.max(T::NTD + 1).max(1 + T::TANGENT as usize);
    let (vals, zero): ([Vec<f64>; 8], bool) = if T::VAL_OD {
        let v = x.coef_od(0).lanes();
        let z = (1..higher).all(|i| x.coef_od(i).lanes().iter().all(|o| o.x.iter().all(|&p| p == 0.0)));
        (std::array::from_fn(|j| v[j].x.to_vec()), z)
    } else if T::VAL_QD {
        let v = x.coef_qd(0).lanes();
        let z = (1..higher).all(|i| x.coef_qd(i).lanes().iter().all(|q| q.x.iter().all(|&p| p == 0.0)));
        (std::array::from_fn(|j| v[j].x.to_vec()), z)
    } else {
        let v = x.val();
        let (h, l) = (v.hi.to_array(), v.lo.to_array());
        let dz = |d: DV| d.hi.to_array().iter().chain(d.lo.to_array().iter()).all(|&p| p == 0.0);
        let mut z = (1..higher).all(|i| dz(x.coef(i)));
        if T::TAN_F64 {
            z &= (0..T::NT).all(|k| x.tan64(k).to_array().iter().all(|&p| p == 0.0));
        }
        if np == 1 {
            z &= l.iter().all(|&p| p == 0.0);
        }
        (std::array::from_fn(|j| if np == 1 { vec![h[j]] } else { vec![h[j], l[j]] }), z)
    };
    (vals, zero)
}

#[derive(Default, Debug)]
struct LaneCheck {
    /// k whose flag-on 1/k (the product of the constant 1) is not recip_parts(k)'s first parts in every lane, or has
    /// a nonzero other coefficient
    on_bad: usize,
    /// k whose quad- or octo-double constant (`recip_lane`, lanes that are not jets) is not recip_parts(k)'s parts
    const_bad: usize,
    /// k whose flag-off 1/k is not recip_int(k)'s parts (zeros past two) in every lane
    off_bad: usize,
    /// k whose flag-on 1/k differs from recip_int(k)'s parts: the control of the flag-off comparison
    on_vs_int: usize,
    /// k whose 1/k under RECIP_MUT_ULP is still recip_parts(k)'s: the control must leave none
    mut_missed: usize,
}

/// The checks of `recip_lanes_exact` on lane type T for k = 1..=KMAX against the verified parts `want[k]`
fn lane_check<T: Lane>(want: &[[f64; 8]]) -> LaneCheck {
    let (mut ctx, _, _) = synth();
    let one = Fl::<T, 1>::splat(DD::ONE);
    let np = recip_lane_parts::<T>();
    let mut c = LaneCheck::default();
    let matches = |x: &T, w: &[f64]| {
        let (v, zero) = lane_parts(x);
        zero && v.iter().all(|lane| bits(lane) == bits(w))
    };
    for k in 1..=KMAX {
        let w = &want[k][..np];
        let d = recip_int(k);
        let wi = &[d.hi, d.lo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..np];
        (ctx.exact_recip, ctx.mutate) = (RECIP_EXACT, 0);
        let on = mul_recip(&ctx, one, k).0[0];
        c.on_bad += !matches(&on, w) as usize;
        c.on_vs_int += !matches(&on, wi) as usize;
        if (T::VAL_QD || T::VAL_OD) && T::NJ == 0 {
            c.const_bad += !matches(&recip_lane::<T>(&want[k]), w) as usize;
        }
        ctx.mutate = RECIP_MUT_ULP;
        c.mut_missed += matches(&mul_recip(&ctx, one, k).0[0], w) as usize;
        (ctx.exact_recip, ctx.mutate) = (0, 0);
        c.off_bad += !matches(&mul_recip(&ctx, one, k).0[0], wi) as usize;
    }
    c
}

/// Pass (fixed before the first run):
/// - recip_parts(k) is 1/k's greedy round-to-nearest expansion in eight parts for k = 1..=KMAX, checked exactly by
///   `first_not_greedy` (multidouble's dyadic arithmetic, no part of recip_parts' integer algorithm);
/// - on every lane type, f64 (F8, Fu, Fun<2>, Dtn<2>), double-double (DV, Du, Dc, Jet<8>), quad-double (QV, Qu,
///   QJet<8>) and octo-double (OV, Ou, Oun<2>, OJet<8>), the flag-on product of the constant 1 and 1/k is
///   recip_parts(k)'s first parts (8, 4, 2 or the hi alone) bit for bit in every lane, its other coefficients zero,
///   at every k; the quad- and octo-double constants of `recip_lane` are those parts too;
/// - with the flag off the same product is recip_int(k)'s two parts, zeros past them, bit for bit at every k.
///
/// Controls, which must fail: RECIP_MUT_ULP (the last part one ulp up) on every lane at every k; recip_int widened
/// to eight parts against `first_not_greedy` at every k that is not a power of two; and the flag-off comparison
/// applied to the flag-on quad- and octo-double lanes, which must differ from recip_int at every such k.
#[test]
fn recip_lanes_exact() {
    let want: Vec<[f64; 8]> = (0..=KMAX).map(|k| if k == 0 { [0.0; 8] } else { recip_parts(k as u64) }).collect();
    let not_greedy: Vec<usize> = (1..=KMAX).filter(|&k| greedy_recip(&want[k], k).is_some()).collect();
    let pow2 = (1..=KMAX).filter(|k| k.is_power_of_two()).count();
    let int_greedy = (1..=KMAX)
        .filter(|&k| {
            let d = recip_int(k);
            greedy_recip(&[d.hi, d.lo, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k).is_none()
        })
        .count();
    let dd_differ = (1..=KMAX).filter(|&k| (want[k][0], want[k][1]) != (recip_int(k).hi, recip_int(k).lo)).count();
    println!(
        "recip_parts, k = 1..{KMAX}: {} not greedy; recip_int widened is greedy at {int_greedy} k (powers of two: \
         {pow2}); recip_int is not 1/k's nearest double-double at {dd_differ} k",
        not_greedy.len()
    );
    assert!(not_greedy.is_empty(), "recip_parts is not greedy at k = {:?}", &not_greedy[..not_greedy.len().min(10)]);
    assert_eq!(int_greedy, pow2, "the control: recip_int widened passed the greedy check where 1/k is not dyadic");
    let lanes: Vec<(&str, LaneCheck)> = vec![
        ("F8", lane_check::<F8>(&want)),
        ("Fu", lane_check::<Fu>(&want)),
        ("Fun<2>", lane_check::<Fun<2>>(&want)),
        ("Dtn<2>", lane_check::<Dtn<2>>(&want)),
        ("DV", lane_check::<DV>(&want)),
        ("Du", lane_check::<Du>(&want)),
        ("Dc", lane_check::<Dc>(&want)),
        ("Jet<8>", lane_check::<Jet<8>>(&want)),
        ("QV", lane_check::<QV>(&want)),
        ("Qu", lane_check::<Qu>(&want)),
        ("QJet<8>", lane_check::<QJet<8>>(&want)),
        ("OV", lane_check::<OV>(&want)),
        ("Ou", lane_check::<Ou>(&want)),
        ("Oun<2>", lane_check::<Oun<2>>(&want)),
        ("OJet<8>", lane_check::<OJet<8>>(&want)),
    ];
    for (name, c) in &lanes {
        println!("{name:8} {c:?}");
    }
    for (name, c) in &lanes {
        assert_eq!((c.on_bad, c.const_bad, c.off_bad), (0, 0, 0), "{name}: a 1/k is not the lane's own rounding or recip_int's");
        assert_eq!(c.mut_missed, 0, "{name}: the control RECIP_MUT_ULP was not seen");
        if name.starts_with('Q') || name.starts_with('O') {
            assert_eq!(c.on_vs_int, KMAX - pow2, "{name}: the flag-off comparison did not see the flag-on 1/k");
        }
    }
}

/// Pass (fixed before the first run): the planted 1/k (Ctx::recip_planted = k0) at k0 = 1, 2, 3, 7, 130, 294, 627 and
/// 4095, flag off and on, is the greedy eight-part expansion of c (1 + 2^-106), c the sum of the unplanted parts
/// (checked exactly), it changes 1/k by 2^-106 c to 2^-400 c, and 1/k at k0 - 1 and k0 + 1 is unchanged bit for bit.
/// Control, which must fail: the unplanted parts against the planted value's greedy check.
#[test]
fn recip_planted_value() {
    let (mut ctx, _, _) = synth();
    let plant = Dy::from_f64(1.0).add(&Dy::from_f64(RECIP_PLANT));
    let tol = Dy::from_f64(2f64.powi(-400));
    for mode in [0, RECIP_EXACT] {
        for k0 in [1, 2, 3, 7, 130, 294, 627, 4095] {
            ctx.exact_recip = mode;
            ctx.recip_planted = 0;
            let base = recip_value(&ctx, k0);
            let nb = [k0 - 1, k0 + 1].map(|k| if k == 0 { [0.0; 8] } else { recip_value(&ctx, k) });
            ctx.recip_planted = k0;
            let p = recip_value(&ctx, k0);
            let na = [k0 - 1, k0 + 1].map(|k| if k == 0 { [0.0; 8] } else { recip_value(&ctx, k) });
            let c = Dy::sum(&base);
            let target = c.mul(&plant);
            let greedy = first_not_greedy(&p, &target, &Dy::from_f64(1.0));
            let control = first_not_greedy(&base, &target, &Dy::from_f64(1.0));
            // (sum p - c) - 2^-106 c, against 2^-400 c
            let err = Dy::sum(&p).sub(&c).sub(&c.mul(&Dy::from_f64(RECIP_PLANT))).abs();
            let within = err.cmp_abs(&tol.mul(&c)) != Ordering::Greater;
            let same_nb = nb.iter().zip(&na).all(|(a, b)| bits(a) == bits(b));
            println!("mode {mode}, k0 {k0}: greedy {greedy:?}, change within 2^-400 {within}, neighbours unchanged {same_nb}, control {control:?}");
            assert!(greedy.is_none() && within && same_nb, "mode {mode}, k0 {k0}: the planted 1/k");
            assert!(control.is_some(), "mode {mode}, k0 {k0}: the control passed");
        }
    }
}

/// residual_in at P on the synthetic state (z as eight parts, P's first) with the flag set, every row in octo-double
fn residual_with<const PARTS: usize>(ctx: &mut Ctx<1>, z: &[Vec<f64>; 8], mode: u8) -> Vec<OD> {
    ctx.exact_recip = mode;
    let zr: Vec<&[f64]> = z[..PARTS].iter().map(|p| &p[..]).collect();
    let f = if PARTS == 8 {
        crate::model::residual_in::<Octo, 1>(ctx, &zr, true).unwrap()
    } else {
        crate::model::residual_in::<Quad, 1>(ctx, &zr, true).unwrap().into_iter().map(OD::from_qd).collect()
    };
    ctx.exact_recip = 0;
    f
}

/// Pass (fixed before the first run), on the synthetic state:
/// - the octo-double residual with the flag on equals the flag-off one bit for bit outside the horizon block, and
///   differs in some of its rows (the flag reaches the horizon series alone);
/// - the quad-double residual with the flag on agrees with the octo-double one to d_lane = rel <= 2^-180 (the
///   quad-double rounding, 2^-209, with room for the series' growth).
///
/// Control, which must fail: the flag-on quad-double residual against the flag-off octo-double one, rel > 2^-180.
#[test]
fn recip_residual_lanes() {
    let (mut ctx, zh, zl) = synth();
    let z = parts8(&zh, &zl);
    let m = Ctx::<1>::M;
    let (h0, h1) = (4 * m * ctx.k(), 4 * m * (ctx.k() + 1));
    let off = residual_with::<8>(&mut ctx, &z, 0);
    let on = residual_with::<8>(&mut ctx, &z, RECIP_EXACT);
    let on_q = residual_with::<4>(&mut ctx, &z, RECIP_EXACT);
    let outside = rows_differ(&on[..h0], &off[..h0]) + rows_differ(&on[h1..], &off[h1..]);
    let inside = rows_differ(&on[h0..h1], &off[h0..h1]);
    let (d_flag, d_lane, d_ctrl) = (rel(&on, &off), rel(&on_q, &on), rel(&on_q, &off));
    let bound = 2f64.powi(-180);
    println!(
        "flag on against off: {outside} rows differ outside the horizon block, {inside} of {} inside, rel {d_flag:.2e}; \
         quad-double against octo-double with the flag on: rel {d_lane:.2e} (bound {bound:.2e}); control (against the \
         flag off) {d_ctrl:.2e}",
        h1 - h0
    );
    assert_eq!(outside, 0, "the flag changed a row outside the horizon block");
    assert!(inside > 0, "the flag changed no horizon row");
    assert!(d_lane <= bound, "the quad- and octo-double lanes disagree with the flag on");
    assert!(d_ctrl > bound, "the control did not fail");
}

/// Pass (fixed before the first run), on the synthetic state with 12 planted orders and the flag on:
/// - the series by order (the session: horizon_series_at) equals residual_jet's (horizon_series on whole jets) bit
///   for bit at every order;
/// - the split form (SPLIT_MIDDLE) is within d_on(k) <= 16 max(d_off(k), 2^-420) of the default form at every order
///   k >= 1, d_off(k) the same difference with the flag off, and its order 1 is the default's bit for bit.
///
/// Controls, which must fail: the flag-on series by order against residual_jet with the flag off (rows differ), and
/// the flag-on split form against the flag-off default form (past 16 max(d_off(k), 2^-420) at some order).
#[test]
fn recip_series_forms() {
    let n = 12;
    let (mut ctx, zs) = planted(n);
    let zv: Vec<Vec<&[f64]>> = zs.iter().map(|z| z.iter().map(|p| &p[..]).collect()).collect();
    let run = |ctx: &mut Ctx<1>, mode: u8, split: u8| {
        (ctx.exact_recip, ctx.series_split) = (mode, split);
        let f = session(ctx, &zs, true);
        (ctx.exact_recip, ctx.series_split) = (0, 0);
        f
    };
    let jet = |ctx: &mut Ctx<1>, mode: u8, k: usize| {
        ctx.exact_recip = mode;
        let f = crate::model::residual_jet_in::<Octo, LANES, 1>(ctx, &zv[..k], true, k + 1).unwrap();
        ctx.exact_recip = 0;
        f
    };
    let (base_on, base_off) = (run(&mut ctx, RECIP_EXACT, 0), run(&mut ctx, 0, 0));
    let (split_on, split_off) = (run(&mut ctx, RECIP_EXACT, SPLIT_MIDDLE), run(&mut ctx, 0, SPLIT_MIDDLE));
    let (mut by_order, mut by_order_ctrl) = (0, 0);
    for k in 1..=n {
        let (on, off) = (jet(&mut ctx, RECIP_EXACT, k), jet(&mut ctx, 0, k));
        by_order += rows_differ(&base_on[k - 1], &on[k]);
        by_order_ctrl += rows_differ(&base_on[k - 1], &off[k]);
    }
    let floor = 2f64.powi(-420);
    let (mut worst, mut ctrl) = (0.0f64, 0.0f64);
    println!(" k   d_on       d_off      d_ctrl");
    for k in 1..=n {
        let (d_on, d_off, d_x) = (rel(&split_on[k - 1], &base_on[k - 1]), rel(&split_off[k - 1], &base_off[k - 1]), rel(&split_on[k - 1], &base_off[k - 1]));
        println!("{k:2}   {d_on:.2e}   {d_off:.2e}   {d_x:.2e}");
        let thr = 16.0 * d_off.max(floor);
        worst = if d_on.is_finite() && d_off.is_finite() { worst.max(d_on / thr) } else { f64::INFINITY };
        ctrl = ctrl.max(d_x / thr);
    }
    let order1 = rows_differ(&split_on[0], &base_on[0]);
    println!(
        "flag on: series by order against residual_jet {by_order} rows differ (control, against the flag off: \
         {by_order_ctrl}); split against default: order 1 {order1} rows differ, worst d_on / threshold {worst:.2e} \
         (control {ctrl:.2e})"
    );
    assert_eq!(by_order, 0, "the series by order is not residual_jet's with the flag on");
    assert!(by_order_ctrl > 0, "the control did not differ");
    assert_eq!(order1, 0, "the split form's order 1 is not the default's");
    assert!(worst <= 1.0, "the split form strays from the default with the flag on");
    assert!(ctrl > 1.0, "the control did not fail");
}
