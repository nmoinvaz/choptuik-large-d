//! Checks and timings of `fastmul::mullow` against the naive octo-double product `mullow_naive`, on planted series.
//!
//! ```text
//! cd hp/fastmul_bench && cargo run --release [-- --no-check | --no-bench | --reps R]
//! cargo test --profile release-dev --lib fastmul_planted -- --nocapture      the checks in hp's tests
//! ```
//!
//! Pass, fixed before the first run. On every planted case (below) at n = 128, 251 and 512, every coefficient of
//! every lane agrees with the naive product to 1e-120 relative, |c_fast - c_naive| <= 1e-120 |c_naive|, and every
//! inverse transform's value lies within 1/4 of its integer (`Stats::max_frac`). On series of small integers the two
//! agree bit for bit (both exact). Control, which must fail: the series spanning 10^+-60 with the slope and the blocks
//! removed (`Cfg { scale: false, blocks: false }`) differ by more than 1e-120 somewhere; a NaN planted in one output
//! coefficient fails `compare`, and one planted in an input makes `Stats::max_frac` infinite. Empty factors, in
//! either order, give zeros in both products. The edge cases (`check_edges`) have the controls
//! `--mutate zero_lane`, `--mutate e2_plain` and `--mutate e2_split_always` of hp/fastmul_bench, each of which
//! must make the run exit 1, and `fastmul_planted` runs all three.
//!
//! Planted cases (8 lanes, random signs, all eight parts random, each lane its own offsets):
//! - `flat`: |a_i| ~ 2^(-0.7 i) and |b_i| ~ 2^(-0.4 i), each times 2^(+-4) at random (a series of the chain's kind);
//! - `ramp`: log10 |a_i| from -60 to +60 over the series and log10 |b_i| from +60 to -60, each +-1 at random (opposite
//!   slopes, which no single scaling flattens);
//! - `bumpy`: log10 |a_i| = 60 sin(2 pi 1.3 i / n + phase), log10 |b_i| = 60 cos(2 pi 0.8 i / n + phase);
//! - `holes`: `flat` with a third of the coefficients and one whole lane zero.
use crate::fastmul::{mullow, mullow_naive, Cfg, Stats};
use multidouble::native::{ODotAcc, OV};
use multidouble::od::OD;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    fn u(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// a random octo-double of magnitude 2^e (within a factor 2), random sign, all eight parts in use
fn rand_od(r: &mut Rng, e: f64) -> OD {
    let sign = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
    let e0 = e.floor();
    let mut x = [0.0f64; 8];
    x[0] = sign * (1.0 + r.u()) * (e0 - 1.0).exp2();
    for j in 1..8 {
        x[j] = (r.u() - 0.5) * x[j - 1].abs() * (-52.0f64).exp2();
    }
    OD::new(x)
}

/// a series of n coefficients on 8 lanes, log2 |coefficient i| on lane l given by lg(i, l), zero where zero(i, l)
fn series(
    r: &mut Rng,
    n: usize,
    lg: impl Fn(usize, usize, &mut Rng) -> f64,
    zero: impl Fn(usize, usize) -> bool,
) -> Vec<OV> {
    (0..n)
        .map(|i| {
            let lanes: [OD; 8] = std::array::from_fn(|l| {
                if zero(i, l) {
                    OD::ZERO
                } else {
                    let e = lg(i, l, r);
                    rand_od(r, e)
                }
            });
            OV::from_lanes(&lanes)
        })
        .collect()
}

const L10: f64 = std::f64::consts::LOG2_10;

/// the planted pair (a, b) of a case
fn case(name: &str, n: usize, seed: u64) -> (Vec<OV>, Vec<OV>) {
    let mut r = Rng(seed);
    let nf = n as f64;
    let no = |_: usize, _: usize| false;
    match name {
        "flat1" => {
            let a = series(&mut r, n, |i, l, r| -1.0 * i as f64 + l as f64 + 8.0 * (r.u() - 0.5), no);
            let b = series(&mut r, n, |i, l, r| -1.0 * i as f64 - l as f64 + 8.0 * (r.u() - 0.5), no);
            (a, b)
        }
        "flat" | "holes" => {
            let holes = name == "holes";
            let z = move |i: usize, l: usize| holes && (l == 5 || (i * 7 + l) % 3 == 0);
            let a = series(&mut r, n, |i, l, r| -0.7 * i as f64 + l as f64 + 8.0 * (r.u() - 0.5), z);
            let b = series(&mut r, n, |i, l, r| -0.4 * i as f64 - l as f64 + 8.0 * (r.u() - 0.5), z);
            (a, b)
        }
        "ramp" => {
            let ramp = |i: usize| 120.0 * i as f64 / (nf - 1.0);
            let a = series(&mut r, n, |i, l, r| L10 * (-60.0 + ramp(i) + 2.0 * (r.u() - 0.5)) + l as f64, no);
            let b = series(&mut r, n, |i, l, r| L10 * (60.0 - ramp(i) + 2.0 * (r.u() - 0.5)) - l as f64, no);
            (a, b)
        }
        "bumpy" => {
            let tp = 2.0 * std::f64::consts::PI;
            let a = series(&mut r, n, |i, l, _| L10 * 60.0 * (tp * 1.3 * i as f64 / nf + 0.3 * l as f64).sin(), no);
            let b = series(&mut r, n, |i, l, _| L10 * 60.0 * (tp * 0.8 * i as f64 / nf + 0.2 * l as f64).cos(), no);
            (a, b)
        }
        "ints" => {
            // small integers: both products exact
            let int = |r: &mut Rng| -> Vec<OV> {
                (0..n)
                    .map(|_| {
                        let lanes: [OD; 8] = std::array::from_fn(|_| OD::from_f64((r.next() % 2001) as f64 - 1000.0));
                        OV::from_lanes(&lanes)
                    })
                    .collect()
            };
            let a = int(&mut r);
            let b = int(&mut r);
            (a, b)
        }
        _ => panic!("case {name}"),
    }
}

/// the fast product into fresh accumulators
fn fast(a: &[OV], b: &[OV], n: usize, cfg: &Cfg) -> (Vec<OV>, Stats) {
    let mut acc = vec![ODotAcc::zero(); n];
    let st = mullow(a, b, n, cfg, &mut acc);
    (acc.into_iter().map(|x| x.finish()).collect(), st)
}

/// max over k and lanes of |x - y| / |y| (y nonzero) and of |x - y| / (|a| * |b|)_k, the difference in octo-double,
/// both infinite when a part of x or y is not finite; and the number of coefficients that differ in any bit
fn compare(x: &[OV], y: &[OV], a: &[OV], b: &[OV]) -> (f64, f64, usize) {
    let n = y.len();
    let ah: Vec<[f64; 8]> = a.iter().map(|v| v.c[0].to_array().map(f64::abs)).collect();
    let bh: Vec<[f64; 8]> = b.iter().map(|v| v.c[0].to_array().map(f64::abs)).collect();
    let (mut rc, mut ra, mut differ) = (0.0f64, 0.0f64, 0);
    for k in 0..n {
        let (xl, yl) = (x[k].lanes(), y[k].lanes());
        for l in 0..8 {
            let d = (xl[l] - yl[l]).to_f64().abs();
            let yv = yl[l].to_f64().abs();
            // a part that is not finite on either side fails the comparison (f64::max would drop a NaN)
            if xl[l].x.iter().chain(yl[l].x.iter()).any(|p| !p.is_finite()) || !d.is_finite() {
                (rc, ra, differ) = (f64::INFINITY, f64::INFINITY, differ + 1);
                continue;
            }
            let ab: f64 = (0..=k).filter(|&i| i < ah.len() && k - i < bh.len()).map(|i| ah[i][l] * bh[k - i][l]).sum();
            if yv > 0.0 {
                rc = rc.max(d / yv);
            } else if d > 0.0 {
                rc = f64::INFINITY;
            }
            if ab > 0.0 {
                ra = ra.max(d / ab);
            }
            if xl[l].x.iter().zip(yl[l].x.iter()).any(|(p, q)| p.to_bits() != q.to_bits()) {
                differ += 1;
            }
        }
    }
    (rc, ra, differ)
}

/// the accuracy checks (module doc); true when every criterion holds
pub fn check() -> bool {
    let mut ok = true;
    let cfg = Cfg::default();
    let ctrl = Cfg { scale: false, blocks: false, ..cfg };
    println!(
        "fastmul check: prec {}, spread {}; pass |c_fast - c_naive| <= 1e-120 |c_naive|, max_frac < 1/4",
        cfg.prec, cfg.spread
    );
    println!(
        "case     n    blocks   pairs  digits  max_frac   rel |c|    rel |a|*|b|  differ | ctrl rel |c|  scale-only  \
         blocks-only"
    );
    for &n in &[128usize, 251, 512] {
        for (ci, name) in ["flat", "holes", "ramp", "bumpy", "ints"].iter().enumerate() {
            let (a, b) = case(name, n, 1000 + 17 * n as u64 + ci as u64);
            let naive = mullow_naive(&a, &b, n);
            let (fx, st) = fast(&a, &b, n, &cfg);
            let (rc, ra, differ) = compare(&fx, &naive, &a, &b);
            let pass = rc <= 1e-120 && st.max_frac < 0.25 && (*name != "ints" || differ == 0);
            ok &= pass;
            let (c0, _, _) = compare(&fast(&a, &b, n, &ctrl).0, &naive, &a, &b);
            let (c1, _, _) = compare(&fast(&a, &b, n, &Cfg { blocks: false, ..cfg }).0, &naive, &a, &b);
            let (c2, _, _) = compare(&fast(&a, &b, n, &Cfg { scale: false, ..cfg }).0, &naive, &a, &b);
            let wide = *name == "ramp" || *name == "bumpy";
            let cfail = c0 > 1e-120;
            if wide {
                ok &= cfail;
            }
            println!(
                "{name:6} {n:4}  {:3}x{:<3}  {:5}  {:5}   {:.2e}   {rc:.2e}   {ra:.2e}     {differ:5} | {c0:.2e}{}    \
                 {c1:.2e}    {c2:.2e}   {}",
                st.blocks.0,
                st.blocks.1,
                st.pairs,
                st.digits,
                st.max_frac,
                if wide { if cfail { " (fails)" } else { " (DID NOT FAIL)" } } else { "" },
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }
    // controls, which must fail: a NaN planted in one output coefficient (compare), and in one input (max_frac)
    let (a, b) = case("flat", 128, 77);
    let naive = mullow_naive(&a, &b, 128);
    let (mut fx, _) = fast(&a, &b, 128, &cfg);
    let mut lanes = fx[40].lanes();
    lanes[3].x[2] = f64::NAN;
    fx[40] = OV::from_lanes(&lanes);
    let (rc, ra, _) = compare(&fx, &naive, &a, &b);
    let mut a2 = a.clone();
    let mut lanes = a2[17].lanes();
    lanes[5].x[0] = f64::NAN;
    a2[17] = OV::from_lanes(&lanes);
    let (fx2, st2) = fast(&a2, &b, 128, &cfg);
    let (rc2, _, _) = compare(&fx2, &mullow_naive(&a2, &b, 128), &a2, &b);
    let fired = !(rc <= 1e-120) && !(ra <= 1e-120) && !(st2.max_frac < 0.25) && !(rc2 <= 1e-120);
    println!(
        "NaN controls: an output's NaN gives rel {rc:.2e} / {ra:.2e}, an input's NaN max_frac {:.2e} and rel \
         {rc2:.2e}: {}",
        st2.max_frac,
        if fired { "fail as they must" } else { "DID NOT FAIL" }
    );
    ok &= fired;
    println!("fastmul check: {}", if ok { "PASS" } else { "FAIL" });
    ok
}

/// Edge cases, under the control `mutate` (`Cfg::mutate`; 0 off), each against the naive product or its exact value:
/// - a product past the FFT tables (8193 ones times one, n = 8193, the long-pair route inside `mullow`);
/// - a lane whose planned scale lies outside f64's range (lane 0 of a = 2^800, 2^-802, 0 and of b zero, the other
///   lanes ones, n = 3), which MUT_ZERO_LANE makes fail;
/// - the long-pair route's 2^e2 at the ends of the range: 8193 copies of 2^-600 times 2^-600 with e2 = 1200, and of
///   2^1023 times 2^-1000 with e2 = 1, exactly 1 and 2^24 each, which MUT_E2_PLAIN makes fail;
/// - outputs at the bottom of the range by the FFT route: 2^-1074 times 2^-1074 (zero) and 2^-537 times 2^-537
///   (2^-1074), n = 1;
/// - a block too wide for the digits (1 and 2^-1024 squared with spread 1024 and no slope), by the direct route,
///   whose first coefficient is exactly 1;
/// - the direct route at e2 = 0 on a wide factor (2^1023 and 2^-900 times one, blocks off, spread 2048), exactly
///   2^1023 and 2^-900, which MUT_E2_SPLIT_ALWAYS makes fail.
pub fn check_edges(mutate: u8) -> bool {
    let cfg = Cfg { mutate, ..Cfg::default() };
    let one = OV::splat_f(1.0);
    let long: Vec<OV> = vec![one; 8193];
    let (fx, st) = fast(&long, &[one], 8193, &cfg);
    let (r1, _, _) = compare(&fx, &mullow_naive(&long, &[one], 8193), &long, &[one]);
    let lane0 = |x: [f64; 3]| -> Vec<OV> {
        x.iter()
            .map(|&v| {
                let mut l = [OD::from_f64(1.0); 8];
                l[0] = OD::from_f64(v);
                OV::from_lanes(&l)
            })
            .collect()
    };
    let a = lane0([800f64.exp2(), (-802f64).exp2(), 0.0]);
    let b = lane0([0.0, 0.0, 0.0]);
    let (fx2, st2) = fast(&a, &b, 3, &cfg);
    let (r2, _, _) = compare(&fx2, &mullow_naive(&a, &b, 3), &a, &b);
    // exactly v on every lane and part 0, zero below
    let exact = |c: &[OV], v: f64| {
        c.iter().all(|x| x.lanes().iter().all(|o| o.x[0] == v && o.x[1..].iter().all(|&p| p == 0.0)))
    };
    let e2case = |x: f64, y: f64, e2: i32, want: f64| -> bool {
        let xs = vec![OV::splat_f(x); 8193];
        let mut acc = vec![ODotAcc::zero(); 8193];
        let st = crate::fastmul::mullow_pow2(&xs, &[OV::splat_f(y)], 8193, &cfg, e2, &mut acc);
        let c: Vec<OV> = acc.into_iter().map(|a| a.finish()).collect();
        st.max_frac < 0.25 && exact(&c, want)
    };
    let e2a = e2case((-600f64).exp2(), (-600f64).exp2(), 1200, 1.0);
    let e2b = e2case(1023f64.exp2(), (-1000f64).exp2(), 1, 24f64.exp2());
    let tiny = |x: f64| -> f64 {
        let v = vec![OV::splat_f(x)];
        let (c, st) = fast(&v, &v, 1, &cfg);
        let (r, _, _) = compare(&c, &mullow_naive(&v, &v, 1), &v, &v);
        if st.max_frac < 0.25 { r } else { f64::INFINITY }
    };
    let (u1, u2) = (tiny((-1074f64).exp2()), tiny((-537f64).exp2()));
    // a spread too wide for the digits (1 and 2^-1024 in one block, no slope): the direct route, c_0 exactly 1
    let wide = Cfg { scale: false, spread: 1024, ..cfg };
    let w2 = vec![one, OV::splat_f((-1024f64).exp2())];
    let mut accw = vec![ODotAcc::zero(); 2];
    let stw = crate::fastmul::mullow(&w2, &w2, 2, &wide, &mut accw);
    let cw: Vec<OV> = accw.into_iter().map(|a| a.finish()).collect();
    let (rw, _, _) = compare(&cw, &mullow_naive(&w2, &w2, 2), &w2, &w2);
    let wide_ok = exact(&cw[..1], 1.0) && rw <= 1e-120 && stw.direct > 0;
    // a wide factor by the direct route at e2 = 0 (blocks off, spread 2048): 2^1023 and 2^-900 times one, exactly
    let flat = Cfg { scale: false, blocks: false, spread: 2048, ..cfg };
    let w3 = vec![OV::splat_f(1023f64.exp2()), OV::splat_f((-900f64).exp2())];
    let mut acc3 = vec![ODotAcc::zero(); 2];
    let st3 = crate::fastmul::mullow(&w3, &[one], 2, &flat, &mut acc3);
    let c3: Vec<OV> = acc3.into_iter().map(|a| a.finish()).collect();
    let wide0_ok = exact(&c3[..1], 1023f64.exp2()) && exact(&c3[1..], (-900f64).exp2()) && st3.direct > 0;
    let ok = r1 <= 1e-120
        && r2 <= 1e-120
        && st.max_frac < 0.25
        && st2.max_frac < 0.25
        && e2a
        && e2b
        && u1 <= 1e-120
        && u2 <= 1e-120
        && wide_ok
        && wide0_ok;
    println!(
        "edges (mutate {mutate}): n 8193 past the FFT tables rel {r1:.2e}; a lane scaled out of range rel {r2:.2e} \
         (max_frac {:.1e}); e2 1200 and 1 on the long route exact {e2a}, {e2b}; 2^-1074 and 2^-537 squared rel \
         {u1:.1e}, {u2:.1e}; spread 1024 by {} direct pairs, c_0 exactly 1 {wide_ok}; 2^1023 and 2^-900 times one \
         at e2 0 exact {wide0_ok}: {}",
        st2.max_frac,
        stw.direct,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// empty factors, in both orders: the naive and the fast product give n zero coefficients, bit for bit alike
pub fn check_empty() -> bool {
    let (a, _) = case("flat", 16, 3);
    let none: Vec<OV> = Vec::new();
    let mut ok = true;
    for (x, y) in [(&none, &a), (&a, &none), (&none, &none)] {
        let naive = mullow_naive(x, y, 16);
        let (fx, _) = fast(x, y, 16, &Cfg::default());
        let zeros = |v: &OV| v.lanes().iter().all(|o| o.x.iter().all(|&p| p == 0.0));
        let zero = naive.len() == 16 && fx.len() == 16 && naive.iter().chain(&fx).all(zeros);
        let (_, _, differ) = compare(&fx, &naive, x, y);
        ok &= zero && differ == 0;
    }
    println!("empty factors (both orders and both empty): {}", if ok { "zeros, PASS" } else { "FAIL" });
    ok
}

/// seconds of the naive and the fast product of a case at n, interleaved, reps times each (median, min, max) and
/// the per-rep ratios (median, min, max)
#[allow(dead_code)] // hp/fastmul_bench's
pub fn bench(name: &str, n: usize, reps: usize, inner: usize) {
    let (a, b) = case(name, n, 7 + n as u64);
    let cfg = Cfg::default();
    let (mut tn, mut tf, mut ratio) = (Vec::new(), Vec::new(), Vec::new());
    let mut sink = 0.0;
    let mut st = Stats::default();
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..inner {
            let c = mullow_naive(&a, &b, n);
            sink += c[n - 1].c[0].to_array()[0];
        }
        let dn = t0.elapsed().as_secs_f64() / inner as f64;
        let t0 = Instant::now();
        for _ in 0..inner {
            let (c, s) = fast(&a, &b, n, &cfg);
            st = s;
            sink += c[n - 1].c[0].to_array()[0];
        }
        let df = t0.elapsed().as_secs_f64() / inner as f64;
        tn.push(dn);
        tf.push(df);
        ratio.push(dn / df);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (a0, b0, c0) = (med(&mut tn), med(&mut tf), med(&mut ratio));
    println!(
        "bench {name:6} n {n:4} ({} blocks x {}, {} digits): naive {:.3} ms ({:.3}-{:.3}), fast {:.3} ms \
         ({:.3}-{:.3}), ratio {:.2} ({:.2}-{:.2}), {reps} reps of {inner} [{}]",
        st.blocks.0,
        st.blocks.1,
        st.digits,
        a0.0 * 1e3,
        a0.1 * 1e3,
        a0.2 * 1e3,
        b0.0 * 1e3,
        b0.1 * 1e3,
        b0.2 * 1e3,
        c0.0,
        c0.1,
        c0.2,
        if sink.is_finite() { "ok" } else { "nan" }
    );
}

/// digits8 against the value: sum_r d_r 2^(16 r) - x 2^k per lane, in octo-double
pub fn check_digits() -> bool {
    let mut r = Rng(5);
    let mut worst = 0.0f64;
    for it in 0..2000 {
        let e = (r.u() * 40.0) as i32 - 20;
        let lanes: [OD; 8] = std::array::from_fn(|_| rand_od(&mut r, e as f64));
        let x = OV::from_lanes(&lanes);
        let t = 28;
        let w = 16 * t as i32 - 2;
        let k: [i32; 8] = std::array::from_fn(|_| w - e - 1 - (it % 7) as i32);
        let mut dv = vec![multidouble::native::V8::zero(); t];
        crate::fastmul::digits8(&x, &k, t, &mut dv);
        let d: Vec<[f64; 8]> = dv.iter().map(|v| v.to_array()).collect();
        for l in 0..8 {
            let mut acc = OD::ZERO;
            for (q, dq) in d.iter().enumerate() {
                acc = acc + OD::from_f64(dq[l]).mul_pwr2((16.0 * q as f64).exp2());
            }
            let xv = OD::new(lanes[l].x.map(|p| p * (k[l] as f64).exp2()));
            let diff = (acc - xv).to_f64().abs();
            // a reconstruction that is not finite fails (f64::max would drop a NaN)
            worst = if diff.is_finite() { worst.max(diff) } else { f64::INFINITY };
        }
    }
    println!("digits8: worst |sum d 2^16r - x 2^k| = {worst:.3e} (pass <= 8)");
    worst <= 8.0
}

/// products with b a single coefficient (c = a b_0 shifted), relative difference per case
pub fn check_unit() -> bool {
    let mut ok = true;
    let cases = [(16usize, 0usize, 1.0f64), (16, 3, 1.0), (16, 0, 3.0), (64, 0, 1.0), (251, 0, 1.0), (251, 0, 0.7)];
    for &(n, j, bv) in &cases {
        let mut r = Rng(9);
        let a = series(&mut r, n, |i, l, r| -1.0 * i as f64 + l as f64 + 8.0 * (r.u() - 0.5), |_, _| false);
        let b: Vec<OV> = (0..n).map(|i| OV::splat(if i == j { OD::from_f64(bv) } else { OD::ZERO })).collect();
        let naive = mullow_naive(&a, &b, n);
        let (fx, st) = fast(&a, &b, n, &Cfg::default());
        let (rc, ra, differ) = compare(&fx, &naive, &a, &b);
        println!(
            "unit n {n} b_{j} = {bv}: rel {rc:.2e} {ra:.2e} differ {differ}, digits {}, blocks {:?}",
            st.digits, st.blocks
        );
        ok &= rc <= 1e-120;
    }
    ok
}

/// the factors dumped by series_split's FASTMUL_DUMP (its doc): (name, n, a, b) per file of dir, sorted by name
#[allow(dead_code)] // hp/fastmul_bench's
pub fn load_dumps(dir: &str) -> Vec<(String, usize, Vec<OV>, Vec<OV>)> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("dump dir")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.ends_with(".bin"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(format!("{dir}/{name}")).expect("dump file");
            let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
            // a header that is not three positive integers matching the payload makes an empty entry (n = 0), which
            // check_dumps fails
            let whole = |x: f64| x >= 1.0 && x.fract() == 0.0 && x < 1e9;
            let valid = bytes.len() % 8 == 0
                && v.len() >= 3
                && whole(v[0])
                && whole(v[1])
                && whole(v[2])
                && v.len() == 3 + 64 * (v[1] as usize + v[2] as usize);
            if !valid {
                return (name, 0, Vec::new(), Vec::new());
            }
            let (n, na, nb) = (v[0] as usize, v[1] as usize, v[2] as usize);
            let ov = |i: usize| -> OV {
                let base = 3 + i * 64;
                let lanes: [OD; 8] = std::array::from_fn(|l| OD { x: std::array::from_fn(|q| v[base + 8 * l + q]) });
                OV::from_lanes(&lanes)
            };
            let a = (0..na).map(ov).collect();
            let b = (na..na + nb).map(ov).collect();
            (name, n, a, b)
        })
        .collect()
}

/// accuracy and time of the fast product against the naive one on dumped chain factors (`load_dumps`); true when
/// there is a dump and every coefficient of every product agrees to 1e-120 of itself, as `check` asks
#[allow(dead_code)] // hp/fastmul_bench's
pub fn check_dumps(dir: &str, reps: usize) -> bool {
    let cfg = Cfg::default();
    let mut ok = true;
    let (mut tn, mut tf) = (0.0, 0.0);
    println!(
        "dumped factors of {dir}: name, n, blocks, pairs, digits, max_frac, rel |c|, rel |a|*|b|, naive ms, fast ms"
    );
    let dumps = load_dumps(dir);
    if dumps.is_empty() {
        println!("dumped factors: none in {dir}; FAIL");
        return false;
    }
    for (name, n, a, b) in dumps {
        if n == 0 {
            println!("{name:18} an invalid dump (its header or length): FAIL");
            ok = false;
            continue;
        }
        let naive = mullow_naive(&a, &b, n);
        let (fx, st) = fast(&a, &b, n, &cfg);
        let (rc, ra, _) = compare(&fx, &naive, &a, &b);
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(mullow_naive(&a, &b, n));
        }
        let dn = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(fast(&a, &b, n, &cfg));
        }
        let df = t0.elapsed().as_secs_f64() / reps as f64;
        tn += dn;
        tf += df;
        let pass = rc <= 1e-120 && st.max_frac < 0.25;
        ok &= pass;
        println!(
            "{name:18} {n:4} {:2}x{:<2} {:4} {:3} {:.1e} {rc:.2e} {ra:.2e} {:8.3} {:8.3} {}",
            st.blocks.0,
            st.blocks.1,
            st.pairs,
            st.digits,
            st.max_frac,
            dn * 1e3,
            df * 1e3,
            if pass { "ok" } else { "FAIL" }
        );
    }
    println!(
        "dumped factors: naive {:.1} ms, fast {:.1} ms in all, ratio {:.2}; {}",
        tn * 1e3,
        tf * 1e3,
        tn / tf,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// `check`, then `bench` of every case at n = 128, 251, 512; exit code 1 when a check fails
#[allow(dead_code)] // hp/fastmul_bench's
pub fn run(args: &[String]) -> i32 {
    let reps: usize = args.iter().position(|a| a == "--reps").map(|i| args[i + 1].parse().unwrap()).unwrap_or(7);
    // --mutate zero_lane | e2_plain | e2_split_always: check_edges under that control, which must then fail
    // (exit 1)
    let mutate = match args.iter().position(|a| a == "--mutate").map(|i| args[i + 1].as_str()) {
        None => 0,
        Some("zero_lane") => crate::fastmul::MUT_ZERO_LANE,
        Some("e2_plain") => crate::fastmul::MUT_E2_PLAIN,
        Some("e2_split_always") => crate::fastmul::MUT_E2_SPLIT_ALWAYS,
        Some(m) => panic!("--mutate {m}: zero_lane, e2_plain or e2_split_always"),
    };
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        return if check_dumps(&args[i + 1], reps.min(3)) { 0 } else { 1 };
    }
    let ok = args.iter().any(|a| a == "--no-check")
        || check_digits() & check_unit() & check_empty() & check_edges(mutate) & check();
    if !args.iter().any(|a| a == "--no-bench") {
        for &n in &[128usize, 251, 512] {
            let inner = (2_000_000 / (n * n)).max(1);
            for name in ["flat1", "flat", "ramp", "bumpy"] {
                for p in crate::fastmul::PROF.iter() {
                    p.store(0, std::sync::atomic::Ordering::Relaxed);
                }
                bench(name, n, reps, inner);
                let per = 1e-6 / (reps * inner) as f64;
                let ms: Vec<String> = crate::fastmul::PROF
                    .iter()
                    .map(|p| format!("{:.3}", p.load(std::sync::atomic::Ordering::Relaxed) as f64 * per))
                    .collect();
                println!(
                    "   ms per product: digits and forward FFT {} (the FFT {}), pointwise {}, inverse {}, \
                     extraction {}",
                    ms[0], ms[4], ms[1], ms[2], ms[3]
                );
            }
        }
    }
    if ok {
        0
    } else {
        1
    }
}

/// the checks of the module doc in hp's tests: the digits, products by one coefficient, empty factors, edges, and every
/// planted case at n = 128, 251 and 512 against the naive product to 1e-120, with the controls
#[cfg(test)]
#[test]
fn fastmul_planted() {
    assert!(check_digits(), "digits8");
    assert!(check_unit(), "products by one coefficient");
    assert!(check_empty(), "empty factors");
    assert!(check_edges(0), "edge cases");
    // their controls, which must fail
    assert!(!check_edges(crate::fastmul::MUT_ZERO_LANE), "the zero-lane control did not fail");
    assert!(!check_edges(crate::fastmul::MUT_E2_PLAIN), "the e2 control did not fail");
    assert!(!check_edges(crate::fastmul::MUT_E2_SPLIT_ALWAYS), "the e2 = 0 split control did not fail");
    assert!(check(), "planted products");
}
