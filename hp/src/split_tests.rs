//! Tests of the split horizon series (`series_split`, Ctx::series_split = SPLIT_MIDDLE, and with the fast middle
//! terms SPLIT_FAST) on the synthetic context of qd_residual::synth, through multishoot's JetSession at `Octo` as
//! Solver.series_o runs it (the fast product itself: fastmul_check).
//!
//! ```text
//! cargo test --profile release-dev --lib split_    criteria fixed before the first run (the doc of each test)
//! cargo test --profile release-dev --lib bench_split -- --ignored --nocapture    the series' time by the forms
//! ```
use crate::model::{Ctx, HALPHA_ON, SPLIT_FAST, SPLIT_MIDDLE, SPLIT_MUT_DROP_L1};
use crate::qd_residual::synth;
use multidouble::od::OD;
use multidouble::precision::Octo;
use std::time::Instant;

/// the session's jet lanes (orders 0..=14)
const LANES: usize = 16;

/// the eight parts of a double-double state (the lower six zero)
fn parts8(zh: &[f64], zl: &[f64]) -> [Vec<f64>; 8] {
    std::array::from_fn(|q| match q {
        0 => zh.to_vec(),
        1 => zl.to_vec(),
        _ => vec![0.0; zh.len()],
    })
}

/// a planted coefficient vector z_k: ((i (k + 3)) mod 11 - 5) / 7 times 2^-3k in octo-double, Delta excluded
/// (prec_tests' coef_k)
fn coef_k(l: usize, k: usize) -> [Vec<f64>; 8] {
    let s = (-3.0 * k as f64).exp2();
    let xs: Vec<OD> = (0..l - 1).map(|i| (OD::from_f64(((i * (k + 3)) % 11) as f64 - 5.0) / OD::from_f64(7.0)).mul_f(s)).collect();
    std::array::from_fn(|q| xs.iter().map(|x| x.x[q]).chain([0.0]).collect())
}

/// z_0 (the synthetic state) and z_1..z_(n-1) planted
fn planted(n: usize) -> (Ctx<1>, Vec<[Vec<f64>; 8]>) {
    let (ctx, zh, zl) = synth();
    let l = ctx.len();
    let zs = std::iter::once(parts8(&zh, &zl)).chain((1..n).map(|k| coef_k(l, k))).collect();
    (ctx, zs)
}

/// F_1..F_n of the residual's series by the session (`Octo`, LANES lanes), given z_0..z_(n-1) one by one, each
/// order k with z_k zero as Solver.series_o hands it to its solve
fn session(ctx: &Ctx<1>, zs: &[[Vec<f64>; 8]], par: bool) -> Vec<Vec<OD>> {
    let sh = ctx.shooting_in::<Octo>();
    let mut s = sh.jet_session::<Octo, LANES>();
    zs.iter()
        .map(|z| {
            let v: Vec<&[f64]> = z.iter().map(|p| &p[..]).collect();
            s.next(&v, par).unwrap()
        })
        .collect()
}

/// the series with the context's series form, horizon form and control set for the call
fn series_with(ctx: &mut Ctx<1>, zs: &[[Vec<f64>; 8]], split: u8, alpha: u8, mutate: u8) -> Vec<Vec<OD>> {
    (ctx.series_split, ctx.horizon_alpha, ctx.mutate) = (split, alpha, mutate);
    let f = session(ctx, zs, true);
    (ctx.series_split, ctx.horizon_alpha, ctx.mutate) = (0, 0, 0);
    f
}

/// max_i |a_i - b_i| / max_i |b_i|, the difference formed in octo-double; infinite when a part or a difference is
/// not finite
fn rel(a: &[OD], b: &[OD]) -> f64 {
    // a row or a difference that is not finite fails every threshold (f64::max would drop a NaN)
    if a.iter().chain(b).any(|x| x.x.iter().any(|p| !p.is_finite())) {
        return f64::INFINITY;
    }
    let mut d = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let v = (*x - *y).to_f64().abs();
        if !v.is_finite() {
            return f64::INFINITY;
        }
        d = d.max(v);
    }
    d / b.iter().map(|y| y.to_f64().abs()).fold(f64::MIN_POSITIVE, f64::max)
}

/// `rel` on a difference that is not finite from finite rows (f64::MAX against -f64::MAX, whose difference
/// overflows) is infinite, and on ordinary rows the relative difference
#[test]
fn split_rel_non_finite() {
    let (big, nbig) = (OD::from_f64(f64::MAX), OD::from_f64(-f64::MAX));
    let r = rel(&[big, OD::from_f64(1.0)], &[nbig, OD::from_f64(1.0)]);
    println!("rel of MAX against -MAX: {r:e}");
    assert_eq!(r, f64::INFINITY, "a difference that is not finite gave {r}");
    let r2 = rel(&[OD::from_f64(3.0)], &[OD::from_f64(2.0)]);
    assert_eq!(r2, 0.5, "rel of 3 against 2 gave {r2}");
}

/// a hash of every part's bits of a series (FNV-1a), to compare a form's bits across builds
fn checksum(f: &[Vec<OD>]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in f.iter().flatten().flat_map(|v| v.x.iter()) {
        h = (h ^ x.to_bits()).wrapping_mul(0x100000001b3);
    }
    h
}

/// the rows of a and b whose parts differ
fn rows_differ(a: &[OD], b: &[OD]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.x.iter().zip(y.x.iter()).any(|(p, q)| p.to_bits() != q.to_bits())).count()
}

/// Pass (fixed before the first run), on the synthetic state with 12 planted orders:
/// - the default form (series_split 0) is residual_jet's series bit for bit (0 rows differ at every order), so the
///   option off leaves the series by order as it was;
/// - the split form's order 1 is the default's bit for bit (it has no middle terms);
/// - at every order k >= 2, d_split(k) = |F_k^split - F_k| / |F_k| <= 1e3 max(d_oa(k), 2^-420), d_oa(k) the same
///   difference of the horizon's alpha form (HALPHA_ON, the same model with other rounding);
/// - the split form's bits do not depend on the thread count (1 thread against 4).
///
/// Control, which must fail by more than 1e3: the middle terms without l = 1 (SPLIT_MUT_DROP_L1), max_k
/// d_ctrl(k) / (1e3 max(d_oa(k), 2^-420)) > 1e3.
#[test]
fn split_series_accuracy() {
    let n = 12;
    let (mut ctx, zs) = planted(n);
    let base = series_with(&mut ctx, &zs, 0, 0, 0);
    let split = series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, 0);
    let alpha = series_with(&mut ctx, &zs, 0, HALPHA_ON, 0);
    let ctrl = series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, SPLIT_MUT_DROP_L1);
    // the default form against residual_jet on the same vectors: order k with z_0..z_(k-1), z_k zero
    let zv: Vec<Vec<&[f64]>> = zs.iter().map(|z| z.iter().map(|p| &p[..]).collect()).collect();
    let off_differ: usize = (1..=n)
        .map(|k| {
            let full = crate::model::residual_jet_in::<Octo, LANES, 1>(&ctx, &zv[..k], true, k + 1).unwrap();
            rows_differ(&base[k - 1], &full[k])
        })
        .sum();
    let split_differ: usize = (0..n).map(|k| rows_differ(&split[k], &base[k])).sum();
    // thread count: the split form in pools of 1 and 4 threads
    let pool = |t: usize| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
    let one = pool(1).install(|| series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, 0));
    let four = pool(4).install(|| series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, 0));
    let thread_differ: usize = (0..n).map(|k| rows_differ(&one[k], &four[k]) + rows_differ(&one[k], &split[k])).sum();
    println!(" k   d_split    d_oa       d_ctrl     |F_k|");
    let floor = (-420.0f64).exp2();
    let (mut worst, mut ctrl_fail) = (0.0f64, 0.0f64);
    for k in 1..=n {
        let (ds, da, dc) = (rel(&split[k - 1], &base[k - 1]), rel(&alpha[k - 1], &base[k - 1]), rel(&ctrl[k - 1], &base[k - 1]));
        let fk = base[k - 1].iter().map(|y| y.to_f64().abs()).fold(0.0, f64::max);
        println!("{k:2}   {ds:.2e}   {da:.2e}   {dc:.2e}   {fk:.2e}");
        let thr = 1e3 * da.max(floor);
        if k >= 2 {
            // a difference or alpha reference that is not finite fails (max drops a NaN); the floor stays for
            // finite ones
            worst = if ds.is_finite() && da.is_finite() { worst.max(ds / thr) } else { f64::INFINITY };
            ctrl_fail = ctrl_fail.max(dc / thr);
        }
    }
    let order1 = rows_differ(&split[0], &base[0]);
    println!("split series' checksum {:016x}", checksum(&split));
    println!(
        "split series: default against residual_jet {off_differ} rows differ; split against default {split_differ} \
         rows differ, order 1 {order1}; worst \
         d_split / threshold {worst:.2e}; threads 1 against 4 and the default pool {thread_differ} rows differ; control \
         max d_ctrl / threshold {ctrl_fail:.2e}"
    );
    assert_eq!(off_differ, 0, "the default form is not residual_jet's");
    assert_eq!(order1, 0, "the split form's order 1 is not the default's");
    assert!(worst <= 1.0, "the split form differs by more than 1e3 d_oa");
    assert_eq!(thread_differ, 0, "the split form depends on the thread count");
    assert!(ctrl_fail > 1e3, "the control did not fail by more than 1e3");
}

/// The series' time by the three forms (default, SPLIT_MIDDLE, SPLIT_FAST) on the synthetic context with its horizon
/// order raised to KS (default 120) and 12 orders, interleaved REPS times (default 3): seconds per run and the ratios.
#[test]
#[ignore]
fn bench_split() {
    let ks: usize = std::env::var("KS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let n = 12;
    let (mut ctx, zs) = planted(n);
    ctx.ks = ks;
    let threads = rayon::current_num_threads();
    let mut t = [Vec::new(), Vec::new(), Vec::new()];
    let forms = [0, SPLIT_MIDDLE, SPLIT_FAST];
    for r in 0..reps {
        // the forms' order rotated each rep, so no form always runs first or after the same one
        for f in (0..forms.len()).map(|i| (i + r) % forms.len()) {
            let split = forms[f];
            let t0 = Instant::now();
            let s = series_with(&mut ctx, &zs, split, 0, 0);
            let dt = t0.elapsed().as_secs_f64();
            assert!(s.iter().all(|v| v.iter().all(|x| x.to_f64().is_finite())), "non-finite series");
            println!("rep {r} split {split}: {dt:.2} s");
            t[f].push(dt);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (a, b, c) = (med(&mut t[0]), med(&mut t[1]), med(&mut t[2]));
    println!(
        "ks {ks}, {n} orders, {threads} threads: default {:.2} s ({:.2}-{:.2}), split {:.2} s ({:.2}-{:.2}), fast {:.2} s \
         ({:.2}-{:.2}); default / split {:.2}, split / fast {:.2}",
        a.0, a.1, a.2, b.0, b.1, b.2, c.0, c.1, c.2, a.0 / b.0, b.0 / c.0
    );
}

/// F_1..F_n by the session on a chain's context (`FILE`, hp/dump_ctx.py; M = 32), given its stored coefficient
/// vectors (`JET`: K, L, then z_0..z_(K-1) as eight part arrays each; from a chain's jet_*.npz) one by one
fn session4(ctx: &Ctx<4>, zs: &[[Vec<f64>; 8]]) -> Vec<Vec<OD>> {
    let sh = ctx.shooting_in::<Octo>();
    let mut s = sh.jet_session::<Octo, 12>();
    zs.iter()
        .map(|z| {
            let v: Vec<&[f64]> = z.iter().map(|p| &p[..]).collect();
            s.next(&v, true).unwrap()
        })
        .collect()
}

/// The accuracy criteria of `split_series_accuracy` and both forms' times on a chain's context and its own series
/// (FILE, JET; NORD orders, default 8, at most 11; KS overrides the context's horizon order; SPLIT the form under
/// test, 1 = SPLIT_MIDDLE by default or 2 = SPLIT_FAST; REPS = 1 runs each timed form once instead of twice):
///
///     FILE=ctx.bin JET=jet.bin cargo test --profile release-dev --lib split_file -- --ignored --nocapture
#[test]
#[ignore]
fn split_file() {
    // the threads' stacks of Solver's pool (python.rs): the global pool's 2 MiB overflowed on this context
    rayon::ThreadPoolBuilder::new().stack_size(256 << 20).build().unwrap().install(split_file_in);
}

fn split_file_in() {
    let v = crate::model::ctxfile::read(&std::env::var("FILE").expect("FILE"));
    let (mut ctx, _, _) = crate::model::ctxfile::load::<4>(&v);
    if let Some(ks) = std::env::var("KS").ok().and_then(|v| v.parse().ok()) {
        ctx.ks = ks;
    }
    let n: usize = std::env::var("NORD").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    assert!(n <= 11, "NORD at most 11 (12 jet lanes)");
    let jet = crate::model::ctxfile::read(&std::env::var("JET").expect("JET"));
    let (kz, l) = (jet[0] as usize, jet[1] as usize);
    assert!(l == ctx.len() && kz >= n, "JET: {kz} vectors of {l}, the context's length {}", ctx.len());
    let zs: Vec<[Vec<f64>; 8]> =
        (0..n).map(|k| std::array::from_fn(|q| jet[2 + (k * 8 + q) * l..2 + (k * 8 + q + 1) * l].to_vec())).collect();
    let run = |ctx: &mut Ctx<4>, split: u8, alpha: u8, mutate: u8| {
        (ctx.series_split, ctx.horizon_alpha, ctx.mutate) = (split, alpha, mutate);
        let t0 = Instant::now();
        use std::sync::atomic::Ordering::Relaxed;
        use crate::series_split::{T_HOR, T_LOOP, T_MID};
        let (h0, m0, l0) = (T_HOR.load(Relaxed), T_MID.load(Relaxed), T_LOOP.load(Relaxed));
        let f = session4(ctx, &zs);
        let dt = t0.elapsed().as_secs_f64();
        let (h1, m1, l1) = (T_HOR.load(Relaxed), T_MID.load(Relaxed), T_LOOP.load(Relaxed));
        (ctx.series_split, ctx.horizon_alpha, ctx.mutate) = (0, 0, 0);
        let s = |a: u64, b: u64| (b - a) as f64 * 1e-9;
        println!(
            "split {split}, alpha {alpha}, mutate {mutate}: {dt:.1} s; the horizon block {:.1} s, of which middle terms \
             {:.1} s and the u-order loop {:.1} s",
            s(h0, h1),
            s(m0, m1),
            s(l0, l1)
        );
        (f, dt)
    };
    let form: u8 = std::env::var("SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(SPLIT_MIDDLE);
    assert!(form == SPLIT_MIDDLE || form == SPLIT_FAST, "SPLIT must be {SPLIT_MIDDLE} or {SPLIT_FAST}");
    let twice = std::env::var("REPS").map(|v| v != "1").unwrap_or(true);
    *crate::series_split::FAST_STATS.lock().unwrap() = (0, 0, 0, 0, 0.0);
    let (base, t0) = run(&mut ctx, 0, 0, 0);
    let (split, t1) = run(&mut ctx, form, 0, 0);
    let (base2, t2) = if twice { run(&mut ctx, 0, 0, 0) } else { (base.clone(), t0) };
    let (split2, t3) = if twice { run(&mut ctx, form, 0, 0) } else { (split.clone(), t1) };
    let (alpha, _) = run(&mut ctx, 0, HALPHA_ON, 0);
    let (ctrl, _) = run(&mut ctx, form, 0, SPLIT_MUT_DROP_L1);
    if form == SPLIT_FAST {
        let g = *crate::series_split::FAST_STATS.lock().unwrap();
        println!(
            "fast middle terms: {} products, {} block pairs, at most {} blocks of a factor and {} digits, max_frac {:.2e}",
            g.0, g.1, g.2, g.3, g.4
        );
    }
    let rerun: usize = (0..n).map(|k| rows_differ(&base[k], &base2[k]) + rows_differ(&split[k], &split2[k])).sum();
    let floor = (-420.0f64).exp2();
    let (mut worst, mut ctrl_fail) = (0.0f64, 0.0f64);
    println!(" k   d_split    d_oa       d_ctrl     |F_k|");
    for k in 1..=n {
        let (ds, da, dc) = (rel(&split[k - 1], &base[k - 1]), rel(&alpha[k - 1], &base[k - 1]), rel(&ctrl[k - 1], &base[k - 1]));
        let fk = base[k - 1].iter().map(|y| y.to_f64().abs()).fold(0.0, f64::max);
        println!("{k:2}   {ds:.2e}   {da:.2e}   {dc:.2e}   {fk:.2e}");
        let thr = 1e3 * da.max(floor);
        if k >= 2 {
            // a difference or alpha reference that is not finite fails (max drops a NaN); the floor stays for
            // finite ones
            worst = if ds.is_finite() && da.is_finite() { worst.max(ds / thr) } else { f64::INFINITY };
            ctrl_fail = ctrl_fail.max(dc / thr);
        }
    }
    let order1 = rows_differ(&split[0], &base[0]);
    println!("checksums: default {:016x}, split {:016x}", checksum(&base), checksum(&split));
    println!(
        "split {form}, ks {}, {n} orders, {} threads: default {t0:.1} s and {t2:.1} s, split {t1:.1} s and {t3:.1} s, ratio {:.2}; \
         order 1 {order1} rows differ; reruns {rerun} rows differ; worst d_split / threshold {worst:.2e}; control \
         {ctrl_fail:.2e}",
        ctx.ks,
        rayon::current_num_threads(),
        (t0 + t2) / (t1 + t3)
    );
    assert_eq!(order1 + rerun, 0, "order 1 or a rerun differs");
    assert!(worst <= 1.0, "the split form differs by more than 1e3 d_oa");
    assert!(ctrl_fail > 1e3, "the control did not fail by more than 1e3");
}

/// Pass (fixed before the first run), on the synthetic state with 12 planted orders, with its horizon order raised to
/// KS (default 60): the fast form (SPLIT_FAST) against the default as `split_series_accuracy` checks the split form:
/// - order 1 the default's bits;
/// - at every order k >= 2, d_fast(k) <= 1e3 max(d_oa(k), 2^-420);
/// - its bits do not depend on the thread count (1 thread against 4);
/// - every exact convolution's values lie within 1/4 of their integers (FAST_STATS);
/// - the default form and SPLIT_MIDDLE are untouched by it: their series after a fast run are the bits of runs before.
///
/// Control, which must fail by more than 1e3: the fast form without l = 1 (SPLIT_MUT_DROP_L1).
#[test]
fn split_fast_accuracy() {
    let n = 12;
    let (mut ctx, zs) = planted(n);
    ctx.ks = std::env::var("KS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    *crate::series_split::FAST_STATS.lock().unwrap() = (0, 0, 0, 0, 0.0);
    let base = series_with(&mut ctx, &zs, 0, 0, 0);
    let mid = series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, 0);
    let fast = series_with(&mut ctx, &zs, SPLIT_FAST, 0, 0);
    let alpha = series_with(&mut ctx, &zs, 0, HALPHA_ON, 0);
    let ctrl = series_with(&mut ctx, &zs, SPLIT_FAST, 0, SPLIT_MUT_DROP_L1);
    let pool = |t: usize| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
    let one = pool(1).install(|| series_with(&mut ctx, &zs, SPLIT_FAST, 0, 0));
    let four = pool(4).install(|| series_with(&mut ctx, &zs, SPLIT_FAST, 0, 0));
    let thread_differ: usize = (0..n).map(|k| rows_differ(&one[k], &four[k]) + rows_differ(&one[k], &fast[k])).sum();
    let base2 = series_with(&mut ctx, &zs, 0, 0, 0);
    let mid2 = series_with(&mut ctx, &zs, SPLIT_MIDDLE, 0, 0);
    let untouched: usize = (0..n).map(|k| rows_differ(&base[k], &base2[k]) + rows_differ(&mid[k], &mid2[k])).sum();
    let g = *crate::series_split::FAST_STATS.lock().unwrap();
    println!(" k   d_fast     d_mid      d_oa       d_ctrl     |F_k|");
    let floor = (-420.0f64).exp2();
    let (mut worst, mut ctrl_fail) = (0.0f64, 0.0f64);
    for k in 1..=n {
        let d = |f: &[Vec<OD>]| rel(&f[k - 1], &base[k - 1]);
        let (df, dm, da, dc) = (d(&fast), d(&mid), d(&alpha), d(&ctrl));
        let fk = base[k - 1].iter().map(|y| y.to_f64().abs()).fold(0.0, f64::max);
        println!("{k:2}   {df:.2e}   {dm:.2e}   {da:.2e}   {dc:.2e}   {fk:.2e}");
        let thr = 1e3 * da.max(floor);
        if k >= 2 {
            // a difference or alpha reference that is not finite fails (max drops a NaN); the floor stays for
            // finite ones
            worst = if df.is_finite() && da.is_finite() { worst.max(df / thr) } else { f64::INFINITY };
            ctrl_fail = ctrl_fail.max(dc / thr);
        }
    }
    let order1 = rows_differ(&fast[0], &base[0]);
    println!("fast series' checksum {:016x}", checksum(&fast));
    println!(
        "fast series, ks {}: order 1 {order1} rows differ; worst d_fast / threshold {worst:.2e}; threads 1 against 4 and \
         the default pool {thread_differ} rows differ; default and split 1 rerun {untouched} rows differ; {} products, \
         {} block pairs, at most {} blocks and {} digits, max_frac {:.2e}; control max d_ctrl / threshold {ctrl_fail:.2e}",
        ctx.ks, g.0, g.1, g.2, g.3, g.4
    );
    assert_eq!(order1, 0, "the fast form's order 1 is not the default's");
    assert!(worst <= 1.0, "the fast form differs by more than 1e3 d_oa");
    assert_eq!(thread_differ, 0, "the fast form depends on the thread count");
    assert_eq!(untouched, 0, "a fast run changed the other forms' bits");
    assert!(g.4 < 0.25, "an exact convolution's value lies {} from its integer", g.4);
    assert!(ctrl_fail > 1e3, "the control did not fail by more than 1e3");
}

/// FNV-1a of f64 bits
fn hash_f64(xs: impl IntoIterator<Item = f64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in xs {
        h = (h ^ x.to_bits()).wrapping_mul(0x100000001b3);
    }
    h
}

/// The inputs of the split and fast checksums, hashed, to tell whether two platforms' checksums part at the inputs
/// (#1034 step 2: win2 and the Mac print different fast and split checksums). `libm` hashes the f64 sin, cos and tan
/// values qd_residual::synth takes from the platform's libm; `state` hashes synth's z_0 (zh, zl), built from them by
/// double-double arithmetic; `planted` hashes split_tests' z_1..z_11. Equal `libm` hashes with unequal `state` ones
/// put the difference in the arithmetic; unequal `libm` hashes put it in the platform's libm. Prints only.
#[test]
fn split_inputs_hash() {
    use std::f64::consts::PI;
    let (m, nn) = (8usize, 16usize);
    let tans = (-(nn as i64) + 1..nn as i64).filter(|k| *k != 0).map(|k| (k as f64 * PI / nn as f64).tan());
    let th: Vec<f64> = (0..m).map(|j| PI * j as f64 / m as f64).collect();
    let trig = th.iter().flat_map(|t| [t.sin(), (2.0 * t).sin(), (4.0 * t).cos()]);
    let libm = hash_f64(tans.chain(trig));
    let (_, zh, zl) = synth();
    let state = hash_f64(zh.iter().chain(zl.iter()).copied());
    let (_, zs) = planted(12);
    let planted_h = hash_f64(zs[1..].iter().flat_map(|z| z.iter().flatten().copied()));
    println!("split inputs: libm {libm:016x}, state {state:016x}, planted {planted_h:016x}");
}
