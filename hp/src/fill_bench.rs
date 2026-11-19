//! Profile of the f64 Jacobian's fill on a saved context (hp/dump_ctx.py): the seconds per evaluation of each block
//! and their extrapolation to all columns, a Taylor step split into its tau derivatives, Cauchy sums and Horner sums,
//! and the lane-vector form of the Taylor recursion (`taylor_coeffs_cols`) against the field-wide form, timed
//! alternately and compared bit for bit.
//!
//!   cd hp && FILE=ctx.bin RUSTFLAGS="-C target-cpu=native" cargo test --release --lib bench_fill -- --ignored --nocapture
//!
//! FORM=fields profiles the field-wide form; ROUNDS (default 6) sets the alternating rounds; THREADS (default 1) the
//! concurrent evaluations of the timed group.
use crate::field::{cs, cs_w, horner, Fl, State};
use crate::lane::{Dtn, Fu, Fun, Lane};
use crate::model::{ctxfile, delta_of, horizon_lus, taylor_coeffs, work, Ctx, HPrep, BY_FIELDS};
use multidouble::dd::DD;
use multidouble::precision::Double;
use multishoot::jacobian::Seed;
use rayon::prelude::*;
use std::time::Instant;

/// run f with the field-wide form (`fields`) or the lane-vector form on this thread
fn with_form<R>(fields: bool, f: impl FnOnce() -> R) -> R {
    BY_FIELDS.with(|c| c.set(fields));
    let r = f();
    BY_FIELDS.with(|c| c.set(false));
    r
}

fn env_usize(name: &str, def: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(def)
}

/// the first NT tasks of block b
fn group<T: Lane, const B: usize>(ctx: &Ctx<B>, b: usize) -> Vec<(usize, Seed)> {
    let tasks: Vec<(usize, Seed)> = ctx.shooting().jac_tasks().into_iter().filter(|t| t.0 == b).collect();
    tasks[..T::NT.min(tasks.len())].to_vec()
}

/// the columns of block b's first group in the given form
fn columns<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], prep: &HPrep, b: usize, fields: bool) -> Vec<Vec<f64>> {
    let sh = ctx.shooting();
    let z = [zh, zl];
    let g = sh.globals::<Double>(&z);
    let gr = group::<T, B>(ctx, b);
    with_form(fields, || sh.jac_columns_f64::<T>(&z, &g, prep, &gr))
}

/// seconds per evaluation of block b's first group in the given form, best of reps
fn t_group<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], prep: &HPrep, b: usize, reps: usize, fields: bool) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let c = columns::<T, B>(ctx, zh, zl, prep, b, fields);
        best = best.min(t0.elapsed().as_secs_f64());
        assert!(c[0].iter().all(|v| v.is_finite()));
    }
    best
}

/// the state of node i with tangent k the values times 1 + k (so that no tangent is zero)
fn node_state<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], i: usize) -> State<T, B> {
    let m = Ctx::<B>::M;
    std::array::from_fn(|f| {
        let v = ctx.fld(zh, zl, ctx.off_node(i) + f * m);
        let mut x = Fl::<T, B>::from_vt(&v, &v);
        for k in 1..T::NT {
            let t = Fl::<T, B>::from_vt(&v, &v.mulf(1.0 + k as f64)).tan64(0);
            x.set_tan64(k, &t);
        }
        x
    })
}

/// seconds of one Taylor step (coefficients and Horner) of segment b's first step in the given form
fn t_step<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], b: usize, fields: bool) -> f64 {
    let s0 = node_state::<T, B>(ctx, zh, zl, b - 1);
    let xs = &ctx.segs[b - 1];
    let omega = T::splat(DD::TWO_PI) / T::splat(delta_of(zh, zl));
    let kt = ctx.kt;
    let mut w = work::<T, B>(kt);
    let h = DD::from_f64(xs[1]) - DD::from_f64(xs[0]);
    with_form(fields, || {
        let t0 = Instant::now();
        for (f, sf) in s0.iter().enumerate() {
            w.c[f][0] = *sf;
        }
        taylor_coeffs(ctx, xs[0], omega, &mut w);
        let s: [Fl<T, B>; 4] = std::array::from_fn(|f| horner(&w.c[f][..=kt], h));
        let t = t0.elapsed().as_secs_f64();
        assert!(s[0].val().amax().is_finite());
        t
    })
}

/// a Taylor step's parts at segment b's first step, seconds per step (best of reps): the whole step, the 2 kt tau
/// derivatives, the Cauchy sums of the recursion (its 12 per order, field-wide), the 4 Horner sums
fn step_split<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], b: usize, reps: usize, fields: bool) -> [f64; 4] {
    let s0 = node_state::<T, B>(ctx, zh, zl, b - 1);
    let xs = &ctx.segs[b - 1];
    let omega = T::splat(DD::TWO_PI) / T::splat(delta_of(zh, zl));
    let kt = ctx.kt;
    let mut w = work::<T, B>(kt);
    let h = DD::from_f64(xs[1]) - DD::from_f64(xs[0]);
    let mut best = [f64::INFINITY; 4];
    let mut sink = 0.0;
    for _ in 0..reps {
        best[0] = best[0].min(t_step::<T, B>(ctx, zh, zl, b, fields));
        for (f, sf) in s0.iter().enumerate() {
            w.c[f][0] = *sf;
        }
        with_form(fields, || taylor_coeffs(ctx, xs[0], omega, &mut w));
        // the tau derivatives alone
        let t0 = Instant::now();
        for k in 0..kt {
            let a = ctx.d.apply(&w.c[0][k]).mul_t(omega);
            let c = ctx.d.apply(&w.c[1][k]).mul_t(omega);
            sink += a.0[0].tan64(0).to_array()[0] + c.0[0].tan64(0).to_array()[0];
        }
        best[1] = best[1].min(t0.elapsed().as_secs_f64());
        // the Cauchy sums alone, on the step's coefficients
        let t0 = Instant::now();
        let [p, m, a, c] = &w.c[..] else { unreachable!() };
        let [a2, q2, g, _h, _pth, _mth, ww, vv] = &w.aux[..8] else { unreachable!() };
        for k in 0..kt {
            let mut acc = cs(a, a, k);
            acc = acc + cs(p, p, k) + cs(m, m, k);
            acc = acc + cs(c, a2, k) + cs(a, q2, k) + cs(a, a2, k);
            acc = acc + cs(c, p, k) + cs(c, m, k) + cs(g, p, k) + cs(g, m, k);
            if k >= 1 {
                acc = acc + cs_w(ww, p, k + 1, 1, k, |j| (k + 1 - j) as f64) + cs_w(vv, m, k + 1, 1, k, |j| (k + 1 - j) as f64);
            }
            sink += acc.0[0].tan64(0).to_array()[0];
        }
        best[2] = best[2].min(t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        let s: [Fl<T, B>; 4] = std::array::from_fn(|f| horner(&w.c[f][..=kt], h));
        best[3] = best[3].min(t0.elapsed().as_secs_f64());
        sink += s[1].val().amax();
    }
    assert!(sink.is_finite() || sink.is_nan());
    best
}

/// bitwise comparison of the lane-vector and field-wide forms on block b's first group: the number of entries whose
/// bits differ
fn diff_bits<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], prep: &HPrep, b: usize) -> (usize, usize) {
    let new = columns::<T, B>(ctx, zh, zl, prep, b, false);
    let old = columns::<T, B>(ctx, zh, zl, prep, b, true);
    let mut n = 0;
    let mut tot = 0;
    for (cn, co) in new.iter().zip(&old) {
        for (x, y) in cn.iter().zip(co) {
            tot += 1;
            n += (x.to_bits() != y.to_bits()) as usize;
        }
    }
    (n, tot)
}

fn profile<const B: usize>(v: &[f64]) {
    let (ctx, zh, zl) = ctxfile::load::<B>(v);
    let m = 8 * B;
    let fields = std::env::var("FORM").map(|s| s == "fields").unwrap_or(false);
    let delta = delta_of(&zh, &zl);
    let prep = HPrep::of(horizon_lus(&ctx, &ctx.fld(&zh, &zl, ctx.off_ah()), delta, false, DD::ZERO).unwrap());
    let sh = ctx.shooting();
    let all = sh.jac_tasks();
    let nb = ctx.k() + 1;
    let steps: usize = ctx.segs.iter().map(|s| s.len() - 1).sum();
    let ws = 12 * (ctx.kt + 2) * m * 5 * 8;
    println!(
        "M = {m}, L = {}, K = {}, {steps} Taylor steps, kt {} kc {} ks {}; Fun<4> work arrays {:.2} MB, D {:.2} MB (hi); form {}",
        ctx.len(),
        ctx.k(),
        ctx.kt,
        ctx.kc,
        ctx.ks,
        ws as f64 / 1e6,
        (m * m * 8) as f64 / 1e6,
        if fields { "fields" } else { "lane vectors" }
    );
    let b = ctx.k() / 2;
    // bit for bit: the lane-vector form against the field-wide one, every block's first group, 1, 4 and 8 tangents
    // and the double-double base
    let (mut nd, mut nt) = (0, 0);
    for bb in [0, 1, b, ctx.k() - 1, ctx.k()] {
        for (d, t) in [
            diff_bits::<Fu, B>(&ctx, &zh, &zl, &prep, bb),
            diff_bits::<Fun<4>, B>(&ctx, &zh, &zl, &prep, bb),
            diff_bits::<Fun<8>, B>(&ctx, &zh, &zl, &prep, bb),
            diff_bits::<Dtn<4>, B>(&ctx, &zh, &zl, &prep, bb),
        ] {
            nd += d;
            nt += t;
        }
    }
    println!("  bitwise, lane vectors vs fields (blocks 0, 1, {b}, {}, {}; Fu, Fun<4>, Fun<8>, Dtn<4>): {nd} of {nt} entries differ", ctx.k() - 1, ctx.k());
    assert_eq!(nd, 0, "the lane-vector form differs from the field-wide form");
    // alternating timing of one Taylor step and of one group of the middle segment
    let rounds = env_usize("ROUNDS", 6);
    let (mut sf, mut sc) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        for _ in 0..3 {
            sf.push(t_step::<Fun<4>, B>(&ctx, &zh, &zl, b, true));
            sc.push(t_step::<Fun<4>, B>(&ctx, &zh, &zl, b, false));
        }
    }
    let threads = env_usize("THREADS", 1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let (mut gf, mut gc) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        for (fl, out) in [(true, &mut gf), (false, &mut gc)] {
            let t0 = Instant::now();
            pool.install(|| {
                (0..threads).into_par_iter().for_each(|_| {
                    let c = columns::<Fun<4>, B>(&ctx, &zh, &zl, &prep, b, fl);
                    assert!(c[0][0].is_finite());
                })
            });
            out.push(t0.elapsed().as_secs_f64());
        }
    }
    let stat = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let (f0, f1, f2) = stat(&mut sf);
    let (c0, c1, c2) = stat(&mut sc);
    println!(
        "  Fun<4> Taylor step (block {b}), {} alternating runs each: fields min {:.3} median {:.3} max {:.3} ms; lane vectors min {:.3} median {:.3} max {:.3} ms; ratio of medians {:.2}x (of minima {:.2}x)",
        sf.len(),
        f0 * 1e3,
        f1 * 1e3,
        f2 * 1e3,
        c0 * 1e3,
        c1 * 1e3,
        c2 * 1e3,
        f1 / c1,
        f0 / c0
    );
    let (f0, f1, f2) = stat(&mut gf);
    let (c0, c1, c2) = stat(&mut gc);
    println!(
        "  Fun<4> group of block {b} ({} steps), {threads} at once, {rounds} alternating runs: fields min {:.3} median {:.3} max {:.3} s; lane vectors min {:.3} median {:.3} max {:.3} s; ratio of medians {:.2}x (of minima {:.2}x)",
        ctx.segs[b - 1].len() - 1,
        f0,
        f1,
        f2,
        c0,
        c1,
        c2,
        f1 / c1,
        f0 / c0
    );
    if std::env::var("BLOCKS").is_err() {
        return;
    }
    // per block, one group per block, extrapolated to the block's groups
    let (mut tc, mut ts, mut th) = (0.0, 0.0, 0.0);
    let mut tstep = Vec::new();
    for bb in 0..nb {
        let ncol = all.iter().filter(|t| t.0 == bb).count();
        let ng = ncol.div_ceil(4);
        let reps = if bb == 0 || bb == ctx.k() { 2 } else { 1 };
        let t = t_group::<Fun<4>, B>(&ctx, &zh, &zl, &prep, bb, reps, fields);
        if bb == 0 {
            tc += t * ng as f64;
        } else if bb < ctx.k() {
            ts += t * ng as f64;
            tstep.push(t / (ctx.segs[bb - 1].len() - 1) as f64);
        } else {
            th += t * ng as f64;
        }
    }
    let tot = tc + ts + th;
    tstep.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  Fun<4> fill, one thread (extrapolated): centre {:.2} s ({:.1}%), segments {:.2} s ({:.1}%), horizon {:.2} s ({:.1}%), total {:.2} s; per Taylor step median {:.3} ms (min {:.3}, max {:.3})",
        tc,
        100.0 * tc / tot,
        ts,
        100.0 * ts / tot,
        th,
        100.0 * th / tot,
        tot,
        tstep[tstep.len() / 2] * 1e3,
        tstep[0] * 1e3,
        tstep[tstep.len() - 1] * 1e3
    );
    let sp = step_split::<Fun<4>, B>(&ctx, &zh, &zl, b, 5, fields);
    let f_d = 2.0 * ctx.kt as f64 * (m * m * 5) as f64;
    let f_c = (0..ctx.kt).map(|k| (10 * (k + 1) + if k >= 1 { 2 * k } else { 0 }) as f64).sum::<f64>() * (m * 9) as f64;
    println!(
        "  Fun<4> step at block {b}: whole {:.3} ms; tau derivatives {:.3} ms ({:.0}%, {:.2} GFMA/s), Cauchy sums (field-wide) {:.3} ms ({:.0}%, {:.2} GFMA/s), Horner {:.3} ms ({:.0}%), rest {:.3} ms",
        sp[0] * 1e3,
        sp[1] * 1e3,
        100.0 * sp[1] / sp[0],
        f_d / sp[1] / 1e9,
        sp[2] * 1e3,
        100.0 * sp[2] / sp[0],
        f_c / sp[2] / 1e9,
        sp[3] * 1e3,
        100.0 * sp[3] / sp[0],
        (sp[0] - sp[1] - sp[2] - sp[3]) * 1e3
    );
}

/// FILE=ctx.bin cargo test --release --lib bench_fill -- --ignored --nocapture
#[test]
#[ignore]
fn bench_fill() {
    let v = ctxfile::read(&std::env::var("FILE").expect("FILE"));
    match ctxfile::m_of(&v) / 8 {
        6 => profile::<6>(&v),
        10 => profile::<10>(&v),
        16 => profile::<16>(&v),
        24 => profile::<24>(&v),
        40 => profile::<40>(&v),
        48 => profile::<48>(&v),
        b => panic!("B = {b} not built into the bench"),
    }
}
