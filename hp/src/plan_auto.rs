//! The Taylor order chosen by the cost of the planned steps: multishoot's `Shooting::plan_segments_auto` (fb68bc3) on
//! hp's equations. Each candidate order is planned from the node states of z, and the one of least estimated cost,
//! the sum of the steps times the step cost k^2 + w k, is kept, the lower order on a tie. The choice counts steps and
//! never times anything, so it is the same on every machine. The planner is multishoot's (`plan_path`, roots from
//! `kth_root`), so at the same order its points differ from hp's own (`model::plan_path`, `powf`) in the last bits.
//!
//! A Jacobian and a residual cost about the plan's steps times a step cost, and at low D a Newton call is 99.5%
//! Jacobian (notes/LOW_D_SPEEDUP.md appendix A), so the weight w is the f64 Jacobian's (`Fun<4>` lanes) there. The
//! weights measured per lane and M are in notes/LOW_D_SPEEDUP.md ("The Taylor order by the cost of planned steps",
//! `bench_step_cost` below).
use crate::model::Ctx;
use multishoot::taylor::{OrderChoice, Plan};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// mutate: the step cost inverted, 2^40 / (k^2 + w k), which prefers the highest order: the control of the checks
/// that the order kept is the cheapest
pub const MUT_INVERT: u8 = 1;

/// The cost of one Taylor step at order k, k^2 + w k (an integer below 2^53, so that the estimates compare exactly);
/// under MUT_INVERT its inverse times 2^40
pub fn step_cost(k: usize, w: usize, mutate: u8) -> f64 {
    let c = (k * k + w * k) as f64;
    if mutate == MUT_INVERT {
        1099511627776.0 / c
    } else {
        c
    }
}

/// The order kept, its points, every pass's candidates, and the orders dropped because a Taylor coefficient was not
/// finite at them (an overflow of the f64 exponent range, which bounds the order from above near x = c at low D)
pub struct Choice {
    pub kt: usize,
    pub segs: Vec<Vec<f64>>,
    pub passes: Vec<OrderChoice>,
    pub dropped: Vec<usize>,
}

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// One pass of the choice over `orders`. multishoot's planner panics on a Taylor coefficient that is not finite, and on
/// a step of zero, where tol / e_k underflows because e_k is near the top of the f64 range (low D near x = c, order 36
/// and up); the pass is then made again without its highest order (the candidates are planned from the highest down,
/// and the coefficients grow with the order), until it plans. Any other panic is an error.
fn pass<const B: usize>(
    ctx: &Ctx<B>,
    zh: &[f64],
    zl: &[f64],
    tol: f64,
    hmax: f64,
    orders: &[usize],
    cost: &dyn Fn(usize) -> f64,
) -> Result<(OrderChoice, usize, Vec<Vec<f64>>, Vec<usize>), String> {
    let mut ks = orders.to_vec();
    ks.sort_unstable();
    ks.dedup();
    let mut dropped = Vec::new();
    loop {
        if ks.is_empty() {
            return Err(format!("plan_auto: a Taylor coefficient is not finite at every candidate order (dropped {dropped:?})"));
        }
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut sh = ctx.shooting();
            let c = sh.plan_segments_auto(zh, zl, tol, hmax, &ks, cost);
            (c, sh.kt, sh.segs)
        }));
        match r {
            Ok((c, kt, segs)) => return Ok((c, kt, segs, dropped)),
            Err(e) => {
                let msg = panic_text(e.as_ref());
                if !(msg.contains("is not finite") || msg.contains("plan_path: step 0e0")) {
                    return Err(format!("plan_auto: {msg}"));
                }
                dropped.push(ks.pop().expect("a candidate"));
            }
        }
    }
}

/// The most passes past the second (`plan_auto`'s edge passes)
pub const EDGE_PASSES: usize = 8;

/// The orders of the next edge pass after a pass whose kept order is its lowest or highest candidate: every order
/// from kt - d to kt (not below 2), or from kt to kt + d (below `cap`), d = max(2, kt / 4); None when the kept order is
/// interior, or at a bound it cannot pass. `OrderChoice::refined_orders` reaches past the lowest candidate only as far
/// as the spacing above it, so a cost that keeps falling toward low orders (the chain's states, where the steps grow
/// slowly as the order falls) has its least below that range.
fn edge_orders(c: &OrderChoice, cap: usize) -> Option<Vec<usize>> {
    let (lo, hi) = (c.candidates.first()?.kt, c.candidates.last()?.kt);
    let d = (c.kt / 4).max(2);
    if c.kt == lo && c.kt > 2 {
        Some(((c.kt.saturating_sub(d)).max(2)..=c.kt).collect())
    } else if c.kt == hi && c.kt + 1 < cap {
        Some((c.kt..=(c.kt + d).min(cap - 1)).collect())
    } else {
        None
    }
}

/// The order of least estimated cost at tol (absolute, the last two Taylor terms) and hmax, and its plan, from the
/// node states of z (hi, lo) and its Delta: the candidates `orders` (multishoot's `Plan::candidate_orders(tol)` when
/// None), then with `refine` every order near the first pass's choice (`OrderChoice::refined_orders`), then, while the
/// order kept is the lowest or highest of its pass, the orders past it (`edge_orders`, at most EDGE_PASSES passes,
/// until one keeps the order it started from). Orders at or above the lowest one dropped are not tried again, nor
/// orders above `kmax` (0: no bound). The step cost is `step_cost(k, w, mutate)`. The context is not changed.
#[allow(clippy::too_many_arguments)]
pub fn plan_auto<const B: usize>(
    ctx: &Ctx<B>,
    zh: &[f64],
    zl: &[f64],
    tol: f64,
    hmax: f64,
    w: usize,
    orders: Option<&[usize]>,
    kmax: usize,
    refine: bool,
    mutate: u8,
) -> Result<Choice, String> {
    let cost = move |k: usize| step_cost(k, w, mutate);
    let kcap = if kmax == 0 { usize::MAX } else { kmax + 1 };
    let first: Vec<usize> = orders.map_or_else(|| Plan::candidate_orders(tol), |o| o.to_vec()).into_iter().filter(|&k| k < kcap).collect();
    if first.is_empty() {
        return Err(format!("plan_auto: no candidate order at or below kmax {kmax}"));
    }
    let (c1, mut kt, mut segs, mut dropped) = pass(ctx, zh, zl, tol, hmax, &first, &cost)?;
    let mut passes = vec![c1];
    if refine {
        for i in 0..=EDGE_PASSES {
            let cap = dropped.iter().copied().min().unwrap_or(usize::MAX).min(kcap);
            let last = passes.last().expect("a pass");
            let ks: Vec<usize> = if i == 0 {
                last.refined_orders().into_iter().filter(|&k| k < cap).collect()
            } else {
                match edge_orders(last, cap) {
                    Some(ks) => ks,
                    None => break,
                }
            };
            if ks.is_empty() {
                break;
            }
            let (c, k, s, d) = pass(ctx, zh, zl, tol, hmax, &ks, &cost)?;
            dropped.extend(d);
            let moved = k != kt;
            (kt, segs) = (k, s);
            passes.push(c);
            // an edge pass that keeps the order it started from has found the least past that edge
            if i > 0 && !moved {
                break;
            }
        }
    }
    Ok(Choice { kt, segs, passes, dropped })
}

/// The estimated cost of a plan at order kt: its steps times `step_cost(kt, w, 0)`
pub fn plan_cost(segs: &[Vec<f64>], kt: usize, w: usize) -> f64 {
    segs.iter().map(|s| s.len() - 1).sum::<usize>() as f64 * step_cost(kt, w, 0)
}

#[cfg(test)]
mod tests {
    //! The planted problem of hp/plan_auto_gain.py (`planted`, the truth in hp/data/plan_auto/planted.txt): p = m = 0,
    //! a^2 = u / (1 + u), u = K2 x^(n-1), c = C (1 + u) / x^(n-1), an exact solution of hp's Taylor recursion whose
    //! steps at every order were counted from the closed forms' Taylor coefficients, in mpmath, by the planner's rule.
    //! The checks: multishoot's planner on hp's equations takes those steps at every order; the order chosen is the
    //! table's least cost; and each segment's Taylor path ends on the exact state of its target node. Each has a
    //! control that must make it fail. The data-driven tests below (ignored) read a context of hp/dump_ctx.py.
    use super::*;
    use crate::dd::DD;
    use crate::dv::DV;
    use crate::field::{DMat, Fl, State};
    use crate::lane::Lane;
    use crate::model::{ctxfile, taylor_coeffs, taylor_path, work};
    use multidouble::field::horner;
    use multidouble::native::{Fun, Oun, Qu, OV};
    use crate::qv::QV;
    use std::time::Instant;

    const TRUTH: &str = include_str!("../data/plan_auto/planted.txt");

    /// the plant: (n, nodes, tol, hmax, Delta, node states (a hi, a lo, c hi, c lo), steps per segment by order)
    struct Planted {
        n: f64,
        nodes: Vec<f64>,
        tol: f64,
        hmax: f64,
        delta: f64,
        states: Vec<[f64; 4]>,
        steps: Vec<(usize, Vec<usize>)>,
    }

    fn planted() -> Planted {
        let mut p = Planted { n: 0.0, nodes: vec![], tol: 0.0, hmax: 0.0, delta: 0.0, states: vec![], steps: vec![] };
        for line in TRUTH.lines().filter(|l| !l.starts_with('#')) {
            let mut it = line.split_whitespace();
            let key = it.next().expect("a key");
            let v: Vec<&str> = it.collect();
            let f = |i: usize| v[i].parse::<f64>().expect("a number");
            match key {
                "n" => p.n = f(0),
                "nodes" => p.nodes = (0..v.len()).map(f).collect(),
                "tol" => p.tol = f(0),
                "hmax" => p.hmax = f(0),
                "delta" => p.delta = f(0),
                "state" => p.states.push([f(1), f(2), f(3), f(4)]),
                "steps" => p.steps.push((v[0].parse().unwrap(), v[1..].iter().map(|s| s.parse().unwrap()).collect())),
                _ => {}
            }
        }
        assert_eq!(p.states.len(), p.nodes.len());
        p
    }

    /// the plant's context at M = 8 (every tau point the same; p = m = 0, so d/dtheta never acts) and its z
    fn planted_ctx(p: &Planted, n: f64) -> (Ctx<1>, Vec<f64>, Vec<f64>) {
        let k = p.nodes.len();
        let segs = (0..k - 1).map(|b| vec![p.nodes[b], p.nodes[b + 1]]).collect();
        let zero = vec![0.0; 64];
        let ctx = Ctx::<1> {
            n: DD::from_f64(n),
            d: DMat::<1>::from_rows(&zero, &zero),
            phase: Fl::<DV, 1>::load(&zero[..8], &zero[..8]),
            nodes: p.nodes.clone(),
            segs,
            kt: 36,
            kc: 4,
            ks: 4,
            freeze: None,
            match_node: k - 1,
            lam: DD::ZERO,
            mutate: 0,
            f64_dd: 0,
            horizon_alpha: 0,
            series_split: 0,
            exact_recip: 0,
            recip_planted: 0,
        };
        let (mut zh, mut zl) = (vec![0.0; ctx.len()], vec![0.0; ctx.len()]);
        for (i, s) in p.states.iter().enumerate() {
            let o = ctx.off_node(i);
            // fields p, m, a, c of 8 points each
            for j in 0..8 {
                zh[o + 16 + j] = s[0];
                zl[o + 16 + j] = s[1];
                zh[o + 24 + j] = s[2];
                zl[o + 24 + j] = s[3];
            }
        }
        let l = zh.len();
        zh[l - 1] = p.delta;
        (ctx, zh, zl)
    }

    fn steps_of(segs: &[Vec<f64>]) -> Vec<usize> {
        segs.iter().map(|s| s.len() - 1).collect()
    }

    /// multishoot's planner on hp's equations at every order of the table, against the table's steps
    fn check_steps(n: f64) -> Result<(), String> {
        let p = planted();
        let (ctx, zh, zl) = planted_ctx(&p, n);
        let mut bad = Vec::new();
        for (k, want) in &p.steps {
            let mut sh = ctx.shooting();
            sh.plan_segments(&zh, &zl, &Plan { kt: *k, tol: p.tol, hmax: p.hmax });
            let got = steps_of(&sh.segs);
            if &got != want {
                bad.push(format!("order {k}: {got:?}, truth {want:?}"));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("{} of {} orders differ: {}", bad.len(), p.steps.len(), bad[..bad.len().min(4)].join("; ")))
        }
    }

    /// the order plan_auto keeps (two passes, step cost k^2 + 20 k) against the table's least cost over every order
    fn check_choice(mutate: u8) -> Result<(), String> {
        let p = planted();
        let (ctx, zh, zl) = planted_ctx(&p, p.n);
        let c = plan_auto(&ctx, &zh, &zl, p.tol, p.hmax, 20, None, p.steps.last().unwrap().0, true, mutate)?;
        let cost = |k: usize, s: &[usize]| s.iter().sum::<usize>() as f64 * step_cost(k, 20, 0);
        let (kb, sb) = p.steps.iter().min_by(|a, b| cost(a.0, &a.1).total_cmp(&cost(b.0, &b.1)).then(a.0.cmp(&b.0))).unwrap();
        let kept = p.steps.iter().find(|s| s.0 == c.kt).ok_or(format!("order {} is not in the table", c.kt))?;
        if steps_of(&c.segs) != kept.1 {
            return Err(format!("order {}: steps {:?}, truth {:?}", c.kt, steps_of(&c.segs), kept.1));
        }
        if cost(c.kt, &kept.1) != cost(*kb, sb) {
            return Err(format!(
                "order {} kept, cost {} (truth); the least is order {kb}, cost {}",
                c.kt,
                cost(c.kt, &kept.1),
                cost(*kb, sb)
            ));
        }
        Ok(())
    }

    /// each segment's Taylor path at order kt on the points segs, from its source node's exact state, against the exact
    /// state of its target node: the largest difference over fields, points and segments, and the bound 10 tol steps
    fn path_error(p: &Planted, kt: usize, segs: &[Vec<f64>]) -> (f64, f64) {
        let (ctx, zh, zl) = planted_ctx(p, p.n);
        let omega = DV::splat(DD::TWO_PI / DD::from_f64(p.delta));
        let mut w = work::<DV, 1>(kt);
        let node = |i: usize| -> State<DV, 1> {
            let o = ctx.off_node(i);
            std::array::from_fn(|f| Fl::<DV, 1>::load(&zh[o + 8 * f..o + 8 * f + 8], &zl[o + 8 * f..o + 8 * f + 8]))
        };
        let mut worst: f64 = 0.0;
        for (b, xs) in segs.iter().enumerate() {
            let end = taylor_path(&ctx, &node(b), xs, omega, &mut w);
            let want = node(b + 1);
            for f in 0..4 {
                let d = (end[f].get(0) - want[f].get(0)).hi.abs();
                worst = worst.max(d);
            }
        }
        let steps: usize = segs.iter().map(|s| s.len() - 1).sum();
        (worst, 10.0 * p.tol * steps as f64)
    }

    #[test]
    fn planted_steps_equal_truth() {
        check_steps(planted().n).unwrap();
    }

    /// control: the plant's equations at n + 0.25 must not take the table's steps
    #[test]
    fn planted_steps_control_n() {
        assert!(check_steps(planted().n + 0.25).is_err());
    }

    #[test]
    fn planted_choice_is_least_cost() {
        check_choice(0).unwrap();
    }

    /// control: the inverted step cost must keep an order of more than the least cost
    #[test]
    fn planted_choice_control_invert() {
        let e = check_choice(MUT_INVERT).unwrap_err();
        assert!(e.contains("the least is"), "{e}");
    }

    /// the auto plan's and the fixed order 36's paths end on the exact node states within 10 tol steps; control: the
    /// auto plan's points summed at an order 12 lower must not
    #[test]
    fn planted_paths_meet_exact_states() {
        let p = planted();
        let (ctx, zh, zl) = planted_ctx(&p, p.n);
        let c = plan_auto(&ctx, &zh, &zl, p.tol, p.hmax, 20, None, p.steps.last().unwrap().0, true, 0).unwrap();
        let mut sh = ctx.shooting();
        sh.plan_segments(&zh, &zl, &Plan { kt: 36, tol: p.tol, hmax: p.hmax });
        for (kt, segs) in [(c.kt, &c.segs), (36, &sh.segs)] {
            let (e, bound) = path_error(&p, kt, segs);
            println!("order {kt}: {} steps, largest end-state error {e:.2e} (bound {bound:.2e})", steps_of(segs).iter().sum::<usize>());
            assert!(e <= bound, "order {kt}: {e:e} > {bound:e}");
        }
        let (e, bound) = path_error(&p, c.kt - 12, &c.segs);
        assert!(e > bound, "control: order {} on order {}'s points: {e:e} <= {bound:e}", c.kt - 12, c.kt);
    }

    // ------------------------------------------------------------------ data-driven (ignored; FILE from hp/dump_ctx.py)

    fn env_or(name: &str, def: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| def.to_string())
    }

    /// "a,b,c" or "lo:hi:step"
    fn orders_of(s: &str) -> Vec<usize> {
        if let Some((lo, rest)) = s.split_once(':') {
            let (hi, st) = rest.split_once(':').unwrap_or((rest, "1"));
            (lo.parse().unwrap()..=hi.parse().unwrap()).step_by(st.parse().unwrap()).collect()
        } else {
            s.split(',').map(|v| v.parse().unwrap()).collect()
        }
    }

    /// hp's planner (model::plan_path, as Solver.plan) at the context's order
    fn hp_plan<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>> {
        use rayon::prelude::*;
        let m = 8 * B;
        let omega = DV::splat(DD::TWO_PI / DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] });
        (1..ctx.k())
            .into_par_iter()
            .map(|b| {
                let (from, to) = if ctx.seg_inward(b) { (b, b - 1) } else { (b - 1, b) };
                let o = ctx.off_node(from);
                let s: State<DV, B> = std::array::from_fn(|q| Fl::<DV, B>::load(&zh[o + q * m..o + (q + 1) * m], &zl[o + q * m..o + (q + 1) * m]));
                crate::model::plan_path(ctx, &s, ctx.nodes[from], ctx.nodes[to], omega, tol, hmax)
            })
            .collect()
    }

    /// max |F| over the Taylor blocks (1..K-1) on the context's plan and order: each segment's double-double path from
    /// its source node's state against its target node's state (the rows of those blocks)
    fn taylor_rows_max<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64]) -> f64 {
        use rayon::prelude::*;
        let m = 8 * B;
        let omega = DV::splat(DD::TWO_PI / DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] });
        let node = |i: usize| -> State<DV, B> {
            let o = ctx.off_node(i);
            std::array::from_fn(|q| Fl::<DV, B>::load(&zh[o + q * m..o + (q + 1) * m], &zl[o + q * m..o + (q + 1) * m]))
        };
        (1..ctx.k())
            .into_par_iter()
            .map(|b| {
                let (from, to) = if ctx.seg_inward(b) { (b, b - 1) } else { (b - 1, b) };
                let mut w = work::<DV, B>(ctx.kt);
                let end = taylor_path(ctx, &node(from), &ctx.segs[b - 1], omega, &mut w);
                let want = node(to);
                (0..4)
                    .flat_map(|f| (0..m).map(move |i| (f, i)))
                    .map(|(f, i)| (end[f].get(i) - want[f].get(i)).hi.abs())
                    .fold(0.0, |a: f64, d| if d.is_nan() { d } else { a.max(d) })
            })
            .reduce(|| 0.0, |a, d| if d.is_nan() || a.is_nan() { f64::NAN } else { a.max(d) })
    }

    /// The planning comparison on a stored state (FILE): hp's planner at the stored order against the stored plan;
    /// multishoot's planner at every order of KS (default 12:96:2; "overflow" where a coefficient is not finite);
    /// plan_auto's choice for each weight of W (default 20; two passes); the residual of the Taylor blocks on the
    /// stored plan, on hp's plan at the stored order and on each choice. TOL (default 1e-30 at M >= 128, else
    /// 1e-32) and HMAX (0.05) as the runs'. MUTATE=1 inverts the step cost (the control). Checks (it fails on any):
    /// the order chosen costs at most 5% more than the least over KS, and its Taylor blocks' max |F| is at most
    /// max(1e-28, 10 times hp's plan at the stored order).
    ///
    ///   cd hp && FILE=ctx.bin RAYON_NUM_THREADS=3 cargo test --profile release-dev --lib plan_orders -- --ignored --nocapture
    #[test]
    #[ignore]
    fn plan_orders() {
        let path = std::env::var("FILE").expect("FILE=<context of hp/dump_ctx.py>");
        let v = ctxfile::read(&path);
        match ctxfile::m_of(&v) {
            8 => plan_orders_on::<1>(&path, &v),
            32 => plan_orders_on::<4>(&path, &v),
            48 => plan_orders_on::<6>(&path, &v),
            320 => plan_orders_on::<40>(&path, &v),
            384 => plan_orders_on::<48>(&path, &v),
            m => panic!("plan_orders: M {m} is not built here (8, 32, 48, 320, 384)"),
        }
    }

    fn plan_orders_on<const B: usize>(path: &str, v: &[f64]) {
        let (mut ctx, zh, zl) = ctxfile::load::<B>(v);
        let m = 8 * B;
        let tol: f64 = env_or("TOL", if m >= 128 { "1e-30" } else { "1e-32" }).parse().unwrap();
        let hmax: f64 = env_or("HMAX", "0.05").parse().unwrap();
        let ks = orders_of(&env_or("KS", "12:96:2"));
        let ws: Vec<usize> = orders_of(&env_or("W", "20"));
        let mutate: u8 = env_or("MUTATE", "0").parse().unwrap();
        let kmax: usize = env_or("KMAX", "0").parse().unwrap();
        let kt0 = ctx.kt;
        let stored = ctx.segs.clone();
        let t0 = Instant::now();
        let f_stored = taylor_rows_max(&ctx, &zh, &zl);
        println!("{path}: M {m}, K {}, kt {kt0}, tol {tol:e}, hmax {hmax}; stored plan {} steps {:?}, Taylor rows max|F| {f_stored:.2e} ({:.1} s)",
            ctx.k(), steps_of(&stored).iter().sum::<usize>(), steps_of(&stored), t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        let own = hp_plan(&ctx, &zh, &zl, tol, hmax);
        ctx.segs = own.clone();
        let f_own = taylor_rows_max(&ctx, &zh, &zl);
        println!("  hp's planner at kt {kt0}: {} steps {:?}, {} the stored plan; Taylor rows max|F| {f_own:.2e} ({:.1} s)",
            steps_of(&own).iter().sum::<usize>(), steps_of(&own), if own == stored { "bit for bit" } else { "not" },
            t0.elapsed().as_secs_f64());
        let mut table = Vec::new();
        for &k in &ks {
            let t0 = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut sh = ctx.shooting();
                sh.plan_segments(&zh, &zl, &Plan { kt: k, tol, hmax });
                steps_of(&sh.segs)
            }));
            match r {
                Ok(s) => {
                    let n: usize = s.iter().sum();
                    println!("  order {k:3}: {n:6} steps {s:?} ({:.1} s)", t0.elapsed().as_secs_f64());
                    table.push((k, n));
                }
                Err(_) => println!("  order {k:3}: overflow (a Taylor coefficient is not finite)"),
            }
        }
        let mut ok = true;
        for &w in &ws {
            let t0 = Instant::now();
            let c = plan_auto(&ctx, &zh, &zl, tol, hmax, w, None, kmax, true, mutate).unwrap();
            let t_choice = t0.elapsed().as_secs_f64();
            let est = plan_cost(&c.segs, c.kt, w);
            let fixed = plan_cost(&own, kt0, w);
            let least = table.iter().map(|&(k, n)| (n as f64 * step_cost(k, w, 0), k)).min_by(|a, b| a.0.total_cmp(&b.0));
            ctx.kt = c.kt;
            ctx.segs = c.segs.clone();
            let f_auto = taylor_rows_max(&ctx, &zh, &zl);
            ctx.kt = kt0;
            let cands: Vec<String> = c.passes.iter().map(|p| {
                p.candidates.iter().map(|q| format!("{}:{}", q.kt, q.steps.as_ref().map_or("-".into(), |s| s.iter().sum::<usize>().to_string()))).collect::<Vec<_>>().join(" ")
            }).collect();
            println!("  w {w}: order {} kept, {} steps {:?}, estimated cost {est:.4e} against kt {kt0}'s {fixed:.4e} (ratio {:.3}); \
                      Taylor rows max|F| {f_auto:.2e}; dropped {:?}; {:.1} s; passes [{}]",
                c.kt, steps_of(&c.segs).iter().sum::<usize>(), steps_of(&c.segs), est / fixed, c.dropped, t_choice, cands.join(" | "));
            if let Some((lc, lk)) = least {
                let gap = est / lc - 1.0;
                println!("    least over KS at w {w}: order {lk}, cost {lc:.4e}; the choice is {:.2}% above it", 100.0 * gap);
                ok &= gap <= 0.05;
            }
            ok &= f_auto <= (10.0 * f_own).max(1e-28);
        }
        assert!(ok, "a check failed (choice more than 5% above the least, or its residual above the bound)");
    }

    /// The octo-double Jacobian (model::jacobian_o_tangents, two tangents, the chain corrector's) of an M 32 state
    /// (FILE, its double-double parts and six zero parts) on its stored plan at its order and on plan_auto's (W, default
    /// 20; TOL 1e-32, HMAX 0.05), REPS (default 2) times each, alternating: the wall seconds of each. Only the Taylor
    /// blocks' columns differ between the two; the context has no freeze projector (the chain's kfreeze changes the
    /// horizon rows alone).
    ///
    ///   cd hp && FILE=ctx.bin RAYON_NUM_THREADS=3 cargo test --profile release-dev --lib jac_o_plans -- --ignored --nocapture
    #[test]
    #[ignore]
    fn jac_o_plans() {
        let path = std::env::var("FILE").expect("FILE=<context of hp/dump_ctx.py>");
        let v = ctxfile::read(&path);
        assert_eq!(ctxfile::m_of(&v), 32, "jac_o_plans: the octo-double kernel is built for M 32");
        let (mut ctx, zh, zl) = ctxfile::load::<4>(&v);
        let tol: f64 = env_or("TOL", "1e-32").parse().unwrap();
        let hmax: f64 = env_or("HMAX", "0.05").parse().unwrap();
        let w: usize = env_or("W", "20").parse().unwrap();
        let reps: usize = env_or("REPS", "2").parse().unwrap();
        let zero = vec![0.0; zh.len()];
        let z: Vec<&[f64]> = [&zh[..], &zl[..]].into_iter().chain(std::iter::repeat_n(&zero[..], 6)).collect();
        let c = plan_auto(&ctx, &zh, &zl, tol, hmax, w, None, 0, true, 0).unwrap();
        let arms = [("stored", ctx.kt, ctx.segs.clone()), ("auto", c.kt, c.segs.clone())];
        let mut t = [Vec::new(), Vec::new()];
        for _ in 0..reps {
            for (i, (_, kt, segs)) in arms.iter().enumerate() {
                ctx.kt = *kt;
                ctx.segs = segs.clone();
                let t0 = Instant::now();
                let (j, _) = crate::model::jacobian_o_tangents(&ctx, &z, true, 2).unwrap();
                t[i].push(t0.elapsed().as_secs_f64());
                assert!(j[0].iter().all(|x| x.is_finite()));
            }
        }
        for (i, (name, kt, segs)) in arms.iter().enumerate() {
            let mut ts = t[i].clone();
            ts.sort_by(f64::total_cmp);
            println!("{path}: {name} plan, order {kt}, {} steps: jacobian_o {:?} s (median {:.2})",
                steps_of(segs).iter().sum::<usize>(), t[i].iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), ts[ts.len() / 2]);
        }
    }

    /// a lane from a double-double value: tangents (f64 and in the value's precision) its multiples 1 + i, so none
    /// is zero
    fn lane_of<T: Lane>(v: DV) -> T {
        let t: Vec<DV> = (0..T::NTD).map(|i| v.mul_f(1.0 + i as f64)).collect();
        let mut x = T::from_vt_n(v, &t);
        for i in 0..T::NT {
            x.set_tan64(i, v.mul_f(1.0 + i as f64).hi);
        }
        x
    }

    /// seconds per Taylor step (coefficients 1..=k and the four Horner sums) of lanes T at order k from node b's
    /// state: (median, least) of reps, each the mean over `threads` concurrent steps
    fn t_step<T: Lane, const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], b: usize, k: usize, reps: usize, threads: usize) -> (f64, f64) {
        use rayon::prelude::*;
        let m = 8 * B;
        let o = ctx.off_node(b);
        let s0: State<T, B> = std::array::from_fn(|q| {
            let f = Fl::<DV, B>::load(&zh[o + q * m..o + (q + 1) * m], &zl[o + q * m..o + (q + 1) * m]);
            Fl(std::array::from_fn(|j| lane_of::<T>(f.0[j])))
        });
        let x0 = ctx.nodes[b];
        let h = DD::from_f64(1e-3);
        let omega = T::splat(DD::TWO_PI) / T::splat(DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] });
        let one = || {
            let mut w = work::<T, B>(k);
            for (f, sf) in s0.iter().enumerate() {
                w.c[f][0] = *sf;
            }
            taylor_coeffs(ctx, x0, omega, &mut w);
            let s: [Fl<T, B>; 4] = std::array::from_fn(|f| horner(&w.c[f][..=k], h));
            s[2].val().amax()
        };
        let mut ts: Vec<f64> = (0..reps)
            .map(|_| {
                let t0 = Instant::now();
                let r: f64 = (0..threads).into_par_iter().map(|_| one()).sum();
                std::hint::black_box(r);
                t0.elapsed().as_secs_f64()
            })
            .collect();
        ts.sort_by(f64::total_cmp);
        (ts[reps / 2], ts[0])
    }

    /// least squares of t ~ a g1(k) + b g2(k) in the relative error: (a, b, largest relative misfit)
    fn fit2(ks: &[usize], t: &[f64], g1: impl Fn(f64) -> f64, g2: impl Fn(f64) -> f64) -> (f64, f64, f64) {
        let (mut s11, mut s12, mut s22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&k, &ti) in ks.iter().zip(t) {
            let (u, v) = (g1(k as f64) / ti, g2(k as f64) / ti);
            s11 += u * u;
            s12 += u * v;
            s22 += v * v;
            r1 += u;
            r2 += v;
        }
        let det = s11 * s22 - s12 * s12;
        let a = (r1 * s22 - r2 * s12) / det;
        let b = (s11 * r2 - s12 * r1) / det;
        let worst = ks.iter().zip(t).map(|(&k, &ti)| ((a * g1(k as f64) + b * g2(k as f64)) / ti - 1.0).abs()).fold(0.0, f64::max);
        (a, b, worst)
    }

    /// The step cost by lane precision: seconds per Taylor step at each order of KS (default 12:96:6) on a stored
    /// state's context (FILE, node K/2's state), for the lanes of the runs at its M (M <= 48: DV residual, QV and Qu
    /// quad-double, OV and Oun<2> octo-double; M >= 128: DV and the f64 Jacobian's Fun<4>), REPS (default 7) each,
    /// THREADS (default 1) concurrent steps per rep. The fit t ~ a (k^2 + w k) gives w per lane, with its largest
    /// relative misfit, and the misfit of multishoot's default w = 20 with a fitted.
    ///
    ///   cd hp && FILE=ctx.bin cargo test --profile release-dev --lib bench_step_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_step_cost() {
        let path = std::env::var("FILE").expect("FILE=<context of hp/dump_ctx.py>");
        let v = ctxfile::read(&path);
        // a large stack, as the Solver's pool: a state of octo-double lanes is large
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || match ctxfile::m_of(&v) {
                32 => bench_small::<4>(&path, &v),
                48 => bench_small::<6>(&path, &v),
                320 => bench_big::<40>(&path, &v),
                384 => bench_big::<48>(&path, &v),
                m => panic!("bench_step_cost: M {m} is not built here (32, 48, 320, 384)"),
            })
            .unwrap()
            .join()
            .unwrap();
    }

    fn bench_cfg() -> (Vec<usize>, usize, usize) {
        (orders_of(&env_or("KS", "12:96:6")), env_or("REPS", "7").parse().unwrap(), env_or("THREADS", "1").parse().unwrap())
    }

    fn bench_lane<T: Lane, const B: usize>(name: &str, ctx: &Ctx<B>, zh: &[f64], zl: &[f64]) {
        let (ks, reps, threads) = bench_cfg();
        let b = ctx.k() / 2;
        let mut med = Vec::new();
        let mut row = Vec::new();
        for &k in &ks {
            let (t, t_min) = t_step::<T, B>(ctx, zh, zl, b, k, reps, threads);
            med.push(t);
            row.push(format!("{k}:{:.3e}({:.2})", t, t_min / t));
        }
        let (a, bb, worst) = fit2(&ks, &med, |k| k * k, |k| k);
        let w = bb / a;
        let g20 = |k: f64| k * k + 20.0 * k;
        let a20 = ks.iter().zip(&med).map(|(&k, &t)| g20(k as f64) / t).sum::<f64>()
            / ks.iter().zip(&med).map(|(&k, &t)| (g20(k as f64) / t).powi(2)).sum::<f64>();
        let worst20 = ks.iter().zip(&med).map(|(&k, &t)| (a20 * g20(k as f64) / t - 1.0).abs()).fold(0.0, f64::max);
        println!("  {name:8} M {}: w {w:.1} (fit misfit {:.1}%; w 20 misfit {:.1}%), node {b}, {threads} thread(s), {reps} reps", 8 * B,
            100.0 * worst, 100.0 * worst20);
        println!("    k:median s (least/median) {}", row.join(" "));
    }

    fn bench_small<const B: usize>(path: &str, v: &[f64]) {
        let (ctx, zh, zl) = ctxfile::load::<B>(v);
        println!("{path}: M {}, K {}", 8 * B, ctx.k());
        bench_lane::<DV, B>("DV", &ctx, &zh, &zl);
        bench_lane::<QV, B>("QV", &ctx, &zh, &zl);
        bench_lane::<Qu, B>("Qu", &ctx, &zh, &zl);
        bench_lane::<OV, B>("OV", &ctx, &zh, &zl);
        bench_lane::<Oun<2>, B>("Oun<2>", &ctx, &zh, &zl);
    }

    fn bench_big<const B: usize>(path: &str, v: &[f64]) {
        let (ctx, zh, zl) = ctxfile::load::<B>(v);
        println!("{path}: M {}, K {}", 8 * B, ctx.k());
        bench_lane::<Fun<4>, B>("Fun<4>", &ctx, &zh, &zl);
        bench_lane::<DV, B>("DV", &ctx, &zh, &zl);
    }
}
