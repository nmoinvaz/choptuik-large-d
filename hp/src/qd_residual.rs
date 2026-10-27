//! The model's generic code with quad-double lanes (notes/QUAD_DOUBLE.md, stage 2) and the residual's operation mix
//! and cost (stage 1, item 5).
//!
//! ```text
//! cargo test --release --lib qv_lane                                   QV through every block of a small synthetic
//!     context agrees with DV to double-double accuracy (control: Delta moved by 1e-20 must fail)
//! FILE=ctx.bin cargo test --release --lib bench_qd_residual -- --ignored --nocapture
//!     (ctx.bin from hp/dump_ctx.py) the residual's operations by type (a counting lane), and the time of the
//!     residual's parts in double-double and with QV lanes, one thread
//! ```
use crate::dd::DD;
use crate::dv::{DotAcc, DV};
use crate::field::{DMat, FieldSolve, Fl, State};
use crate::lane::{BaseLane, Lane};
use crate::model::{block_state, centre_state, horizon_lus, residual, residual_qd, residual_qdh, taylor_path, Ctx, TWork, QD_MUT_NOREFINE, QD_MUT_REFINE4};
use crate::qd::QD;
use crate::qv::{QDotAcc, QV};
use crate::v8::V8;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Instant;

// -------------------------------------------------------------------------------------------------------------
// a counting double-double lane
// -------------------------------------------------------------------------------------------------------------

const NC: usize = 11;
const CNAMES: [&str; NC] = ["add/sub", "mul", "div", "neg", "x const", "x f64", "+ const", "acc a b", "acc a c", "acc a", "finish"];
thread_local!(static CNT: RefCell<[u64; NC]> = const { RefCell::new([0; NC]) });

fn bump(i: usize) {
    CNT.with(|c| c.borrow_mut()[i] += 1);
}

fn take_counts() -> [u64; NC] {
    CNT.with(|c| std::mem::replace(&mut *c.borrow_mut(), [0; NC]))
}

/// DV with every lane operation counted (one count per 8-lane operation)
#[derive(Clone, Copy, Debug)]
struct CDV(DV);

impl Add for CDV {
    type Output = CDV;
    fn add(self, o: CDV) -> CDV {
        bump(0);
        CDV(self.0 + o.0)
    }
}
impl Sub for CDV {
    type Output = CDV;
    fn sub(self, o: CDV) -> CDV {
        bump(0);
        CDV(self.0 - o.0)
    }
}
impl Mul for CDV {
    type Output = CDV;
    fn mul(self, o: CDV) -> CDV {
        bump(1);
        CDV(self.0 * o.0)
    }
}
impl Div for CDV {
    type Output = CDV;
    fn div(self, o: CDV) -> CDV {
        bump(2);
        CDV(self.0 / o.0)
    }
}
impl Neg for CDV {
    type Output = CDV;
    fn neg(self) -> CDV {
        bump(3);
        CDV(-self.0)
    }
}

impl Lane for CDV {}

impl BaseLane for CDV {
    type S = multidouble::simd::Native;
    type Acc = DotAcc;
    const TANGENT: bool = false;
    fn zero() -> Self {
        CDV(DV::zero())
    }
    fn cst(c: DV) -> Self {
        CDV(c)
    }
    fn mulc(self, c: DV) -> Self {
        bump(4);
        CDV(self.0 * c)
    }
    fn mulf(self, s: f64) -> Self {
        bump(5);
        CDV(self.0.mul_f(s))
    }
    fn mul_pwr2(self, s: f64) -> Self {
        bump(5);
        CDV(self.0.mul_pwr2(s))
    }
    fn addc(self, c: DV) -> Self {
        bump(6);
        CDV(self.0 + c)
    }
    fn bcast(self, j: usize) -> Self {
        CDV(self.0.bcast(j))
    }
    fn val(self) -> DV {
        self.0
    }
    fn tan(self) -> DV {
        DV::zero()
    }
    fn from_vt(v: DV, _t: DV) -> Self {
        CDV(v)
    }
    fn acc0() -> DotAcc {
        DotAcc::zero()
    }
    fn acc(a: DotAcc, x: Self, y: Self) -> DotAcc {
        bump(7);
        a.add_prod(x.0, y.0)
    }
    fn accc(a: DotAcc, x: Self, c: DV) -> DotAcc {
        bump(8);
        a.add_prod(x.0, c)
    }
    fn acc1(a: DotAcc, x: Self) -> DotAcc {
        bump(9);
        a.add(x.0)
    }
    fn fin(a: DotAcc) -> Self {
        bump(10);
        CDV(a.finish())
    }
}

// -------------------------------------------------------------------------------------------------------------
// a small synthetic context (M = 8)
// -------------------------------------------------------------------------------------------------------------

/// The odd-sector d/dtheta of hp/python/hpkernel/driver.py dmat_odd in f64 (lo parts zero), M = 8; nodes
/// 0.005..0.1, four Taylor steps per segment; the state: centre data p1 = 0.3 sin(theta), c0 = 0.6 + 0.05 cos(2
/// theta), the nodes integrated outward from the centre in double-double, the horizon data from the last node.
pub(crate) fn synth() -> (Ctx<1>, Vec<f64>, Vec<f64>) {
    let m = 8usize;
    let nn = 2 * m;
    let mut dh = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut v = 0.0;
            for (jj, sgn) in [(j, 1.0), (j + m, -1.0)] {
                if jj != i {
                    let k = i as i64 - jj as i64;
                    let d = k as f64 * PI / nn as f64;
                    v += sgn * 0.5 * if k.rem_euclid(2) == 0 { 1.0 } else { -1.0 } / d.tan();
                }
            }
            dh[i * m + j] = v;
        }
    }
    let d = DMat::<1>::from_rows(&dh, &vec![0.0; m * m]);
    let th: Vec<f64> = (0..m).map(|j| PI * j as f64 / m as f64).collect();
    let phase = Fl::<DV, 1>::load(&th.iter().map(|t| t.sin()).collect::<Vec<_>>(), &vec![0.0; m]);
    let nodes = vec![0.005, 0.01, 0.02, 0.05, 0.1];
    let segs: Vec<Vec<f64>> = nodes.windows(2).map(|w| (0..=4).map(|k| w[0] + (w[1] - w[0]) * k as f64 / 4.0).collect()).collect();
    let k = nodes.len();
    let ctx = Ctx::<1> { n: DD::from_f64(1.7), d, phase, nodes, segs, kt: 20, kc: 30, ks: 30, freeze: None, match_node: k - 1, lam: DD::ZERO, mutate: 0, f64_dd: 0, horizon_alpha: 0 };
    let l = ctx.len();
    let (mut zh, mut zl) = (vec![0.0; l], vec![0.0; l]);
    let p1: Vec<f64> = th.iter().map(|t| 0.3 * (2.0 * t).sin()).collect();
    let c0: Vec<f64> = th.iter().map(|t| 0.6 + 0.05 * (4.0 * t).cos()).collect();
    zh[..m].copy_from_slice(&p1);
    zh[m..2 * m].copy_from_slice(&c0);
    let delta = DD::new(3.4645583951137432, 1.7408255559455386e-17);
    zh[l - 1] = delta.hi;
    zl[l - 1] = delta.lo;
    let omega = DV::splat(DD::TWO_PI / delta);
    let put = |zh: &mut Vec<f64>, zl: &mut Vec<f64>, off: usize, f: &Fl<DV, 1>| f.store(&mut zh[off..off + m], &mut zl[off..off + m]);
    let mut s: State<DV, 1> = centre_state(&ctx, &ctx.fld(&zh, &zl, 0), &ctx.fld(&zh, &zl, m), omega);
    let mut w = TWork::<DV, 1>::new(ctx.kt);
    for b in 0..k {
        if b > 0 {
            s = taylor_path(&ctx, &s, &ctx.segs[b - 1], omega, &mut w);
        }
        for q in 0..4 {
            put(&mut zh, &mut zl, ctx.off_node(b) + q * m, &s[q]);
        }
    }
    put(&mut zh, &mut zl, ctx.off_mh(), &s[1].mulf(0.1));
    put(&mut zh, &mut zl, ctx.off_ah(), &s[2]);
    (ctx, zh, zl)
}

/// max over blocks, fields and points of |QV block state - DV block state| / max(1, |DV|), the QV lanes run with
/// Delta (1 + rel)
fn qv_vs_dv<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], rel: f64) -> f64 {
    let l = zh.len();
    let delta = DD { hi: zh[l - 1], lo: zl[l - 1] };
    let lus = horizon_lus(ctx, &ctx.fld(zh, zl, ctx.off_ah()), delta, false, DD::ZERO).unwrap();
    let mut worst = 0.0f64;
    for b in 0..ctx.k() + 1 {
        let inp: Vec<Fl<DV, B>> = ctx.block_inputs(b).iter().map(|&o| ctx.fld(zh, zl, o)).collect();
        let sd = block_state::<DV, B>(ctx, b, &inp, DV::splat(delta), &lus, &lus);
        let inq: Vec<Fl<QV, B>> = inp.iter().map(Fl::cst).collect();
        let dq = QV::splat(QD::from_dd(delta) + QD::from_f64(delta.hi * rel));
        let sq = block_state::<QV, B>(ctx, b, &inq, dq, &lus, &lus);
        for q in 0..4 {
            for j in 0..Ctx::<B>::M {
                let (a, c) = (sd[q].get(j), sq[q].0[j / 8].lane(j % 8));
                let diff = (c - QD::from_dd(a)).to_f64().abs();
                assert!(diff.is_finite(), "block {b} field {q} point {j}: {a:?} vs {c:?}");
                worst = worst.max(diff / (a.hi.abs().max(1.0)));
            }
        }
    }
    worst
}

/// QV runs every block of the generic model (centre series, Taylor stepping, horizon series with the double-double
/// solves) and agrees with DV to double-double accuracy; moving Delta by 1e-20 in the QV run must break the check.
#[test]
fn qv_lane_blocks_agree() {
    let (ctx, zh, zl) = synth();
    let d = qv_vs_dv(&ctx, &zh, &zl, 0.0);
    let dc = qv_vs_dv(&ctx, &zh, &zl, 1e-20);
    println!("QV vs DV, all blocks of the synthetic context: max relative difference {d:.2e}; with Delta x (1 + 1e-20): {dc:.2e}");
    assert!(d < 1e-26, "QV and DV differ by {d:e}");
    assert!(dc > 1e-26, "control: Delta moved by 1e-20 not seen ({dc:e})");
}

/// The horizon block's QV state from a four-part state (z2, z3 given), with ctx.mutate as set.
fn horizon_qv<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4]) -> crate::field::State<QV, B> {
    let l = z[0].len();
    let delta = DD { hi: z[0][l - 1], lo: z[1][l - 1] };
    let lus = horizon_lus(ctx, &ctx.fld(z[0], z[1], ctx.off_ah()), delta, false, DD::ZERO).unwrap();
    let kb = ctx.k();
    let inq: Vec<Fl<QV, B>> = ctx
        .block_inputs(kb)
        .iter()
        .map(|&o| {
            let mut r = [QV::zero(); B];
            for (b, rb) in r.iter_mut().enumerate() {
                let i = o + 8 * b;
                *rb = QV::load(&z[0][i..i + 8], &z[1][i..i + 8], &z[2][i..i + 8], &z[3][i..i + 8]);
            }
            Fl(r)
        })
        .collect();
    let dq = QV::splat(QD::new(z[0][l - 1], z[1][l - 1], z[2][l - 1], z[3][l - 1]));
    block_state::<QV, B>(ctx, kb, &inq, dq, &lus, &lus)
}

/// max over fields and points of |a - b| / max(1, |b|) for two QV states
fn qv_diff<const B: usize>(a: &crate::field::State<QV, B>, b: &crate::field::State<QV, B>) -> f64 {
    let mut worst = 0.0f64;
    for q in 0..4 {
        for j in 0..Ctx::<B>::M {
            let (x, y) = (a[q].0[j / 8].lane(j % 8), b[q].0[j / 8].lane(j % 8));
            worst = worst.max((x - y).to_f64().abs() / y.to_f64().abs().max(1.0));
        }
    }
    worst
}

/// residual_qdh (the horizon block in quad-double) on the synthetic context:
/// 1. at a double-double state (z2 = z3 = 0) every row outside the horizon block is bitwise `residual`'s, and the
///    horizon rows agree with it to double-double accuracy (1e-26);
/// 2. the refined horizon solves have converged: two steps (the default) against four agree to 1e-55; control: no
///    refinement against four must differ by more than 1e-40 (the double-double LU alone is not quad-double);
/// 3. the four parts are read: a_h moved by 1e-40 relative in z2 must move the horizon block's QV state by more
///    than 1e-48 (and z2 = 0 gives it bit for bit again).
#[test]
fn residual_qdh_agrees() {
    let (mut ctx, zh, zl) = synth();
    let l = ctx.len();
    let m = Ctx::<1>::M;
    let zero = vec![0.0; l];
    let (fh, fl) = residual(&ctx, &zh, &zl, false).unwrap();
    let (qh, ql) = residual_qdh(&ctx, [&zh, &zl, &zero, &zero], false).unwrap();
    let hb = ctx.k() * 4 * m;
    let outside = (0..l).filter(|&i| i < hb || i == l - 1).all(|i| qh[i] == fh[i] && ql[i] == fl[i]);
    let dh = (hb..l - 1).map(|i| ((qh[i] - fh[i]) + (ql[i] - fl[i])).abs()).fold(0.0, f64::max);
    let fmax = (hb..l - 1).map(|i| fh[i].abs()).fold(0.0, f64::max);
    println!("residual_qdh at z2 = z3 = 0: rows outside the horizon block bitwise residual's: {outside}; horizon rows differ by {dh:.2e} (max |F| there {fmax:.2e})");
    assert!(outside, "residual_qdh changed rows outside the horizon block");
    assert!(dh < 1e-26, "horizon rows differ by {dh:e}");
    let z: [&[f64]; 4] = [&zh, &zl, &zero, &zero];
    ctx.mutate = QD_MUT_REFINE4;
    let r4 = horizon_qv(&ctx, z);
    ctx.mutate = 0;
    let r2 = horizon_qv(&ctx, z);
    ctx.mutate = QD_MUT_NOREFINE;
    let r0 = horizon_qv(&ctx, z);
    ctx.mutate = 0;
    let (d2, d0) = (qv_diff(&r2, &r4), qv_diff(&r0, &r4));
    println!("horizon block, refinement: 2 steps against 4: {d2:.2e}; control, none against 4: {d0:.2e}");
    assert!(d2 < 1e-55, "the refinement has not converged in two steps ({d2:e})");
    assert!(d0 > 1e-40, "control: the unrefined solves agree with the refined ones ({d0:e})");
    let mut z2 = zero.clone();
    for i in ctx.off_ah()..ctx.off_ah() + m {
        z2[i] = 1e-40 * zh[i];
    }
    let rp = horizon_qv(&ctx, [&zh, &zl, &z2, &zero]);
    let dp = qv_diff(&rp, &r2);
    let same = qv_diff(&horizon_qv(&ctx, z), &r2);
    println!("a_h moved by 1e-40 in z2: the horizon state moves by {dp:.2e}; again with z2 = 0: {same:.1e}");
    assert!(dp > 1e-48, "the third part of a_h was not read ({dp:e})");
    assert!(same == 0.0, "not reproducible ({same:e})");
}

/// Block b's QV state from a four-part state (as horizon_qv, for any block)
fn block_qv<const B: usize>(ctx: &Ctx<B>, z: [&[f64]; 4], b: usize) -> crate::field::State<QV, B> {
    let l = z[0].len();
    let delta = DD { hi: z[0][l - 1], lo: z[1][l - 1] };
    let lus = horizon_lus(ctx, &ctx.fld(z[0], z[1], ctx.off_ah()), delta, false, DD::ZERO).unwrap();
    let inq: Vec<Fl<QV, B>> = ctx
        .block_inputs(b)
        .iter()
        .map(|&o| {
            let mut r = [QV::zero(); B];
            for (k, rk) in r.iter_mut().enumerate() {
                let i = o + 8 * k;
                *rk = QV::load(&z[0][i..i + 8], &z[1][i..i + 8], &z[2][i..i + 8], &z[3][i..i + 8]);
            }
            Fl(r)
        })
        .collect();
    let dq = QV::splat(QD::new(z[0][l - 1], z[1][l - 1], z[2][l - 1], z[3][l - 1]));
    block_state::<QV, B>(ctx, b, &inq, dq, &lus, &lus)
}

/// residual_qd (every block in quad-double) on the synthetic context:
/// 1. at a double-double state (z2 = z3 = 0) every row agrees with `residual`'s to double-double accuracy (1e-26), the
///    horizon rows and the phase row bitwise with residual_qdh's; control: some row outside the horizon block differs
///    from `residual`'s in its last bits (the quad-double path ran there);
/// 2. the four parts are read in the centre block and in a Taylor block: their inputs moved by 1e-40 relative in z2
///    must move the block's QV state by more than 1e-48, and z2 = 0 gives it bit for bit again.
#[test]
fn residual_qd_agrees() {
    let (ctx, zh, zl) = synth();
    let l = ctx.len();
    let m = Ctx::<1>::M;
    let zero = vec![0.0; l];
    let (fh, fl) = residual(&ctx, &zh, &zl, false).unwrap();
    let (hh, hl) = residual_qdh(&ctx, [&zh, &zl, &zero, &zero], false).unwrap();
    let (qh, ql) = residual_qd(&ctx, [&zh, &zl, &zero, &zero], false).unwrap();
    let hb = ctx.k() * 4 * m;
    let d = (0..l).map(|i| ((qh[i] - fh[i]) + (ql[i] - fl[i])).abs()).fold(0.0, f64::max);
    let same_h = (hb..l).all(|i| qh[i] == hh[i] && ql[i] == hl[i]);
    let changed = (0..hb).filter(|&i| qh[i] != fh[i] || ql[i] != fl[i]).count();
    println!("residual_qd at z2 = z3 = 0: max difference from residual {d:.2e}; horizon and phase rows bitwise residual_qdh's: \
              {same_h}; rows outside the horizon block that differ from residual's in their last bits: {changed} of {hb}");
    assert!(d < 1e-26, "residual_qd differs from residual by {d:e}");
    assert!(same_h, "residual_qd's horizon rows are not residual_qdh's");
    assert!(changed > 0, "control: no row outside the horizon block took the quad-double path");
    for b in [0, 1] {
        let z: [&[f64]; 4] = [&zh, &zl, &zero, &zero];
        let r0 = block_qv(&ctx, z, b);
        let mut z2 = zero.clone();
        for &o in ctx.block_inputs(b).iter() {
            for i in o..o + m {
                z2[i] = 1e-40 * zh[i];
            }
        }
        let dp = qv_diff(&block_qv(&ctx, [&zh, &zl, &z2, &zero], b), &r0);
        let again = qv_diff(&block_qv(&ctx, z, b), &r0);
        println!("block {b}: inputs moved by 1e-40 in z2 move its state by {dp:.2e}; again with z2 = 0: {again:.1e}");
        assert!(dp > 1e-48, "block {b}: the third part of its inputs was not read ({dp:e})");
        assert!(again == 0.0, "block {b}: not reproducible ({again:e})");
    }
}

// -------------------------------------------------------------------------------------------------------------
// the residual: operations, times, and the quad-double estimate
// -------------------------------------------------------------------------------------------------------------

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn secs(reps: usize, mut f: impl FnMut()) -> f64 {
    median(
        (0..reps)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .collect(),
    )
}

/// ns per 8-lane operation of each counted type for DV and for QV (throughput, 16 independent vectors)
fn op_costs() -> ([f64; NC], [f64; NC]) {
    use std::hint::black_box;
    const NV: usize = 16;
    let reps = 20000;
    let per = (reps * NV) as f64;
    let mk = |k: usize| QV::splat(QD::new(1.0 + 0.01 * k as f64, 3e-18 * k as f64, -1e-34, 2e-51));
    let qa: Vec<QV> = (0..NV).map(mk).collect();
    let qb: Vec<QV> = (0..NV).map(|k| mk(k + 3)).collect();
    let da: Vec<DV> = qa.iter().map(|q| q.to_dv()).collect();
    let db: Vec<DV> = qb.iter().map(|q| q.to_dv()).collect();
    macro_rules! t {
        ($x:expr, $f:expr) => {{
            let mut x = $x.clone();
            let t0 = Instant::now();
            for _ in 0..reps {
                for i in 0..NV {
                    x[i] = ($f)(x[i], i);
                }
                black_box(&mut x);
            }
            t0.elapsed().as_secs_f64() * 1e9 / per
        }};
    }
    macro_rules! tacc {
        ($a:expr, $b:expr, $z:expr, $f:expr) => {{
            let t0 = Instant::now();
            let mut s = 0.0;
            for _ in 0..reps {
                for i in 0..NV {
                    let mut acc = $z;
                    for k in 0..16 {
                        acc = ($f)(acc, $a[(i + k) % NV], $b[k]);
                    }
                    s += black_box(&acc).s0();
                }
            }
            black_box(s);
            t0.elapsed().as_secs_f64() * 1e9 / (per * 16.0)
        }};
    }
    let half = 0.5;
    let mut d = [0.0; NC];
    let mut q = [0.0; NC];
    d[0] = t!(da, |x: DV, i: usize| (x + db[i]).mul_f(half)) - t!(da, |x: DV, _i: usize| x.mul_f(half));
    q[0] = t!(qa, |x: QV, i: usize| (x + qb[i]).mul_pwr2(half)) - t!(qa, |x: QV, _i: usize| x.mul_pwr2(half));
    d[1] = t!(da, |x: DV, i: usize| x * db[i]);
    q[1] = t!(qa, |x: QV, i: usize| x * qb[i]);
    d[2] = t!(da, |x: DV, i: usize| x / db[i]);
    q[2] = t!(qa, |x: QV, i: usize| x / qb[i]);
    d[3] = t!(da, |x: DV, _i: usize| -x);
    q[3] = t!(qa, |x: QV, _i: usize| -x);
    d[4] = d[1];
    q[4] = q[1];
    d[5] = t!(da, |x: DV, _i: usize| x.mul_f(1.0000001));
    q[5] = t!(qa, |x: QV, _i: usize| x.mul_f(1.0000001));
    d[6] = d[0];
    q[6] = q[0];
    d[7] = tacc!(da, db, DotAcc::zero(), |a: DotAcc, x: DV, y: DV| a.add_prod(x, y));
    q[7] = tacc!(qa, qb, QDotAcc::zero(), |a: QDotAcc, x: QV, y: QV| a.add_prod(x, y));
    d[8] = d[7];
    q[8] = q[7];
    d[9] = tacc!(da, db, DotAcc::zero(), |a: DotAcc, x: DV, _y: DV| a.add(x));
    q[9] = tacc!(qa, qb, QDotAcc::zero(), |a: QDotAcc, x: QV, _y: QV| a.add(x));
    d[10] = t!(da, |x: DV, i: usize| { let a = DotAcc { s: x.hi, e: db[i].lo }; a.finish() });
    q[10] = t!(qa, |x: QV, i: usize| { let a = QDotAcc { s: [x.c[0], qb[i].c[1], x.c[2], qb[i].c[3], x.c[3]] }; a.finish() });
    (d.map(|v| v.max(0.0)), q.map(|v| v.max(0.0)))
}

trait S0 {
    fn s0(&self) -> f64;
}
impl S0 for DotAcc {
    fn s0(&self) -> f64 {
        self.s.to_array()[0]
    }
}
impl S0 for QDotAcc {
    fn s0(&self) -> f64 {
        self.s[0].to_array()[0]
    }
}

fn run_residual<const B: usize>(v: &[f64]) {
    let (ctx, zh, zl) = crate::model::ctxfile::load::<B>(v);
    let m = Ctx::<B>::M;
    let l = ctx.len();
    let delta = DD { hi: zh[l - 1], lo: zl[l - 1] };
    let nb = ctx.k() + 1;
    let ah = ctx.fld(&zh, &zl, ctx.off_ah());
    println!("M = {m}, L = {l}, kt {}, kc {}, ks {}, {} nodes, {} Taylor steps; isa {}; one thread", ctx.kt, ctx.kc, ctx.ks, ctx.k(), ctx.segs.iter().map(|s| s.len() - 1).sum::<usize>(), V8::ISA);
    let t_res = secs(3, || {
        std::hint::black_box(residual(&ctx, &zh, &zl, false).unwrap());
    });
    let mut lus = Vec::new();
    let t_lus = secs(3, || lus = horizon_lus(&ctx, &ah, delta, false, DD::ZERO).unwrap());
    // one horizon solve (double-double, scalar), as in psolve
    let f = ctx.fld(&zh, &zl, ctx.off_ah());
    let t_solve = secs(5, || {
        for lu in &lus {
            std::hint::black_box(lu.solve_lanes(&f));
        }
    }) / lus.len() as f64;
    let inputs = |b: usize| -> Vec<Fl<DV, B>> { ctx.block_inputs(b).iter().map(|&o| ctx.fld(&zh, &zl, o)).collect() };
    let (cd, cq) = op_costs();
    println!("ns per 8-lane operation (dd, qd): {}", (0..NC).filter(|&i| i != 4 && i != 6 && i != 8).map(|i| format!("{} {:.1}/{:.1}", CNAMES[i], cd[i], cq[i])).collect::<Vec<_>>().join(", "));
    println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}   operations (8-lane) by type", "block", "dd s", "qd lanes s", "ratio", "model dd", "model qd", "solves");
    let mut tot = [0.0f64; 4];
    let mut counts_tot = [0u64; NC];
    for b in 0..nb {
        let inp = inputs(b);
        let td = secs(3, || {
            std::hint::black_box(block_state::<DV, B>(&ctx, b, &inp, DV::splat(delta), &lus, &lus));
        });
        let inq: Vec<Fl<QV, B>> = inp.iter().map(Fl::cst).collect();
        let tq = secs(3, || {
            std::hint::black_box(block_state::<QV, B>(&ctx, b, &inq, QV::splat(QD::from_dd(delta)), &lus, &lus));
        });
        let inc: Vec<Fl<CDV, B>> = inp.iter().map(Fl::cst).collect();
        take_counts();
        std::hint::black_box(block_state::<CDV, B>(&ctx, b, &inc, CDV(DV::splat(delta)), &lus, &lus));
        let c = take_counts();
        let solves = if b == ctx.k() { ctx.ks + 1 } else { 0 };
        // model: counts x per-operation cost, plus the double-double solves (their cost measured above)
        let md: f64 = (0..NC).map(|i| c[i] as f64 * cd[i]).sum::<f64>() * 1e-9 + solves as f64 * t_solve;
        let mq: f64 = (0..NC).map(|i| c[i] as f64 * cq[i]).sum::<f64>() * 1e-9 + solves as f64 * t_solve;
        for i in 0..NC {
            counts_tot[i] += c[i];
        }
        tot[0] += td;
        tot[1] += tq;
        tot[2] += md;
        tot[3] += mq;
        let cs: Vec<String> = (0..NC).filter(|&i| c[i] > 0).map(|i| format!("{} {}", CNAMES[i], c[i])).collect();
        println!("{b:>6} {td:>10.4} {tq:>10.4} {:>10.1} {md:>10.4} {mq:>10.4} {solves:>7}   {}", tq / td, cs.join(", "));
    }
    println!("{:>6} {:>10.4} {:>10.4} {:>10.1} {:>10.4} {:>10.4}", "all", tot[0], tot[1], tot[1] / tot[0], tot[2], tot[3]);
    let cs: Vec<String> = (0..NC).filter(|&i| counts_tot[i] > 0).map(|i| format!("{} {}", CNAMES[i], counts_tot[i])).collect();
    println!("operations in all blocks: {}", cs.join(", "));
    println!(
        "residual (double-double, one thread) {t_res:.4} s = horizon LUs {t_lus:.4} s ({} of {m} x {m}) + blocks {:.4} s (of which {} horizon solves {:.4} s) + assembly",
        ctx.ks + 1,
        tot[0],
        ctx.ks + 1,
        (ctx.ks + 1) as f64 * t_solve
    );
}

/// FILE=ctx.bin (hp/dump_ctx.py) cargo test --release --lib bench_qd_residual -- --ignored --nocapture
#[test]
#[ignore]
fn bench_qd_residual() {
    let v = crate::model::ctxfile::read(&std::env::var("FILE").expect("FILE"));
    match crate::model::ctxfile::m_of(&v) / 8 {
        6 => run_residual::<6>(&v),
        10 => run_residual::<10>(&v),
        b => panic!("M = {} not built into the bench (M = 48, 80)", 8 * b),
    }
}
