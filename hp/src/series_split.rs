//! The horizon series one jet order at a time with each Cauchy sum's middle terms formed ahead of the order
//! (`Ctx::series_split` = SPLIT_MIDDLE; off by default), on the octo-double jets of `Solver.series_o`.
//!
//! At jet order j (the dimension), a Cauchy sum over the u-order, sum_i x_i y_(k-i), has the terms
//! x_(i,l) y_(k-i,j-l), l = 0..j. The middle terms, 1 <= l <= j - 1, read only orders below j, which `keep` holds at
//! every u-order from the earlier calls. The edge terms, l = 0 and l = j, read order j, which the u-order loop forms.
//! So the middle terms of all sums at all u-orders are summed first, in parallel, into open accumulators, and the loop
//! adds only the two edge terms per index (85% of the terms are middle ones at N = 24). The sums are
//! `horizon_series_at`'s in another order. Their accumulator (multidouble's `ODotAcc`, nine levels) holds each sum far
//! below the octo-double rounding, so a reordered sum finishes to other bits only when it lies that close to a
//! rounding boundary; on the synthetic context and on a D 253 chain context the series were the same bits.
//! `split_tests` and hp/series_split_check.py bound the difference by the two horizon forms' (Ctx::horizon_alpha).
//!
//! With SPLIT_FAST the middle terms are instead truncated u-products by block-scaled exact convolution
//! (`middles_fast`, `fastmul`), one per sum, lane vector and order l, the sum's terms grouped by l.
//!
//! The split of a product into the terms on known coefficients and those on the newest one is that of relaxed
//! multiplication: J. van der Hoeven, Relax, but don't be too lazy, J. Symbolic Comput. 34(6), 479-542 (2002),
//! doi:10.1006/jsco.2002.0562.
use crate::field::Fl;
use crate::lane::Lane;
use crate::fastmul;
use crate::model::{
    dapply_at, dim, horizon_series_at, mul_recip, psolve_at, series_scale, Ctx, HPrep, H_SERIES, JET_MUT_KAPPA,
    SPLIT_FAST, SPLIT_MUT_DROP_L1,
};
use crate::dd::DD;
use multidouble::dual::{Lane as BaseLane, Param};
use multidouble::native::{ODotAcc, OJet, OV};
use rayon::prelude::*;
use std::any::Any;

type F<const N: usize, const B: usize> = Fl<OJet<N>, B>;
/// one jet coefficient of a field
type V<const B: usize> = Fl<OV, B>;

/// nanoseconds in the horizon series by order, in the split form's middle terms and in its u-order loop, summed
/// over calls (tests' timing only)
#[cfg(test)]
pub(crate) static T_HOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static T_MID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static T_LOOP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// the horizon series' coefficient j in the context's form: `horizon_series_at`, or with Ctx::series_split its split
/// form where it applies (`horizon_series_at_split`)
pub fn horizon_series_at_form<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    delta: &T,
    prep: &HPrep,
    keep: &mut Vec<Vec<Fl<T, B>>>,
    j: usize,
) {
    #[cfg(test)]
    let t0 = std::time::Instant::now();
    if ctx.series_split == 0 || !horizon_series_at_split(ctx, m0, a0, delta, prep, keep, j) {
        horizon_series_at(ctx, m0, a0, delta, prep, keep, j);
    }
    #[cfg(test)]
    T_HOR.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

// the series of `keep` that enter a Cauchy sum (model::H_SERIES's order)
const PP: usize = 0;
const MM: usize = 1;
const AA: usize = 2;
const CC: usize = 3;
const A2: usize = 4;
const Q2: usize = 5;
const PM: usize = 8;
const VV: usize = 9;
const WW: usize = 10;
const ZC: usize = 11;
const AM1: usize = 12;
const PMP: usize = 13;

/// A Cauchy sum of the horizon recursion at u-order kk: sum_(i=lo..=k-hoff) (w_i x_i) y_(k-i), k = kk + dk, with
/// w_i = i when `w`, else 1, and y_kk read from series `yk` when given (p + m before p_kk is solved).
#[derive(Clone, Copy)]
struct HSum {
    x: usize,
    y: usize,
    yk: Option<usize>,
    dk: usize,
    lo: usize,
    hoff: usize,
    w: bool,
    /// formed only for kk < ks (after the loop's last p_kk)
    inner: bool,
}

const fn hs(x: usize, y: usize, dk: usize, lo: usize, hoff: usize, w: bool, inner: bool) -> HSum {
    HSum { x, y, yk: None, dk, lo, hoff, w, inner }
}

// the sums in the order the u-order loop forms them
const S_CA: usize = 0;
const S_WPU: usize = 1;
const S_S1P: usize = 2;
const S_S2P: usize = 3;
const S_CPMR: usize = 4;
const S_PP: usize = 5;
const S_MM: usize = 6;
const S_AA2: usize = 7;
const S_AQ2: usize = 8;
const S_A2N: usize = 9;
const S_ZMU: usize = 10;
const S_S1M: usize = 11;
const S_S2M: usize = 12;
const S_CPMF: usize = 13;
const NSUMS: usize = 14;

/// the sums of `horizon_series_at`, as it forms them. p_kk is zero in those before its solve, so their ranges leave
/// it out: (i p_i) w_(kk+1-i) for i < kk, v_i p_(kk-i) and c_i p_(kk-i) for i >= 1, and c (p + m) with p_kk's m_kk.
fn sums(alpha: bool) -> [HSum; NSUMS] {
    let mut s = [
        hs(CC, if alpha { AM1 } else { A2 }, 0, 0, 0, false, false),
        hs(PP, WW, 1, 0, 2, true, false),
        hs(VV, PP, 0, 1, 0, false, false),
        hs(CC, PP, 0, 1, 0, false, false),
        hs(CC, PM, 0, 0, 0, false, false),
        hs(PP, PP, 0, 0, 0, false, false),
        hs(MM, MM, 0, 0, 0, false, false),
        hs(AA, A2, 0, 0, 0, false, true),
        hs(AA, Q2, 0, 0, 0, false, true),
        hs(AA, AA, 1, 0, 0, false, true),
        hs(MM, ZC, 1, 0, 1, true, true),
        hs(VV, MM, 0, 0, 0, false, true),
        hs(CC, MM, 0, 0, 0, false, true),
        hs(CC, PM, 0, 0, 0, false, true),
    ];
    s[S_CPMR].yk = Some(PMP);
    s
}

/// (k, lo, hi) of sum d at u-order kk, None when it has no terms
#[inline]
fn range(d: &HSum, kk: usize) -> Option<(usize, usize, usize)> {
    let k = kk + d.dk;
    (k >= d.lo + d.hoff).then(|| (k, d.lo, k - d.hoff))
}

/// terms of an edge sum from which its B lane vectors are summed on separate threads
const PAR_EDGE: usize = 16;

/// The middle terms (1 <= l <= j - 1) of every sum at every u-order, from the orders below j in `keep`: entry
/// s (ks + 1) + kk, one open accumulator per lane vector, the terms in the order i, then l. drop1: SPLIT_MUT_DROP_L1.
fn middles<const N: usize, const B: usize>(
    sums: &[HSum; NSUMS],
    keep: &[Vec<F<N, B>>],
    ks: usize,
    j: usize,
    drop1: bool,
) -> Vec<[ODotAcc; B]> {
    let per = ks + 1;
    let mut mids = vec![[ODotAcc::zero(); B]; NSUMS * per];
    let l0 = if drop1 { 2 } else { 1 };
    mids.par_iter_mut().enumerate().for_each(|(t, m)| {
        let (s, kk) = (t / per, t % per);
        let d = &sums[s];
        if d.inner && kk == ks {
            return;
        }
        let Some((k, lo, hi)) = range(d, kk) else { return };
        let (x, y) = (&keep[d.x], &keep[d.y]);
        for (b, acc) in m.iter_mut().enumerate() {
            for i in lo..=hi {
                let xi = &x[i].0[b];
                let yi = match d.yk {
                    Some(o) if k - i == kk => &keep[o][kk].0[b],
                    _ => &y[k - i].0[b],
                };
                if d.w {
                    let wi = i as f64;
                    for l in l0..j {
                        acc.add_prod_mut(&xi.c[l].mul_f(wi), &yi.c[j - l]);
                    }
                } else {
                    for l in l0..j {
                        acc.add_prod_mut(&xi.c[l], &yi.c[j - l]);
                    }
                }
            }
        }
    });
    mids
}

/// what the fast middle terms did, summed over calls (tests' report only): products, block pairs, the largest
/// block count of a factor and of digits, and the largest distance of an exact convolution's value from its integer
#[cfg(test)]
pub(crate) static FAST_STATS: std::sync::Mutex<(usize, usize, usize, usize, f64)> =
    std::sync::Mutex::new((0, 0, 0, 0, 0.0));

/// with FASTMUL_DUMP=DIR (tests only; not from the control's runs), the factors of lane vector 0's products at l = 1,
/// j/2 and j - 1 into
/// DIR/j{j}_s{s}_l{l}.bin: n, the two lengths, then each factor's coefficients as eight lanes of eight parts (f64,
/// little-endian), for hp/fastmul_bench's --dump
#[cfg(test)]
fn dump(xs: &[OV], ys: &[OV], n: usize, j: usize, s: usize, b: usize, l: usize) {
    let Ok(dir) = std::env::var("FASTMUL_DUMP") else { return };
    if b != 0 || (l != 1 && l != j / 2 && l != j - 1) {
        return;
    }
    let mut out: Vec<f64> = vec![n as f64, xs.len() as f64, ys.len() as f64];
    for v in xs.iter().chain(ys) {
        for od in v.lanes() {
            out.extend_from_slice(&od.x);
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/j{j}_s{s}_l{l}.bin"), bytes).expect("FASTMUL_DUMP");
}

/// `middles` by block-scaled exact products (Ctx::series_split = SPLIT_FAST): for each sum and lane vector, the
/// middle terms at every u-order as sum_(l) of the truncated u-products of (w_i x_(i,l), zero for i < lo) and
/// (y_(m,j-l), zero for m < hoff) by `fastmul::mullow`, each into the sum's per-index accumulators in the order l.
/// S_CPMR's terms at i = 0, which read p + m before p_kk's solve (`yk`), are added directly, and a square's products
/// at l and j - l (S_PP, S_MM, S_A2N) are formed once and added twice. Each product errs by
/// 2^-424 of its terms' absolute sum or less, about the octo-double rounding of its terms. The bits depend on no
/// thread count (each sum and lane vector is one task) and on no libm (the convolutions are exact).
fn middles_fast<const N: usize, const B: usize>(
    sums: &[HSum; NSUMS],
    keep: &[Vec<F<N, B>>],
    ks: usize,
    j: usize,
    drop1: bool,
) -> Vec<[ODotAcc; B]> {
    let per = ks + 1;
    let l0 = if drop1 { 2 } else { 1 };
    let cfg = fastmul::Cfg::default();
    let tasks: Vec<(usize, usize)> = (0..NSUMS).flat_map(|s| (0..B).map(move |b| (s, b))).collect();
    let out: Vec<(Vec<ODotAcc>, fastmul::Stats)> = tasks
        .par_iter()
        .map(|&(s, b)| {
            let d = &sums[s];
            let n = per + d.dk;
            let mut acc = vec![ODotAcc::zero(); n];
            let mut st = fastmul::Stats::default();
            let (x, y) = (&keep[d.x], &keep[d.y]);
            let (nx, ny) = (n.min(x.len()), n.min(y.len()));
            // with yk the term i = 0 reads another series (below)
            let ilo = if d.yk.is_some() { d.lo.max(1) } else { d.lo };
            let mut xs = vec![OV::zero(); nx];
            let mut ys = vec![OV::zero(); ny];
            // a square's products at l and j - l are the same: the pair formed once, at l < j - l, and added twice
            let square = d.x == d.y && !d.w && d.yk.is_none() && d.lo == 0 && d.hoff == 0;
            for l in l0..j {
                let e2 = match (square, j - l) {
                    (true, m) if m < l && m >= l0 => continue,
                    (true, m) if m > l => 1,
                    _ => 0,
                };
                for (i, v) in xs.iter_mut().enumerate() {
                    let c = x[i].0[b].c[l];
                    *v = if i < ilo {
                        OV::zero()
                    } else if d.w {
                        c.mul_f(i as f64)
                    } else {
                        c
                    };
                }
                for (m, v) in ys.iter_mut().enumerate() {
                    *v = if m < d.hoff { OV::zero() } else { y[m].0[b].c[j - l] };
                }
                #[cfg(test)]
                if !drop1 {
                    dump(&xs, &ys, n, j, s, b, l);
                }
                st.merge(&fastmul::mullow_pow2(&xs, &ys, n, &cfg, e2, &mut acc));
            }
            if let Some(o) = d.yk {
                debug_assert!(d.dk == 0 && d.lo == 0 && d.hoff == 0 && !d.w, "series_split: yk on S_CPMR's form only");
                for kk in 0..per {
                    for l in l0..j {
                        acc[kk].add_prod_mut(&x[0].0[b].c[l], &keep[o][kk].0[b].c[j - l]);
                    }
                }
            }
            assert!(st.max_frac < 0.25, "series_split: an exact convolution's value {} from its integer", st.max_frac);
            (acc, st)
        })
        .collect();
    let mut mids = vec![[ODotAcc::zero(); B]; NSUMS * per];
    #[cfg(test)]
    let mut tot = fastmul::Stats::default();
    for ((s, b), (acc, _st)) in tasks.iter().zip(out) {
        let d = &sums[*s];
        for kk in 0..per {
            if (d.inner && kk == ks) || range(d, kk).is_none() {
                continue;
            }
            mids[s * per + kk][*b] = acc[kk + d.dk];
        }
        #[cfg(test)]
        tot.merge(&_st);
    }
    #[cfg(test)]
    {
        let mut g = FAST_STATS.lock().unwrap();
        let most = tot.blocks.0.max(tot.blocks.1);
        *g = (g.0 + tot.calls, g.1 + tot.pairs, g.2.max(most), g.3.max(tot.digits), g.4.max(tot.max_frac));
    }
    mids
}

/// The sums' edge terms in the u-order loop: coefficient j of sum `s` at u-order kk, its middle accumulators plus
/// x_(i,0) y_(k-i,j) and x_(i,j) y_(k-i,0) for i = lo..=hi. x and y must be the series the sum names (checked by
/// address against `base`, the series' storage), y_kk read from yk where the sum says.
struct Edges<const B: usize> {
    sums: [HSum; NSUMS],
    mids: Vec<[ODotAcc; B]>,
    base: [usize; H_SERIES],
    per: usize,
    j: usize,
}

impl<const B: usize> Edges<B> {
    fn sum<const N: usize>(&self, s: usize, kk: usize, x: &[F<N, B>], y: &[F<N, B>], yk: Option<&F<N, B>>) -> V<B> {
        let d = &self.sums[s];
        assert!(
            x.as_ptr() as usize == self.base[d.x] && y.as_ptr() as usize == self.base[d.y] && yk.is_some() == d.yk.is_some(),
            "series_split: sum {s} given other series than its own"
        );
        let mid = &self.mids[s * self.per + kk];
        let j = self.j;
        let Some((k, lo, hi)) = range(d, kk) else {
            return Fl(std::array::from_fn(|b| mid[b].finish()));
        };
        let lane = |b: usize| -> OV {
            let mut acc = mid[b];
            for i in lo..=hi {
                let xi = &x[i].0[b];
                let yi = match yk {
                    Some(o) if k - i == kk => &o.0[b],
                    _ => &y[k - i].0[b],
                };
                if d.w {
                    let wi = i as f64;
                    acc.add_prod_mut(&xi.c[0].mul_f(wi), &yi.c[j]);
                    acc.add_prod_mut(&xi.c[j].mul_f(wi), &yi.c[0]);
                } else {
                    acc.add_prod_mut(&xi.c[0], &yi.c[j]);
                    acc.add_prod_mut(&xi.c[j], &yi.c[0]);
                }
            }
            acc.finish()
        };
        let mut out = V::<B>::zero();
        if hi + 1 - lo >= PAR_EDGE {
            out.0.par_iter_mut().enumerate().for_each(|(b, o)| *o = lane(b));
        } else {
            for (b, o) in out.0.iter_mut().enumerate() {
                *o = lane(b);
            }
        }
        out
    }
}

/// coefficient j of every lane vector of f
#[inline]
fn cj<const N: usize, const B: usize>(f: &F<N, B>, j: usize) -> V<B> {
    Fl(std::array::from_fn(|b| f.0[b].c[j]))
}

/// coefficient j of every lane vector of f set to v
#[inline]
fn setj<const N: usize, const B: usize>(f: &mut F<N, B>, j: usize, v: &V<B>) {
    for b in 0..B {
        f.0[b].c[j] = v.0[b];
    }
}

/// `horizon_series_at` with the middle terms formed ahead (module doc) when T is an octo-double jet of one of the
/// session's lane counts (8, 12, ..., 32); false for any other lane, which the caller forms by `horizon_series_at`
// keep is downcast as a whole, so it stays a Vec
#[allow(clippy::ptr_arg)]
pub fn horizon_series_at_split<T: Lane, const B: usize>(
    ctx: &Ctx<B>,
    m0: &Fl<T, B>,
    a0: &Fl<T, B>,
    delta: &T,
    prep: &HPrep,
    keep: &mut Vec<Vec<Fl<T, B>>>,
    j: usize,
) -> bool {
    macro_rules! lanes {
        ($($n:literal),*) => {$(
            if let Some(kp) = (keep as &mut dyn Any).downcast_mut::<Vec<Vec<F<$n, B>>>>() {
                let m0 = (m0 as &dyn Any).downcast_ref::<F<$n, B>>().expect("m0 on the keep's lanes");
                let a0 = (a0 as &dyn Any).downcast_ref::<F<$n, B>>().expect("a0 on the keep's lanes");
                let delta = (delta as &dyn Any).downcast_ref::<OJet<$n>>().expect("delta on the keep's lanes");
                series_od::<$n, B>(ctx, m0, a0, delta, prep, kp, j);
                return true;
            }
        )*};
    }
    lanes!(8, 12, 16, 20, 24, 28, 32);
    false
}

/// the split form on octo-double jets of N coefficients: `horizon_series_at`'s operations in its order, each Cauchy
/// sum as its middle accumulators plus its edge terms (`Edges`), and each operation that it forms on whole jets and
/// copies at j on coefficient j alone. Orders 0 and 1 have no middle terms and are `horizon_series_at`'s, bit for bit.
fn series_od<const N: usize, const B: usize>(
    ctx: &Ctx<B>,
    m0: &F<N, B>,
    a0: &F<N, B>,
    delta: &OJet<N>,
    prep: &HPrep,
    keep: &mut Vec<Vec<F<N, B>>>,
    j: usize,
) {
    if j < 2 {
        horizon_series_at(ctx, m0, a0, delta, prep, keep, j);
        return;
    }
    debug_assert!(prep.rate.is_zero() && ctx.horizon_alpha != crate::model::HALPHA_MUT, "series_split: jet lanes");
    let ks = ctx.ks;
    assert!(keep.len() >= H_SERIES && keep[0].len() == ks + 2, "series_split: order {j} before orders 0 and 1");
    let alpha = ctx.horizon_alpha != 0;
    let sums = sums(alpha);
    #[cfg(test)]
    let t0 = std::time::Instant::now();
    let drop1 = ctx.mutate == SPLIT_MUT_DROP_L1;
    let mids = if ctx.series_split == SPLIT_FAST {
        middles_fast(&sums, keep, ks, j, drop1)
    } else {
        middles(&sums, keep, ks, j, drop1)
    };
    #[cfg(test)]
    let t1 = std::time::Instant::now();
    let ed = Edges::<B> {
        sums,
        mids,
        base: std::array::from_fn(|s| keep[s].as_ptr() as usize),
        per: ks + 1,
        j,
    };
    let z = F::<N, B>::zero();
    let [pp, mm, aa, cc, a2, q2, dp, dm, pm, vv, ww, zc, am1, pmp, ca1, dpp, dmm, cpmr, agu, cpmf, rest, kept] =
        &mut keep[..H_SERIES]
    else {
        unreachable!("twenty-two series")
    };
    let sig = series_scale(ctx.u0().hi);
    let isig = 1.0 / sig;
    let n1 = dim::<OJet<N>, B>(ctx) - <OJet<N> as BaseLane>::P::one();
    let nh = dim::<OJet<N>, B>(ctx).mul_f(0.5);
    let nh_kappa = if ctx.mutate == JET_MUT_KAPPA { <OJet<N> as BaseLane>::P::from_dd(ctx.n.mul_f(0.5)) } else { nh };
    let one = F::<N, B>::splat(DD::ONE);
    <OJet<N> as BaseLane>::div_at(&<OJet<N> as BaseLane>::splat(DD::TWO_PI), delta, j, &mut kept[0].0[0]);
    let omega = kept[0].0[0];
    let a00 = if alpha { *a0 - one } else { *a0 };
    aa[0].copy_at(&a00, j);
    cc[0].copy_at(&one, j);
    mm[0].copy_at(m0, j);
    aa[0].mul_at(&aa[0], j, &mut a2[0]);
    dapply_at(ctx, &mm[0], j, &mut dmm[0]);
    dmm[0].mul_t_at(&omega, j, &mut dm[0]);
    let sone = one.mul_pwr2(sig);
    let wfun = |c: &[F<N, B>], i: usize| -> F<N, B> {
        match i {
            0 => one - c[0],
            1 => -c[1] - sone,
            _ => -c[i],
        }
    };
    let zfun = |c: &[F<N, B>], i: usize| -> F<N, B> {
        match i {
            0 => c[0] + one,
            1 => c[1] - sone,
            _ => c[i],
        }
    };
    ww[0].copy_at(&wfun(cc, 0), j);
    zc[0].copy_at(&zfun(cc, 0), j);
    one.div_at(&zc[0], j, &mut kept[1]);
    // coefficient j alone of every expression that horizon_series_at forms on whole jets and copies at j: the
    // same operations on that coefficient (OV's Lane arithmetic is OJet's coefficient by coefficient)
    let c = |f: &F<N, B>| cj(f, j);
    let (onej, sonej) = (c(&one), c(&sone));
    let wj = |cs: &[F<N, B>], i: usize| -> V<B> {
        match i {
            0 => onej - c(&cs[0]),
            1 => -c(&cs[1]) - sonej,
            _ => -c(&cs[i]),
        }
    };
    let zj = |cs: &[F<N, B>], i: usize| -> V<B> {
        match i {
            0 => c(&cs[0]) + onej,
            1 => c(&cs[1]) - sonej,
            _ => c(&cs[i]),
        }
    };
    let (mut t, mut u, mut rhs) = (z, z, z);
    for kk in 0..=ks {
        let kf = kk as f64;
        if alpha {
            setj(&mut am1[kk], j, &(c(&aa[kk]).mulf(2.0) + c(&a2[kk])));
            setj(&mut ca1[kk], j, &ed.sum(S_CA, kk, cc, am1, None));
        } else {
            let s = ed.sum(S_CA, kk, cc, a2, None);
            setj(&mut ca1[kk], j, &(s - c(&cc[kk])));
        }
        ca1[kk].mulp_at(&n1, j, &mut t);
        let c_next = mul_recip(ctx, c(&cc[kk]).mulf(kf) - c(&t), kk + 1).mul_pwr2(sig);
        setj(&mut cc[kk + 1], j, &c_next);
        setj(&mut ww[kk + 1], j, &(wj(cc, kk + 1) - wj(cc, kk).mul_pwr2(sig)));
        setj(&mut vv[kk], j, &(c(&cc[kk + 1]).mulf((kk + 1) as f64).mul_pwr2(isig) - c(&cc[kk]).mulf(kf)));
        setj(&mut zc[kk + 1], j, &(zj(cc, kk + 1) - zj(cc, kk).mul_pwr2(sig)));
        // p_k is zero where horizon_series reads it before its solve (the sums' ranges leave it out)
        let pk = std::mem::replace(&mut pp[kk], z);
        setj(&mut pmp[kk], j, &(c(&pp[kk]) + c(&mm[kk])));
        let wpu = ed.sum(S_WPU, kk, pp, ww, None).mul_pwr2(isig);
        let tau = if kk >= 1 { -c(&dp[kk - 1]).mul_pwr2(sig) } else { c(&z) };
        let s1 = ed.sum(S_S1P, kk, vv, pp, None);
        let s2 = ed.sum(S_S2P, kk, cc, pp, None);
        setj(&mut cpmr[kk], j, &ed.sum(S_CPMR, kk, cc, pm, Some(&pmp[kk])));
        cpmr[kk].mulp_at(&nh, j, &mut t);
        let r = (((tau - wpu) + s1) + s2) - c(&t);
        pp[kk] = pk;
        let c1 = cc[1].mul_pwr2(isig);
        let kappa = ((c1 + one).mulf(kf) + c1 + one).addp(-nh_kappa);
        // psolve_at reads the right-hand side's coefficient j alone
        setj(&mut rhs, j, &(-r));
        psolve_at(ctx, prep, kk, &kappa, &rhs, &omega, &mut pp[kk], j);
        dapply_at(ctx, &pp[kk], j, &mut dpp[kk]);
        dpp[kk].mul_t_at(&omega, j, &mut dp[kk]);
        let (tq, uq) = (ed.sum(S_PP, kk, pp, pp, None), ed.sum(S_MM, kk, mm, mm, None));
        setj(&mut q2[kk], j, &(tq + uq));
        if kk == ks {
            break;
        }
        let ag = if alpha {
            let ta = ed.sum(S_AA2, kk, aa, a2, None);
            setj(&mut agu[kk], j, &(-((c(&aa[kk]).mulf(2.0) + c(&a2[kk]).mulf(3.0)) + ta)));
            let ua = ed.sum(S_AQ2, kk, aa, q2, None);
            agu[kk].mulp_at(&n1, j, &mut t);
            (c(&q2[kk]) + ua) + c(&t)
        } else {
            let ua = ed.sum(S_AQ2, kk, aa, q2, None);
            let ta = ed.sum(S_AA2, kk, aa, a2, None);
            setj(&mut agu[kk], j, &(c(&aa[kk]) - ta));
            agu[kk].mulp_at(&n1, j, &mut t);
            ua + c(&t)
        };
        let a_next = mul_recip(ctx, c(&aa[kk]).mulf(kf) - ag.mulf(0.5), kk + 1).mul_pwr2(sig);
        setj(&mut aa[kk + 1], j, &a_next);
        setj(&mut a2[kk + 1], j, &ed.sum(S_A2N, kk, aa, aa, None));
        let tau = if kk >= 1 { c(&dm[kk]) - c(&dm[kk - 1]).mul_pwr2(sig) } else { c(&dm[kk]) };
        setj(&mut pm[kk], j, &(c(&pp[kk]) + c(&mm[kk])));
        let zmu = ed.sum(S_ZMU, kk, mm, zc, None).mul_pwr2(isig);
        let s1 = ed.sum(S_S1M, kk, vv, mm, None);
        let s2 = ed.sum(S_S2M, kk, cc, mm, None);
        setj(&mut cpmf[kk], j, &ed.sum(S_CPMF, kk, cc, pm, None));
        cpmf[kk].mulp_at(&nh, j, &mut t);
        setj(&mut rest[kk], j, &((((tau - zmu) - s1) - s2) + c(&t)));
        rest[kk].mul_at(&kept[1], j, &mut u);
        setj(&mut mm[kk + 1], j, &mul_recip(ctx, c(&u), kk + 1).mul_pwr2(sig));
        dapply_at(ctx, &mm[kk + 1], j, &mut dmm[kk + 1]);
        dmm[kk + 1].mul_t_at(&omega, j, &mut dm[kk + 1]);
    }
    #[cfg(test)]
    {
        use std::sync::atomic::Ordering::Relaxed;
        T_MID.fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
        T_LOOP.fetch_add(t1.elapsed().as_nanos() as u64, Relaxed);
    }
}
