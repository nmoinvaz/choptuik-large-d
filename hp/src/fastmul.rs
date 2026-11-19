//! Truncated products of octo-double series, c_k = sum_(i=0..=k) a_i b_(k-i) for k < n on the 8 lanes of `OV`, by
//! block scaling and exact integer convolution (`mullow`), with the naive product `mullow_naive` as its reference.
//!
//! Scaling. a_i and b_i are multiplied by 2^(s i) on each lane, s an integer slope from the series' leading exponents
//! (least squares), which multiplies c_k by 2^(s k) exactly.
//!
//! Blocks. Each scaled series is cut into runs of indices whose leading exponents lie within h bits on every lane. In
//! a block of largest exponent F (per lane) the coefficient becomes the integer X_i = a_i 2^(s i + W - F) with each
//! part rounded, W = 16 t - 2 bits for t digits, so it keeps W - h - 5 bits or more. Every pair of blocks whose
//! product reaches below n is multiplied exactly, and the pair's terms then err by 2^-prec of its smallest term or
//! less. The slope (from both series, from a or from b) and h (`Cfg::spread` times 1, 2 or 4) are those of least
//! estimated cost (`cost`): flat series are one block each.
//!
//! The exact product. Each integer is t balanced 16-bit digits. The digit sequences are transformed by f64 FFTs (two
//! real sequences per complex transform), the digit products summed by weight r + r' in the frequency domain, and the
//! weights transformed back. Every value is then an integer below 2^44, whose rounding error is checked to be under
//! 1/4 (`Stats::max_frac`), so the convolution is exact and its bits depend on no libm. Weights too small to reach
//! 2^-prec of the smallest term are not formed. The integer at each index is carried into balanced digits, cut into
//! eight 53-bit parts from its top bit (`parts_of`, 2^-424 of it or less dropped) and added into the index's
//! octo-double accumulator (`ODotAcc`).
//!
//! J. van der Hoeven, Making fast multiplication of polynomials numerically stable, preprint, Univ. Paris-Sud (2008);
//! the blocks are those of F. Johansson's Arb (`arb_poly_mullow_block`), IEEE Trans. Comput. 66, 1281-1292 (2017).
use multidouble::native::{ODotAcc, OV, V8};
use std::sync::OnceLock;

/// PROF: nanoseconds in load (digits, FFT), pointwise, inverse, extraction
pub static PROF: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
fn prof(i: usize, t0: std::time::Instant) -> std::time::Instant {
    let t1 = std::time::Instant::now();
    PROF[i].fetch_add((t1 - t0).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    t1
}

/// lanes of an `OV`
const LN: usize = 8;
/// bits of a digit
const DB: i32 = 16;
const R16: f64 = 65536.0;
const IR16: f64 = 1.0 / 65536.0;
/// largest FFT length (a block pair of up to 2^LMAX_LOG2 - 1 terms)
const LMAX_LOG2: usize = 13;

/// lane-wise f64 values
type L8 = [f64; LN];

/// The product's parameters.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    /// bits kept below the smallest term of a block pair (the product's accuracy, relative to sum |a_i| |b_(k-i)|)
    pub prec: i32,
    /// the largest spread in bits of the scaled leading exponents within a block
    pub spread: i32,
    /// the slope 2^(s i) on; off, s = 0
    pub scale: bool,
    /// blocks on; off, each series is one block and its digits are those of `spread` (the control: the blocks' and
    /// the slope's guarantee removed)
    pub blocks: bool,
    /// the controls of fastmul_check's edge cases, 0 off: MUT_ZERO_LANE scales zero lanes as the others,
    /// MUT_E2_PLAIN scales the long-pair route's first factor by 2^e2 as one f64
    pub mutate: u8,
}

/// `Cfg::mutate`: zero lanes scaled like the others (their scale can lie outside f64's range)
pub const MUT_ZERO_LANE: u8 = 1;
/// `Cfg::mutate`: the long-pair route's 2^e2 formed as one f64 on the first factor (it can overflow)
pub const MUT_E2_PLAIN: u8 = 2;

impl Default for Cfg {
    fn default() -> Self {
        Cfg { prec: 424, spread: 32, scale: true, blocks: true, mutate: 0 }
    }
}

/// What one product did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// products formed
    pub calls: usize,
    /// block pairs multiplied
    pub pairs: usize,
    /// blocks of a and of b
    pub blocks: (usize, usize),
    /// the largest distance of an inverse transform's value from its integer (exact while below 1/2); infinite when a
    /// factor is not finite
    pub max_frac: f64,
    /// digits per coefficient (largest over the pairs)
    pub digits: usize,
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.calls += o.calls;
        self.pairs += o.pairs;
        self.blocks = (self.blocks.0.max(o.blocks.0), self.blocks.1.max(o.blocks.1));
        self.max_frac = self.max_frac.max(o.max_frac);
        self.digits = self.digits.max(o.digits);
    }
}

/// c_k = sum_(i=0..=k) a_i b_(k-i), k < n, each sum in an `ODotAcc` (the reference); an empty factor gives zeros
pub fn mullow_naive(a: &[OV], b: &[OV], n: usize) -> Vec<OV> {
    (0..n)
        .map(|k| {
            let mut acc = ODotAcc::zero();
            for i in 0..a.len().min(k + 1) {
                if k - i < b.len() {
                    acc.add_prod_mut(&a[i], &b[k - i]);
                }
            }
            acc.finish()
        })
        .collect()
}

/// c_k for k < n added into acc[k] (acc of n entries or more) by the block-scaled exact product (module doc)
pub fn mullow(a: &[OV], b: &[OV], n: usize, cfg: &Cfg, acc: &mut [ODotAcc]) -> Stats {
    mullow_pow2(a, b, n, cfg, 0, acc)
}

/// `mullow` with c_k 2^e2 added (exactly: e2 enters the parts' exponents)
pub fn mullow_pow2(a: &[OV], b: &[OV], n: usize, cfg: &Cfg, e2: i32, acc: &mut [ODotAcc]) -> Stats {
    let (na, nb) = (a.len().min(n), b.len().min(n));
    let mut st = Stats { calls: 1, ..Stats::default() };
    if na == 0 || nb == 0 || n == 0 {
        return st;
    }
    // a factor that is not finite is reported (max_frac infinite, which every check fails) and acc left as it was
    let finite = |v: &[OV]| v.iter().all(|x| x.c.iter().all(|p| p.to_array().iter().all(|q| q.is_finite())));
    if !finite(&a[..na]) || !finite(&b[..nb]) {
        st.max_frac = f64::INFINITY;
        return st;
    }
    let (ea, eb) = (exponents(&a[..na]), exponents(&b[..nb]));
    // the slope and the blocks' spread limit of least estimated cost: the slope from both series, from a alone and
    // from b alone, and the limit cfg.spread, twice and four times it (one limit and no slope without the blocks)
    let (fa, fb): ([f64; LN], [f64; LN]) = (std::array::from_fn(|l| fit(&ea, l)), std::array::from_fn(|l| fit(&eb, l)));
    let slopes: Vec<[i32; LN]> = if cfg.scale {
        vec![
            std::array::from_fn(|l| (-(fa[l] + fb[l]) * 0.5).round() as i32),
            std::array::from_fn(|l| (-fa[l]).round() as i32),
            std::array::from_fn(|l| (-fb[l]).round() as i32),
        ]
    } else {
        vec![[0; LN]]
    };
    let limits: &[i32] = if cfg.blocks { &[1, 2, 4] } else { &[1] };
    let mut best: Option<(f64, [i32; LN], Vec<Block>, Vec<Block>)> = None;
    for sl in &slopes {
        for &m in limits {
            let h = cfg.spread * m;
            let (ba, bb) = (blocks(&ea, sl, h, cfg), blocks(&eb, sl, h, cfg));
            let c = cost(&ba, &bb, n, cfg);
            if best.as_ref().is_none_or(|b| c < b.0) {
                best = Some((c, *sl, ba, bb));
            }
        }
    }
    let (_, s, ba, bb) = best.expect("a slope and a limit");
    let (ba, bb) = if cfg.blocks { shorten(ba, bb, &ea, &eb, &s, n, cfg) } else { (ba, bb) };
    st.blocks = (ba.len(), bb.len());
    // the scratch of this thread, kept between calls (its slabs are megabytes)
    thread_local! {
        static WS: std::cell::RefCell<Work> = std::cell::RefCell::new(Work::default());
    }
    WS.with(|w| {
        let ws = &mut *w.borrow_mut();
        mullow_pairs(a, b, n, cfg, e2, acc, &ba, &bb, &s, &mut st, ws);
    });
    st
}

/// every block pair of `mullow` reaching below n
#[allow(clippy::too_many_arguments)]
fn mullow_pairs(
    a: &[OV],
    b: &[OV],
    n: usize,
    cfg: &Cfg,
    e2: i32,
    acc: &mut [ODotAcc],
    ba: &[Block],
    bb: &[Block],
    s: &[i32; LN],
    st: &mut Stats,
    ws: &mut Work,
) {
    for xa in ba {
        for xb in bb {
            let o = xa.start + xb.start;
            if o >= n {
                continue;
            }
            st.pairs += 1;
            let (x, y) = (&a[xa.start..xa.end], &b[xb.start..xb.end]);
            if x.len().min(n - o) + y.len().min(n - o) - 1 > 1 << LMAX_LOG2 {
                // past the FFT tables: this pair by ODotAcc, 2^e2 split per lane between the factors so that both
                // and their products keep the range of the result (each factor scaled exactly, in steps)
                let (xs, ys) = if cfg.mutate == MUT_E2_PLAIN {
                    (x.iter().map(|v| v.mul_pwr2(2f64.powi(e2))).collect::<Vec<_>>(), y.to_vec())
                } else {
                    let top = |v: &[OV]| -> [i32; LN] {
                        std::array::from_fn(|l| {
                            v.iter().map(|c| c.c[0].to_array()[l]).filter(|&p| p != 0.0).map(expo).max().unwrap_or(0)
                        })
                    };
                    let (tx, ty) = (top(x), top(y));
                    let al: [i32; LN] = std::array::from_fn(|l| (ty[l] - tx[l] + e2).div_euclid(2));
                    let be: [i32; LN] = std::array::from_fn(|l| e2 - al[l]);
                    (x.iter().map(|v| scale_lanes(v, &al)).collect(), y.iter().map(|v| scale_lanes(v, &be)).collect())
                };
                for k in 0..n - o {
                    for i in 0..xs.len().min(k + 1) {
                        if k - i < ys.len() {
                            acc[o + k].add_prod_mut(&xs[i], &ys[k - i]);
                        }
                    }
                }
                continue;
            }
            pair(x, y, xa, xb, s, n - o, o, cfg, e2, acc, st, ws);
        }
    }
}

/// the leading exponent e (|x| in [2^(e-1), 2^e)) of every coefficient's leading part per lane, i32::MIN for zero
fn exponents(a: &[OV]) -> Vec<[i32; LN]> {
    a.iter().map(|v| v.c[0].to_array().map(|x| if x == 0.0 { i32::MIN } else { expo(x) })).collect()
}

#[inline]
fn expo(x: f64) -> i32 {
    let b = x.to_bits();
    let ex = ((b >> 52) & 0x7ff) as i32;
    if ex == 0 {
        -1010 - (b & ((1u64 << 52) - 1)).leading_zeros() as i32
    } else {
        ex - 1022
    }
}

/// the least-squares slope of the nonzero exponents of a series on lane l (0 with fewer than two)
fn fit(e: &[[i32; LN]], l: usize) -> f64 {
    let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, v) in e.iter().enumerate() {
        if v[l] != i32::MIN {
            let (x, y) = (i as f64, v[l] as f64);
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    let d = n * sxx - sx * sx;
    if n < 2.0 || d <= 0.0 {
        0.0
    } else {
        (n * sxy - sx * sy) / d
    }
}

/// a block: indices start..end, its largest scaled exponent per lane, and its spread (largest over the lanes)
#[derive(Clone)]
struct Block {
    start: usize,
    end: usize,
    frame: [i32; LN],
    spread: i32,
}

/// the blocks of a series (module doc), each of spread h or less; one block when cfg.blocks is off, taken to have the
/// spread cfg.spread
fn blocks(e: &[[i32; LN]], s: &[i32; LN], h: i32, cfg: &Cfg) -> Vec<Block> {
    let mut out = Vec::new();
    let mut start = 0;
    let (mut lo, mut hi) = ([i32::MAX; LN], [i32::MIN; LN]);
    let close = |start: usize, end: usize, lo: &[i32; LN], hi: &[i32; LN], out: &mut Vec<Block>| {
        let spread = (0..LN).filter(|&l| hi[l] != i32::MIN).map(|l| hi[l] - lo[l]).max().unwrap_or(0);
        let frame = std::array::from_fn(|l| if hi[l] == i32::MIN { 0 } else { hi[l] });
        out.push(Block { start, end, frame, spread: if cfg.blocks { spread } else { cfg.spread } });
    };
    for (i, v) in e.iter().enumerate() {
        let mut split = false;
        let mut nlo = lo;
        let mut nhi = hi;
        for l in 0..LN {
            if v[l] != i32::MIN {
                let x = v[l] + s[l] * i as i32;
                nlo[l] = nlo[l].min(x);
                nhi[l] = nhi[l].max(x);
                split |= nhi[l] - nlo[l] > h;
            }
        }
        if split && cfg.blocks && i > start {
            close(start, i, &lo, &hi, &mut out);
            start = i;
            for l in 0..LN {
                let x = v[l].saturating_add(s[l] * i as i32);
                (lo[l], hi[l]) = if v[l] == i32::MIN { (i32::MAX, i32::MIN) } else { (x, x) };
            }
        } else {
            (lo, hi) = (nlo, nhi);
        }
    }
    close(start, e.len(), &lo, &hi, &mut out);
    out
}

/// the block of indices start..end of a series (its frame and spread, as `blocks` closes one)
fn block_of(e: &[[i32; LN]], s: &[i32; LN], start: usize, end: usize, cfg: &Cfg) -> Block {
    let (mut lo, mut hi) = ([i32::MAX; LN], [i32::MIN; LN]);
    for (i, v) in e.iter().enumerate().take(end).skip(start) {
        for l in 0..LN {
            if v[l] != i32::MIN {
                let x = v[l] + s[l] * i as i32;
                lo[l] = lo[l].min(x);
                hi[l] = hi[l].max(x);
            }
        }
    }
    let spread = (0..LN).filter(|&l| hi[l] != i32::MIN).map(|l| hi[l] - lo[l]).max().unwrap_or(0);
    let frame = std::array::from_fn(|l| if hi[l] == i32::MIN { 0 } else { hi[l] });
    Block { start, end, frame, spread: if cfg.blocks { spread } else { cfg.spread } }
}

/// The blocks with a pair's terms that just exceed a power of two cut down: each block of one series whose pair with
/// the other's first block has P > L/2 terms (L = P rounded up to a power of two) while its outputs fit in L/2 is cut
/// where that pair's terms become L/2, its last few coefficients a block of their own (a small pair). Kept for each
/// series when the estimated cost falls (an FFT length halved, n = 131 at ks 130 for one).
fn shorten(
    ba: Vec<Block>,
    bb: Vec<Block>,
    ea: &[[i32; LN]],
    eb: &[[i32; LN]],
    s: &[i32; LN],
    n: usize,
    cfg: &Cfg,
) -> (Vec<Block>, Vec<Block>) {
    let (mut ba, mut bb) = (ba, bb);
    for side in 0..2 {
        let (xs, ys, ex) = if side == 0 { (&ba, &bb, ea) } else { (&bb, &ba, eb) };
        let y0 = &ys[0];
        let mut cut: Vec<Block> = Vec::with_capacity(xs.len() + 1);
        for x in xs {
            let o = x.start + y0.start;
            let fits = o < n && {
                let (la, lb) = ((x.end - x.start).min(n - o), (y0.end - y0.start).min(n - o));
                let p = la + lb - 1;
                let l = p.next_power_of_two();
                let at = x.start + (l / 2 + 1).saturating_sub(lb);
                if p > l / 2 && n - o <= l / 2 && at > x.start && at < x.start + la {
                    cut.push(block_of(ex, s, x.start, at, cfg));
                    cut.push(block_of(ex, s, at, x.end, cfg));
                    true
                } else {
                    false
                }
            };
            if !fits {
                cut.push(x.clone());
            }
        }
        if cut.len() > xs.len() {
            let c0 = cost(&ba, &bb, n, cfg);
            let c1 = if side == 0 { cost(&cut, &bb, n, cfg) } else { cost(&ba, &cut, n, cfg) };
            if c1 < c0 {
                if side == 0 {
                    ba = cut;
                } else {
                    bb = cut;
                }
            }
        }
    }
    (ba, bb)
}

/// A block pair's digits: t per coefficient (even), W = 16 t - 2 >= prec + spread + 5 bits, which keeps every
/// coefficient to 2^-(prec + 1) of itself, and the weights q0..=2t-2 formed (nq of them, even): those below sum to
/// under 2^-prec of the smallest term, 2^(2W - h_a - h_b - 2).
struct Plan {
    t: usize,
    w: i32,
    q0: usize,
    nq: usize,
}

fn plan(cfg: &Cfg, ha: i32, hb: i32, la: usize, lb: usize) -> Plan {
    let t = ((cfg.prec + ha.max(hb) + 7 + DB - 1) / DB) as usize;
    let t = t + t % 2;
    let w = DB * t as i32 - 2;
    let lg = ((t * la.min(lb)) as f64).log2().ceil() as i32;
    let q0 = (((2 * w - ha - hb - 2 - cfg.prec - 14 - lg) / DB).max(0) as usize).min(2 * t - 2);
    let nq = (2 * t - 1 - q0).div_ceil(2) * 2;
    Plan { t, w, q0, nq }
}

/// The estimated seconds of the block pairs reaching below n, per pair: 0.76 us per coefficient's digits, 6.5 ns
/// per butterfly of a transform's slab (t forward slabs, nq / 2 inverse), 6.2 ns per digit product at each of the
/// L/2 + 1 frequencies, and 1.1 us per output index (the parts of `pair`, measured on an M5 at n = 251).
fn cost(ba: &[Block], bb: &[Block], n: usize, cfg: &Cfg) -> f64 {
    let mut c = 0.0;
    for xa in ba {
        for xb in bb {
            let o = xa.start + xb.start;
            if o >= n {
                continue;
            }
            let (la, lb) = ((xa.end - xa.start).min(n - o), (xb.end - xb.start).min(n - o));
            let lc = (la + lb - 1).min(n - o);
            let l = (la + lb - 1).next_power_of_two();
            let pl = plan(cfg, xa.spread, xb.spread, la, lb);
            let prods: usize = (pl.q0..=2 * pl.t - 2).map(|q| q.min(pl.t - 1) - q.saturating_sub(pl.t - 1) + 1).sum();
            let bfly = (l / 2) as f64 * (l as f64).log2().max(1.0);
            c += 0.76e-6 * (la + lb) as f64
                + 6.5e-9 * (pl.t + pl.nq / 2) as f64 * bfly
                + 6.2e-9 * (l / 2 + 1) as f64 * prods as f64
                + 1.1e-6 * lc as f64;
        }
    }
    c
}

/// 2^e for e in [-1022, 1023]
#[inline(always)]
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// x 2^n with one rounding (musl's scalbn): the factor applied in steps that keep the intermediate normal, so a
/// result past f64's range is infinite and one below it rounds to a subnormal or zero once
fn scalbn(mut x: f64, mut n: i32) -> f64 {
    const SUB: i32 = 1022 - 53;
    if n > 1023 {
        x *= pow2(1023);
        n -= 1023;
        if n > 1023 {
            x *= pow2(1023);
            n = (n - 1023).min(1023);
        }
    } else if n < -1022 {
        x *= pow2(-SUB);
        n += SUB;
        if n < -1022 {
            x *= pow2(-SUB);
            n = (n + SUB).max(-1022);
        }
    }
    x * pow2(n)
}

/// lane l of every part of v times 2^e\[l\] (`scalbn`)
fn scale_lanes(v: &OV, e: &[i32; LN]) -> OV {
    OV { c: v.c.map(|p| V8::from_array(std::array::from_fn(|l| scalbn(p.to_array()[l], e[l])))) }
}

/// per-lane factors f1, f2, each a normal power of two, with f1 f2 = 2^e (|e| <= 2044)
#[inline(always)]
fn pow2_split(e: &[i32; LN]) -> (L8, L8) {
    (std::array::from_fn(|l| pow2(e[l] / 2)), std::array::from_fn(|l| pow2(e[l] - e[l] / 2)))
}

/// 1.5 2^52: (x + MAGIC) - MAGIC is x rounded to an integer (to nearest, ties to even) for |x| < 2^51
const MAGIC: f64 = 6755399441055744.0;

/// x rounded to an integer on every lane (|x| < 2^51)
#[inline(always)]
fn rnd(x: V8) -> V8 {
    let m = V8::splat(MAGIC);
    x.add(m).sub(m)
}

/// The digits of a coefficient's 8 lanes: lane l of x times 2^k\[l\], each part rounded to an integer (so the
/// lane is within 4 of x 2^k), as t balanced 16-bit digits d\[r\] (|d| <= 2^15). Each part p is split exactly into
/// digits from its largest lane's top digit down to its smallest lane's lowest one, each digit the nearest integer to
/// the remainder over 2^(16 r) and the remainder then reduced by it (exact, since it is the remainder's low bits); the
/// digit sums are then carried into balanced form. |x 2^k| must be below 2^(16 t - 1).
pub(crate) fn digits8(x: &OV, k: &[i32; LN], t: usize, d: &mut [V8]) {
    let (f1, f2) = pow2_split(k);
    let (f1, f2) = (V8::from_array(f1), V8::from_array(f2));
    for dr in d.iter_mut().take(t) {
        *dr = V8::zero();
    }
    let (one, inf) = (V8::splat(1.0), V8::splat(f64::INFINITY));
    for part in &x.c {
        let mut rem = part.mul(f1).mul(f2);
        let a = rem.abs();
        let amax = a.hmax();
        if amax < 0.5 {
            continue;
        }
        let amin = -a.ge(one).sel(a, inf).neg().hmax();
        let rhi = (((expo(amax) - 1).max(0) / DB) as usize).min(t - 1);
        let rlo = if amin.is_finite() { ((expo(amin) - 54).max(0) / DB) as usize } else { 0 };
        for r in (rlo.min(rhi)..=rhi).rev() {
            let q = rnd(rem.mul(V8::splat(pow2(-DB * r as i32))));
            d[r] = d[r].add(q);
            rem = rem.sub(q.mul(V8::splat(pow2(DB * r as i32))));
        }
    }
    let (ir, r16) = (V8::splat(IR16), V8::splat(R16));
    let mut carry = V8::zero();
    for dr in d.iter_mut().take(t) {
        let v = dr.add(carry);
        let hi = rnd(v.mul(ir));
        *dr = v.sub(hi.mul(r16));
        carry = hi;
    }
    debug_assert!(carry.to_array().iter().all(|&c| c == 0.0), "digits8: |x 2^k| too large");
}

/// The integer v = sum_q d\[q\]\[ln\] 2^(16 q) (balanced digits of lane ln) times 2^e as eight parts: |v| cut into
/// chunks of 53 bits from its top bit down, each part a chunk with v's sign, so the parts do not overlap and their sum
/// is v 2^e truncated below the eighth chunk (2^-424 of v or less).
fn parts_of(d: &[L8], ln: usize, e: i32) -> [f64; 8] {
    const NL: usize = 18;
    let nl = d.len().div_ceil(4) + 1;
    assert!(nl <= NL, "parts_of: {} digits", d.len());
    let mut w = [0u64; NL];
    let mut carry: i128 = 0;
    for (li, wl) in w.iter_mut().enumerate().take(nl) {
        let mut v = carry;
        for j in 0..4 {
            if let Some(dq) = d.get(4 * li + j) {
                v += (dq[ln] as i64 as i128) << (16 * j);
            }
        }
        *wl = v as u64;
        carry = v >> 64;
    }
    let neg = carry < 0;
    if neg {
        let mut c = true;
        for wl in w.iter_mut().take(nl) {
            let (r, cc) = (!*wl).overflowing_add(c as u64);
            *wl = r;
            c = cc;
        }
    }
    let Some(top) = (0..nl).rev().find(|&i| w[i] != 0) else { return [0.0; 8] };
    let hb = 64 * top as i64 + 63 - w[top].leading_zeros() as i64;
    let bit = |b: i64| -> u64 {
        // bits b..b + 53 of |v| (zero below bit 0)
        if b + 53 <= 0 {
            return 0;
        }
        let (bb, sh) = if b < 0 { (0, -b) } else { (b, 0) };
        let (li, bo) = ((bb / 64) as usize, (bb % 64) as u32);
        let lo = w[li] >> bo;
        let hi = if bo > 0 && li + 1 < nl { w[li + 1] << (64 - bo) } else { 0 };
        (((lo | hi) & ((1u64 << 53) - 1)) << sh) & ((1u64 << 53) - 1)
    };
    let sg = if neg { -1.0 } else { 1.0 };
    std::array::from_fn(|j| {
        let b = hb - 52 - 53 * j as i64;
        let c = bit(b);
        if c == 0 {
            0.0
        } else {
            let ex = (b + e as i64).clamp(-4000, 4000) as i32;
            scalbn(sg * c as f64, ex)
        }
    })
}

/// FFT tables of length 2^m: twiddles exp(-2 pi i k / L), k < L/2, and the bit reversal
struct Tables {
    tw: Vec<(f64, f64)>,
    rev: Vec<usize>,
}

fn tables(m: usize) -> &'static Tables {
    static T: OnceLock<Vec<Tables>> = OnceLock::new();
    &T.get_or_init(|| {
        (0..=LMAX_LOG2)
            .map(|m| {
                let l = 1usize << m;
                let tw = (0..l / 2)
                    .map(|k| {
                        let a = -2.0 * std::f64::consts::PI * k as f64 / l as f64;
                        (a.cos(), a.sin())
                    })
                    .collect();
                let rev =
                    (0..l).map(|i| if m == 0 { 0 } else { i.reverse_bits() >> (usize::BITS as usize - m) }).collect();
                Tables { tw, rev }
            })
            .collect()
    })[m]
}

/// a row of a slab: 8 lanes' real parts, then their imaginary parts
type C8 = [V8; 2];

#[inline(always)]
fn c0() -> C8 {
    [V8::zero(), V8::zero()]
}

/// in-place forward DIF FFT of one slab of L rows (rows nz.. zero on input); output in bit-reversed row order
fn fft_dif(x: &mut [C8], tb: &Tables, nz: usize) {
    let l = x.len();
    let mut h = l / 2;
    let mut first = true;
    while h >= 1 {
        let step = l / (2 * h);
        for g in (0..l).step_by(2 * h) {
            let (lo, hi) = x[g..g + 2 * h].split_at_mut(h);
            for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let (wr, wi) = tb.tw[k * step];
                let (wr, wi) = (V8::splat(wr), V8::splat(wi));
                if first && g + k + h >= nz {
                    if g + k >= nz {
                        continue;
                    }
                    let (ur, ui) = (a[0], a[1]);
                    *b = [ur.fma(wr, ui.mul(wi).neg()), ur.fma(wi, ui.mul(wr))];
                } else {
                    let (ur, ui, vr, vi) = (a[0], a[1], b[0], b[1]);
                    *a = [ur.add(vr), ui.add(vi)];
                    let (dr, di) = (ur.sub(vr), ui.sub(vi));
                    *b = [dr.fma(wr, di.mul(wi).neg()), dr.fma(wi, di.mul(wr))];
                }
            }
        }
        first = false;
        h /= 2;
    }
}

/// in-place inverse DIT FFT (unnormalized, conjugate twiddles) of one slab, bit-reversed rows to natural rows
fn ifft_dit(x: &mut [C8], tb: &Tables) {
    let l = x.len();
    let mut h = 1;
    while h < l {
        let step = l / (2 * h);
        for g in (0..l).step_by(2 * h) {
            let (lo, hi) = x[g..g + 2 * h].split_at_mut(h);
            for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let (wr, wi) = tb.tw[k * step];
                let (wr, wi) = (V8::splat(wr), V8::splat(wi));
                let (vr, vi) = (b[0], b[1]);
                let tr = vr.fma(wr, vi.mul(wi));
                let ti = vi.fma(wr, vr.mul(wi).neg());
                let (ur, ui) = (a[0], a[1]);
                *a = [ur.add(tr), ui.add(ti)];
                *b = [ur.sub(tr), ui.sub(ti)];
            }
        }
        h *= 2;
    }
}

/// scratch of a block pair
#[derive(Default)]
struct Work {
    xs: Vec<C8>,
    ys: Vec<C8>,
    vs: Vec<C8>,
    dig: Vec<V8>,
    dl: Vec<L8>,
    xh: Vec<[V8; 3]>,
    yh: Vec<[V8; 2]>,
    cq: Vec<V8>,
}

/// a block's first `len` coefficients as t digits in t/2 slabs of L rows (digit 2u in the real parts and 2u + 1 in
/// the imaginary parts of slab u), then each slab's forward FFT
#[allow(clippy::too_many_arguments)]
fn load(
    a: &[OV],
    len: usize,
    start: usize,
    frame: &[i32; LN],
    s: &[i32; LN],
    t: usize,
    l: usize,
    xs: &mut Vec<C8>,
    dig: &mut Vec<V8>,
    tb: &Tables,
    mutate: u8,
) {
    let w = DB * t as i32 - 2;
    xs.clear();
    xs.resize(t / 2 * l, c0());
    dig.resize(t, V8::zero());
    for (i, v) in a.iter().enumerate().take(len) {
        // a zero lane is not scaled: its scale can lie outside f64's range, and 0 times infinity is not zero
        let hi = v.c[0].to_array();
        let k: [i32; LN] =
            std::array::from_fn(|ln| {
                if hi[ln] == 0.0 && mutate != MUT_ZERO_LANE {
                    0
                } else {
                    s[ln] * (start + i) as i32 + w - frame[ln]
                }
            });
        digits8(v, &k, t, dig);
        for u in 0..t / 2 {
            xs[u * l + i] = [dig[2 * u], dig[2 * u + 1]];
        }
    }
    let t0 = std::time::Instant::now();
    for slab in xs.chunks_exact_mut(l) {
        fft_dif(slab, tb, len);
    }
    prof(4, t0);
}

/// the spectra (twice them) of digits 2u and 2u + 1 at frequency f from slab rows f (z) and -f (w): z + conj w, and
/// -i (z - conj w), each as (re, im, -im)
#[inline(always)]
fn unpack3(z: &C8, w: &C8) -> ([V8; 3], [V8; 3]) {
    let (zr, zi, wr, wi) = (z[0], z[1], w[0], w[1]);
    let (ei, oi) = (zi.sub(wi), wr.sub(zr));
    ([zr.add(wr), ei, ei.neg()], [zi.add(wi), oi, oi.neg()])
}

/// `unpack3` without the negated imaginary parts
#[inline(always)]
fn unpack2(z: &C8, w: &C8) -> ([V8; 2], [V8; 2]) {
    let (zr, zi, wr, wi) = (z[0], z[1], w[0], w[1]);
    ([zr.add(wr), zi.sub(wi)], [zi.add(wi), wr.sub(zr)])
}

/// (ar, ai) += x y, x = (re, im, -im), y = (re, im)
#[inline(always)]
fn cmac(ar: &mut V8, ai: &mut V8, x: &[V8; 3], y: &[V8; 2]) {
    *ar = x[0].fma(y[0], x[2].fma(y[1], *ar));
    *ai = x[0].fma(y[1], x[1].fma(y[0], *ai));
}

/// C_q = sum_(r + r' = q) X_r Y_r' for q = q0..=2t-2 at one frequency, x\[r\] holding (re, im, -im) of digit r and
/// y\[r\] (re, im); C_q into c\[2(q - q0)\] (re), c\[2(q - q0) + 1\] (im). Weights in pairs q, q + 1, which share
/// each x_r, and y_(q+1-r) is the y_(q-r') of the step before.
#[inline(always)]
fn conv(x: &[[V8; 3]], y: &[[V8; 2]], t: usize, q0: usize, c: &mut [V8]) {
    let qe = 2 * t - 2;
    let mut q = q0;
    while q < qe {
        let (lo, hi) = (q.saturating_sub(t - 1), q.min(t - 1));
        let (lo1, hi1) = ((q + 1).saturating_sub(t - 1), (q + 1).min(t - 1));
        let (mut ar, mut ai, mut br, mut bi) = (V8::zero(), V8::zero(), V8::zero(), V8::zero());
        if lo < lo1 {
            cmac(&mut ar, &mut ai, &x[lo], &y[q - lo]);
        }
        let mut p = y[q + 1 - lo1];
        for (xv, y0) in x[lo1..=hi].iter().zip(y[q - hi..=q - lo1].iter().rev()) {
            ar = xv[0].fma(y0[0], xv[2].fma(y0[1], ar));
            ai = xv[0].fma(y0[1], xv[1].fma(y0[0], ai));
            br = xv[0].fma(p[0], xv[2].fma(p[1], br));
            bi = xv[0].fma(p[1], xv[1].fma(p[0], bi));
            p = *y0;
        }
        if hi1 > hi {
            cmac(&mut br, &mut bi, &x[hi1], &y[q + 1 - hi1]);
        }
        let j = 2 * (q - q0);
        c[j] = ar;
        c[j + 1] = ai;
        c[j + 2] = br;
        c[j + 3] = bi;
        q += 2;
    }
    if q == qe {
        let (lo, hi) = (q.saturating_sub(t - 1), q.min(t - 1));
        let (mut ar, mut ai) = (V8::zero(), V8::zero());
        for (xv, yv) in x[lo..=hi].iter().zip(y[q - hi..=q - lo].iter().rev()) {
            cmac(&mut ar, &mut ai, xv, yv);
        }
        let j = 2 * (q - q0);
        c[j] = ar;
        c[j + 1] = ai;
    }
}

/// one block pair: x (block xa of a, from xa.start) times y (block xb of b), output indices o..o + nout
#[allow(clippy::too_many_arguments)]
fn pair(
    x: &[OV],
    y: &[OV],
    xa: &Block,
    xb: &Block,
    s: &[i32; LN],
    nout: usize,
    o: usize,
    cfg: &Cfg,
    e2: i32,
    acc: &mut [ODotAcc],
    st: &mut Stats,
    ws: &mut Work,
) {
    let (la, lb) = (x.len().min(nout), y.len().min(nout));
    let lc = (la + lb - 1).min(nout);
    let m = (la + lb - 1).next_power_of_two().trailing_zeros() as usize;
    debug_assert!(m <= LMAX_LOG2, "fastmul: block pair of {} terms", la + lb - 1);
    let l = 1usize << m;
    let tb = tables(m);
    let Plan { t, w, q0, nq } = plan(cfg, xa.spread, xb.spread, la, lb);
    st.digits = st.digits.max(t);
    let t0 = std::time::Instant::now();
    load(x, la, xa.start, &xa.frame, s, t, l, &mut ws.xs, &mut ws.dig, tb, cfg.mutate);
    load(y, lb, xb.start, &xb.frame, s, t, l, &mut ws.ys, &mut ws.dig, tb, cfg.mutate);
    let t0 = prof(0, t0);
    ws.vs.clear();
    ws.vs.resize(nq / 2 * l, c0());
    ws.xh.resize(t, [V8::zero(); 3]);
    ws.yh.resize(t, [V8::zero(); 2]);
    ws.cq.resize(2 * nq, V8::zero());
    // the padding weight (nq past 2t - 1 - q0), which conv does not write
    for c in ws.cq[2 * (2 * t - 1 - q0)..].iter_mut() {
        *c = V8::zero();
    }
    for f in 0..=l / 2 {
        let (pf, pg) = (tb.rev[f], tb.rev[(l - f) & (l - 1)]);
        for u in 0..t / 2 {
            (ws.xh[2 * u], ws.xh[2 * u + 1]) = unpack3(&ws.xs[u * l + pf], &ws.xs[u * l + pg]);
            (ws.yh[2 * u], ws.yh[2 * u + 1]) = unpack2(&ws.ys[u * l + pf], &ws.ys[u * l + pg]);
        }
        conv(&ws.xh, &ws.yh, t, q0, &mut ws.cq);
        // the inverse's input at rows f and -f: C_(2u) + i C_(2u+1), and at -f the conjugates' combination
        for u in 0..nq / 2 {
            let (er, ei, or, oi) = (ws.cq[4 * u], ws.cq[4 * u + 1], ws.cq[4 * u + 2], ws.cq[4 * u + 3]);
            let (rf, rg) = (u * l + pf, u * l + pg);
            if pf == pg {
                ws.vs[rf] = [er, or];
            } else {
                ws.vs[rf] = [er.sub(oi), ei.add(or)];
                ws.vs[rg] = [er.add(oi), or.sub(ei)];
            }
        }
    }
    let t0 = prof(1, t0);
    for slab in ws.vs.chunks_exact_mut(l) {
        ifft_dit(slab, tb);
    }
    let t0 = prof(2, t0);
    // each unpacked spectrum was twice its value, so each product four times: the inverse is scaled by 1/(4L)
    let sc = 0.25 / l as f64;
    let nd = 2 * t + 2 - q0;
    ws.dig.resize(nd, V8::zero());
    ws.dl.resize(nd, [0.0; LN]);
    let mut frac = V8::zero();
    // lanes whose distance from the integer is not a number (a NaN or infinity in a factor), which V8::max can drop
    let mut bad = V8::zero().ne(V8::zero());
    let (ir, r16, scv) = (V8::splat(IR16), V8::splat(R16), V8::splat(sc));
    for mi in 0..lc {
        let mut carry = V8::zero();
        for qi in 0..nd {
            let mut v = carry;
            if qi < nq {
                let z = ws.vs[qi / 2 * l + mi][qi % 2].mul(scv);
                let zr = rnd(z);
                let e = z.sub(zr).abs();
                bad = bad.or(e.ne(e));
                frac = frac.max(e);
                v = v.add(zr);
            }
            let hi = rnd(v.mul(ir));
            ws.dig[qi] = v.sub(hi.mul(r16));
            carry = hi;
        }
        debug_assert!(carry.to_array().iter().all(|&c| c == 0.0), "fastmul: digits past the top");
        // the integer at index k on each lane, digit 0 of weight 2^(16 q0 + e0), as eight 53-bit parts
        let k = o + mi;
        for (dl, d) in ws.dl.iter_mut().zip(&ws.dig) {
            *dl = d.to_array();
        }
        let mut parts = [[0.0f64; LN]; 8];
        for ln in 0..LN {
            let e0 = xa.frame[ln] + xb.frame[ln] - 2 * w - s[ln] * k as i32 + DB * q0 as i32 + e2;
            let p = parts_of(&ws.dl[..nd], ln, e0);
            for q in 0..8 {
                parts[q][ln] = p[q];
            }
        }
        acc[k] = acc[k].add(OV { c: parts.map(V8::from_array) });
    }
    // a factor that is not finite makes max_frac infinite, which fails every check of it
    st.max_frac = if bad.bits() != 0 { f64::INFINITY } else { st.max_frac.max(frac.hmax()) };
    prof(3, t0);
}
