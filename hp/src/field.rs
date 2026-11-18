//! A tau field on the half grid: M = 8 B collocation points theta_j = 2 pi j / (2M), j < M, stored as B lanes.
//! p, m are odd under theta -> theta + pi and a, c even, so the other half period is implied and pointwise products
//! keep the right parity. d/dtheta of an odd field is the folded spectral matrix (`DMat`).
//!
//! The field type and its arithmetic are multidouble's `field` module (this module's code, moved there): `Fl` and its
//! operations, the Cauchy product coefficients (`cs`, `cs_range`, `cs_w`), `horner`, the constant matrix `DMat` (on
//! the compile-time backend, multidouble's `native::DMat`) and `FieldSolve`. This crate adds the four-field state, and
//! the Cauchy products of jet lanes with their B lane vectors summed on separate threads, and the order-k sums of a
//! series formed one jet order at a time, several independent sums at once with every lane vector on its own task
//! (`sums_at`).
use multidouble::dual::Lane;
use multidouble::field::{cs_at_splits, cs_lane_at, cs_range_at, cs_w_at, cs_w_lane_at};
use multidouble::simd::Simd;
pub use multidouble::field::{horner, FieldSolve, Fl, TangentTail};
pub use multidouble::native::DMat;
use rayon::prelude::*;

/// the state at a point: the four fields p, m, a, c
pub type State<T, const B: usize> = [Fl<T, B>; 4];

/// terms times jet coefficients from which a Cauchy product of jet lanes is split over its lane vectors
const PAR_MIN: usize = 48;

/// whether the sum of `terms` products of T is split over the B lane vectors: jet lanes (each product a truncated
/// convolution of NJ coefficients) with enough terms, not the f64 lanes, whose multidouble sums interleave the vectors
#[inline(always)]
fn split<T: Lane>(terms: usize) -> bool {
    T::NJ > 0 && !(T::VAL_F64 || T::TAN_F64) && terms * T::NJ >= PAR_MIN
}

/// `sum_{j=lo..=hi} (w(j) x[j]) y[k-j]`, each lane vector's terms in the order j = lo..=hi as multidouble's sums
/// take them (its in-place lane-vector sums), the B vectors on separate threads
fn split_sum<T: Lane, const B: usize>(
    x: &[Fl<T, B>],
    y: &[Fl<T, B>],
    k: usize,
    lo: usize,
    hi: usize,
    w: Option<&(dyn Fn(usize) -> f64 + Sync)>,
) -> Fl<T, B> {
    let mut out = Fl::<T, B>::zero();
    out.0.par_iter_mut().enumerate().for_each(|(b, o)| match w {
        Some(w) => multidouble::field::cs_w_lane_into(o, b, x, y, k, lo, hi, w),
        None => multidouble::field::cs_lane_into(o, b, x, y, k, lo, hi),
    });
    out
}

/// `sum_{j=0..=k} x[j] y[k-j]` (multidouble's `cs`, bitwise), the accumulator borrowed rather than moved
#[inline(always)]
pub fn cs<T: Lane, const B: usize>(x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    if split::<T>(k + 1) {
        split_sum(x, y, k, 0, k, None)
    } else {
        let mut out = Fl::<T, B>::zero();
        multidouble::field::cs_into(&mut out, x, y, k);
        out
    }
}

/// `sum_{j=lo..=hi} x[j] y[k-j]` (multidouble's `cs_range`, bitwise)
#[inline(always)]
pub fn cs_range<T: Lane, const B: usize>(x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize, lo: usize, hi: usize) -> Fl<T, B> {
    if hi >= lo && split::<T>(hi - lo + 1) {
        split_sum(x, y, k, lo, hi, None)
    } else {
        let mut out = Fl::<T, B>::zero();
        multidouble::field::cs_range_into(&mut out, x, y, k, lo, hi);
        out
    }
}

/// `sum_{j=lo..=hi} (w(j) x[j]) y[k-j]` with exact f64 weights w(j) (multidouble's `cs_w`, bitwise)
#[inline(always)]
pub fn cs_w<T: Lane, const B: usize>(
    x: &[Fl<T, B>],
    y: &[Fl<T, B>],
    k: usize,
    lo: usize,
    hi: usize,
    w: impl Fn(usize) -> f64 + Sync,
) -> Fl<T, B> {
    if hi >= lo && split::<T>(hi - lo + 1) {
        split_sum(x, y, k, lo, hi, Some(&w))
    } else {
        let mut out = Fl::<T, B>::zero();
        multidouble::field::cs_w_into(&mut out, x, y, k, lo, hi, w);
        out
    }
}

/// Tests only: `sums_at` forms every sum serially, one after another, as the series did before (the baseline of
/// prec_tests' bench_oseries_by_order)
#[cfg(test)]
pub static SERIAL_AT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One order-k Cauchy sum for `sums_at`: coefficient k of `sum_{i=lo..=hi} (w(i) x[i]) y[j-i]` into out, weighted only
/// if w is given (multidouble's `cs_range_at`, `cs_w_at`)
pub struct SumAt<'a, T: Lane, const B: usize> {
    out: &'a mut Fl<T, B>,
    x: &'a [Fl<T, B>],
    y: &'a [Fl<T, B>],
    j: usize,
    lo: usize,
    hi: usize,
    w: Option<&'a (dyn Fn(usize) -> f64 + Sync)>,
}

impl<'a, T: Lane, const B: usize> SumAt<'a, T, B> {
    /// `sum_{i=0..=j} x[i] y[j-i]` (`cs_at`)
    pub fn new(out: &'a mut Fl<T, B>, x: &'a [Fl<T, B>], y: &'a [Fl<T, B>], j: usize) -> Self {
        SumAt { out, x, y, j, lo: 0, hi: j, w: None }
    }
    /// the terms lo..=hi alone (`cs_range_at`)
    pub fn range(self, lo: usize, hi: usize) -> Self {
        SumAt { lo, hi, ..self }
    }
    /// each term i weighted by w(i), an exact f64 that is the same on every call and thread (`cs_w_at`)
    pub fn weighted(self, w: &'a (dyn Fn(usize) -> f64 + Sync)) -> Self {
        SumAt { w: Some(w), ..self }
    }
    fn terms(&self) -> usize {
        (self.hi + 1).saturating_sub(self.lo)
    }
}

/// Coefficient k of each of several independent sums (no sum reads another's output), each bitwise its own
/// `cs_range_at` or `cs_w_at`. When their terms together are long enough (multidouble's `cs_at_splits`), each lane vector
/// of each sum is one task of rayon's current pool, which sums its terms in their order, so neither the split nor the
/// pool's size changes a bit; otherwise they are formed one after another.
pub fn sums_at<T: Lane, const B: usize>(sums: Vec<SumAt<'_, T, B>>, k: usize) {
    #[cfg(test)]
    let serial = SERIAL_AT.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    let serial = false;
    if serial || !cs_at_splits::<T>(sums.iter().map(SumAt::terms).sum(), k) {
        for SumAt { out, x, y, j, lo, hi, w } in sums {
            match w {
                Some(w) => cs_w_at(out, x, y, j, lo, hi, w, k),
                None => cs_range_at(out, x, y, j, lo, hi, k),
            }
        }
        return;
    }
    let tasks: Vec<_> = sums
        .into_iter()
        .flat_map(|SumAt { out, x, y, j, lo, hi, w }| out.0.iter_mut().enumerate().map(move |(b, o)| (o, b, x, y, j, lo, hi, w)))
        .collect();
    tasks.into_par_iter().for_each(|(o, b, x, y, j, lo, hi, w)| {
        T::S::vectorize(
            #[inline(always)]
            || match w {
                Some(w) => cs_w_lane_at(o, b, x, y, j, lo, hi, w, k),
                None => cs_lane_at(o, b, x, y, j, lo, hi, k),
            },
        )
    });
}
