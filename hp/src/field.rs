//! A tau field on the half grid: M = 8 B collocation points theta_j = 2 pi j / (2M), j < M, stored as B lanes.
//! p, m are odd under theta -> theta + pi and a, c even, so the other half period is implied and pointwise products
//! keep the right parity. d/dtheta of an odd field is the folded spectral matrix (`DMat`).
//!
//! The field type and its arithmetic are multidouble's `field` module (this module's code, moved there): `Fl` and its
//! operations, the Cauchy product coefficients (`cs`, `cs_range`, `cs_w`), `horner`, the constant matrix `DMat` (on
//! the compile-time backend, multidouble's `native::DMat`) and `FieldSolve`. This crate adds the four-field state, and
//! the Cauchy products of jet lanes with their B lane vectors summed on separate threads.
use multidouble::dual::Lane;
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
