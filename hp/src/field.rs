//! A tau field on the half grid: M = 8 B collocation points theta_j = 2 pi j / (2M), j < M, stored as B lanes.
//! p, m are odd under theta -> theta + pi and a, c even, so the other half period is implied and pointwise products
//! keep the right parity. d/dtheta of an odd field is the folded spectral matrix (`DMat`).
//!
//! The field type and its arithmetic are multidouble's `field` module (this module's code, moved there): `Fl` and its
//! operations, the Cauchy product coefficients (`cs`, `cs_range`, `cs_w`), `horner`, the constant matrix `DMat` (on
//! the compile-time backend, multidouble's `native::DMat`) and `FieldSolve`. This crate adds the four-field state.
pub use multidouble::field::{cs, cs_range, cs_w, horner, FieldSolve, Fl, TangentTail};
pub use multidouble::native::DMat;

/// the state at a point: the four fields p, m, a, c
pub type State<T, const B: usize> = [Fl<T, B>; 4];
