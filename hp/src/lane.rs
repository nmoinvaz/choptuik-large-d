//! The model's lane trait: multidouble's `dual::Lane` (values, tangents, accumulators; re-exported here as
//! `BaseLane`) with its `field::MatLane` (`dmat_apply`, the product with a constant tau-derivative matrix), on the
//! compile-time backend. Every lane type of multidouble's is one (`DV`, `Du`, `Dc`, `F8`, `Fu`, `Fun<N>`, `Dtn<N>`,
//! `QV`, `Qu`, `Counted<L>`); the quad-double ones refine the horizon solves (`BaseLane::VAL_QD`, model::psolve).
pub use multidouble::dual::Lane as BaseLane;
pub use multidouble::field::MatLane;
pub use multidouble::native::{Dc, Dtn, Du, Fu, Fun, Qu, F8};
use multidouble::simd::Native;

/// multidouble's lanes on the compile-time backend (`Native`), with the tau-derivative product (`MatLane::dmat_apply`:
/// the double-double columns through the dot accumulator, the f64 lanes the columns' f64 copy)
pub trait Lane: MatLane<S = Native> {}

impl<T: MatLane<S = Native>> Lane for T {}
