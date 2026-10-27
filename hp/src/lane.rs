//! The model's lane trait: multidouble's `dual::Lane` (values, tangents, accumulators; re-exported here as
//! `BaseLane`) with its `field::MatLane` (`dmat_apply`, the product with a constant tau-derivative matrix), on the
//! compile-time backend. The lane types (`DV`, `Du`, `F8`, `Fu`, `Fun<N>`, `Dtn<N>`, `QV`) and their matrix products
//! are multidouble's.
use crate::dv::DV;
use crate::qv::QV;
pub use multidouble::dual::Lane as BaseLane;
pub use multidouble::field::MatLane;
pub use multidouble::native::{Dtn, Du, Fu, Fun, F8};
use multidouble::simd::Native;

/// multidouble's lanes on the compile-time backend (`Native`), with the tau-derivative product (`MatLane::dmat_apply`:
/// the double-double columns through the dot accumulator, the f64 lanes the columns' f64 copy)
pub trait Lane: MatLane<S = Native> {
    /// The horizon solves (model::psolve) refine the double-double LU's solution with the residual of the system formed
    /// in this lane's precision (quad-double lanes: model::residual_qdh). The other lanes solve as before.
    const QD_SOLVE: bool = false;
}

impl Lane for DV {}
impl Lane for Du {}
impl Lane for QV {
    const QD_SOLVE: bool = true;
}
impl Lane for F8 {}
impl Lane for Fu {}
impl<const N: usize> Lane for Dtn<N> {}
impl<const N: usize> Lane for Fun<N> {}
