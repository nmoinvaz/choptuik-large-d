//! The self-gravitating hedgehog sigma-model's DSS (echoing) critical solution on the multishoot engine.
//!
//! * `model`: the equations (centre series, Taylor recursion in x, horizon series) as multishoot's `Equations`, and
//!   the entry points (residual, exact Jacobian, Newton step) in double-double and quad-double.
//! * `planted`: the source terms of a manufactured solution, for the planted-truth check (`planted.py`).
//! * `python` (feature `python`): the pyo3 bindings, `sigmams.Solver`.
//!
//! The arithmetic is multidouble's and the multiple-shooting engine multishoot's, at hp's versions. The equations are
//! copied and adapted from hp's `model.rs` (the Choptuik scalar) with the sigma-model terms of
//! `solver/matter/sigdss_solver.py`; hp itself is not touched.
pub mod model;
pub mod planted;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests;
