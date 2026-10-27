//! Block storage for the multiple-shooting Newton system: a staircase LU that keeps and factors only the nonzero node
//! blocks of the Jacobian, so the dense n x n matrix is never built (n = 4M(K+1) + 1; at M = 256 and K = 13 nodes a
//! dense f64 Jacobian would be 1.6 GB, 93% zeros).
//!
//! Structure. Residual block b (4M rows) depends on two neighbouring groups of unknowns and on Delta: block 0 on the
//! centre data (p1, c0) and node 0; block b = 1..K-1 on nodes b-1 and b (whichever way its segment runs); block K on
//! node K-1 and the horizon data (m_h, a_h). The phase row depends on p1 only. Delta enters every block, so the
//! factorization works on an augmented system with one copy delta_b of Delta per residual block and link rows
//! delta_{b-1} - delta_b = 0. With the column groups
//!     g_0 = (p1, c0),  g_t = (delta_{t-1}, node t-1) for t = 1..K,  g_{K+1} = (delta_K, m_h, a_h)
//! and the row groups (phase, block 0), (link, block 1), ..., (link, block K), row group s touches g_s and g_{s+1}
//! only. Gaussian elimination with partial pivoting then runs in steps: step s factors the columns of g_s, its pivot
//! candidates being the rows left over from step s-1 (all of them zero outside g_s) and row group s, and passes the
//! rows it does not pivot on, now holding the Schur complement on g_{s+1}, to step s+1; the last step is a square LU
//! on g_{K+1}. This is exactly partial pivoting on the augmented matrix in this row and column order (rows below a
//! step never hold the step's columns), so it has that algorithm's stability, and the fill stays inside the steps.
//! Stored per step: L and U11 (rows x pivot columns) and U12 (pivot rows x next group), f64.
//!
//! Solves are in double-double with the f64 factors, as the dense mixed-precision path does (DDLu with lo = 0).
//! With E the copy of Delta into every delta_b and S the selection of the non-link rows, J = S J_aug E, so
//! J^-1 = P J_aug^-1 S^T and J^-T = S J_aug^-T P^T for any P with P E = I; P reads Delta from delta_0.
//!
//! The factorization and the solves are multidouble's `linalg::StaircaseLu` (this module's code, moved there with
//! generic names: `ColumnGroup` with `param` for the Delta copy, row0 for the phase row): each step's factorization is
//! `linalg::lu_rect_f64`, LAPACK's blocked right-looking getrf (Anderson et al., LAPACK Users' Guide, 3rd ed., SIAM
//! 1999) restricted to the first np columns of a rectangular block, and the solves round as `DDLu`'s own do.
pub use multidouble::linalg::{ColumnGroup as Group, StaircaseControl, StaircaseLu as BlockLu};

/// Deliberate breakage for the controls (hp/highm_check.py): 1 one pivot of the middle step scaled by 1 + 1e-6;
/// 2 the link rows replaced by delta_s = 0 (the copies are no longer tied to Delta).
pub const MUT_PIVOT: u8 = 1;
pub const MUT_LINK: u8 = 2;

/// The staircase LU's control for a mutate value (MUT_PIVOT, MUT_LINK; any other value: none)
pub fn control(mutate: u8) -> StaircaseControl {
    match mutate {
        MUT_PIVOT => StaircaseControl::Pivot,
        MUT_LINK => StaircaseControl::Link,
        _ => StaircaseControl::None,
    }
}
