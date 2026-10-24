//! The model's lane trait: multidouble's `dual::Lane` (values, tangents, accumulators; re-exported here as
//! `BaseLane`) plus `dmat_apply`, the product with a constant tau-derivative matrix, which needs this crate's tau-field
//! types. The lane types themselves (`DV`, `Du`, `F8`, `Fu`, `Fun<N>`, `Dtn<N>`, `QV`) come from multidouble.
use crate::field::{DMat, Fl};
use crate::dv::{DotAcc, DV};
use crate::qv::QV;
use crate::v8::V8;
pub use multidouble::dual::Lane as BaseLane;
pub use multidouble::native::{Dtn, Du, Fu, Fun, F8};
use multidouble::simd::Native;

/// multidouble's lanes on the compile-time backend (`Native`), with the tau-derivative product
pub trait Lane: BaseLane<S = Native> {
    /// D f for a constant matrix D (field::DMat::apply). The default multiplies by the double-double columns; the
    /// f64 lanes use the matrix's f64 copy.
    #[inline(always)]
    fn dmat_apply<const B: usize>(d: &DMat<B>, f: &Fl<Self, B>) -> Fl<Self, B> {
        d.apply_dd(f)
    }
}

impl Lane for DV {}
impl Lane for Du {}
impl Lane for QV {}

impl Lane for F8 {
    #[inline(always)]
    fn dmat_apply<const B: usize>(d: &DMat<B>, f: &Fl<Self, B>) -> Fl<Self, B> {
        let mut acc = [V8::zero(); B];
        for jb in 0..B {
            for jl in 0..8 {
                let s = f.0[jb].0.bcast(jl);
                let col = &d.cols64[8 * jb + jl];
                for b in 0..B {
                    acc[b] = s.fma(col[b], acc[b]);
                }
            }
        }
        Fl(acc.map(multidouble::dual::F8))
    }
}

impl Lane for Fu {
    #[inline(always)]
    fn dmat_apply<const B: usize>(d: &DMat<B>, f: &Fl<Self, B>) -> Fl<Self, B> {
        let mut av = [V8::zero(); B];
        let mut ad = [V8::zero(); B];
        for jb in 0..B {
            for jl in 0..8 {
                let sv = f.0[jb].v.bcast(jl);
                let sd = f.0[jb].d.bcast(jl);
                let col = &d.cols64[8 * jb + jl];
                for b in 0..B {
                    av[b] = sv.fma(col[b], av[b]);
                    ad[b] = sd.fma(col[b], ad[b]);
                }
            }
        }
        let mut out = [Fu::zero(); B];
        for b in 0..B {
            out[b] = Fu { v: av[b], d: ad[b] };
        }
        Fl(out)
    }
}

impl<const N: usize> Lane for Dtn<N> {
    #[inline(always)]
    fn dmat_apply<const B: usize>(dm: &DMat<B>, f: &Fl<Self, B>) -> Fl<Self, B> {
        let mut av = [DotAcc::zero(); B];
        let mut ad = [[V8::zero(); N]; B];
        for jb in 0..B {
            for jl in 0..8 {
                let sv = f.0[jb].v.bcast(jl);
                let sd = f.0[jb].d.map(|x| x.bcast(jl));
                let col = &dm.cols[8 * jb + jl];
                let col64 = &dm.cols64[8 * jb + jl];
                for b in 0..B {
                    av[b] = av[b].add_prod(sv, col.0[b]);
                    for i in 0..N {
                        ad[b][i] = sd[i].fma(col64[b], ad[b][i]);
                    }
                }
            }
        }
        let mut out = [Self::zero(); B];
        for b in 0..B {
            out[b] = Dtn { v: av[b].finish(), d: ad[b] };
        }
        Fl(out)
    }
}

impl<const N: usize> Lane for Fun<N> {
    #[inline(always)]
    fn dmat_apply<const B: usize>(dm: &DMat<B>, f: &Fl<Self, B>) -> Fl<Self, B> {
        let mut av = [V8::zero(); B];
        let mut ad = [[V8::zero(); N]; B];
        for jb in 0..B {
            for jl in 0..8 {
                let sv = f.0[jb].v.bcast(jl);
                let sd = f.0[jb].d.map(|x| x.bcast(jl));
                let col64 = &dm.cols64[8 * jb + jl];
                for b in 0..B {
                    av[b] = sv.fma(col64[b], av[b]);
                    for i in 0..N {
                        ad[b][i] = sd[i].fma(col64[b], ad[b][i]);
                    }
                }
            }
        }
        let mut out = [Self::zero(); B];
        for b in 0..B {
            out[b] = Fun { v: av[b], d: ad[b] };
        }
        Fl(out)
    }
}
