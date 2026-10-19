//! AVX2 + FMA lane type (4 x f64 in a ymm register) and the `Avx2` backend.

use super::{simd_backend, Lanes};
use crate::model::Ctx;
use crate::scalar::{Fld, State};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct F64x4(__m256d);

// SAFETY: only instantiated inside `#[target_feature(enable = "avx2,fma")]` kernels.
unsafe impl Lanes for F64x4 {
    const L: usize = 4;
    // 16 ymm registers: 12 accumulators + 3 operands, or 2 x 6 accumulators + 2 broadcasts + 1 row
    const G1: usize = 12;
    const G2: usize = 6;

    #[inline(always)]
    unsafe fn zero() -> Self {
        F64x4(_mm256_setzero_pd())
    }
    #[inline(always)]
    unsafe fn splat(v: f64) -> Self {
        F64x4(_mm256_set1_pd(v))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        F64x4(_mm256_load_pd(p))
    }
    #[inline(always)]
    unsafe fn store(p: *mut f64, v: Self) {
        _mm256_store_pd(p, v.0)
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        F64x4(_mm256_add_pd(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        F64x4(_mm256_sub_pd(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        F64x4(_mm256_mul_pd(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn div(a: Self, b: Self) -> Self {
        F64x4(_mm256_div_pd(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn fmadd(a: Self, b: Self, c: Self) -> Self {
        F64x4(_mm256_fmadd_pd(a.0, b.0, c.0))
    }
    #[inline(always)]
    unsafe fn fnmadd(a: Self, b: Self, c: Self) -> Self {
        F64x4(_mm256_fnmadd_pd(a.0, b.0, c.0))
    }
    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        F64x4(_mm256_xor_pd(a.0, _mm256_set1_pd(-0.0)))
    }
}

simd_backend!(Avx2, F64x4, "avx2,fma");
