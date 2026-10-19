//! NEON lane type (2 x f64 in a q register) and the `Neon` backend.

use super::{simd_backend, Lanes};
use crate::model::Ctx;
use crate::scalar::{Fld, State};
use core::arch::aarch64::*;

#[derive(Clone, Copy)]
pub struct F64x2(float64x2_t);

// SAFETY: NEON is part of every aarch64 target this crate builds for; the kernels also enable it.
unsafe impl Lanes for F64x2 {
    const L: usize = 2;
    // 32 v registers: 16 accumulators + 3 operands, or 2 x 12 accumulators + 2 broadcasts + 1 row
    const G1: usize = 16;
    const G2: usize = 12;

    #[inline(always)]
    unsafe fn zero() -> Self {
        F64x2(vdupq_n_f64(0.0))
    }
    #[inline(always)]
    unsafe fn splat(v: f64) -> Self {
        F64x2(vdupq_n_f64(v))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        F64x2(vld1q_f64(p))
    }
    #[inline(always)]
    unsafe fn store(p: *mut f64, v: Self) {
        vst1q_f64(p, v.0)
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        F64x2(vaddq_f64(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        F64x2(vsubq_f64(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn mul(a: Self, b: Self) -> Self {
        F64x2(vmulq_f64(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn div(a: Self, b: Self) -> Self {
        F64x2(vdivq_f64(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn fmadd(a: Self, b: Self, c: Self) -> Self {
        // vfmaq_f64(c, a, b) = c + a * b
        F64x2(vfmaq_f64(c.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn fnmadd(a: Self, b: Self, c: Self) -> Self {
        // vfmsq_f64(c, a, b) = c - a * b
        F64x2(vfmsq_f64(c.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn neg(a: Self) -> Self {
        F64x2(vnegq_f64(a.0))
    }
}

simd_backend!(Neon, F64x2, "neon");
