//! Scalar abstraction (f64 and Complex64) and the 64-byte aligned per-tau-point field type.

use num_complex::Complex64;
use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// Scalar type the residual is generic over. All operations used by the residual are analytic
/// (+, -, *, /), so instantiating with Complex64 gives a complex-step derivative.
pub trait Sc:
    Copy
    + Send
    + Sync
    + Debug
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
{
    fn zero() -> Self;
    fn re(x: f64) -> Self;
    /// acc + self * a  (a real); fused for f64.
    fn fma_r(self, a: f64, acc: Self) -> Self;
    /// acc + self * b; fused for f64.
    fn fma(self, b: Self, acc: Self) -> Self;
    /// Magnitude used for pivoting (|x| for f64, |re|+|im| like LAPACK's cabs1 for complex).
    fn mag(self) -> f64;
    /// Exact bit pattern (for de-duplicating identical block inputs within a batch).
    fn push_bits(self, v: &mut Vec<u64>);
}

impl Sc for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn re(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn fma_r(self, a: f64, acc: Self) -> Self {
        self.mul_add(a, acc)
    }
    #[inline(always)]
    fn fma(self, b: Self, acc: Self) -> Self {
        self.mul_add(b, acc)
    }
    #[inline(always)]
    fn mag(self) -> f64 {
        self.abs()
    }
    #[inline(always)]
    fn push_bits(self, v: &mut Vec<u64>) {
        v.push(self.to_bits());
    }
}

impl Sc for Complex64 {
    #[inline(always)]
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    #[inline(always)]
    fn re(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    #[inline(always)]
    fn fma_r(self, a: f64, acc: Self) -> Self {
        // real matrix acting on real and imaginary parts separately (as the Python _cplx wrapper does)
        Complex64::new(self.re.mul_add(a, acc.re), self.im.mul_add(a, acc.im))
    }
    #[inline(always)]
    fn fma(self, b: Self, acc: Self) -> Self {
        acc + self * b
    }
    #[inline(always)]
    fn mag(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    #[inline(always)]
    fn push_bits(self, v: &mut Vec<u64>) {
        v.push(self.re.to_bits());
        v.push(self.im.to_bits());
    }
}

/// One field sampled on N tau points (N padded to a multiple of 8), 64-byte aligned so that every
/// 8-lane chunk is one aligned AVX-512 vector.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Fld<T, const N: usize>(pub [T; N]);

impl<T: Sc, const N: usize> Fld<T, N> {
    #[inline(always)]
    pub fn zero() -> Self {
        Fld([T::zero(); N])
    }
    #[inline(always)]
    pub fn splat(v: T) -> Self {
        Fld([v; N])
    }
}

impl<T, const N: usize> Index<usize> for Fld<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T, const N: usize> IndexMut<usize> for Fld<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

/// State (p, m, a, c) at one x.
pub type State<T, const N: usize> = [Fld<T, N>; 4];

#[inline(always)]
pub fn zero_state<T: Sc, const N: usize>() -> State<T, N> {
    [Fld::zero(); 4]
}
