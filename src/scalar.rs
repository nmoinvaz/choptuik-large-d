//! Scalar abstraction (f64, Complex64, the forward-mode `Dual` and the double-double `DD`) and the
//! 64-byte aligned per-tau-point field type.

use num_complex::Complex64;
use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

/// Scalar type the residual is generic over. All operations used by the residual are analytic
/// (+, -, *, /), so instantiating with Complex64 gives a complex-step derivative and with `Dual` an
/// exact forward-mode derivative.
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

    /// True for tangent scalars (`Dual`). Only those carry the growth-mode shift: with `Ctx::lam`
    /// set, every tau-derivative of the tangent gains `lam * tangent` (perturbations ~ exp(lam tau)).
    const TANGENT: bool = false;
    /// The tangent part, with zero value ((0, d) for `Dual`; zero for the other scalars).
    #[inline(always)]
    fn tangent(self) -> Self {
        Self::zero()
    }
    /// (value, tangent) of a tangent scalar.
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        unreachable!("parts() is only called for tangent scalars")
    }
    #[inline(always)]
    fn from_parts(_v: f64, _d: f64) -> Self {
        unreachable!("from_parts() is only called for tangent scalars")
    }
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

/// Forward-mode dual number v + d ε (ε² = 0). The value part is computed with exactly the f64
/// operations of the plain-loop path, so it reproduces the f64 residual bit for bit; the tangent part
/// is the exact directional derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub d: f64,
}

impl Dual {
    #[inline(always)]
    pub fn new(v: f64, d: f64) -> Self {
        Dual { v, d }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline(always)]
    fn add(self, o: Dual) -> Dual {
        Dual::new(self.v + o.v, self.d + o.d)
    }
}
impl Sub for Dual {
    type Output = Dual;
    #[inline(always)]
    fn sub(self, o: Dual) -> Dual {
        Dual::new(self.v - o.v, self.d - o.d)
    }
}
impl Mul for Dual {
    type Output = Dual;
    #[inline(always)]
    fn mul(self, o: Dual) -> Dual {
        Dual::new(self.v * o.v, self.v.mul_add(o.d, self.d * o.v))
    }
}
impl Div for Dual {
    type Output = Dual;
    #[inline(always)]
    fn div(self, o: Dual) -> Dual {
        let q = self.v / o.v;
        Dual::new(q, (self.d - q * o.d) / o.v)
    }
}
impl Neg for Dual {
    type Output = Dual;
    #[inline(always)]
    fn neg(self) -> Dual {
        Dual::new(-self.v, -self.d)
    }
}
impl Mul<f64> for Dual {
    type Output = Dual;
    #[inline(always)]
    fn mul(self, a: f64) -> Dual {
        Dual::new(self.v * a, self.d * a)
    }
}
impl Div<f64> for Dual {
    type Output = Dual;
    #[inline(always)]
    fn div(self, a: f64) -> Dual {
        Dual::new(self.v / a, self.d / a)
    }
}
impl Add<f64> for Dual {
    type Output = Dual;
    #[inline(always)]
    fn add(self, a: f64) -> Dual {
        Dual::new(self.v + a, self.d)
    }
}
impl Sub<f64> for Dual {
    type Output = Dual;
    #[inline(always)]
    fn sub(self, a: f64) -> Dual {
        Dual::new(self.v - a, self.d)
    }
}

impl Sc for Dual {
    const TANGENT: bool = true;
    #[inline(always)]
    fn zero() -> Self {
        Dual::new(0.0, 0.0)
    }
    #[inline(always)]
    fn re(x: f64) -> Self {
        Dual::new(x, 0.0)
    }
    #[inline(always)]
    fn fma_r(self, a: f64, acc: Self) -> Self {
        Dual::new(self.v.mul_add(a, acc.v), self.d.mul_add(a, acc.d))
    }
    #[inline(always)]
    fn fma(self, b: Self, acc: Self) -> Self {
        Dual::new(self.v.mul_add(b.v, acc.v), self.v.mul_add(b.d, self.d.mul_add(b.v, acc.d)))
    }
    #[inline(always)]
    fn mag(self) -> f64 {
        self.v.abs()
    }
    #[inline(always)]
    fn push_bits(self, v: &mut Vec<u64>) {
        v.push(self.v.to_bits());
        v.push(self.d.to_bits());
    }
    #[inline(always)]
    fn tangent(self) -> Self {
        Dual::new(0.0, self.d)
    }
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        (self.v, self.d)
    }
    #[inline(always)]
    fn from_parts(v: f64, d: f64) -> Self {
        Dual::new(v, d)
    }
}

/// Double-double number hi + lo (|lo| <= ulp(hi)/2), about 32 significant digits, with the usual
/// error-free transformations (Dekker / Knuth two-sum, FMA two-product). Used to evaluate the residual
/// far below f64 rounding noise, where the weakly selected period Delta is no longer pinned in f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DD {
    pub hi: f64,
    pub lo: f64,
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl DD {
    #[inline(always)]
    pub fn new(hi: f64, lo: f64) -> Self {
        let (h, l) = quick_two_sum(hi, lo);
        DD { hi: h, lo: l }
    }
    #[inline(always)]
    pub fn from_f64(x: f64) -> Self {
        DD { hi: x, lo: 0.0 }
    }
    #[inline(always)]
    fn mul_f64(self, b: f64) -> DD {
        let (p, e) = two_prod(self.hi, b);
        let (h, l) = quick_two_sum(p, self.lo.mul_add(b, e));
        DD { hi: h, lo: l }
    }
    #[inline(always)]
    fn add_f64(self, b: f64) -> DD {
        let (s, e) = two_sum(self.hi, b);
        let (h, l) = quick_two_sum(s, e + self.lo);
        DD { hi: h, lo: l }
    }
}

impl Add for DD {
    type Output = DD;
    #[inline(always)]
    fn add(self, o: DD) -> DD {
        let (s, e) = two_sum(self.hi, o.hi);
        let (t, f) = two_sum(self.lo, o.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (h, l) = quick_two_sum(s, e + f);
        DD { hi: h, lo: l }
    }
}
impl Neg for DD {
    type Output = DD;
    #[inline(always)]
    fn neg(self) -> DD {
        DD { hi: -self.hi, lo: -self.lo }
    }
}
impl Sub for DD {
    type Output = DD;
    #[inline(always)]
    fn sub(self, o: DD) -> DD {
        self + (-o)
    }
}
impl Mul for DD {
    type Output = DD;
    #[inline(always)]
    fn mul(self, o: DD) -> DD {
        let (p, e) = two_prod(self.hi, o.hi);
        let e = self.hi.mul_add(o.lo, self.lo.mul_add(o.hi, e));
        let (h, l) = quick_two_sum(p, e);
        DD { hi: h, lo: l }
    }
}
impl Div for DD {
    type Output = DD;
    #[inline(always)]
    fn div(self, o: DD) -> DD {
        // long division: q1 + q2 with one correction (Bailey's QD dd_real accurate division)
        let q1 = self.hi / o.hi;
        let r = self - o.mul_f64(q1);
        let q2 = r.hi / o.hi;
        let r = r - o.mul_f64(q2);
        let q3 = r.hi / o.hi;
        let (h, l) = quick_two_sum(q1, q2);
        DD { hi: h, lo: l }.add_f64(q3)
    }
}
impl Mul<f64> for DD {
    type Output = DD;
    #[inline(always)]
    fn mul(self, a: f64) -> DD {
        self.mul_f64(a)
    }
}
impl Div<f64> for DD {
    type Output = DD;
    #[inline(always)]
    fn div(self, a: f64) -> DD {
        self / DD::from_f64(a)
    }
}
impl Add<f64> for DD {
    type Output = DD;
    #[inline(always)]
    fn add(self, a: f64) -> DD {
        self.add_f64(a)
    }
}
impl Sub<f64> for DD {
    type Output = DD;
    #[inline(always)]
    fn sub(self, a: f64) -> DD {
        self.add_f64(-a)
    }
}

impl Sc for DD {
    #[inline(always)]
    fn zero() -> Self {
        DD { hi: 0.0, lo: 0.0 }
    }
    #[inline(always)]
    fn re(x: f64) -> Self {
        DD::from_f64(x)
    }
    #[inline(always)]
    fn fma_r(self, a: f64, acc: Self) -> Self {
        acc + self.mul_f64(a)
    }
    #[inline(always)]
    fn fma(self, b: Self, acc: Self) -> Self {
        acc + self * b
    }
    #[inline(always)]
    fn mag(self) -> f64 {
        self.hi.abs()
    }
    #[inline(always)]
    fn push_bits(self, v: &mut Vec<u64>) {
        v.push(self.hi.to_bits());
        v.push(self.lo.to_bits());
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
