//! The scalar abstraction the model is written over. A `Lane` is 8 tau points of a field.
//!
//! * `DV`: values (double-double).
//! * `Du`: value + one forward-mode tangent (v + d eps, eps^2 = 0), both double-double. One evaluation with a
//!   unit tangent on one unknown gives one exact Jacobian column.
use crate::dd::DD;
use crate::dv::{DotAcc, DV};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Lane:
    Copy + Send + Sync + 'static + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    type Acc: Copy;
    const TANGENT: bool;
    fn zero() -> Self;
    /// a constant (zero tangent)
    fn cst(c: DV) -> Self;
    #[inline(always)]
    fn splat(x: DD) -> Self {
        Self::cst(DV::splat(x))
    }
    /// times a constant lane vector
    fn mulc(self, c: DV) -> Self;
    /// times a scalar constant
    #[inline(always)]
    fn muls(self, s: DD) -> Self {
        self.mulc(DV::splat(s))
    }
    /// times an exact f64 (small integers, 1/2)
    fn mulf(self, s: f64) -> Self;
    /// plus a constant lane vector
    fn addc(self, c: DV) -> Self;
    /// all 8 lanes = lane j
    fn bcast(self, j: usize) -> Self;
    fn val(self) -> DV;
    fn tan(self) -> DV;
    fn from_vt(v: DV, t: DV) -> Self;
    fn acc0() -> Self::Acc;
    /// a + x y
    fn acc(a: Self::Acc, x: Self, y: Self) -> Self::Acc;
    /// a + x c, c constant
    fn accc(a: Self::Acc, x: Self, c: DV) -> Self::Acc;
    /// a + x
    fn acc1(a: Self::Acc, x: Self) -> Self::Acc;
    fn fin(a: Self::Acc) -> Self;
}

impl Lane for DV {
    type Acc = DotAcc;
    const TANGENT: bool = false;
    #[inline(always)]
    fn zero() -> Self {
        DV::zero()
    }
    #[inline(always)]
    fn cst(c: DV) -> Self {
        c
    }
    #[inline(always)]
    fn mulc(self, c: DV) -> Self {
        self * c
    }
    #[inline(always)]
    fn mulf(self, s: f64) -> Self {
        self.mul_f(s)
    }
    #[inline(always)]
    fn addc(self, c: DV) -> Self {
        self + c
    }
    #[inline(always)]
    fn bcast(self, j: usize) -> Self {
        DV::bcast(self, j)
    }
    #[inline(always)]
    fn val(self) -> DV {
        self
    }
    #[inline(always)]
    fn tan(self) -> DV {
        DV::zero()
    }
    #[inline(always)]
    fn from_vt(v: DV, _t: DV) -> Self {
        v
    }
    #[inline(always)]
    fn acc0() -> DotAcc {
        DotAcc::zero()
    }
    #[inline(always)]
    fn acc(a: DotAcc, x: Self, y: Self) -> DotAcc {
        a.add_prod(x, y)
    }
    #[inline(always)]
    fn accc(a: DotAcc, x: Self, c: DV) -> DotAcc {
        a.add_prod(x, c)
    }
    #[inline(always)]
    fn acc1(a: DotAcc, x: Self) -> DotAcc {
        a.add(x)
    }
    #[inline(always)]
    fn fin(a: DotAcc) -> Self {
        a.finish()
    }
}

/// Forward-mode dual number over double-double lanes.
#[derive(Clone, Copy, Debug)]
pub struct Du {
    pub v: DV,
    pub d: DV,
}

impl Add for Du {
    type Output = Du;
    #[inline(always)]
    fn add(self, o: Du) -> Du {
        Du { v: self.v + o.v, d: self.d + o.d }
    }
}
impl Sub for Du {
    type Output = Du;
    #[inline(always)]
    fn sub(self, o: Du) -> Du {
        Du { v: self.v - o.v, d: self.d - o.d }
    }
}
impl Neg for Du {
    type Output = Du;
    #[inline(always)]
    fn neg(self) -> Du {
        Du { v: -self.v, d: -self.d }
    }
}
impl Mul for Du {
    type Output = Du;
    #[inline(always)]
    fn mul(self, o: Du) -> Du {
        let d = DotAcc::zero().add_prod(self.v, o.d).add_prod(self.d, o.v).finish();
        Du { v: self.v * o.v, d }
    }
}
impl Div for Du {
    type Output = Du;
    #[inline(always)]
    fn div(self, o: Du) -> Du {
        let q = self.v / o.v;
        Du { v: q, d: (self.d - q * o.d) / o.v }
    }
}

impl Lane for Du {
    type Acc = (DotAcc, DotAcc);
    const TANGENT: bool = true;
    #[inline(always)]
    fn zero() -> Self {
        Du { v: DV::zero(), d: DV::zero() }
    }
    #[inline(always)]
    fn cst(c: DV) -> Self {
        Du { v: c, d: DV::zero() }
    }
    #[inline(always)]
    fn mulc(self, c: DV) -> Self {
        Du { v: self.v * c, d: self.d * c }
    }
    #[inline(always)]
    fn mulf(self, s: f64) -> Self {
        Du { v: self.v.mul_f(s), d: self.d.mul_f(s) }
    }
    #[inline(always)]
    fn addc(self, c: DV) -> Self {
        Du { v: self.v + c, d: self.d }
    }
    #[inline(always)]
    fn bcast(self, j: usize) -> Self {
        Du { v: self.v.bcast(j), d: self.d.bcast(j) }
    }
    #[inline(always)]
    fn val(self) -> DV {
        self.v
    }
    #[inline(always)]
    fn tan(self) -> DV {
        self.d
    }
    #[inline(always)]
    fn from_vt(v: DV, t: DV) -> Self {
        Du { v, d: t }
    }
    #[inline(always)]
    fn acc0() -> Self::Acc {
        (DotAcc::zero(), DotAcc::zero())
    }
    #[inline(always)]
    fn acc(a: Self::Acc, x: Self, y: Self) -> Self::Acc {
        (a.0.add_prod(x.v, y.v), a.1.add_prod(x.v, y.d).add_prod(x.d, y.v))
    }
    #[inline(always)]
    fn accc(a: Self::Acc, x: Self, c: DV) -> Self::Acc {
        (a.0.add_prod(x.v, c), a.1.add_prod(x.d, c))
    }
    #[inline(always)]
    fn acc1(a: Self::Acc, x: Self) -> Self::Acc {
        (a.0.add(x.v), a.1.add(x.d))
    }
    #[inline(always)]
    fn fin(a: Self::Acc) -> Self {
        Du { v: a.0.finish(), d: a.1.finish() }
    }
}
