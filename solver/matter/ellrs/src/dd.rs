//! Double-double arrays in ddnp.py's algorithms, so that every value is bitwise ddnp's: the accurate addition of Hida,
//! Li and Bailey's QD library (2001), the product with its two cross terms rounded apart and then summed, mul_d and
//! add_d, and the three-quotient division. The error-free transforms are Dekker's (Numer. Math. 18, 224 (1971)) and
//! Knuth's two-sum; the product's error comes from one fused multiply-subtract, which equals Dekker's split product
//! exactly barring overflow and underflow.
//!
//! The lanes are multidouble's 8-lane f64 vectors (`multidouble::native::V8`, the backend of the build's target
//! features) with an f64 tail per row. multidouble's own double-double product fuses the cross terms (one rounding
//! fewer than ddnp), so its `DV` would not give ddnp's bits; only its f64 lanes are used here.
use multidouble::native::V8;
use rayon::prelude::*;

/// f64 lanes: f64 itself (the tails) and V8
pub trait Ln: Copy {
    fn sp(x: f64) -> Self;
    fn ld(s: &[f64]) -> Self;
    fn st(self, s: &mut [f64]);
    fn a(self, o: Self) -> Self;
    fn s(self, o: Self) -> Self;
    fn m(self, o: Self) -> Self;
    fn d(self, o: Self) -> Self;
    /// self * b - c with one rounding
    fn fms(self, b: Self, c: Self) -> Self;
    fn ng(self) -> Self;
}

impl Ln for f64 {
    #[inline(always)]
    fn sp(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn ld(s: &[f64]) -> Self {
        s[0]
    }
    #[inline(always)]
    fn st(self, s: &mut [f64]) {
        s[0] = self
    }
    #[inline(always)]
    fn a(self, o: Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn s(self, o: Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn m(self, o: Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn d(self, o: Self) -> Self {
        self / o
    }
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self.mul_add(b, -c)
    }
    #[inline(always)]
    fn ng(self) -> Self {
        -self
    }
}

impl Ln for V8 {
    #[inline(always)]
    fn sp(x: f64) -> Self {
        V8::splat(x)
    }
    #[inline(always)]
    fn ld(s: &[f64]) -> Self {
        V8::load(s)
    }
    #[inline(always)]
    fn st(self, s: &mut [f64]) {
        self.store(s)
    }
    #[inline(always)]
    fn a(self, o: Self) -> Self {
        self.add(o)
    }
    #[inline(always)]
    fn s(self, o: Self) -> Self {
        self.sub(o)
    }
    #[inline(always)]
    fn m(self, o: Self) -> Self {
        self.mul(o)
    }
    #[inline(always)]
    fn d(self, o: Self) -> Self {
        self.div(o)
    }
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        V8::fms(self, b, c)
    }
    #[inline(always)]
    fn ng(self) -> Self {
        self.neg()
    }
}

#[inline(always)]
pub fn two_sum<T: Ln>(a: T, b: T) -> (T, T) {
    let s = a.a(b);
    let bb = s.s(a);
    (s, a.s(s.s(bb)).a(b.s(bb)))
}

/// ddnp.qts (quick two-sum)
#[inline(always)]
pub fn qts<T: Ln>(a: T, b: T) -> (T, T) {
    let s = a.a(b);
    (s, b.s(s.s(a)))
}

#[inline(always)]
pub fn two_prod<T: Ln>(a: T, b: T) -> (T, T) {
    let p = a.m(b);
    (p, a.fms(b, p))
}

/// ddnp.add
#[inline(always)]
pub fn add<T: Ln>(ah: T, al: T, bh: T, bl: T) -> (T, T) {
    let (s, e) = two_sum(ah, bh);
    let (t, f) = two_sum(al, bl);
    let e = e.a(t);
    let (s, e) = qts(s, e);
    let e = e.a(f);
    qts(s, e)
}

/// ddnp.mul: e + (a.h b.l + a.l b.h), each product rounded
#[inline(always)]
pub fn mul<T: Ln>(ah: T, al: T, bh: T, bl: T) -> (T, T) {
    let (p, e) = two_prod(ah, bh);
    let e = e.a(ah.m(bl).a(al.m(bh)));
    qts(p, e)
}

/// ddnp.mul_d (a double d)
#[inline(always)]
pub fn mul_d<T: Ln>(ah: T, al: T, d: T) -> (T, T) {
    let (p, e) = two_prod(ah, d);
    let e = e.a(al.m(d));
    qts(p, e)
}

/// ddnp.add_d (a double d)
#[inline(always)]
pub fn add_d<T: Ln>(ah: T, al: T, d: T) -> (T, T) {
    let (s, e) = two_sum(ah, d);
    let e = e.a(al);
    qts(s, e)
}

/// ddnp.div: three quotients by b.h, the remainders by mul_d and sub
#[inline(always)]
pub fn div<T: Ln>(ah: T, al: T, bh: T, bl: T) -> (T, T) {
    let q1 = ah.d(bh);
    let (mh, ml) = mul_d(bh, bl, q1);
    let (rh, rl) = add(ah, al, mh.ng(), ml.ng());
    let q2 = rh.d(bh);
    let (mh, ml) = mul_d(bh, bl, q2);
    let (rh, _) = add(rh, rl, mh.ng(), ml.ng());
    let q3 = rh.d(bh);
    let (q1, q2) = qts(q1, q2);
    add(q1, q2, q3, T::sp(0.0))
}

/// a double-double array, r x c row-major
#[derive(Clone, Debug)]
pub struct A {
    pub h: Vec<f64>,
    pub l: Vec<f64>,
    pub r: usize,
    pub c: usize,
}

/// arrays from this many entries up have their rows split over the thread pool
const PAR_MIN: usize = 1 << 15;

impl A {
    pub fn zeros(r: usize, c: usize) -> A {
        A { h: vec![0.0; r * c], l: vec![0.0; r * c], r, c }
    }
    pub fn from_h(h: Vec<f64>, r: usize, c: usize) -> A {
        assert_eq!(h.len(), r * c);
        A { l: vec![0.0; r * c], h, r, c }
    }
    pub fn rows(&self, i0: usize, i1: usize) -> A {
        let c = self.c;
        A { h: self.h[i0 * c..i1 * c].to_vec(), l: self.l[i0 * c..i1 * c].to_vec(), r: i1 - i0, c }
    }
    /// columns j0..j1 of every row
    pub fn cols(&self, j0: usize, j1: usize) -> A {
        let w = j1 - j0;
        let mut o = A::zeros(self.r, w);
        for i in 0..self.r {
            o.h[i * w..(i + 1) * w].copy_from_slice(&self.h[i * self.c + j0..i * self.c + j1]);
            o.l[i * w..(i + 1) * w].copy_from_slice(&self.l[i * self.c + j0..i * self.c + j1]);
        }
        o
    }
    pub fn maxabs_h(&self) -> f64 {
        self.h.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
    }
}

/// rows stacked (equal column counts)
pub fn vstack(parts: &[&A]) -> A {
    let c = parts[0].c;
    let mut h = Vec::new();
    let mut l = Vec::new();
    let mut r = 0;
    for p in parts {
        assert_eq!(p.c, c);
        h.extend_from_slice(&p.h);
        l.extend_from_slice(&p.l);
        r += p.r;
    }
    A { h, l, r, c }
}

/// columns concatenated (equal row counts)
pub fn hcat(parts: &[&A]) -> A {
    let r = parts[0].r;
    let c: usize = parts.iter().map(|p| p.c).sum();
    let mut o = A::zeros(r, c);
    for i in 0..r {
        let mut j0 = 0;
        for p in parts {
            assert_eq!(p.r, r);
            o.h[i * c + j0..i * c + j0 + p.c].copy_from_slice(&p.h[i * p.c..(i + 1) * p.c]);
            o.l[i * c + j0..i * c + j0 + p.c].copy_from_slice(&p.l[i * p.c..(i + 1) * p.c]);
            j0 += p.c;
        }
    }
    o
}

/// f(i0, i1, h, l) on groups of whole rows of (h, l), in parallel when the array is large
pub fn for_rows<F>(r: usize, c: usize, h: &mut [f64], l: &mut [f64], f: F)
where
    F: Fn(usize, usize, &mut [f64], &mut [f64]) + Sync,
{
    if r * c < PAR_MIN || r < 2 || c == 0 || crate::prof::seq() {
        f(0, r, h, l);
        return;
    }
    let g = (PAR_MIN / 4 / c).max(1);
    h.par_chunks_mut(g * c).zip(l.par_chunks_mut(g * c)).enumerate().for_each(|(k, (hh, ll))| {
        let i0 = k * g;
        f(i0, i0 + hh.len() / c, hh, ll)
    });
}

/// a binary double-double operation
pub trait Op2: Sync {
    fn f<T: Ln>(&self, ah: T, al: T, bh: T, bl: T) -> (T, T);
}
/// a unary one (with its double or double-double constants)
pub trait Op1: Sync {
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T);
}

pub struct Add;
pub struct Sub;
pub struct Mul;
pub struct Div;
impl Op2 for Add {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T, bh: T, bl: T) -> (T, T) {
        add(ah, al, bh, bl)
    }
}
/// ddnp.sub = add(a, neg(b)), neg exact
impl Op2 for Sub {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T, bh: T, bl: T) -> (T, T) {
        add(ah, al, bh.ng(), bl.ng())
    }
}
impl Op2 for Mul {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T, bh: T, bl: T) -> (T, T) {
        mul(ah, al, bh, bl)
    }
}
impl Op2 for Div {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T, bh: T, bl: T) -> (T, T) {
        div(ah, al, bh, bl)
    }
}

/// mul_d(a, d)
pub struct MulD(pub f64);
/// add_d(a, d)
pub struct AddD(pub f64);
/// add_d(neg(a), d): d - a as ell_ddm's DT.__rsub__
pub struct RsubD(pub f64);
/// div(a, R(x)), the divisor's low part +0
pub struct DivR(pub f64);
/// mul(a, R(h, l)) by one double-double
pub struct MulR(pub f64, pub f64);
pub struct Neg;
impl Op1 for MulD {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        mul_d(ah, al, T::sp(self.0))
    }
}
impl Op1 for AddD {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        add_d(ah, al, T::sp(self.0))
    }
}
impl Op1 for RsubD {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        add_d(ah.ng(), al.ng(), T::sp(self.0))
    }
}
impl Op1 for DivR {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        div(ah, al, T::sp(self.0), T::sp(0.0))
    }
}
impl Op1 for MulR {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        mul(ah, al, T::sp(self.0), T::sp(self.1))
    }
}
impl Op1 for Neg {
    #[inline(always)]
    fn f<T: Ln>(&self, ah: T, al: T) -> (T, T) {
        (ah.ng(), al.ng())
    }
}

#[inline(always)]
fn row2<O: Op2>(o: &O, ah: &[f64], al: &[f64], bh: &[f64], bl: &[f64], oh: &mut [f64], ol: &mut [f64]) {
    let c = oh.len();
    let n8 = c / 8 * 8;
    let mut j = 0;
    while j < n8 {
        let (h, l) = o.f(V8::ld(&ah[j..]), V8::ld(&al[j..]), V8::ld(&bh[j..]), V8::ld(&bl[j..]));
        h.st(&mut oh[j..]);
        l.st(&mut ol[j..]);
        j += 8;
    }
    for j in n8..c {
        let (h, l) = o.f(ah[j], al[j], bh[j], bl[j]);
        oh[j] = h;
        ol[j] = l;
    }
}

#[inline(always)]
fn row1<O: Op1>(o: &O, ah: &[f64], al: &[f64], oh: &mut [f64], ol: &mut [f64]) {
    let c = oh.len();
    let n8 = c / 8 * 8;
    let mut j = 0;
    while j < n8 {
        let (h, l) = o.f(V8::ld(&ah[j..]), V8::ld(&al[j..]));
        h.st(&mut oh[j..]);
        l.st(&mut ol[j..]);
        j += 8;
    }
    for j in n8..c {
        let (h, l) = o.f(ah[j], al[j]);
        oh[j] = h;
        ol[j] = l;
    }
}

/// o(a, b) elementwise; a (1 x c) or b (1 x c) is broadcast over the other's rows (numpy's broadcasting)
pub fn bin<O: Op2>(o: &O, a: &A, b: &A) -> A {
    assert_eq!(a.c, b.c, "bin: columns {} and {}", a.c, b.c);
    assert!(a.r == b.r || a.r == 1 || b.r == 1, "bin: rows {} and {}", a.r, b.r);
    let r = a.r.max(b.r);
    let c = a.c;
    let mut out = A::zeros(r, c);
    let (ab, bb) = (a.r == 1 && r > 1, b.r == 1 && r > 1);
    for_rows(r, c, &mut out.h, &mut out.l, |i0, i1, oh, ol| {
        for i in i0..i1 {
            let ia = if ab { 0 } else { i };
            let ib = if bb { 0 } else { i };
            let k = (i - i0) * c;
            row2(
                o,
                &a.h[ia * c..(ia + 1) * c],
                &a.l[ia * c..(ia + 1) * c],
                &b.h[ib * c..(ib + 1) * c],
                &b.l[ib * c..(ib + 1) * c],
                &mut oh[k..k + c],
                &mut ol[k..k + c],
            );
        }
    });
    out
}

/// o(a) elementwise
pub fn un<O: Op1>(o: &O, a: &A) -> A {
    let (r, c) = (a.r, a.c);
    let mut out = A::zeros(r, c);
    for_rows(r, c, &mut out.h, &mut out.l, |i0, i1, oh, ol| {
        row1(o, &a.h[i0 * c..i1 * c], &a.l[i0 * c..i1 * c], oh, ol);
    });
    out
}

/// mul_d(a, d) with one double per column (ddnp.mul_d broadcasting a (c,) array)
pub fn mul_dcol(a: &A, d: &[f64]) -> A {
    assert_eq!(d.len(), a.c);
    let (r, c) = (a.r, a.c);
    let mut out = A::zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            let (h, l) = mul_d(a.h[i * c + j], a.l[i * c + j], d[j]);
            out.h[i * c + j] = h;
            out.l[i * c + j] = l;
        }
    }
    out
}

/// 2^k exactly (0 below the subnormals, infinity above the range)
pub fn exp2i(k: i64) -> f64 {
    if k > 1023 {
        f64::INFINITY
    } else if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else if k >= -1074 {
        f64::from_bits(1u64 << (k + 1074))
    } else {
        0.0
    }
}
