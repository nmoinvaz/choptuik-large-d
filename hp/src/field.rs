//! A tau field on the half grid: M = 8 B collocation points theta_j = 2 pi j / (2M), j < M, stored as B lanes.
//! p, m are odd under theta -> theta + pi and a, c even, so the other half period is implied and pointwise products
//! keep the right parity. d/dtheta of an odd field is the folded spectral matrix (`DMat`).
use crate::dd::DD;
use crate::dv::DV;
use crate::lane::Lane;
use crate::v8::V8;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Fl<T: Lane, const B: usize>(pub [T; B]);

pub type State<T, const B: usize> = [Fl<T, B>; 4];

macro_rules! fl_binop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl<T: Lane, const B: usize> $tr for Fl<T, B> {
            type Output = Fl<T, B>;
            #[inline(always)]
            fn $f(self, o: Fl<T, B>) -> Fl<T, B> {
                let mut r = self.0;
                for b in 0..B {
                    r[b] = self.0[b] $op o.0[b];
                }
                Fl(r)
            }
        }
    };
}
fl_binop!(Add, add, +);
fl_binop!(Sub, sub, -);
fl_binop!(Mul, mul, *);
fl_binop!(Div, div, /);

impl<T: Lane, const B: usize> Neg for Fl<T, B> {
    type Output = Fl<T, B>;
    #[inline(always)]
    fn neg(self) -> Fl<T, B> {
        let mut r = self.0;
        for b in 0..B {
            r[b] = -self.0[b];
        }
        Fl(r)
    }
}

impl<T: Lane, const B: usize> Fl<T, B> {
    #[inline(always)]
    pub fn zero() -> Self {
        Fl([T::zero(); B])
    }
    #[inline(always)]
    pub fn splat(x: DD) -> Self {
        Fl([T::splat(x); B])
    }
    #[inline(always)]
    pub fn cst(c: &Fl<DV, B>) -> Self {
        let mut r = [T::zero(); B];
        for b in 0..B {
            r[b] = T::cst(c.0[b]);
        }
        Fl(r)
    }
    #[inline(always)]
    pub fn muls(self, s: DD) -> Self {
        let sv = DV::splat(s);
        let mut r = self.0;
        for b in 0..B {
            r[b] = self.0[b].mulc(sv);
        }
        Fl(r)
    }
    #[inline(always)]
    pub fn mulf(self, s: f64) -> Self {
        let mut r = self.0;
        for b in 0..B {
            r[b] = self.0[b].mulf(s);
        }
        Fl(r)
    }
    /// times a power of two, every part exact (multidouble's Lane::mul_pwr2): bitwise `mulf(s)` for such s
    #[inline(always)]
    pub fn mul_pwr2(self, s: f64) -> Self {
        let mut r = self.0;
        for b in 0..B {
            r[b] = self.0[b].mul_pwr2(s);
        }
        Fl(r)
    }
    /// times a lane-constant scalar of the generic type (e.g. omega, which carries a tangent)
    #[inline(always)]
    pub fn mul_t(self, s: T) -> Self {
        let mut r = self.0;
        for b in 0..B {
            r[b] = self.0[b] * s;
        }
        Fl(r)
    }
    #[inline(always)]
    pub fn adds(self, s: DD) -> Self {
        let sv = DV::splat(s);
        let mut r = self.0;
        for b in 0..B {
            r[b] = self.0[b].addc(sv);
        }
        Fl(r)
    }
    pub fn val(&self) -> Fl<DV, B> {
        let mut r = [DV::zero(); B];
        for b in 0..B {
            r[b] = self.0[b].val();
        }
        Fl(r)
    }
    pub fn tan(&self) -> Fl<DV, B> {
        let mut r = [DV::zero(); B];
        for b in 0..B {
            r[b] = self.0[b].tan();
        }
        Fl(r)
    }
    pub fn from_vt(v: &Fl<DV, B>, t: &Fl<DV, B>) -> Self {
        let mut r = [T::zero(); B];
        for b in 0..B {
            r[b] = T::from_vt(v.0[b], t.0[b]);
        }
        Fl(r)
    }
}

impl<const B: usize> Fl<DV, B> {
    /// from M = 8 B values (hi, lo)
    pub fn load(hi: &[f64], lo: &[f64]) -> Self {
        let mut r = [DV::zero(); B];
        for b in 0..B {
            r[b] = DV::load(&hi[8 * b..8 * b + 8], &lo[8 * b..8 * b + 8]);
        }
        Fl(r)
    }
    pub fn store(&self, hi: &mut [f64], lo: &mut [f64]) {
        for b in 0..B {
            self.0[b].store(&mut hi[8 * b..8 * b + 8], &mut lo[8 * b..8 * b + 8]);
        }
    }
    pub fn get(&self, j: usize) -> DD {
        self.0[j / 8].lane(j % 8)
    }
    /// max |hi| over the lanes
    pub fn amax(&self) -> f64 {
        let mut m = 0.0f64;
        for b in 0..B {
            m = m.max(self.0[b].hi.abs().hmax());
        }
        m
    }
}

/// Cauchy product coefficient sum_{j=0..k} x[j] y[k-j]
#[inline(always)]
pub fn cs<T: Lane, const B: usize>(x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize) -> Fl<T, B> {
    if T::VAL_F64 || T::TAN_F64 {
        return cs_il(x, y, k, 0, k, |_| 1.0, false);
    }
    let mut out = [T::zero(); B];
    for b in 0..B {
        let mut a = T::acc0();
        for j in 0..=k {
            a = T::acc(a, x[j].0[b], y[k - j].0[b]);
        }
        out[b] = T::fin(a);
    }
    Fl(out)
}

/// sum_{j=lo..=hi} x[j] y[k-j]
#[inline(always)]
pub fn cs_range<T: Lane, const B: usize>(x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize, lo: usize, hi: usize) -> Fl<T, B> {
    if T::VAL_F64 || T::TAN_F64 {
        return cs_il(x, y, k, lo, hi, |_| 1.0, false);
    }
    let mut out = [T::zero(); B];
    for b in 0..B {
        let mut a = T::acc0();
        for j in lo..=hi {
            a = T::acc(a, x[j].0[b], y[k - j].0[b]);
        }
        out[b] = T::fin(a);
    }
    Fl(out)
}

/// sum_{j=lo..=hi} (w(j) x[j]) y[k-j] with exact f64 weights w(j)
#[inline(always)]
pub fn cs_w<T: Lane, const B: usize>(
    x: &[Fl<T, B>],
    y: &[Fl<T, B>],
    k: usize,
    lo: usize,
    hi: usize,
    w: impl Fn(usize) -> f64,
) -> Fl<T, B> {
    if T::VAL_F64 || T::TAN_F64 {
        return cs_il(x, y, k, lo, hi, w, true);
    }
    let mut out = [T::zero(); B];
    for b in 0..B {
        let mut a = T::acc0();
        for j in lo..=hi {
            a = T::acc(a, x[j].0[b].mulf(w(j)), y[k - j].0[b]);
        }
        out[b] = T::fin(a);
    }
    Fl(out)
}

/// sum_{j=lo..=hi} (w(j) x[j]) y[k-j] (weights only if weighted) with the B lane vectors' accumulations
/// interleaved, for the f64 lanes, whose single accumulation chains are latency-bound; each lane vector still sums its
/// terms in the order j = lo..=hi.
#[inline(always)]
fn cs_il<T: Lane, const B: usize>(x: &[Fl<T, B>], y: &[Fl<T, B>], k: usize, lo: usize, hi: usize, w: impl Fn(usize) -> f64, weighted: bool) -> Fl<T, B> {
    let mut a = [T::acc0(); B];
    for j in lo..=hi {
        let (xj, yj) = (&x[j].0, &y[k - j].0);
        if weighted {
            let wj = w(j);
            for b in 0..B {
                a[b] = T::acc(a[b], xj[b].mulf(wj), yj[b]);
            }
        } else {
            for b in 0..B {
                a[b] = T::acc(a[b], xj[b], yj[b]);
            }
        }
    }
    Fl(a.map(T::fin))
}

/// Horner evaluation sum_k s[k] h^k
#[inline(always)]
pub fn horner<T: Lane, const B: usize>(s: &[Fl<T, B>], h: DD) -> Fl<T, B> {
    let mut out = Fl::<T, B>::zero();
    for k in (0..s.len()).rev() {
        out = out.muls(h) + s[k];
    }
    out
}

/// Constant M x M double-double matrix stored by columns (column j as a field over the rows), with a copy of the
/// columns' hi parts for the f64 lanes.
#[derive(Clone)]
pub struct DMat<const B: usize> {
    pub cols: Vec<Fl<DV, B>>,
    /// cols64[j][b] = cols[j].0[b].hi
    pub cols64: Vec<[V8; B]>,
}

impl<const B: usize> DMat<B> {
    /// from row-major (M, M) hi / lo arrays
    pub fn from_rows(hi: &[f64], lo: &[f64]) -> Self {
        let m = 8 * B;
        let mut cols = Vec::with_capacity(m);
        for j in 0..m {
            let ch: Vec<f64> = (0..m).map(|i| hi[i * m + j]).collect();
            let cl: Vec<f64> = (0..m).map(|i| lo[i * m + j]).collect();
            cols.push(Fl::<DV, B>::load(&ch, &cl));
        }
        let cols64 = cols.iter().map(|c| c.0.map(|v| v.hi)).collect();
        DMat { cols, cols64 }
    }

    /// out = D f (in the lane type's precision: `Lane::dmat_apply`)
    #[inline(always)]
    pub fn apply<T: Lane>(&self, f: &Fl<T, B>) -> Fl<T, B> {
        T::dmat_apply(self, f)
    }

    /// out = D f with the double-double columns (accumulated by `Lane::accc`)
    #[inline(always)]
    pub fn apply_dd<T: Lane>(&self, f: &Fl<T, B>) -> Fl<T, B> {
        let mut acc = [T::acc0(); B];
        for jb in 0..B {
            for jl in 0..8 {
                let s = f.0[jb].bcast(jl);
                let col = &self.cols[8 * jb + jl];
                for b in 0..B {
                    acc[b] = T::accc(acc[b], s, col.0[b]);
                }
            }
        }
        let mut out = [T::zero(); B];
        for b in 0..B {
            out[b] = T::fin(acc[b]);
        }
        Fl(out)
    }
}

/// Solves of a `multidouble::linalg::DDLu` for an M-point field given as B lane vectors (M = n): `solve_lanes` in
/// double-double, `solve_lanes_hi` in plain f64 with the hi parts of the factors (`DDLu::solve_hi`; lo parts out zero).
pub trait FieldSolve {
    fn solve_lanes<const B: usize>(&self, f: &Fl<DV, B>) -> Fl<DV, B>;
    fn solve_lanes_hi<const B: usize>(&self, f: &Fl<DV, B>) -> Fl<DV, B>;
}

impl FieldSolve for multidouble::linalg::DDLu {
    fn solve_lanes<const B: usize>(&self, f: &Fl<DV, B>) -> Fl<DV, B> {
        let m = 8 * B;
        let mut h = vec![0.0; m];
        let mut l = vec![0.0; m];
        f.store(&mut h, &mut l);
        let mut b: Vec<DD> = (0..m).map(|i| DD { hi: h[i], lo: l[i] }).collect();
        self.solve(&mut b);
        for i in 0..m {
            h[i] = b[i].hi;
            l[i] = b[i].lo;
        }
        Fl::<DV, B>::load(&h, &l)
    }

    fn solve_lanes_hi<const B: usize>(&self, f: &Fl<DV, B>) -> Fl<DV, B> {
        let m = 8 * B;
        let mut h = vec![0.0; m];
        let mut l = vec![0.0; m];
        f.store(&mut h, &mut l);
        self.solve_hi(&mut h);
        l.iter_mut().for_each(|v| *v = 0.0);
        Fl::<DV, B>::load(&h, &l)
    }
}
