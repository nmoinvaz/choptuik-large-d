//! ell_ddm.DT: a double-double value (1 x c) and optional tangents (rows x c, one row per input direction), with
//! ell_ddm's operator semantics transcribed one for one (which operand's tangent is formed first, a float operand by
//! mul_d / add_d, division by a float as division by the double-double (x, 0), the None tangents).
use crate::dd::{self, bin, mul_dcol, un, vstack, A};
use crate::oz::Oz;

#[derive(Clone, Debug)]
pub struct DT {
    pub v: A,
    pub t: Option<A>,
}

impl DT {
    pub fn val(v: A) -> DT {
        DT { v, t: None }
    }
}

impl AsRef<DT> for DT {
    fn as_ref(&self) -> &DT {
        self
    }
}

/// x + y
pub fn add(x: &DT, y: &DT) -> DT {
    let t = match (&x.t, &y.t) {
        (Some(a), Some(b)) => Some(bin(&dd::Add, a, b)),
        (Some(a), None) => Some(a.clone()),
        (None, Some(b)) => Some(b.clone()),
        (None, None) => None,
    };
    DT { v: bin(&dd::Add, &x.v, &y.v), t }
}

/// -x
pub fn neg(x: &DT) -> DT {
    DT { v: un(&dd::Neg, &x.v), t: x.t.as_ref().map(|t| un(&dd::Neg, t)) }
}

/// x - y = x + (-y)
pub fn sub(x: &DT, y: &DT) -> DT {
    let t = match (&x.t, &y.t) {
        (Some(a), Some(b)) => Some(bin(&dd::Sub, a, b)),
        (Some(a), None) => Some(a.clone()),
        (None, Some(b)) => Some(un(&dd::Neg, b)),
        (None, None) => None,
    };
    DT { v: bin(&dd::Sub, &x.v, &y.v), t }
}

/// x + f (f a double)
pub fn addf(x: &DT, f: f64) -> DT {
    DT { v: un(&dd::AddD(f), &x.v), t: x.t.clone() }
}

/// x - f = x + (-f)
pub fn subf(x: &DT, f: f64) -> DT {
    addf(x, -f)
}

/// f - x = (-x) + f
pub fn rsubf(f: f64, x: &DT) -> DT {
    DT { v: un(&dd::RsubD(f), &x.v), t: x.t.as_ref().map(|t| un(&dd::Neg, t)) }
}

/// x * y: value, then x.v * y.t, then x.t * y.v, summed
pub fn mul(x: &DT, y: &DT) -> DT {
    let v = bin(&dd::Mul, &x.v, &y.v);
    let a = y.t.as_ref().map(|t| bin(&dd::Mul, &x.v, t));
    let b = x.t.as_ref().map(|t| bin(&dd::Mul, t, &y.v));
    let t = match (a, b) {
        (Some(a), Some(b)) => Some(bin(&dd::Add, &a, &b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    DT { v, t }
}

/// x * f (f a double: mul_d)
pub fn mulf(x: &DT, f: f64) -> DT {
    DT { v: un(&dd::MulD(f), &x.v), t: x.t.as_ref().map(|t| un(&dd::MulD(f), t)) }
}

/// x * (h, l), a double-double scalar
pub fn mulr(x: &DT, h: f64, l: f64) -> DT {
    DT { v: un(&dd::MulR(h, l), &x.v), t: x.t.as_ref().map(|t| un(&dd::MulR(h, l), t)) }
}

/// x / y: q = x.v / y.v; tangents (x.t - q y.t) / y.v
pub fn div(x: &DT, y: &DT) -> DT {
    let q = bin(&dd::Div, &x.v, &y.v);
    let mut t = x.t.clone();
    if let Some(yt) = &y.t {
        let qt = bin(&dd::Mul, &q, yt);
        t = Some(match t {
            None => un(&dd::Neg, &qt),
            Some(xt) => bin(&dd::Sub, &xt, &qt),
        });
    }
    let t = t.map(|t| bin(&dd::Div, &t, &y.v));
    DT { v: q, t }
}

/// x / f, as x / R(f)
pub fn divf(x: &DT, f: f64) -> DT {
    DT { v: un(&dd::DivR(f), &x.v), t: x.t.as_ref().map(|t| un(&dd::DivR(f), t)) }
}

/// x * d with one double per column (mul_d), value and tangents
pub fn mul_dcol_dt(x: &DT, d: &[f64]) -> DT {
    DT { v: mul_dcol(&x.v, d), t: x.t.as_ref().map(|t| mul_dcol(t, d)) }
}

/// op applied to every value and tangent of xs (one Ozaki product: rows are independent, so stacking them gives
/// the bits of separate products)
pub fn lin_many(op: &Oz, xs: &[&DT]) -> Vec<DT> {
    let mut parts: Vec<&A> = Vec::new();
    for x in xs {
        parts.push(&x.v);
        if let Some(t) = &x.t {
            parts.push(t);
        }
    }
    let y = op.rmul(&vstack(&parts));
    let mut out = Vec::with_capacity(xs.len());
    let mut i = 0;
    for x in xs {
        let v = y.rows(i, i + x.v.r);
        i += x.v.r;
        let t = x.t.as_ref().map(|t| {
            let a = y.rows(i, i + t.r);
            i += t.r;
            a
        });
        out.push(DT { v, t });
    }
    out
}

/// ell_ddm._cs: sum over j of X[j] Y[k - j], from 0.0 (so the first term passes through add_d(., 0.0))
pub fn cs<X: AsRef<DT>, Y: AsRef<DT>>(x: &[X], y: &[Y], k: usize) -> DT {
    let mut s: Option<DT> = None;
    for j in 0..=k {
        let p = mul(x[j].as_ref(), y[k - j].as_ref());
        s = Some(match s {
            None => addf(&p, 0.0),
            Some(s) => add(&s, &p),
        });
    }
    s.unwrap()
}

/// Python's sum() of DT terms (from 0: the first term passes through add_d(., 0.0))
pub fn pysum<I: Iterator<Item = DT>>(it: I) -> DT {
    let mut s: Option<DT> = None;
    for p in it {
        s = Some(match s {
            None => addf(&p, 0.0),
            Some(s) => add(&s, &p),
        });
    }
    s.expect("pysum of nothing")
}
