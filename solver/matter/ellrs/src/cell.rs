//! ell_ddm._rk4's combinations of the stage derivatives in one pass each (the same operations on the same values as
//! the DT arrays', so the same bits).
use crate::dd::{self, Ln, A};
use crate::dt::DT;
use multidouble::native::V8;

/// s + k a (ell_ddm's _axpy: add(s, mul_d(k, a))), value and tangents
pub fn axpy(s: &DT, k: &DT, a: f64) -> DT {
    let op = AxpyD(a);
    DT {
        v: dd::bin(&op, &s.v, &k.v),
        t: Some(dd::bin(&op, s.t.as_ref().unwrap(), k.t.as_ref().unwrap())),
    }
}

struct AxpyD(f64);
impl dd::Op2 for AxpyD {
    #[inline(always)]
    fn f<T: Ln>(&self, sh: T, sl: T, kh: T, kl: T) -> (T, T) {
        let (mh, ml) = dd::mul_d(kh, kl, T::sp(self.0));
        dd::add(sh, sl, mh, ml)
    }
}

/// ell_ddm._rk4's update s + (((k1 + k2 2) + k3 2) + k4) h6, value and tangents, in one pass
pub fn rk_update(s: &DT, k: [&DT; 4], h6h: f64, h6l: f64) -> DT {
    let one = |s: &A, k1: &A, k2: &A, k3: &A, k4: &A| -> A {
        assert!([k1, k2, k3, k4].iter().all(|x| x.r == s.r && x.c == s.c), "rk_update: shapes");
        let mut o = A::uninit(s.r, s.c);
        let len = s.h.len();
        let n8 = len / 8 * 8;
        let mut j = 0;
        while j < n8 {
            let (h, l) = upd(
                [V8::ld(&s.h[j..]), V8::ld(&k1.h[j..]), V8::ld(&k2.h[j..]), V8::ld(&k3.h[j..]), V8::ld(&k4.h[j..])],
                [V8::ld(&s.l[j..]), V8::ld(&k1.l[j..]), V8::ld(&k2.l[j..]), V8::ld(&k3.l[j..]), V8::ld(&k4.l[j..])],
                h6h,
                h6l,
            );
            h.st(&mut o.h[j..]);
            l.st(&mut o.l[j..]);
            j += 8;
        }
        for j in n8..len {
            let (h, l) = upd([s.h[j], k1.h[j], k2.h[j], k3.h[j], k4.h[j]], [s.l[j], k1.l[j], k2.l[j], k3.l[j], k4.l[j]], h6h, h6l);
            o.h[j] = h;
            o.l[j] = l;
        }
        o
    };
    let tk: Vec<&A> = k.iter().map(|d| d.t.as_ref().unwrap()).collect();
    DT {
        v: one(&s.v, &k[0].v, &k[1].v, &k[2].v, &k[3].v),
        t: Some(one(s.t.as_ref().unwrap(), tk[0], tk[1], tk[2], tk[3])),
    }
}

#[inline(always)]
fn upd<T: Ln>(h: [T; 5], l: [T; 5], h6h: f64, h6l: f64) -> (T, T) {
    let (bh, bl) = dd::mul_d(h[2], l[2], T::sp(2.0));
    let (ah, al) = dd::add(h[1], l[1], bh, bl);
    let (ch, cl) = dd::mul_d(h[3], l[3], T::sp(2.0));
    let (ah, al) = dd::add(ah, al, ch, cl);
    let (ah, al) = dd::add(ah, al, h[4], l[4]);
    let (mh, ml) = dd::mul(ah, al, T::sp(h6h), T::sp(h6l));
    dd::add(h[0], l[0], mh, ml)
}
