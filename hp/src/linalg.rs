//! Dense double-double LU with partial pivoting (LAPACK getf2 order), column-major with the leading dimension
//! padded to a multiple of 8 so the trailing update runs on 8-row lane vectors. Used for the per-order periodic
//! solves of the horizon series (M x M) and for the Newton system (L x L, columns updated in parallel).
use crate::dd::{two_prod, two_sum, DD};
use crate::dv::{self, DV};
use crate::v8::V8;
use rayon::prelude::*;

#[derive(Clone)]
pub struct DDLu {
    pub n: usize,
    pub ld: usize,
    pub hi: Vec<f64>,
    pub lo: Vec<f64>,
    pub perm: Vec<usize>,
}

#[inline(always)]
pub fn ld_for(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// a - l t on 8 rows (l, a lane vectors; t scalar), as one DD product added with an error-free sum
#[inline(always)]
fn axpy8(a: DV, l: DV, nt: DV) -> DV {
    let (p, pe) = dv::two_prod(l.hi, nt.hi);
    let pe = l.hi.fma(nt.lo, pe);
    let pe = l.lo.fma(nt.hi, pe);
    let (s, t) = dv::two_sum(a.hi, p);
    let e = a.lo.add(t.add(pe));
    let (h, lo) = dv::two_sum(s, e);
    DV { hi: h, lo }
}

#[inline(always)]
fn dd_axpy(a: DD, l: DD, nt: DD) -> DD {
    let (p, pe) = two_prod(l.hi, nt.hi);
    let pe = l.hi.mul_add(nt.lo, pe);
    let pe = l.lo.mul_add(nt.hi, pe);
    let (s, t) = two_sum(a.hi, p);
    let e = a.lo + (t + pe);
    let (h, lo) = two_sum(s, e);
    DD { hi: h, lo }
}

impl DDLu {
    /// Factor the n x n matrix given column-major with leading dimension ld (ld % 8 == 0, padding rows zero).
    /// par: update the trailing columns in parallel (rayon's current pool).
    pub fn factor(n: usize, ld: usize, mut hi: Vec<f64>, mut lo: Vec<f64>, par: bool) -> Result<DDLu, String> {
        assert!(ld % 8 == 0 && ld >= n && hi.len() >= ld * n && lo.len() >= ld * n);
        let mut perm = vec![0usize; n];
        let mut mh = vec![0.0f64; ld];
        let mut ml = vec![0.0f64; ld];
        for k in 0..n {
            let ck = k * ld;
            let mut p = k;
            let mut best = hi[ck + k].abs();
            for i in k + 1..n {
                let v = hi[ck + i].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best == 0.0 || !best.is_finite() {
                return Err(format!("singular or non-finite pivot at column {k} ({best})"));
            }
            perm[k] = p;
            if p != k {
                for j in 0..n {
                    hi.swap(j * ld + k, j * ld + p);
                    lo.swap(j * ld + k, j * ld + p);
                }
            }
            let inv = DD::ONE / DD { hi: hi[ck + k], lo: lo[ck + k] };
            for i in k + 1..n {
                let v = DD { hi: hi[ck + i], lo: lo[ck + i] } * inv;
                hi[ck + i] = v.hi;
                lo[ck + i] = v.lo;
            }
            if k + 1 == n {
                break;
            }
            let r0 = 8 * ((k + 1) / 8);
            for i in r0..ld {
                if i > k && i < n {
                    mh[i] = hi[ck + i];
                    ml[i] = lo[ck + i];
                } else {
                    mh[i] = 0.0;
                    ml[i] = 0.0;
                }
            }
            let (_, rh) = hi.split_at_mut((k + 1) * ld);
            let (_, rl) = lo.split_at_mut((k + 1) * ld);
            let ncols = n - k - 1;
            let upd = |(ch, cl): (&mut [f64], &mut [f64])| {
                let nt = DV::splat(DD { hi: -ch[k], lo: -cl[k] });
                let mut i = r0;
                while i < ld {
                    let a = DV::load(&ch[i..i + 8], &cl[i..i + 8]);
                    let l = DV::load(&mh[i..i + 8], &ml[i..i + 8]);
                    axpy8(a, l, nt).store(&mut ch[i..i + 8], &mut cl[i..i + 8]);
                    i += 8;
                }
            };
            if par && ncols * (ld - r0) > 32768 {
                rh[..ncols * ld].par_chunks_mut(ld).zip(rl[..ncols * ld].par_chunks_mut(ld)).for_each(upd);
            } else {
                rh[..ncols * ld].chunks_mut(ld).zip(rl[..ncols * ld].chunks_mut(ld)).for_each(upd);
            }
        }
        Ok(DDLu { n, ld, hi, lo, perm })
    }

    #[inline(always)]
    fn at(&self, i: usize, j: usize) -> DD {
        DD { hi: self.hi[j * self.ld + i], lo: self.lo[j * self.ld + i] }
    }

    /// Solve A x = b in place (b: n values).
    pub fn solve(&self, b: &mut [DD]) {
        let n = self.n;
        for k in 0..n {
            let p = self.perm[k];
            if p != k {
                b.swap(k, p);
            }
        }
        for k in 0..n {
            let nt = -b[k];
            if nt.hi == 0.0 {
                continue;
            }
            for i in k + 1..n {
                b[i] = dd_axpy(b[i], self.at(i, k), nt);
            }
        }
        for k in (0..n).rev() {
            b[k] = b[k] / self.at(k, k);
            let nt = -b[k];
            for i in 0..k {
                b[i] = dd_axpy(b[i], self.at(i, k), nt);
            }
        }
    }

    /// Solve A^T y = b in place. With P A = L U (P the recorded row swaps): U^T w = b, L^T v = w, y = P^T v.
    pub fn solve_t(&self, b: &mut [DD]) {
        let n = self.n;
        for k in 0..n {
            let mut acc = b[k];
            for i in 0..k {
                acc = dd_axpy(acc, self.at(i, k), -b[i]);
            }
            b[k] = acc / self.at(k, k);
        }
        for k in (0..n).rev() {
            let mut acc = b[k];
            for i in k + 1..n {
                acc = dd_axpy(acc, self.at(i, k), -b[i]);
            }
            b[k] = acc;
        }
        for k in (0..n).rev() {
            let p = self.perm[k];
            if p != k {
                b.swap(k, p);
            }
        }
    }

    /// Solve for an M-point field given as B lane vectors (M = n).
    pub fn solve_lanes<const B: usize>(&self, f: &crate::field::Fl<DV, B>) -> crate::field::Fl<DV, B> {
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
        crate::field::Fl::<DV, B>::load(&h, &l)
    }

    /// min and max |u_kk| (hi parts)
    pub fn pivot_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for k in 0..self.n {
            let v = self.hi[k * self.ld + k].abs();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }
}

/// Gram matrix G = J^T J and g = J^T b for a column-major J (n x n, leading dimension ld): G column-major with
/// leading dimension ld (only what LU needs; G is symmetric and filled completely).
pub fn gram(n: usize, ld: usize, jh: &[f64], jl: &[f64], bh: &[f64], bl: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<DD>) {
    let dot = |a: usize, bh2: &[f64], bl2: &[f64]| -> DD {
        let mut acc = dv::DotAcc::zero();
        let mut i = 0;
        while i < ld {
            let x = DV::load(&jh[a * ld + i..a * ld + i + 8], &jl[a * ld + i..a * ld + i + 8]);
            let y = DV::load(&bh2[i..i + 8], &bl2[i..i + 8]);
            acc = acc.add_prod(x, y);
            i += 8;
        }
        let v = acc.finish();
        let (h, l) = (v.hi.to_array(), v.lo.to_array());
        let mut s = DD::ZERO;
        for k in 0..8 {
            s = s + DD { hi: h[k], lo: l[k] };
        }
        s
    };
    let cols: Vec<Vec<DD>> = (0..n)
        .into_par_iter()
        .map(|c| (0..=c).map(|r| dot(r, &jh[c * ld..c * ld + ld], &jl[c * ld..c * ld + ld])).collect())
        .collect();
    let mut gh = vec![0.0; ld * n];
    let mut gl = vec![0.0; ld * n];
    for c in 0..n {
        for r in 0..=c {
            let v = cols[c][r];
            gh[c * ld + r] = v.hi;
            gl[c * ld + r] = v.lo;
            gh[r * ld + c] = v.hi;
            gl[r * ld + c] = v.lo;
        }
    }
    let mut bpad_h = vec![0.0; ld];
    let mut bpad_l = vec![0.0; ld];
    bpad_h[..n].copy_from_slice(&bh[..n]);
    bpad_l[..n].copy_from_slice(&bl[..n]);
    let g: Vec<DD> = (0..n).into_par_iter().map(|c| dot(c, &bpad_h, &bpad_l)).collect();
    (gh, gl, g)
}

#[allow(dead_code)]
fn _unused(_: V8) {}
