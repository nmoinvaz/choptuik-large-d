//! ell_ddm.Op: X @ W for a fixed double-double operator W, by Ozaki's error-free splitting into f64 products (Ozaki,
//! Ogita, Oishi and Rump, Numer. Algorithms 59, 95 (2012)). Both factors are split into NSLICE slices whose products,
//! summed over the inner dimension n, are exact in f64, so the f64 products (matrixmultiply) give the same bits in any
//! summation order. The levels are then combined as ell_ddm does: level 0 exact, level 1 by a two-sum, the levels >= 2
//! summed in f64 in ell_ddm's order and added last.
use crate::dd::{add, add_d, exp2i, two_sum, A};
use rayon::prelude::*;

pub const NSLICE: usize = 5;

/// a fixed operator W (n x m), held as ell_ddm.Op's column-scaled slices
pub struct Oz {
    pub n: usize,
    pub m: usize,
    /// NSLICE slices, each n x m row-major
    pub sl: Vec<Vec<f64>>,
    rho: i64,
}

/// ell_ddm._slices' rho: slices of ceil((53 + log2 n)/2) bits below their scale
fn rho_of(n: usize) -> i64 {
    ((53.0 + (n.max(2) as f64).log2()) / 2.0).ceil() as i64
}

/// the rows of the f64 products of this size and up are split over the thread pool
const PAR_FLOPS: usize = 1 << 21;

impl Oz {
    pub fn new(n: usize, m: usize, sl: Vec<Vec<f64>>) -> Oz {
        assert_eq!(sl.len(), NSLICE);
        for s in &sl {
            assert_eq!(s.len(), n * m);
        }
        Oz { n, m, sl, rho: rho_of(n) }
    }

    /// X @ W for X (r x n)
    pub fn rmul(&self, x: &A) -> A {
        assert_eq!(x.c, self.n, "Oz::rmul: X has {} columns, W {} rows", x.c, self.n);
        let (r, n, m) = (x.r, self.n, self.m);
        let mut out = A::zeros(r, m);
        if r == 0 {
            return out;
        }
        let nt = rayon::current_num_threads();
        if r * n * m < PAR_FLOPS || nt < 2 || r < 2 {
            self.rmul_rows(&x.h, &x.l, r, &mut out.h, &mut out.l);
            return out;
        }
        let g = r.div_ceil(2 * nt).max(4);
        out.h.par_chunks_mut(g * m).zip(out.l.par_chunks_mut(g * m)).enumerate().for_each(|(k, (oh, ol))| {
            let i0 = k * g;
            let rr = oh.len() / m;
            self.rmul_rows(&x.h[i0 * n..(i0 + rr) * n], &x.l[i0 * n..(i0 + rr) * n], rr, oh, ol);
        });
        out
    }

    fn rmul_rows(&self, xh: &[f64], xl: &[f64], r: usize, oh: &mut [f64], ol: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        // ell_ddm._slices along rows (axis=-1): each row's scale from its own largest |entry|
        let mut xs = vec![0.0; NSLICE * r * n];
        let mut rh = xh.to_vec();
        let mut rl = xl.to_vec();
        for i in 0..r {
            let rhi = &mut rh[i * n..(i + 1) * n];
            let rli = &mut rl[i * n..(i + 1) * n];
            for p in 0..NSLICE {
                let mx = rhi.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                let tau = if mx > 0.0 { mx.log2().ceil() } else { 0.0 };
                let sig = 0.75 * exp2i(self.rho + tau as i64);
                let o = &mut xs[(p * r + i) * n..(p * r + i + 1) * n];
                for j in 0..n {
                    let s = (rhi[j] + sig) - sig;
                    o[j] = s;
                    let d = rhi[j] - s;
                    let (a, b) = two_sum(d, rli[j]);
                    rhi[j] = a;
                    rli[j] = b;
                }
            }
        }
        // q[s] = [xs_0; ...; xs_{NSLICE-1-s}] @ sl[s]: the products P_ps = xs_p @ sl_s for p + s < NSLICE
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(NSLICE);
        for s in 0..NSLICE {
            let rows = (NSLICE - s) * r;
            let mut c = vec![0.0; rows * m];
            unsafe {
                matrixmultiply::dgemm(
                    rows,
                    n,
                    m,
                    1.0,
                    xs.as_ptr(),
                    n as isize,
                    1,
                    self.sl[s].as_ptr(),
                    m as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    m as isize,
                    1,
                );
            }
            q.push(c);
        }
        let pq = |p: usize, s: usize, k: usize| q[s][p * r * m + k];
        for k in 0..r * m {
            // level 0, then level 1 by a two-sum
            let (s1, e1) = two_sum(pq(0, 1, k), pq(1, 0, k));
            let (ah, al) = add(pq(0, 0, k), 0.0, s1, e1);
            // levels >= 2 summed in f64 in ell_ddm's loop order
            let mut rest = 0.0;
            let mut first = true;
            for p in 0..NSLICE {
                for s in 2usize.saturating_sub(p)..NSLICE - p {
                    let v = pq(p, s, k);
                    if first {
                        rest = v;
                        first = false;
                    } else {
                        rest += v;
                    }
                }
            }
            let (h, l) = add_d(ah, al, rest);
            oh[k] = h;
            ol[k] = l;
        }
    }
}
