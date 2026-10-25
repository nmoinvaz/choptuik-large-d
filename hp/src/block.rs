//! Block storage for the multiple-shooting Newton system: a staircase LU that keeps and factors only the nonzero node
//! blocks of the Jacobian, so the dense n x n matrix is never built (n = 4M(K+1) + 1; at M = 256 and K = 13 nodes a
//! dense f64 Jacobian would be 1.6 GB, 93% zeros).
//!
//! Structure. Residual block b (4M rows) depends on two neighbouring groups of unknowns and on Delta: block 0 on the
//! centre data (p1, c0) and node 0; block b = 1..K-1 on nodes b-1 and b (whichever way its segment runs); block K on
//! node K-1 and the horizon data (m_h, a_h). The phase row depends on p1 only. Delta enters every block, so the
//! factorization works on an augmented system with one copy delta_b of Delta per residual block and link rows
//! delta_{b-1} - delta_b = 0. With the column groups
//!     g_0 = (p1, c0),  g_t = (delta_{t-1}, node t-1) for t = 1..K,  g_{K+1} = (delta_K, m_h, a_h)
//! and the row groups (phase, block 0), (link, block 1), ..., (link, block K), row group s touches g_s and g_{s+1}
//! only. Gaussian elimination with partial pivoting then runs in steps: step s factors the columns of g_s, its pivot
//! candidates being the rows left over from step s-1 (all of them zero outside g_s) and row group s, and passes the
//! rows it does not pivot on, now holding the Schur complement on g_{s+1}, to step s+1; the last step is a square LU
//! on g_{K+1}. This is exactly partial pivoting on the augmented matrix in this row and column order (rows below a
//! step never hold the step's columns), so it has that algorithm's stability, and the fill stays inside the steps.
//! Stored per step: L and U11 (rows x pivot columns) and U12 (pivot rows x next group), f64.
//!
//! Solves are in double-double with the f64 factors, as the dense mixed-precision path does (DDLu with lo = 0).
//! With E the copy of Delta into every delta_b and S the selection of the non-link rows, J = S J_aug E, so
//! J^-1 = P J_aug^-1 S^T and J^-T = S J_aug^-T P^T for any P with P E = I; P reads Delta from delta_0.
//!
//! Each step's factorization is multidouble's `linalg::lu_rect_f64`: LAPACK's blocked right-looking getrf (Anderson et
//! al., LAPACK Users' Guide, 3rd ed., SIAM 1999) restricted to the first np columns of a rectangular block, as
//! `linalg::DDLu::factor_f64` does it for the square matrix; the solves round as its own solves do (`linalg::dd_axpy`).
use crate::dd::DD;
use crate::linalg::{dd_axpy, ld_for, lu_rect_f64};
use rayon::prelude::*;

/// A column group of the augmented system: a Delta copy first if `delta`, then z[zoff..zoff + zlen].
#[derive(Clone, Copy, Debug)]
pub struct Group {
    pub delta: bool,
    pub zoff: usize,
    pub zlen: usize,
}

impl Group {
    pub fn len(&self) -> usize {
        self.zlen + self.delta as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One elimination step: rows 0..nc carried from the previous step, row nc the extra row (phase or link), rows
/// nc + 1..nr the residual rows roff..roff + rlen; np pivot columns (this step's group), nrc right columns (the next).
struct Step {
    nc: usize,
    nr: usize,
    np: usize,
    nrc: usize,
    ld: usize,
    /// ld x np, column-major: U11 on and above the diagonal, the multipliers of L below (rows up to nr)
    l: Vec<f64>,
    /// np x nrc, column-major (leading dimension np): U12
    u12: Vec<f64>,
    perm: Vec<usize>,
    roff: usize,
    rlen: usize,
}

pub struct BlockLu {
    /// size of the original system (z)
    pub n: usize,
    groups: Vec<Group>,
    steps: Vec<Step>,
    /// index of the group whose Delta copy is read as Delta (P), and z's Delta index
    gdelta: usize,
    idelta: usize,
    /// the phase row's index in the original residual
    iphase: usize,
    /// seconds spent in fill (building the Jacobian's blocks) and in the factorization
    pub t_fill: f64,
    pub t_factor: f64,
}

/// Deliberate breakage for the controls (hp/highm_check.py): 1 one pivot of the middle step scaled by 1 + 1e-6;
/// 2 the link rows replaced by delta_s = 0 (the copies are no longer tied to Delta).
pub const MUT_PIVOT: u8 = 1;
pub const MUT_LINK: u8 = 2;

#[inline(always)]
fn f(x: f64) -> DD {
    DD { hi: x, lo: 0.0 }
}

impl BlockLu {
    /// Build and factor. groups: g_0..g_S; rblocks[s] (s < S): the residual rows (offset, count) of row group s;
    /// iphase: the phase row's index in the residual. fill(s, w, ld, r0) writes row group s's residual rows into rows
    /// r0..r0 + rlen of the step's ld x (np + nrc) column-major array w (columns: g_s, then g_{s+1}; it may be
    /// called concurrently with nothing else), and for s = 0 also the phase row at row r0 - 1. The link rows are
    /// written here. Steps are factored in order, each in parallel over column chunks when par.
    pub fn factor<F>(n: usize, groups: Vec<Group>, rblocks: &[(usize, usize)], iphase: usize, idelta: usize, mut fill: F, par: bool, mutate: u8) -> Result<BlockLu, String>
    where
        F: FnMut(usize, &mut [f64], usize, usize) -> Result<(), String>,
    {
        let ns = groups.len();
        if ns < 2 || rblocks.len() != ns - 1 || groups[0].delta {
            return Err("block LU: bad layout".into());
        }
        let gdelta = groups.iter().position(|g| g.delta).ok_or("block LU: no Delta copy")?;
        let mut steps: Vec<Step> = Vec::with_capacity(ns);
        // carried rows: (count, column-major nc x np of the coming step, leading dimension nc)
        let mut carry: Vec<f64> = Vec::new();
        let mut nc = 0usize;
        let (mut t_fill, mut t_factor) = (0.0f64, 0.0f64);
        for s in 0..ns {
            let np = groups[s].len();
            let nrc = if s + 1 < ns { groups[s + 1].len() } else { 0 };
            let (roff, rlen) = if s + 1 < ns { rblocks[s] } else { (0, 0) };
            let nnew = if s + 1 < ns { 1 + rlen } else { 0 };
            let nr = nc + nnew;
            if nr < np {
                return Err(format!("block LU: step {s} has {nr} rows for {np} pivots"));
            }
            if s + 1 == ns && nr != np {
                return Err(format!("block LU: last step not square ({nr} x {np})"));
            }
            let ld = ld_for(nr);
            let ncol = np + nrc;
            let mut w = vec![0.0f64; ld * ncol];
            for c in 0..np {
                w[c * ld..c * ld + nc].copy_from_slice(&carry[c * nc..(c + 1) * nc]);
            }
            if s + 1 < ns {
                if s > 0 {
                    // link row delta_{s-1} - delta_s = 0 (both groups start with their Delta copy)
                    if !(groups[s].delta && groups[s + 1].delta) {
                        return Err("block LU: a link row needs Delta copies on both sides".into());
                    }
                    if mutate != MUT_LINK {
                        w[nc] = 1.0;
                    }
                    w[np * ld + nc] = -1.0;
                }
                let t = std::time::Instant::now();
                fill(s, &mut w, ld, nc + 1)?;
                t_fill += t.elapsed().as_secs_f64();
            }
            let mut perm = vec![0usize; np];
            let t = std::time::Instant::now();
            lu_rect_f64(nr, ncol, np, ld, &mut w, &mut perm, par)?;
            t_factor += t.elapsed().as_secs_f64();
            if mutate == MUT_PIVOT && s == ns / 2 {
                let k = np / 2;
                w[k * ld + k] *= 1.0 + 1e-6;
            }
            // U12 (rows 0..np of the right columns) and the next carry (rows np..nr of the right columns)
            let mut u12 = vec![0.0f64; np * nrc];
            let nc2 = nr - np;
            let mut carry2 = vec![0.0f64; nc2 * nrc];
            for c in 0..nrc {
                let col = &w[(np + c) * ld..(np + c) * ld + nr];
                u12[c * np..(c + 1) * np].copy_from_slice(&col[..np]);
                carry2[c * nc2..(c + 1) * nc2].copy_from_slice(&col[np..nr]);
            }
            w.truncate(ld * np);
            w.shrink_to_fit();
            steps.push(Step { nc, nr, np, nrc, ld, l: w, u12, perm, roff, rlen });
            carry = carry2;
            nc = nc2;
        }
        Ok(BlockLu { n, groups, steps, gdelta, idelta, iphase, t_fill, t_factor })
    }

    /// bytes held by the factors
    pub fn bytes(&self) -> usize {
        self.steps.iter().map(|s| 8 * (s.l.len() + s.u12.len()) + 8 * s.perm.len()).sum()
    }

    #[inline(always)]
    fn lu(st: &Step, i: usize, k: usize) -> f64 {
        st.l[k * st.ld + i]
    }

    /// min and max |u_kk| over all steps
    pub fn pivot_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for st in &self.steps {
            for k in 0..st.np {
                let v = Self::lu(st, k, k).abs();
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo, hi)
    }

    /// Solve J x = b in place (b: n values, the original row order in, the original unknowns out).
    pub fn solve(&self, b: &mut [DD]) {
        assert_eq!(b.len(), self.n);
        let ns = self.steps.len();
        let mut carry: Vec<DD> = Vec::new();
        let mut ys: Vec<Vec<DD>> = Vec::with_capacity(ns);
        for (s, st) in self.steps.iter().enumerate() {
            let mut v = carry;
            if s + 1 < ns {
                v.push(if s == 0 { b[self.iphase] } else { DD::ZERO });
                v.extend_from_slice(&b[st.roff..st.roff + st.rlen]);
            }
            debug_assert_eq!(v.len(), st.nr);
            for k in 0..st.np {
                let p = st.perm[k];
                if p != k {
                    v.swap(k, p);
                }
            }
            for k in 0..st.np {
                let nt = -v[k];
                if nt.hi == 0.0 {
                    continue;
                }
                let col = &st.l[k * st.ld..k * st.ld + st.nr];
                for i in k + 1..st.nr {
                    v[i] = dd_axpy(v[i], f(col[i]), nt);
                }
            }
            carry = v.split_off(st.np);
            ys.push(v);
        }
        let mut xr: Vec<DD> = Vec::new();
        let mut xs: Vec<Vec<DD>> = vec![Vec::new(); ns];
        for s in (0..ns).rev() {
            let st = &self.steps[s];
            let mut y = std::mem::take(&mut ys[s]);
            for j in (0..st.nrc).rev() {
                let nt = -xr[j];
                if nt.hi == 0.0 {
                    continue;
                }
                let col = &st.u12[j * st.np..(j + 1) * st.np];
                for i in 0..st.np {
                    y[i] = dd_axpy(y[i], f(col[i]), nt);
                }
            }
            for k in (0..st.np).rev() {
                y[k] = y[k] / f(Self::lu(st, k, k));
                let nt = -y[k];
                let col = &st.l[k * st.ld..k * st.ld + k];
                for i in 0..k {
                    y[i] = dd_axpy(y[i], f(col[i]), nt);
                }
            }
            xr = y.clone();
            xs[s] = y;
        }
        for (s, g) in self.groups.iter().enumerate() {
            let o = g.delta as usize;
            b[g.zoff..g.zoff + g.zlen].copy_from_slice(&xs[s][o..o + g.zlen]);
            if s == self.gdelta {
                b[self.idelta] = xs[s][0];
            }
        }
    }

    /// Solve J^T y = b in place (b: n values indexed by the unknowns in, by the residual rows out).
    pub fn solve_t(&self, b: &mut [DD]) {
        assert_eq!(b.len(), self.n);
        let ns = self.steps.len();
        // U^T w = P^T-placed b, forward over the groups
        let mut ws: Vec<Vec<DD>> = Vec::with_capacity(ns);
        for (s, g) in self.groups.iter().enumerate() {
            let st = &self.steps[s];
            let mut v: Vec<DD> = Vec::with_capacity(st.np);
            if g.delta {
                v.push(if s == self.gdelta { b[self.idelta] } else { DD::ZERO });
            }
            v.extend_from_slice(&b[g.zoff..g.zoff + g.zlen]);
            if s > 0 {
                let sp = &self.steps[s - 1];
                let wp = &ws[s - 1];
                for c in 0..st.np {
                    let mut acc = v[c];
                    let col = &sp.u12[c * sp.np..(c + 1) * sp.np];
                    for i in 0..sp.np {
                        acc = dd_axpy(acc, f(col[i]), -wp[i]);
                    }
                    v[c] = acc;
                }
            }
            for k in 0..st.np {
                let mut acc = v[k];
                let col = &st.l[k * st.ld..k * st.ld + k];
                for i in 0..k {
                    acc = dd_axpy(acc, f(col[i]), -v[i]);
                }
                v[k] = acc / f(Self::lu(st, k, k));
            }
            ws.push(v);
        }
        // L^T and P^T, backward over the steps; each step's leading entries are the adjoint of the rows it carried in
        let mut adj: Vec<DD> = Vec::new();
        let mut out = vec![DD::ZERO; self.n];
        for s in (0..ns).rev() {
            let st = &self.steps[s];
            let mut v = std::mem::take(&mut ws[s]);
            v.extend_from_slice(&adj);
            debug_assert_eq!(v.len(), st.nr);
            for k in (0..st.np).rev() {
                let mut acc = v[k];
                let col = &st.l[k * st.ld..k * st.ld + st.nr];
                for i in k + 1..st.nr {
                    acc = dd_axpy(acc, f(col[i]), -v[i]);
                }
                v[k] = acc;
            }
            for k in (0..st.np).rev() {
                let p = st.perm[k];
                if p != k {
                    v.swap(k, p);
                }
            }
            if s + 1 < ns {
                if s == 0 {
                    out[self.iphase] = v[st.nc];
                }
                out[st.roff..st.roff + st.rlen].copy_from_slice(&v[st.nc + 1..st.nr]);
            }
            v.truncate(st.nc);
            adj = v;
        }
        b.copy_from_slice(&out);
    }

    /// Several right-hand sides, row-major (nrhs, n) as (hi, lo), in place; each row as solve / solve_t gives it.
    pub fn solve_many(&self, bh: &mut [f64], bl: &mut [f64], nrhs: usize, transpose: bool, par: bool) {
        let n = self.n;
        let one = |(h, l): (&mut [f64], &mut [f64])| {
            let mut b: Vec<DD> = h.iter().zip(l.iter()).map(|(&a, &c)| DD { hi: a, lo: c }).collect();
            if transpose {
                self.solve_t(&mut b);
            } else {
                self.solve(&mut b);
            }
            for i in 0..n {
                h[i] = b[i].hi;
                l[i] = b[i].lo;
            }
        };
        let (bh, bl) = (&mut bh[..nrhs * n], &mut bl[..nrhs * n]);
        if par {
            bh.par_chunks_mut(n).zip(bl.par_chunks_mut(n)).for_each(one);
        } else {
            bh.chunks_mut(n).zip(bl.chunks_mut(n)).for_each(one);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::DDLu;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
    }

    /// A random system with the multiple-shooting structure: groups of sizes a (centre), b (nodes, k of them), a
    /// (horizon); residual blocks of b rows; a Delta column and a phase row on the first a/2 unknowns. Returns the
    /// dense column-major matrix (ld_for(n)), its size, and the layout.
    #[allow(clippy::type_complexity)]
    fn system(a: usize, b: usize, k: usize, inward_from: usize, seed: u64) -> (Vec<f64>, usize, usize, Vec<Group>, Vec<(usize, usize)>) {
        let n = 2 * a + k * b + 1;
        let ld = ld_for(n);
        let mut rng = Rng(seed);
        let mut m = vec![0.0f64; ld * n];
        let goff: Vec<usize> = (0..k + 2).map(|t| if t == 0 { 0 } else { a + (t - 1) * b }).collect();
        let glen: Vec<usize> = (0..k + 2).map(|t| if t == 0 || t == k + 1 { a } else { b }).collect();
        for blk in 0..k + 1 {
            // rows blk*b..: input group and target group (target -I), Delta column
            let (inp, tgt) = if blk == 0 {
                (0, 1)
            } else if blk == k {
                (k + 1, k)
            } else if blk >= inward_from {
                (blk + 1, blk)
            } else {
                (blk, blk + 1)
            };
            // near-identity propagators (as for short segments), so a long chain stays well conditioned
            let sc = 1.0 / (glen[inp] as f64).sqrt();
            for r in 0..b {
                let row = blk * b + r;
                for c in 0..glen[inp] {
                    m[(goff[inp] + c) * ld + row] = rng.next() * sc + if c == r % glen[inp] { 1.0 } else { 0.0 };
                }
                m[(goff[tgt] + r) * ld + row] -= 1.0;
                m[(n - 1) * ld + row] = rng.next() * 10.0;
            }
        }
        for c in 0..a / 2 {
            m[c * ld + n - 1] = rng.next() + 0.7;
        }
        let mut groups = vec![Group { delta: false, zoff: 0, zlen: a }];
        for t in 1..=k {
            groups.push(Group { delta: true, zoff: goff[t], zlen: b });
        }
        groups.push(Group { delta: true, zoff: goff[k + 1], zlen: a });
        let rblocks = (0..k + 1).map(|blk| (blk * b, b)).collect();
        (m, n, ld, groups, rblocks)
    }

    fn block_lu(m: &[f64], n: usize, ld: usize, groups: &[Group], rblocks: &[(usize, usize)], mutate: u8) -> BlockLu {
        let gs = groups.to_vec();
        let fill = |s: usize, w: &mut [f64], wld: usize, r0: usize| -> Result<(), String> {
            let (p, q) = (gs[s], gs[s + 1]);
            let np = p.len();
            let (roff, rlen) = rblocks[s];
            // column of z index zi in this step: panel group p, right group q; Delta -> q's copy
            let pos = |zi: usize| -> Option<usize> {
                if zi == n - 1 {
                    Some(np)
                } else if zi >= p.zoff && zi < p.zoff + p.zlen {
                    Some(p.delta as usize + zi - p.zoff)
                } else if zi >= q.zoff && zi < q.zoff + q.zlen {
                    Some(np + q.delta as usize + zi - q.zoff)
                } else {
                    None
                }
            };
            for zi in 0..n {
                for r in 0..rlen {
                    let v = m[zi * ld + roff + r];
                    if v != 0.0 {
                        let c = pos(zi).ok_or(format!("entry outside the step's groups: row {} col {zi}", roff + r))?;
                        w[c * wld + r0 + r] = v;
                    }
                }
                if s == 0 {
                    let v = m[zi * ld + n - 1];
                    if v != 0.0 {
                        w[pos(zi).unwrap() * wld + r0 - 1] = v;
                    }
                }
            }
            Ok(())
        };
        BlockLu::factor(n, groups.to_vec(), rblocks, n - 1, n - 1, fill, true, mutate).unwrap()
    }

    fn rel(x: &[DD], y: &[DD]) -> f64 {
        let ym = y.iter().map(|v| v.hi.abs()).fold(0.0, f64::max);
        x.iter().zip(y).map(|(a, b)| ((a.hi - b.hi) + (a.lo - b.lo)).abs()).fold(0.0, f64::max) / ym
    }

    /// The block LU's solves (f64 factors, double-double solves) agree with the double-double LU of the same f64
    /// matrix (the reference, exact to ~cond x 1e-32) as well as the dense f64 LU's solves do (DDLu::factor_f64, the
    /// dense mixed-precision path): error <= 100 x the dense f64 error + 1e-15, plain and transposed, with outward and
    /// inward segments. The controls (a pivot scaled by 1 + 1e-6; the Delta copies untied) must fail the check.
    #[test]
    fn block_lu_matches_dense() {
        for (a, b, k, inward, seed) in [(10, 20, 4, 9, 1), (16, 32, 6, 4, 2), (40, 80, 12, 99, 3), (24, 48, 13, 8, 4)] {
            let (m, n, ld, groups, rblocks) = system(a, b, k, inward, seed);
            let dense = DDLu::factor(n, ld, m.clone(), vec![0.0; m.len()], false).unwrap();
            let dense64 = DDLu::factor_f64(n, ld, m.clone(), false).unwrap();
            let rhs: Vec<DD> = (0..n).map(|i| DD::from_f64(((i * 37) % 11) as f64 - 5.0 + 0.1 * (i % 3) as f64)).collect();
            let mut res = [[0.0f64; 2]; 3];
            let mut d64 = [0.0f64; 2];
            for (ti, t) in [false, true].into_iter().enumerate() {
                let mut x = rhs.clone();
                let mut y = rhs.clone();
                if t {
                    dense64.solve_t(&mut x);
                    dense.solve_t(&mut y);
                } else {
                    dense64.solve(&mut x);
                    dense.solve(&mut y);
                }
                d64[ti] = rel(&x, &y);
            }
            for (mi, mutate) in [0u8, MUT_PIVOT, MUT_LINK].into_iter().enumerate() {
                let blu = block_lu(&m, n, ld, &groups, &rblocks, mutate);
                for (ti, t) in [false, true].into_iter().enumerate() {
                    let mut x = rhs.clone();
                    let mut y = rhs.clone();
                    if t {
                        blu.solve_t(&mut x);
                        dense.solve_t(&mut y);
                    } else {
                        blu.solve(&mut x);
                        dense.solve(&mut y);
                    }
                    res[mi][ti] = rel(&x, &y);
                }
            }
            eprintln!(
                "a {a} b {b} k {k} (n {n}): error against the dd LU: block {:.1e} (transposed {:.1e}), dense f64 LU {:.1e} ({:.1e}); pivot control {:.1e} / {:.1e}; link control {:.1e} / {:.1e}",
                res[0][0], res[0][1], d64[0], d64[1], res[1][0], res[1][1], res[2][0], res[2][1]
            );
            let lim = [100.0 * d64[0] + 1e-15, 100.0 * d64[1] + 1e-15];
            assert!(res[0][0] <= lim[0] && res[0][1] <= lim[1], "block LU less accurate than the dense f64 LU: {:?} vs {d64:?}", res[0]);
            assert!(res[1][0] > lim[0] && res[1][1] > lim[1], "the pivot control does not fire: {:?}", res[1]);
            assert!(res[2][0] > lim[0] && res[2][1] > lim[1], "the link control does not fire: {:?}", res[2]);
        }
    }

    /// solve_many gives each row bitwise as solve / solve_t
    #[test]
    fn block_solve_many_is_solve() {
        let (m, n, ld, groups, rblocks) = system(12, 24, 5, 3, 7);
        let blu = block_lu(&m, n, ld, &groups, &rblocks, 0);
        let k = 5;
        let mut rng = Rng(3);
        let bh: Vec<f64> = (0..k * n).map(|_| rng.next()).collect();
        let bl = vec![0.0; k * n];
        for t in [false, true] {
            let (mut xh, mut xl) = (bh.clone(), bl.clone());
            blu.solve_many(&mut xh, &mut xl, k, t, true);
            for r in 0..k {
                let mut v: Vec<DD> = (0..n).map(|i| DD { hi: bh[r * n + i], lo: 0.0 }).collect();
                if t {
                    blu.solve_t(&mut v);
                } else {
                    blu.solve(&mut v);
                }
                for i in 0..n {
                    assert_eq!(v[i].hi.to_bits(), xh[r * n + i].to_bits());
                    assert_eq!(v[i].lo.to_bits(), xl[r * n + i].to_bits());
                }
            }
        }
    }
}
