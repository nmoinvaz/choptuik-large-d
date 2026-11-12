//! The pivots of multidouble's `linalg::clu_logdet`: its LU copied from multidouble 4.1.0 (`clu_logdet_on`, the
//! version hp pins) with each pivot recorded as it is formed and nothing else changed, so ln|det|, arg det, min |u_kk|
//! and max |u_kk| are clu_logdet's bit for bit (`mode_pivots` returns them for that check, #552). Those are f64
//! aggregates, so the comparison sees changes above their resolution; the double-double identity rests on the
//! source, which differs from clu_logdet_on only by the three pushes and the returned lists.
use multidouble::dd::{pow2_near, DD};
use multidouble::dv::{DotAcc, DV};
use multidouble::simd::{self, Kernel, Simd};
use rayon::prelude::*;

/// clu_logdet's (ln|det|, arg det, min |u_kk|, max |u_kk|), then ln|u_kk|, arg u_kk and the pivot row of each column
pub type Pivots = (f64, f64, f64, f64, Vec<f64>, Vec<f64>, Vec<usize>);

/// The LU of the n x n complex double-double matrix (column-major, leading dimension ld) with its pivots
pub fn clu_pivots(n: usize, ld: usize, rh: Vec<f64>, rl: Vec<f64>, ih: Vec<f64>, il: Vec<f64>, par: bool) -> Result<Pivots, String> {
    simd::dispatch(CluPivots { n, ld, rh, rl, ih, il, par })
}

struct CluPivots {
    n: usize,
    ld: usize,
    rh: Vec<f64>,
    rl: Vec<f64>,
    ih: Vec<f64>,
    il: Vec<f64>,
    par: bool,
}

impl Kernel for CluPivots {
    type Output = Result<Pivots, String>;
    #[inline(always)]
    fn run<S: Simd>(self) -> Self::Output {
        clu_pivots_on::<S>(self.n, self.ld, self.rh, self.rl, self.ih, self.il, self.par)
    }
}

#[allow(clippy::type_complexity)]
fn clu_pivots_on<S: Simd>(
    n: usize,
    ld: usize,
    mut rh: Vec<f64>,
    mut rl: Vec<f64>,
    mut ih: Vec<f64>,
    mut il: Vec<f64>,
    par: bool,
) -> Result<Pivots, String> {
    S::vectorize(
        #[inline(always)]
        move || {
            assert!(ld.is_multiple_of(8) && ld >= n);
            assert!(rh.len() >= ld * n && rl.len() >= ld * n && ih.len() >= ld * n && il.len() >= ld * n);
            let at = |h: &[f64], l: &[f64], i: usize| DD { hi: h[i], lo: l[i] };
            let mut la = 0.0f64;
            let mut arg = 0.0f64;
            let mut pmin = f64::INFINITY;
            let mut pmax = 0.0f64;
            let (mut lnu, mut argu, mut rows) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
            let mut lrh = vec![0.0f64; ld];
            let mut lrl = vec![0.0f64; ld];
            let mut lih = vec![0.0f64; ld];
            let mut lil = vec![0.0f64; ld];
            for k in 0..n {
                let ck = k * ld;
                let mut p = k;
                let mut best = -1.0f64;
                for i in k..n {
                    let v = rh[ck + i].hypot(ih[ck + i]);
                    if v > best {
                        best = v;
                        p = i;
                    }
                }
                // a NaN diagonal entry is a non-finite pivot (as `pivot_search`); a column of NaN leaves best at -1
                // (a NaN modulus never replaces it, so best is never NaN itself)
                if rh[ck + k].hypot(ih[ck + k]).is_nan() || best <= 0.0 || !best.is_finite() {
                    return Err(format!("singular or non-finite pivot at column {k} ({best})"));
                }
                if p != k {
                    for j in 0..n {
                        rh.swap(j * ld + k, j * ld + p);
                        rl.swap(j * ld + k, j * ld + p);
                        ih.swap(j * ld + k, j * ld + p);
                        il.swap(j * ld + k, j * ld + p);
                    }
                    arg += std::f64::consts::PI;
                }
                let ur = at(&rh, &rl, ck + k);
                let ui = at(&ih, &il, ck + k);
                let mag = (ur.hi + ur.lo).hypot(ui.hi + ui.lo);
                // the pivot's lo parts can be NaN with finite hi parts
                if mag <= 0.0 || !mag.is_finite() {
                    return Err(format!("singular or non-finite pivot at column {k} ({mag})"));
                }
                // |u| and its phase to f64 are all ln|u| and arg u need
                la += mag.ln();
                arg += (ui.hi + ui.lo).atan2(ur.hi + ur.lo);
                lnu.push(mag.ln());
                argu.push((ui.hi + ui.lo).atan2(ur.hi + ur.lo));
                rows.push(p);
                pmin = pmin.min(mag);
                pmax = pmax.max(mag);
                if k + 1 == n {
                    break;
                }
                // The multipliers a_ik / u with u and every a_ik scaled by the same power of two (exact) to bring u
                // to order one: 1/u_s = conj(u_s) / |u_s|^2 then neither overflows nor underflows for any pivot,
                // and a_ik,s / u_s is a_ik / u, which can only overflow when the multiplier itself does (a
                // subnormal pivot under an ordinary entry). The quotients' significands are those of the unscaled
                // formula, so the factors are bitwise the same wherever that one stays in range.
                let isc = 1.0 / pow2_near(mag.clamp(f64::MIN_POSITIVE, 2f64.powi(1022)));
                let (us_r, us_i) = (ur.mul_pwr2(isc), ui.mul_pwr2(isc));
                let den = us_r * us_r + us_i * us_i;
                let vr = us_r / den;
                let vi = -(us_i / den);
                for i in k + 1..n {
                    let ar = at(&rh, &rl, ck + i).mul_pwr2(isc);
                    let ai = at(&ih, &il, ck + i).mul_pwr2(isc);
                    let xr = ar * vr - ai * vi;
                    let xi = ar * vi + ai * vr;
                    rh[ck + i] = xr.hi;
                    rl[ck + i] = xr.lo;
                    ih[ck + i] = xi.hi;
                    il[ck + i] = xi.lo;
                }
                let r0 = 8 * ((k + 1) / 8);
                for i in r0..ld {
                    if i > k && i < n {
                        lrh[i] = rh[ck + i];
                        lrl[i] = rl[ck + i];
                        lih[i] = ih[ck + i];
                        lil[i] = il[ck + i];
                    } else {
                        lrh[i] = 0.0;
                        lrl[i] = 0.0;
                        lih[i] = 0.0;
                        lil[i] = 0.0;
                    }
                }
                let ncols = n - k - 1;
                let (_, arh) = rh.split_at_mut((k + 1) * ld);
                let (_, arl) = rl.split_at_mut((k + 1) * ld);
                let (_, aih) = ih.split_at_mut((k + 1) * ld);
                let (_, ail) = il.split_at_mut((k + 1) * ld);
                let upd = |(((ch, cl), dh), dl): (((&mut [f64], &mut [f64]), &mut [f64]), &mut [f64])| {
                    S::vectorize(
                        #[inline(always)]
                        || {
                            let tr = DD { hi: ch[k], lo: cl[k] };
                            let ti = DD { hi: dh[k], lo: dl[k] };
                            if tr.hi == 0.0 && ti.hi == 0.0 {
                                return;
                            }
                            let ntr = DV::<S>::splat(-tr);
                            let nti = DV::<S>::splat(-ti);
                            let pti = DV::<S>::splat(ti);
                            let mut i = r0;
                            while i < ld {
                                let ar = DV::<S>::load(&ch[i..i + 8], &cl[i..i + 8]);
                                let ai = DV::<S>::load(&dh[i..i + 8], &dl[i..i + 8]);
                                let lr = DV::<S>::load(&lrh[i..i + 8], &lrl[i..i + 8]);
                                let li = DV::<S>::load(&lih[i..i + 8], &lil[i..i + 8]);
                                // a -= l t:  re: ar - lr tr + li ti,  im: ai - lr ti - li tr
                                let nr = DotAcc::<S>::zero().add(ar).add_prod(lr, ntr).add_prod(li, pti).finish();
                                let ni = DotAcc::<S>::zero().add(ai).add_prod(lr, nti).add_prod(li, ntr).finish();
                                nr.store(&mut ch[i..i + 8], &mut cl[i..i + 8]);
                                ni.store(&mut dh[i..i + 8], &mut dl[i..i + 8]);
                                i += 8;
                            }
                        },
                    )
                };
                if par && ncols * (ld - r0) > 32768 {
                    arh[..ncols * ld]
                        .par_chunks_mut(ld)
                        .zip(arl[..ncols * ld].par_chunks_mut(ld))
                        .zip(aih[..ncols * ld].par_chunks_mut(ld))
                        .zip(ail[..ncols * ld].par_chunks_mut(ld))
                        .for_each(upd);
                } else {
                    arh[..ncols * ld]
                        .chunks_mut(ld)
                        .zip(arl[..ncols * ld].chunks_mut(ld))
                        .zip(aih[..ncols * ld].chunks_mut(ld))
                        .zip(ail[..ncols * ld].chunks_mut(ld))
                        .for_each(upd);
                }
            }
            let two_pi = 2.0 * std::f64::consts::PI;
            let mut a = arg.rem_euclid(two_pi);
            if a > std::f64::consts::PI {
                a -= two_pi;
            }
            Ok((la, a, pmin, pmax, lnu, argu, rows))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use multidouble::linalg::clu_logdet;

    /// an n x n complex double-double matrix (leading dimension ld) of deterministic entries in [-1, 1], with lo parts
    fn matrix(n: usize, ld: usize, seed: u64) -> [Vec<f64>; 4] {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut m = [vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n]];
        for j in 0..n {
            for i in 0..n {
                let (h, g) = (next(), next());
                m[0][j * ld + i] = h;
                m[1][j * ld + i] = h * 1e-17 * next();
                m[2][j * ld + i] = g;
                m[3][j * ld + i] = g * 1e-17 * next();
            }
        }
        m
    }

    /// Pass: ln|det|, arg det, min and max |u_kk| equal clu_logdet's bit for bit, with and without the parallel update,
    /// and the pivots' ln|u_kk| summed in order give ln|det| bit for bit.
    #[test]
    fn pivots_reproduce_clu_logdet() {
        for (n, par) in [(37usize, false), (37, true), (300, true)] {
            let ld = n.div_ceil(8) * 8;
            let [rh, rl, ih, il] = matrix(n, ld, 7);
            let want = clu_logdet(n, ld, rh.clone(), rl.clone(), ih.clone(), il.clone(), par).unwrap();
            let (la, arg, pmin, pmax, lnu, argu, rows) = clu_pivots(n, ld, rh, rl, ih, il, par).unwrap();
            assert_eq!(
                [la.to_bits(), arg.to_bits(), pmin.to_bits(), pmax.to_bits()],
                [want.0.to_bits(), want.1.to_bits(), want.2.to_bits(), want.3.to_bits()],
                "n {n}, par {par}"
            );
            assert_eq!(lnu.iter().fold(0.0, |a, x| a + x).to_bits(), la.to_bits(), "n {n}: sum of ln|u_kk|");
            assert_eq!((lnu.len(), argu.len(), rows.len()), (n, n, n));
        }
    }

    /// The control of that comparison: one entry moved by 1e-12 relative changes ln|det|. The comparison sees changes
    /// above ln|det|'s f64 resolution (about 1e-15 here; one ulp of one entry's hi part does not move it); that the
    /// copy is clu_logdet in double-double rests on its source, which differs only by the recording.
    #[test]
    fn control_entry_change_moves_ln_det() {
        let (n, ld) = (37, 40);
        let [mut rh, rl, ih, il] = matrix(n, ld, 7);
        let a = clu_pivots(n, ld, rh.clone(), rl.clone(), ih.clone(), il.clone(), false).unwrap().0;
        rh[5 * ld + 11] *= 1.0 + 1e-12;
        let b = clu_pivots(n, ld, rh, rl, ih, il, false).unwrap().0;
        assert_ne!(a.to_bits(), b.to_bits());
    }

    /// Planted: a diagonal matrix with decreasing |d_k| has no row swaps and pivots d_k, so ln|u_kk| = ln|d_k| and
    /// the rows are 0, 1, ..., n - 1.
    #[test]
    fn planted_diagonal_pivots() {
        let (n, ld) = (12, 16);
        let mut m = [vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n]];
        let d: Vec<(f64, f64)> = (0..n).map(|k| (0.5f64.powi(k as i32), -(0.25f64.powi(k as i32)))).collect();
        for (k, &(re, im)) in d.iter().enumerate() {
            m[0][k * ld + k] = re;
            m[2][k * ld + k] = im;
        }
        let [rh, rl, ih, il] = m;
        let (_, _, _, _, lnu, argu, rows) = clu_pivots(n, ld, rh, rl, ih, il, false).unwrap();
        for (k, &(re, im)) in d.iter().enumerate() {
            assert!((lnu[k] - re.hypot(im).ln()).abs() < 1e-15, "k {k}");
            assert!((argu[k] - im.atan2(re)).abs() < 1e-15, "k {k}");
        }
        assert_eq!(rows, (0..n).collect::<Vec<_>>());
    }
}
