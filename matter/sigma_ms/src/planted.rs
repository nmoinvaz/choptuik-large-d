//! The source terms of a manufactured solution, for the planted-truth check (planted.py).
//!
//! With sources S_f added to the right-hand sides of the five equations (model.rs: x c_x = G + S_c, 2 x a_x = H + S_a,
//! x psi_x = (p + m)/2 + S_psi, and S_p, S_m on the right of the p and m equations, where x c_x is the full derivative
//! G + S_c), a chosen set of fields solves the equations exactly. The planted fields of planted.py are polynomials in
//! x, with psi = q x, so sigma = sin(beta x)/kappa and gamma = cos(beta x) with beta = kappa q, and every source is
//!
//!   S_f(x) = P_f0(x) + P_f1(x) cos(2 beta x) + P_f2(x) sin(2 beta x),
//!
//! a polynomial part and two trigonometric parts with polynomial coefficients, per collocation point. planted.py forms
//! the coefficients in mpmath from closed forms of the fields (sharing no code with the recursions); this module
//! expands S_f in Taylor series about any point, for the Taylor steps (about x0), the centre series (about x = 0)
//! and the horizon series (in u = 1 - x).
use multidouble::dd::{split, DD};
use multidouble::field::Fl;
use multidouble::native::DV;
use multishoot::periodic::sin_cos;

pub struct Planted<const B: usize> {
    /// beta = kappa q at each collocation point
    pub beta: Vec<DD>,
    /// coef[f][t][d][j]: the coefficient of x^d of P_ft at point j (f: p, m, a, c, psi; t: 1, cos 2 beta x,
    /// sin 2 beta x)
    pub coef: Vec<Vec<Vec<Vec<DD>>>>,
}

fn binom(d: usize, e: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..e {
        r = r * (d - i) as f64 / (i + 1) as f64;
    }
    r
}

impl<const B: usize> Planted<B> {
    /// Taylor coefficients 0..=order of S_f (f = 0..5) in h, x = x0 + dir h, each times scale^k (the scaled variable
    /// of the series; dir = 1 or -1, scale a power of two, so both factors are exact)
    pub fn series(&self, x0: DD, dir: f64, order: usize, scale: f64) -> [Vec<Fl<DV, B>>; 5] {
        let m = 8 * B;
        let mut out: Vec<Vec<Vec<DD>>> = vec![vec![vec![DD::ZERO; m]; order + 1]; 5];
        let ds = dir * scale;
        for j in 0..m {
            let b2 = self.beta[j].mul_f(2.0);
            let (s, c) = sin_cos(b2 * x0);
            // cos(2 beta x) and sin(2 beta x) about x0: (2 beta dir scale)^k / k! times cos and sin of 2 beta x0 + k pi/2
            let mut tc = vec![DD::ZERO; order + 1];
            let mut ts = vec![DD::ZERO; order + 1];
            let mut fac = DD::ONE;
            for k in 0..=order {
                let (ck, sk) = match k % 4 {
                    0 => (c, s),
                    1 => (-s, c),
                    2 => (-c, -s),
                    _ => (s, -c),
                };
                tc[k] = fac * ck;
                ts[k] = fac * sk;
                fac = fac * b2.mul_f(ds) / DD::from_f64((k + 1) as f64);
            }
            for f in 0..5 {
                for t in 0..3 {
                    let poly = &self.coef[f][t];
                    let deg = poly.len();
                    if deg == 0 {
                        continue;
                    }
                    // the polynomial about x0: sum_e ph_e h^e, ph_e = (dir scale)^e sum_{d >= e} C(d, e) s_d x0^(d-e)
                    let mut ph = vec![DD::ZERO; deg];
                    for (e, phe) in ph.iter_mut().enumerate() {
                        let mut acc = DD::ZERO;
                        let mut xp = DD::ONE;
                        for d in e..deg {
                            acc = acc + (poly[d][j] * xp).mul_f(binom(d, e));
                            xp = xp * x0;
                        }
                        *phe = acc.mul_f(ds.powi(e as i32));
                    }
                    for k in 0..=order {
                        let mut acc = DD::ZERO;
                        for (e, &phe) in ph.iter().enumerate().take(k + 1) {
                            let tr = match t {
                                0 => {
                                    if k == e {
                                        DD::ONE
                                    } else {
                                        DD::ZERO
                                    }
                                }
                                1 => tc[k - e],
                                _ => ts[k - e],
                            };
                            acc = acc + phe * tr;
                        }
                        out[f][k][j] = out[f][k][j] + acc;
                    }
                }
            }
        }
        std::array::from_fn(|f| {
            out[f]
                .iter()
                .map(|v| {
                    let (h, l) = split(v);
                    Fl::<DV, B>::load(&h, &l)
                })
                .collect()
        })
    }
}
