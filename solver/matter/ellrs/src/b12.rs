//! Block 12 (horizon) of ell_ddm.DDEll.mode_matrix: DDEll.block(12, X) transcribed operation for operation. The
//! horizon series (DDEll.ssh_series, with its periodic solves) runs on all tangent rows at once; the frozen-step RK4
//! of the horizon segment (DDEll.integrate_u) runs on groups of tangent rows in parallel, each group carrying its own
//! copy of the value row (rows never mix in DDEll's arithmetic, so the groups give the bits of one run).
use crate::dd::{self, bin, hcat, un, A};
use crate::dt::{self, cs, lin_many, pysum, DT};
use crate::oz::Oz;
use rayon::prelude::*;
use std::sync::Arc;

/// the f64 LU of (om Dm^T + diag(kappa + lam)) behind DDEll._solve
pub trait LuFactory: Sync {
    fn factor(&self, diag: &[f64]) -> Result<Box<dyn LuSolve>, String>;
}
/// solves of one factored matrix: rhs (b x N) row-major, one system per row
pub trait LuSolve {
    fn solve(&self, rhs: &[f64], b: usize) -> Result<Vec<f64>, String>;
}

/// LU with partial pivoting in f64 (not LAPACK's blocked order, so its last bits differ from scipy's)
pub struct NativeLu {
    pub n: usize,
    /// om Dm_h^T, row-major
    pub a0: Vec<f64>,
}

struct NativeFac {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl LuFactory for NativeLu {
    fn factor(&self, diag: &[f64]) -> Result<Box<dyn LuSolve>, String> {
        let n = self.n;
        let mut a = self.a0.clone();
        for i in 0..n {
            a[i * n + i] += diag[i];
        }
        let mut piv = vec![0usize; n];
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let d = a[k * n + k];
            if d == 0.0 {
                return Err("native LU: singular matrix".into());
            }
            for i in k + 1..n {
                let f = a[i * n + k] / d;
                a[i * n + k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= f * a[k * n + j];
                    }
                }
            }
        }
        Ok(Box::new(NativeFac { n, lu: a, piv }))
    }
}

impl LuSolve for NativeFac {
    fn solve(&self, rhs: &[f64], b: usize) -> Result<Vec<f64>, String> {
        let n = self.n;
        let mut out = rhs.to_vec();
        out.par_chunks_mut(n).take(b).for_each(|x| {
            for k in 0..n {
                x.swap(k, self.piv[k]);
            }
            for i in 0..n {
                let mut s = x[i];
                for j in 0..i {
                    s -= self.lu[i * n + j] * x[j];
                }
                x[i] = s;
            }
            for i in (0..n).rev() {
                let mut s = x[i];
                for j in i + 1..n {
                    s -= self.lu[i * n + j] * x[j];
                }
                x[i] = s / self.lu[i * n + i];
            }
        });
        Ok(out)
    }
}

/// the controls
#[derive(Clone, Copy, Default)]
pub struct Mutate {
    /// tangents without lambda (ell_ddm's --mutate=lam)
    pub nolam: bool,
    /// the top odd harmonic dropped from the tau operators (its rows of Bodd and columns of Sodd zeroed)
    pub harm: bool,
}

pub struct B12 {
    pub n: f64,
    pub ll: f64,
    pub nn: usize,
    pub no: usize,
    pub ne: usize,
    /// the odd harmonics as doubles (DDEll.kvec["odd"])
    pub kodd: Vec<f64>,
    pub ks: usize,
    pub u0: f64,
    pub bodd: Oz,
    pub beven: Oz,
    pub sodd: Oz,
    pub seven: Oz,
    pub dm: Oz,
    pub mutate: Mutate,
}

/// one evaluation's constants
struct Ev<'a> {
    b: &'a B12,
    om: f64,
    lam: f64,
}

impl<'a> Ev<'a> {
    /// DDEll._dcoef (odd): (re, im) of harmonic k -> (-k im, k re)
    fn dcoef(&self, x: &A) -> A {
        let m = self.b.kodd.len();
        assert_eq!(x.c, 2 * m);
        let mut o = A::zeros(x.r, x.c);
        for i in 0..x.r {
            for j in 0..m {
                let k = self.b.kodd[j];
                let (h, l) = dd::mul_d(-x.h[i * x.c + m + j], -x.l[i * x.c + m + j], k);
                o.h[i * x.c + j] = h;
                o.l[i * x.c + j] = l;
                let (h, l) = dd::mul_d(x.h[i * x.c + j], x.l[i * x.c + j], k);
                o.h[i * x.c + m + j] = h;
                o.l[i * x.c + m + j] = l;
            }
        }
        o
    }

    /// DDEll.Dspec (odd): omega d/dtau on coefficients, + lambda on the tangents
    fn dspec(&self, c: &DT) -> DT {
        let v = un(&dd::MulD(self.om), &self.dcoef(&c.v));
        let t = c.t.as_ref().map(|t| {
            let x = un(&dd::MulD(self.om), &self.dcoef(t));
            if self.lam != 0.0 {
                bin(&dd::Add, &x, &un(&dd::MulD(self.lam), t))
            } else {
                x
            }
        });
        DT { v, t }
    }

    /// DDEll.Dphys: omega d/dtau on grid values (f @ Dm), + lambda on the tangents
    fn dphys(&self, f: &DT) -> DT {
        let y = lin_many(&self.b.dm, &[f]).pop().unwrap();
        let v = un(&dd::MulD(self.om), &y.v);
        let t = y.t.as_ref().map(|yt| {
            let x = un(&dd::MulD(self.om), yt);
            if self.lam != 0.0 {
                bin(&dd::Add, &x, &un(&dd::MulD(self.lam), f.t.as_ref().unwrap()))
            } else {
                x
            }
        });
        DT { v, t }
    }

    /// DDEll._solve: (omega d/dtau + kappa + lam) f = rhs on the grid, a float64 LU refined with double-double
    /// residuals (at most 8 refinements, stopping when max |df| <= 1e-33 max |f.h|)
    fn solve(&self, kap: &A, lam: f64, rhs: &A, lu: &dyn LuFactory) -> Result<A, String> {
        let nn = self.b.nn;
        let diag: Vec<f64> = (0..nn).map(|j| kap.h[j] + lam).collect();
        let fac = lu.factor(&diag)?;
        let mut f = A::from_h(fac.solve(&rhs.h, rhs.r)?, rhs.r, nn);
        for _ in 0..8 {
            let dmf = self.b.dm.rmul(&f);
            let mut af = bin(&dd::Add, &un(&dd::MulD(self.om), &dmf), &bin(&dd::Mul, kap, &f));
            if lam != 0.0 {
                af = bin(&dd::Add, &af, &un(&dd::MulD(lam), &f));
            }
            let r = bin(&dd::Sub, rhs, &af);
            let df = A::from_h(fac.solve(&r.h, r.r)?, r.r, nn);
            f = bin(&dd::Add, &f, &df);
            let mdf = df.maxabs_h();
            if mdf <= 1e-33 * f.maxabs_h().max(1e-300) {
                break;
            }
        }
        Ok(f)
    }

    /// DDEll.solve_periodic on dual numbers
    fn solve_periodic(&self, kappa: &DT, rhs: &DT, lu: &dyn LuFactory) -> Result<DT, String> {
        let fv = self.solve(&kappa.v, 0.0, &rhs.v, lu)?;
        let mut ft = None;
        if rhs.t.is_some() || kappa.t.is_some() {
            let mut rt = match &rhs.t {
                Some(t) => t.clone(),
                None => A::zeros(kappa.t.as_ref().unwrap().r, self.b.nn),
            };
            if let Some(kt) = &kappa.t {
                rt = bin(&dd::Sub, &rt, &bin(&dd::Mul, kt, &fv));
            }
            ft = Some(self.solve(&kappa.v, self.lam, &rt, lu)?);
        }
        Ok(DT { v: fv, t: ft })
    }

    /// DDEll.ssh_series (orders 0..Ks): P, M, A, C, Q
    fn ssh_series(&self, m0: DT, a0: DT, q0: DT, lu: &dyn LuFactory) -> Result<[Vec<Arc<DT>>; 5], String> {
        let (n, ll, k) = (self.b.n, self.b.ll, self.b.ks);
        let nn = self.b.nn;
        let z = Arc::new(DT::val(A::zeros(1, nn)));
        let one = Arc::new(DT::val(A { h: vec![1.0; nn], l: vec![0.0; nn], r: 1, c: nn }));
        let mk = || vec![z.clone(); k + 2];
        let (mut p, mut m, mut a, mut c, mut q) = (mk(), mk(), mk(), mk(), mk());
        let (mut a2, mut q2, mut qq, mut g, mut h) = (mk(), mk(), mk(), mk(), mk());
        let (mut dp, mut dm) = (mk(), mk());
        a[0] = Arc::new(a0);
        c[0] = one.clone();
        m[0] = Arc::new(m0);
        q[0] = Arc::new(q0);
        a2[0] = Arc::new(dt::mul(&a[0], &a[0]));
        qq[0] = Arc::new(dt::mul(&q[0], &q[0]));
        g[0] = Arc::new(dt::mul(&a2[0], &qq[0]));
        h[0] = Arc::new(dt::mul(&a2[0], &q[0]));
        dm[0] = Arc::new(self.dphys(&m[0]));
        for kk in 0..=k {
            let fk = kk as f64;
            let fk1 = (kk + 1) as f64;
            // C[Kk+1] = (C[Kk] Kk - (cs(C, A2) - C[Kk]) (n-1) + cs(C, G) 2L) / (Kk+1)
            let t1 = dt::mulf(&c[kk], fk);
            let t2 = dt::mulf(&dt::sub(&cs(&c, &a2, kk), &c[kk]), n - 1.0);
            let t4 = dt::mulf(&cs(&c, &g, kk), 2.0 * ll);
            c[kk + 1] = Arc::new(dt::divf(&dt::add(&dt::sub(&t1, &t2), &t4), fk1));
            let mut w: Vec<Arc<DT>> = (0..kk + 2).map(|i| Arc::new(dt::neg(&c[i]))).collect();
            w[0] = Arc::new(dt::addf(&w[0], 1.0));
            w[1] = Arc::new(dt::subf(&w[1], 1.0));
            let mut ww = vec![w[0].clone()];
            for i in 1..kk + 2 {
                ww.push(Arc::new(dt::sub(&w[i], &w[i - 1])));
            }
            let v: Vec<Arc<DT>> =
                (0..kk + 1).map(|i| Arc::new(dt::sub(&dt::mulf(&c[i + 1], (i + 1) as f64), &dt::mulf(&c[i], i as f64)))).collect();
            let pm: Vec<Arc<DT>> = (0..kk + 1).map(|i| Arc::new(dt::add(&p[i], &m[i]))).collect();
            let tau_term = if kk >= 1 { dt::neg(&dp[kk - 1]) } else { (*z).clone() };
            let wpu = pysum((0..kk + 1).map(|i| dt::mul(&dt::mulf(&p[i], i as f64), &ww[kk + 1 - i])));
            let r = dt::add(
                &dt::sub(
                    &dt::add(&dt::add(&dt::sub(&tau_term, &wpu), &cs(&v, &p, kk)), &cs(&c, &p, kk)),
                    &dt::mulf(&cs(&c, &pm, kk), 0.5 * n),
                ),
                &dt::mulf(&cs(&c, &h, kk), ll),
            );
            let kappa = dt::addf(&dt::add(&dt::mulf(&dt::addf(&c[1], 1.0), fk), &c[1]), 1.0 - 0.5 * n);
            p[kk] = Arc::new(self.solve_periodic(&kappa, &dt::neg(&r), lu)?);
            dp[kk] = Arc::new(self.dphys(&p[kk]));
            q2[kk] = Arc::new(dt::add(&cs(&p, &p, kk), &cs(&m, &m, kk)));
            if kk == k {
                break;
            }
            let ag = dt::add(
                &dt::add(&cs(&a, &q2, kk), &dt::mulf(&dt::sub(&a[kk], &cs(&a, &a2, kk)), n - 1.0)),
                &dt::mulf(&cs(&a, &g, kk), 2.0 * ll),
            );
            a[kk + 1] = Arc::new(dt::divf(&dt::sub(&dt::mulf(&a[kk], fk), &dt::mulf(&ag, 0.5)), fk1));
            a2[kk + 1] = Arc::new(cs(&a, &a, kk + 1));
            let mut zc: Vec<Arc<DT>> = (0..kk + 2).map(|i| c[i].clone()).collect();
            zc[0] = Arc::new(dt::addf(&zc[0], 1.0));
            zc[1] = Arc::new(dt::subf(&zc[1], 1.0));
            let mut zz = vec![zc[0].clone()];
            for i in 1..kk + 2 {
                zz.push(Arc::new(dt::sub(&zc[i], &zc[i - 1])));
            }
            let tau_term = dt::sub(&dm[kk], if kk >= 1 { &dm[kk - 1] } else { &z });
            let pm: Vec<Arc<DT>> = (0..kk + 1).map(|i| Arc::new(dt::add(&p[i], &m[i]))).collect();
            let zmu = pysum((0..kk + 1).map(|i| dt::mul(&dt::mulf(&m[i], i as f64), &zz[kk + 1 - i])));
            let rest = dt::sub(
                &dt::add(
                    &dt::sub(&dt::sub(&dt::sub(&tau_term, &zmu), &cs(&v, &m, kk)), &cs(&c, &m, kk)),
                    &dt::mulf(&cs(&c, &pm, kk), 0.5 * n),
                ),
                &dt::mulf(&cs(&c, &h, kk), ll),
            );
            m[kk + 1] = Arc::new(dt::div(&rest, &dt::mulf(&zz[0], fk1)));
            dm[kk + 1] = Arc::new(self.dphys(&m[kk + 1]));
            q[kk + 1] = Arc::new(dt::divf(&dt::sub(&dt::mulf(&q[kk], fk), &dt::mulf(&dt::add(&p[kk], &m[kk]), 0.5)), fk1));
            qq[kk + 1] = Arc::new(cs(&q, &q, kk + 1));
            g[kk + 1] = Arc::new(cs(&a2, &qq, kk + 1));
            h[kk + 1] = Arc::new(cs(&a2, &q, kk + 1));
        }
        for x in [&mut p, &mut m, &mut a, &mut c, &mut q] {
            x.truncate(k + 1);
        }
        Ok([p, m, a, c, q])
    }

    /// DDEll.rhs: the x-ODE on coefficients (fields p, m, a, c, psi)
    fn rhs(&self, x: f64, s: &[DT]) -> Vec<DT> {
        let b = self.b;
        let (n, ll) = (b.n, b.ll);
        let ds0 = self.dspec(&s[0]);
        let ds1 = self.dspec(&s[1]);
        let mut od = lin_many(&b.bodd, &[&s[0], &s[1], &s[4], &ds0, &ds1]).into_iter();
        let (p, m, q, dpp, dmm) = (od.next().unwrap(), od.next().unwrap(), od.next().unwrap(), od.next().unwrap(), od.next().unwrap());
        let mut ev = lin_many(&b.beven, &[&s[2], &s[3]]).into_iter();
        let (a, c) = (ev.next().unwrap(), ev.next().unwrap());
        let a2 = dt::mul(&a, &a);
        let g = dt::mul(&dt::mul(&a2, &q), &q);
        // cx = c ((n-1)(a2 - 1) - 2L g) / x
        let cx = dt::divf(&dt::mul(&c, &dt::sub(&dt::mulf(&dt::subf(&a2, 1.0), n - 1.0), &dt::mulf(&g, 2.0 * ll))), x);
        // ax = a ((p p + m m) + 2L g + (n-1)(1 - a2)) / (2x)
        let ax = dt::divf(
            &dt::mul(
                &a,
                &dt::add(
                    &dt::add(&dt::add(&dt::mul(&p, &p), &dt::mul(&m, &m)), &dt::mulf(&g, 2.0 * ll)),
                    &dt::mulf(&dt::rsubf(1.0, &a2), n - 1.0),
                ),
            ),
            2.0 * x,
        );
        let qx = dt::divf(&dt::add(&p, &m), 2.0 * x);
        let pc = dt::divf(&dt::mul(&dt::mulf(&c, n), &dt::add(&p, &m)), 2.0 * x);
        let lt = dt::divf(&dt::mul(&dt::mul(&dt::mulf(&c, ll), &a2), &q), x);
        // px = (-Dp + cx p - c p / x + pc - lt) / (x - c)
        let px = dt::div(
            &dt::sub(
                &dt::add(&dt::sub(&dt::add(&dt::neg(&dpp), &dt::mul(&cx, &p)), &dt::divf(&dt::mul(&c, &p), x)), &pc),
                &lt,
            ),
            &dt::rsubf(x, &c),
        );
        // mx = (-Dmm - cx m + c m / x - pc + lt) / (x + c)
        let mx = dt::div(
            &dt::add(
                &dt::sub(&dt::add(&dt::sub(&dt::neg(&dmm), &dt::mul(&cx, &m)), &dt::divf(&dt::mul(&c, &m), x)), &pc),
                &lt,
            ),
            &dt::addf(&c, x),
        );
        let mut so = lin_many(&b.sodd, &[&px, &mx, &qx]).into_iter();
        let (sp, sm, sq) = (so.next().unwrap(), so.next().unwrap(), so.next().unwrap());
        let mut se = lin_many(&b.seven, &[&ax, &cx]).into_iter();
        let (sa, sc) = (se.next().unwrap(), se.next().unwrap());
        vec![sp, sm, sa, sc, sq]
    }

    /// DDEll.integrate_u (its _rk4 in y = ln(1 - x)): e holds exp(y), exp(y + h/2) and exp(y + h) of each step,
    /// formed by numpy as DDEll does
    fn rk4_u(&self, s0: Vec<DT>, h: f64, e: &[[f64; 3]]) -> Vec<DT> {
        let (h6h, h6l) = dd::div(h, 0.0, 6.0, 0.0);
        let f = |ee: f64, st: &[DT]| -> Vec<DT> { self.rhs(1.0 - ee, st).iter().map(|r| dt::mulf(r, -ee)).collect() };
        let axpy = |s: &[DT], k: &[DT], a: f64| -> Vec<DT> { s.iter().zip(k).map(|(si, ki)| dt::add(si, &dt::mulf(ki, a))).collect() };
        let mut s = s0;
        for st in e {
            let k1 = f(st[0], &s);
            let k2 = f(st[1], &axpy(&s, &k1, h / 2.0));
            let k3 = f(st[1], &axpy(&s, &k2, h / 2.0));
            let k4 = f(st[2], &axpy(&s, &k3, h));
            s = (0..5)
                .map(|i| {
                    let inc = dt::add(&dt::add(&dt::add(&k1[i], &dt::mulf(&k2[i], 2.0)), &dt::mulf(&k3[i], 2.0)), &k4[i]);
                    dt::add(&s[i], &dt::mulr(&inc, h6h, h6l))
                })
                .collect();
        }
        s
    }
}

/// _eval_series: Horner from 0.0
fn eval_series(sx: &[Arc<DT>], x: f64) -> DT {
    let mut out: Option<DT> = None;
    for k in (0..sx.len()).rev() {
        out = Some(match out {
            None => dt::addf(&sx[k], 0.0),
            Some(o) => dt::add(&dt::mulf(&o, x), &sx[k]),
        });
    }
    out.unwrap()
}

/// block 12's result: the end state's tangents (nf x blk, fields p, m, a, c, psi) and its value row (1 x blk)
pub struct Out {
    pub t: A,
    pub v: A,
}

impl B12 {
    /// DDEll.block(12, x) at om = 2 pi / Delta and lam; rk: None for the state after the series and its evaluation at
    /// u0 (DDEll's S.long case), else (h, e) of the horizon segment's RK4 in y; chunks: tangent-row groups of the RK4
    pub fn block(&self, x: &[f64], om: f64, lam: f64, rk: Option<(f64, &[[f64; 3]])>, lu: &dyn LuFactory, chunks: usize) -> Result<Out, String> {
        let (no, ne) = (self.no, self.ne);
        let nf = 2 * no + ne;
        if x.len() != nf {
            return Err(format!("block 12 takes {nf} inputs, not {}", x.len()));
        }
        let ev = Ev { b: self, om, lam: if self.mutate.nolam { 0.0 } else { lam } };
        // DDEll._inputs: values x, tangents the identity columns
        let input = |o: usize, sz: usize| -> DT {
            let mut t = A::zeros(nf, sz);
            for j in 0..sz {
                t.h[(o + j) * sz + j] = 1.0;
            }
            DT { v: A::from_h(x[o..o + sz].to_vec(), 1, sz), t: Some(t) }
        };
        let (ms, as_, qs) = (input(0, no), input(no, ne), input(no + ne, no));
        let mut od = lin_many(&self.bodd, &[&ms, &qs]).into_iter();
        let (m0, q0) = (od.next().unwrap(), od.next().unwrap());
        let a0 = lin_many(&self.beven, &[&as_]).pop().unwrap();
        let sx = ev.ssh_series(m0, a0, q0, lu)?;
        let ev5: Vec<DT> = sx.iter().map(|f| eval_series(f, self.u0)).collect();
        let mut so = lin_many(&self.sodd, &[&ev5[0], &ev5[1], &ev5[4]]).into_iter();
        let (sp, sm, sq) = (so.next().unwrap(), so.next().unwrap(), so.next().unwrap());
        let mut se = lin_many(&self.seven, &[&ev5[2], &ev5[3]]).into_iter();
        let (sa, sc) = (se.next().unwrap(), se.next().unwrap());
        let s = vec![sp, sm, sa, sc, sq];
        let s = match rk {
            None => s,
            Some((h, e)) => {
                let nc = chunks.clamp(1, nf);
                let g = nf.div_ceil(nc);
                let groups: Vec<(usize, usize)> = (0..nf).step_by(g).map(|i0| (i0, (i0 + g).min(nf))).collect();
                let parts: Vec<Vec<DT>> = groups
                    .par_iter()
                    .map(|&(i0, i1)| {
                        let si: Vec<DT> =
                            s.iter().map(|f| DT { v: f.v.clone(), t: Some(f.t.as_ref().unwrap().rows(i0, i1)) }).collect();
                        ev.rk4_u(si, h, e)
                    })
                    .collect();
                (0..5)
                    .map(|f| {
                        let ts: Vec<&A> = parts.iter().map(|pp| pp[f].t.as_ref().unwrap()).collect();
                        DT { v: parts[0][f].v.clone(), t: Some(dd::vstack(&ts)) }
                    })
                    .collect()
            }
        };
        let ts: Vec<&A> = s.iter().map(|f| f.t.as_ref().unwrap()).collect();
        let vs: Vec<&A> = s.iter().map(|f| &f.v).collect();
        Ok(Out { t: hcat(&ts), v: hcat(&vs) })
    }
}
