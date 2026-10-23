//! Python bindings: `Solver(M, n, dmat, phase, nodes, segs, kt, kc, ks, threads)`.
//! Double-double arrays are passed as (hi, lo) pairs of float64 numpy arrays.
use crate::dd::DD;
use crate::dv::DV;
use crate::field::{DMat, Fl, State};
use crate::lane::Lane;
use crate::linalg::DDLu;
use crate::model::{self, Ctx, TWork};
use crate::v8::V8;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

type V = (Vec<f64>, Vec<f64>);

/// How the f64 Jacobian is evaluated: `tangents` columns per evaluation (1: `Fu` lanes, N > 1: `Fun<N>`), and with
/// dd_base the base trajectory in double-double (`Dtn<N>` lanes: N f64 tangents on a double-double value).
#[derive(Clone, Copy)]
struct F64Jac {
    tangents: usize,
    dd_base: bool,
}

trait Api: Send + Sync {
    fn len(&self) -> usize;
    fn m(&self) -> usize;
    fn residual(&self, zh: &[f64], zl: &[f64]) -> Result<V, String>;
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
    /// f64 Jacobian, column-major: (J, ld); `F64Jac` the lanes; mutate: Ctx::mutate for this call (0 = none)
    fn jacobian_f64(&mut self, zh: &[f64], zl: &[f64], mutate: u8, how: F64Jac) -> Result<(Vec<f64>, usize), String>;
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>>;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn segs(&self) -> Vec<Vec<f64>>;
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V;
    fn series(&self, zh: &[f64], zl: &[f64], which: u8) -> Result<V, String>;
    /// Mode matrix M(lam): the Jacobian of the residual for perturbations ~ exp(lam tau) (every tau-derivative of a
    /// perturbation gains + lam), Delta held fixed: the leading (L-1) x (L-1) block (no Delta column, no phase row).
    /// Column-major with leading dimension ld_for(L-1): (hi, lo, ld).
    fn mode_matrix(&mut self, zh: &[f64], zl: &[f64], lam: DD) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
}

fn state_of<const B: usize>(sh: &[f64], sl: &[f64]) -> State<DV, B> {
    let m = 8 * B;
    [0, 1, 2, 3].map(|q| Fl::<DV, B>::load(&sh[q * m..(q + 1) * m], &sl[q * m..(q + 1) * m]))
}

fn store_state<const B: usize>(s: &State<DV, B>) -> V {
    let m = 8 * B;
    let mut h = vec![0.0; 4 * m];
    let mut l = vec![0.0; 4 * m];
    for q in 0..4 {
        s[q].store(&mut h[q * m..(q + 1) * m], &mut l[q * m..(q + 1) * m]);
    }
    (h, l)
}

impl<const B: usize> Api for Ctx<B> {
    fn len(&self) -> usize {
        Ctx::<B>::len(self)
    }
    fn m(&self) -> usize {
        8 * B
    }
    fn residual(&self, zh: &[f64], zl: &[f64]) -> Result<V, String> {
        model::residual(self, zh, zl, true)
    }
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
        model::jacobian(self, zh, zl, true)
    }
    fn jacobian_f64(&mut self, zh: &[f64], zl: &[f64], mutate: u8, how: F64Jac) -> Result<(Vec<f64>, usize), String> {
        self.mutate = mutate;
        let r = match (how.dd_base, how.tangents) {
            (false, 1) => model::jacobian_f64(self, zh, zl, true),
            (false, 2) => model::jacobian_fun::<2, B>(self, zh, zl, true),
            (false, 4) => model::jacobian_fun::<4, B>(self, zh, zl, true),
            (false, 8) => model::jacobian_fun::<8, B>(self, zh, zl, true),
            (true, 1) => model::jacobian_dtn::<1, B>(self, zh, zl, true),
            (true, 2) => model::jacobian_dtn::<2, B>(self, zh, zl, true),
            (true, 4) => model::jacobian_dtn::<4, B>(self, zh, zl, true),
            (true, 8) => model::jacobian_dtn::<8, B>(self, zh, zl, true),
            (_, t) => Err(format!("tangents must be 1, 2, 4 or 8, not {t}")),
        };
        self.mutate = 0;
        r
    }
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>> {
        use rayon::prelude::*;
        let delta = DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] };
        let omega = DV::splat(DD::TWO_PI / delta);
        let m = 8 * B;
        (1..self.k())
            .into_par_iter()
            .map(|b| {
                let (from, to) = if self.seg_inward(b) { (b, b - 1) } else { (b - 1, b) };
                let o = self.off_node(from);
                let s = state_of::<B>(&zh[o..o + 4 * m], &zl[o..o + 4 * m]);
                model::plan_path(self, &s, self.nodes[from], self.nodes[to], omega, tol, hmax)
            })
            .collect()
    }
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String> {
        if segs.len() + 1 != self.k() {
            return Err(format!("need {} segments, got {}", self.k() - 1, segs.len()));
        }
        for (b, s) in segs.iter().enumerate() {
            let (from, to) = if self.seg_inward(b + 1) { (b + 1, b) } else { (b, b + 1) };
            if s.len() < 2 || s[0] != self.nodes[from] || *s.last().unwrap() != self.nodes[to] {
                return Err(format!("segment {} must run from node {} to node {}", b + 1, self.nodes[from], self.nodes[to]));
            }
        }
        self.segs = segs;
        Ok(())
    }
    fn segs(&self) -> Vec<Vec<f64>> {
        self.segs.clone()
    }
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V {
        let delta = DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] };
        let omega = DV::splat(DD::TWO_PI / delta);
        let s = state_of::<B>(sh, sl);
        let mut w = TWork::<DV, B>::new(self.kt);
        store_state(&model::taylor_path(self, &s, xs, omega, &mut w))
    }
    /// which = 0: centre series coefficients (kc+1, 4, M); 1: horizon series (ks+1, 4, M)
    fn series(&self, zh: &[f64], zl: &[f64], which: u8) -> Result<V, String> {
        let delta = DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] };
        let omega = DV::splat(DD::TWO_PI / delta);
        let s = if which == 0 {
            model::centre_series(self, &self.fld(zh, zl, 0), &self.fld(zh, zl, 8 * B), omega)
        } else {
            let lus = model::horizon_lus(self, &self.fld(zh, zl, self.off_ah()), delta, true, DD::ZERO)?;
            model::horizon_series(self, &self.fld(zh, zl, self.off_mh()), &self.fld(zh, zl, self.off_ah()), omega, &lus, &lus)
        };
        let m = 8 * B;
        let nk = s[0].len();
        let mut h = vec![0.0; nk * 4 * m];
        let mut l = vec![0.0; nk * 4 * m];
        for k in 0..nk {
            for q in 0..4 {
                let o = (k * 4 + q) * m;
                s[q][k].val().store(&mut h[o..o + m], &mut l[o..o + m]);
            }
        }
        Ok((h, l))
    }
    fn mode_matrix(&mut self, zh: &[f64], zl: &[f64], lam: DD) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
        self.lam = lam;
        let r = model::jacobian(self, zh, zl, true);
        self.lam = DD::ZERO;
        let (jh, jl, ld) = r?;
        let n = Ctx::<B>::len(self) - 1;
        let ld2 = crate::linalg::ld_for(n);
        let mut h = vec![0.0; ld2 * n];
        let mut l = vec![0.0; ld2 * n];
        for c in 0..n {
            h[c * ld2..c * ld2 + n].copy_from_slice(&jh[c * ld..c * ld + n]);
            l[c * ld2..c * ld2 + n].copy_from_slice(&jl[c * ld..c * ld + n]);
        }
        Ok((h, l, ld2))
    }
}

fn slice<'a>(a: &'a PyReadonlyArray1<f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

fn arr<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    v.into_pyarray(py)
}

#[pyclass]
struct Solver {
    api: Box<dyn Api>,
    pool: Arc<rayon::ThreadPool>,
    lu: Option<DDLu>,
}

fn freeze_pair<const B: usize>(p: &Option<(Vec<f64>, Vec<f64>)>) -> Option<(DMat<B>, DMat<B>)> {
    p.as_ref().map(|(h, l)| {
        let m = 8 * B;
        let lo = DMat::<B>::from_rows(h, l);
        // Pi_high = I - Pi_low
        let mut hh = vec![0.0; m * m];
        let mut hl = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                let v = DD::from_f64(if i == j { 1.0 } else { 0.0 }) - DD { hi: h[i * m + j], lo: l[i * m + j] };
                hh[i * m + j] = v.hi;
                hl[i * m + j] = v.lo;
            }
        }
        (lo, DMat::<B>::from_rows(&hh, &hl))
    })
}

macro_rules! make_ctx {
    ($b:expr, $n:expr, $dh:expr, $dl:expr, $ph:expr, $pl:expr, $nodes:expr, $segs:expr, $kt:expr, $kc:expr, $ks:expr, $plow:expr, $mn:expr, [$($bb:literal),*]) => {
        match $b {
            $( $bb => Box::new(Ctx::<$bb> {
                n: $n,
                d: DMat::<$bb>::from_rows($dh, $dl),
                phase: Fl::<DV, $bb>::load($ph, $pl),
                nodes: $nodes,
                segs: $segs,
                kt: $kt,
                kc: $kc,
                ks: $ks,
                freeze: freeze_pair::<$bb>(&$plow),
                match_node: $mn,
                lam: DD::ZERO,
                mutate: 0,
            }) as Box<dyn Api>, )*
            _ => return Err(PyValueError::new_err("M must be one of 8, 16, 32, 48, 64, 80, 96, 128")),
        }
    };
}

#[pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (m, n_hi, n_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc, ks, threads=0, plow_hi=None, plow_lo=None, match_node=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        m: usize,
        n_hi: f64,
        n_lo: f64,
        dmat_hi: PyReadonlyArray2<f64>,
        dmat_lo: PyReadonlyArray2<f64>,
        phase_hi: PyReadonlyArray1<f64>,
        phase_lo: PyReadonlyArray1<f64>,
        nodes: Vec<f64>,
        segs: Vec<Vec<f64>>,
        kt: usize,
        kc: usize,
        ks: usize,
        threads: usize,
        plow_hi: Option<PyReadonlyArray2<f64>>,
        plow_lo: Option<PyReadonlyArray2<f64>>,
        match_node: Option<usize>,
    ) -> PyResult<Self> {
        if m % 8 != 0 {
            return Err(PyValueError::new_err("M must be a multiple of 8"));
        }
        let dh = dmat_hi.as_slice().map_err(|_| PyValueError::new_err("dmat_hi must be C-contiguous"))?;
        let dl = dmat_lo.as_slice().map_err(|_| PyValueError::new_err("dmat_lo must be C-contiguous"))?;
        if dh.len() != m * m || dl.len() != m * m {
            return Err(PyValueError::new_err("dmat must be (M, M)"));
        }
        let ph = slice(&phase_hi, "phase_hi")?;
        let pl = slice(&phase_lo, "phase_lo")?;
        if ph.len() != m || pl.len() != m {
            return Err(PyValueError::new_err("phase must have M entries"));
        }
        if nodes.len() < 2 {
            return Err(PyValueError::new_err("need at least two nodes"));
        }
        let n = DD::new(n_hi, n_lo);
        let plow: Option<(Vec<f64>, Vec<f64>)> = match (plow_hi, plow_lo) {
            (Some(a), Some(b)) => {
                let (a, b) = (a.as_slice().map_err(|_| PyValueError::new_err("plow"))?, b.as_slice().map_err(|_| PyValueError::new_err("plow"))?);
                if a.len() != m * m || b.len() != m * m {
                    return Err(PyValueError::new_err("plow must be (M, M)"));
                }
                Some((a.to_vec(), b.to_vec()))
            }
            _ => None,
        };
        let mut api: Box<dyn Api> =
            make_ctx!(m / 8, n, dh, dl, ph, pl, nodes.clone(), vec![], kt, kc, ks, plow, match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1), [1, 2, 4, 6, 8, 10, 12, 16]);
        let mn = match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1);
        let segs = if segs.is_empty() {
            (1..nodes.len()).map(|b| if b > mn { vec![nodes[b], nodes[b - 1]] } else { vec![nodes[b - 1], nodes[b]] }).collect()
        } else {
            segs
        };
        api.set_segs(segs).map_err(PyValueError::new_err)?;
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap());
        Ok(Solver { api, pool, lu: None })
    }

    #[getter]
    fn size(&self) -> usize {
        self.api.len()
    }

    #[getter]
    fn m(&self) -> usize {
        self.api.m()
    }

    fn get_segs(&self) -> Vec<Vec<f64>> {
        self.api.segs()
    }

    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> PyResult<()> {
        self.lu = None;
        self.api.set_segs(segs).map_err(PyValueError::new_err)
    }

    fn residual<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        if h.len() != self.api.len() || l.len() != self.api.len() {
            return Err(PyValueError::new_err(format!("z must have length {}", self.api.len())));
        }
        let (api, pool) = (&self.api, &self.pool);
        let (fh, fl) = py.detach(|| pool.install(|| api.residual(h, l))).map_err(PyValueError::new_err)?;
        Ok((arr(py, fh), arr(py, fl)))
    }

    /// Full Jacobian as two (L, L) arrays (row index = residual, column = unknown).
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len();
        let (jh, jl, ld) = py.detach(|| pool.install(|| api.jacobian(h, l))).map_err(PyValueError::new_err)?;
        let to2 = |v: Vec<f64>| {
            let mut a = Array2::<f64>::zeros((n, n));
            for c in 0..n {
                for r in 0..n {
                    a[[r, c]] = v[c * ld + r];
                }
            }
            a
        };
        Ok((to2(jh).into_pyarray(py), to2(jl).into_pyarray(py)))
    }

    /// The Jacobian in plain f64 as one (L, L) array: f64 vector lanes, constants and the base state rounded to f64
    /// (the horizon solves with the hi parts of the double-double LU factors; the centre and horizon series stopped
    /// once their tangent terms are below f64 rounding). tangents: columns per evaluation (1, 2, 4 (default), 8; the
    /// columns agree to rounding whatever the count); dd_base: the base trajectory in double-double (a more accurate
    /// Jacobian where the f64 base trajectory's rounding is amplified, but no faster Newton convergence; slower). mutate
    /// (default 0): a deliberate breakage of the f64 path for the controls of hp/mixed_check.py (1 no tau derivative,
    /// 2 omega scaled by 1 + 1e-9, 3 the horizon solves' kappa tangent term dropped).
    #[pyo3(signature = (zh, zl, mutate=0, tangents=4, dd_base=false))]
    #[allow(clippy::too_many_arguments)]
    fn jacobian_f64<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        mutate: u8,
        tangents: usize,
        dd_base: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let how = F64Jac { tangents, dd_base };
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        if h.len() != self.api.len() || l.len() != self.api.len() {
            return Err(PyValueError::new_err(format!("z must have length {}", self.api.len())));
        }
        let (api, pool) = (&mut self.api, &self.pool);
        let n = api.len();
        let (jh, ld) = py.detach(|| pool.install(|| api.jacobian_f64(h, l, mutate, how))).map_err(PyValueError::new_err)?;
        let mut a = Array2::<f64>::zeros((n, n));
        for c in 0..n {
            for r in 0..n {
                a[[r, c]] = jh[c * ld + r];
            }
        }
        Ok(a.into_pyarray(py))
    }

    /// Mixed-precision Newton step: residual F in double-double, the Jacobian in f64 (jacobian_f64 with tangents,
    /// dd_base), its LU (lu = "f64": the blocked LU in plain f64, DDLu::factor_f64; "dd": promoted to double-double
    /// with lo = 0 and factored by the double-double LU), kept for chord steps, and dz = -J^-1 F solved in
    /// double-double with those factors. Same return as newton: (dz_hi, dz_lo, F_hi, F_lo, min |u_kk|, max |u_kk|).
    /// Iterated, it converges to the same solution as newton (linearly, with a contraction factor of about the f64
    /// Jacobian's relative error times the condition number; notes/MIXED_PRECISION.md).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (zh, zl, tangents=4, dd_base=false, lu="f64"))]
    fn newton_mixed<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        tangents: usize,
        dd_base: bool,
        lu: &str,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        if h.len() != self.api.len() || l.len() != self.api.len() {
            return Err(PyValueError::new_err(format!("z must have length {}", self.api.len())));
        }
        let lu_f64 = match lu {
            "f64" => true,
            "dd" => false,
            _ => return Err(PyValueError::new_err("lu must be \"f64\" or \"dd\"")),
        };
        let (api, pool) = (&mut self.api, &self.pool);
        let n = api.len();
        let res = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let (fh, fl) = api.residual(h, l)?;
                    let (jh, ld) = api.jacobian_f64(h, l, 0, F64Jac { tangents, dd_base })?;
                    let lu = if lu_f64 {
                        DDLu::factor_f64(n, ld, jh, true)?
                    } else {
                        let jl = vec![0.0; jh.len()];
                        DDLu::factor(n, ld, jh, jl, true)?
                    };
                    let mut b: Vec<DD> = (0..n).map(|i| -DD { hi: fh[i], lo: fl[i] }).collect();
                    lu.solve(&mut b);
                    Ok((b, fh, fl, lu))
                })
            })
            .map_err(PyValueError::new_err)?;
        let (b, fh, fl, lu) = res;
        let (pmin, pmax) = lu.pivot_range();
        self.lu = Some(lu);
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect()), arr(py, fh), arr(py, fl), pmin, pmax))
    }

    /// Mode matrix M(lam) (Delta fixed, no phase row) as two (L-1, L-1) arrays.
    fn mode_matrix<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lam_hi: f64,
        lam_lo: f64,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&mut self.api, &self.pool);
        let n = api.len() - 1;
        let lam = DD::new(lam_hi, lam_lo);
        let (mh, ml, ld) = py.detach(|| pool.install(|| api.mode_matrix(h, l, lam))).map_err(PyValueError::new_err)?;
        let to2 = |v: Vec<f64>| {
            let mut a = Array2::<f64>::zeros((n, n));
            for c in 0..n {
                for r in 0..n {
                    a[[r, c]] = v[c * ld + r];
                }
            }
            a
        };
        Ok((to2(mh).into_pyarray(py), to2(ml).into_pyarray(py)))
    }

    /// det M(lam) by double-double LU: (sign, ln|det|, min |u_kk|, max |u_kk|). Its sign changes at the growth rate of
    /// an unstable mode; the critical exponent is gamma = 1/lam.
    fn mode_det(&mut self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, lam_hi: f64, lam_lo: f64) -> PyResult<(f64, f64, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&mut self.api, &self.pool);
        let n = api.len() - 1;
        let lam = DD::new(lam_hi, lam_lo);
        let r = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let (mh, ml, ld) = api.mode_matrix(h, l, lam)?;
                    let lu = DDLu::factor(n, ld, mh, ml, true)?;
                    let (s, la) = lu.logdet();
                    let (pmin, pmax) = lu.pivot_range();
                    Ok((s, la, pmin, pmax))
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(r)
    }

    /// Newton step: residual F, exact Jacobian J, its LU (kept for chord steps), dz = -J^-1 F.
    /// Returns (dz_hi, dz_lo, F_hi, F_lo, min |u_kk|, max |u_kk|).
    #[allow(clippy::type_complexity)]
    fn newton<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len();
        let res = py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let (fh, fl) = api.residual(h, l)?;
                let (jh, jl, ld) = api.jacobian(h, l)?;
                let lu = DDLu::factor(n, ld, jh, jl, true)?;
                let mut b: Vec<DD> = (0..n).map(|i| -DD { hi: fh[i], lo: fl[i] }).collect();
                lu.solve(&mut b);
                Ok((b, fh, fl, lu))
            })
        })
        .map_err(PyValueError::new_err)?;
        let (b, fh, fl, lu) = res;
        let (pmin, pmax) = lu.pivot_range();
        self.lu = Some(lu);
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect()), arr(py, fh), arr(py, fl), pmin, pmax))
    }

    /// Factor a given (L, L) matrix (hi, and lo or None for an f64 matrix) by the double-double LU and store it as the
    /// LU for chord / lu_solve / lu_solve_t / lu_solve_many (as newton does with the Jacobian). Returns
    /// (min |u_kk|, max |u_kk|).
    /// f64: factor the hi part by the f64 LU (DDLu::factor_f64) instead.
    #[pyo3(signature = (ah, al=None, f64=false))]
    fn lu_factor(&mut self, py: Python<'_>, ah: PyReadonlyArray2<'_, f64>, al: Option<PyReadonlyArray2<'_, f64>>, f64: bool) -> PyResult<(f64, f64)> {
        let n = self.api.len();
        if ah.as_array().shape() != [n, n] || al.as_ref().is_some_and(|a| a.as_array().shape() != [n, n]) {
            return Err(PyValueError::new_err(format!("the matrix must be ({n}, {n})")));
        }
        let ld = crate::linalg::ld_for(n);
        let (a, b) = (ah.as_array(), al.as_ref().map(|a| a.as_array()));
        let mut h = vec![0.0; ld * n];
        let mut l = vec![0.0; ld * n];
        for c in 0..n {
            for r in 0..n {
                h[c * ld + r] = a[[r, c]];
                if let Some(b) = &b {
                    l[c * ld + r] = b[[r, c]];
                }
            }
        }
        let pool = &self.pool;
        let lu = py
            .detach(|| pool.install(|| if f64 { DDLu::factor_f64(n, ld, h, true) } else { DDLu::factor(n, ld, h, l, true) }))
            .map_err(PyValueError::new_err)?;
        let r = lu.pivot_range();
        self.lu = Some(lu);
        Ok(r)
    }

    /// Chord step with the stored LU: dz = -J0^-1 F(z). Returns (dz_hi, dz_lo, F_hi, F_lo).
    #[allow(clippy::type_complexity)]
    fn chord<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let lu = self.lu.as_ref().ok_or_else(|| PyValueError::new_err("no stored LU: call newton first"))?;
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len();
        let (b, fh, fl) = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let (fh, fl) = api.residual(h, l)?;
                    let mut b: Vec<DD> = (0..n).map(|i| -DD { hi: fh[i], lo: fl[i] }).collect();
                    lu.solve(&mut b);
                    Ok((b, fh, fl))
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect()), arr(py, fh), arr(py, fl)))
    }

    /// Solve with the stored LU: x = J0^-1 r.
    fn lu_solve<'py>(
        &self,
        py: Python<'py>,
        rh: PyReadonlyArray1<'py, f64>,
        rl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let lu = self.lu.as_ref().ok_or_else(|| PyValueError::new_err("no stored LU: call newton first"))?;
        let (h, l) = (slice(&rh, "rh")?, slice(&rl, "rl")?);
        let mut b: Vec<DD> = h.iter().zip(l).map(|(&a, &c)| DD { hi: a, lo: c }).collect();
        lu.solve(&mut b);
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect())))
    }

    /// Levenberg-Marquardt steps: J at z, G = J^T J, g = -J^T F; for each mu, dz = (G + mu^2 diag(G))^-1 g.
    /// Returns ([(dz_hi, dz_lo) per mu], F_hi, F_lo). Also stores the plain LU of J (for chord / transpose solves).
    #[allow(clippy::type_complexity)]
    fn lm_steps<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        mus: Vec<f64>,
    ) -> PyResult<(Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len();
        let res = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let (fh, fl) = api.residual(h, l)?;
                    let (jh, jl, ld) = api.jacobian(h, l)?;
                    let nfh: Vec<f64> = fh.iter().map(|v| -v).collect();
                    let nfl: Vec<f64> = fl.iter().map(|v| -v).collect();
                    let (gh, gl, g) = crate::linalg::gram(n, ld, &jh, &jl, &nfh, &nfl);
                    let mut steps = Vec::new();
                    for &mu in &mus {
                        let mut ah = gh.clone();
                        let mut al = gl.clone();
                        for i in 0..n {
                            let d = DD { hi: ah[i * ld + i], lo: al[i * ld + i] };
                            let v = d + d.mul_f(mu * mu);
                            ah[i * ld + i] = v.hi;
                            al[i * ld + i] = v.lo;
                        }
                        let lu = DDLu::factor(n, ld, ah, al, true)?;
                        let mut b = g.clone();
                        lu.solve(&mut b);
                        steps.push(b);
                    }
                    let lu = DDLu::factor(n, ld, jh, jl, true)?;
                    Ok((steps, fh, fl, lu))
                })
            })
            .map_err(PyValueError::new_err)?;
        let (steps, fh, fl, lu) = res;
        self.lu = Some(lu);
        let out = steps
            .into_iter()
            .map(|b| (arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect())))
            .collect();
        Ok((out, arr(py, fh), arr(py, fl)))
    }

    /// Solve with the transpose of the stored LU: y = J0^-T r. With r = e_Delta, y is dDelta/dF: the change of the
    /// converged Delta caused by a residual perturbation dF is -y . dF.
    fn lu_solve_t<'py>(
        &self,
        py: Python<'py>,
        rh: PyReadonlyArray1<'py, f64>,
        rl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let lu = self.lu.as_ref().ok_or_else(|| PyValueError::new_err("no stored LU: call newton first"))?;
        let (h, l) = (slice(&rh, "rh")?, slice(&rl, "rl")?);
        let mut b: Vec<DD> = h.iter().zip(l).map(|(&a, &c)| DD { hi: a, lo: c }).collect();
        py.detach(|| lu.solve_t(&mut b));
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect())))
    }

    /// Solve with the stored LU for several right-hand sides at once: the rows of R (k, L) as (hi, lo) -> the rows of
    /// X with J0 X^T = R^T (transpose: J0^T X^T = R^T). Eight right-hand sides per lane vector, groups of eight in
    /// parallel; every row is bitwise what lu_solve / lu_solve_t gives for it.
    #[pyo3(signature = (rh, rl, transpose=false))]
    #[allow(clippy::type_complexity)]
    fn lu_solve_many<'py>(
        &self,
        py: Python<'py>,
        rh: PyReadonlyArray2<'py, f64>,
        rl: PyReadonlyArray2<'py, f64>,
        transpose: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let lu = self.lu.as_ref().ok_or_else(|| PyValueError::new_err("no stored LU: call newton first"))?;
        let n = lu.n;
        let sh = rh.as_array().shape().to_vec();
        if sh[1] != n || rl.as_array().shape() != sh.as_slice() {
            return Err(PyValueError::new_err(format!("R must be (k, {n}) for hi and lo")));
        }
        let k = sh[0];
        let mut h = rh.as_slice().map_err(|_| PyValueError::new_err("rh must be C-contiguous"))?.to_vec();
        let mut l = rl.as_slice().map_err(|_| PyValueError::new_err("rl must be C-contiguous"))?.to_vec();
        let pool = &self.pool;
        py.detach(|| pool.install(|| lu.solve_many(&mut h, &mut l, k, transpose, true)));
        let to2 = |v: Vec<f64>| Array2::from_shape_vec((k, n), v).unwrap().into_pyarray(py);
        Ok((to2(h), to2(l)))
    }

    /// Adaptive Taylor step points for every segment (from the node states in z); does not change the solver.
    fn plan(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, tol: f64, hmax: f64) -> PyResult<Vec<Vec<f64>>> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        Ok(py.detach(|| pool.install(|| api.plan(h, l, tol, hmax))))
    }

    /// Taylor-integrate a state (hi, lo; 4M) along the x points xs with Delta from z; returns the final state.
    #[allow(clippy::too_many_arguments)]
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        sh: PyReadonlyArray1<'py, f64>,
        sl: PyReadonlyArray1<'py, f64>,
        xs: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (a, b) = (slice(&sh, "sh")?, slice(&sl, "sl")?);
        if a.len() != 4 * self.api.m() {
            return Err(PyValueError::new_err("state must have 4M entries"));
        }
        let api = &self.api;
        let (oh, ol) = py.detach(|| api.integrate(h, l, a, b, &xs));
        Ok((arr(py, oh), arr(py, ol)))
    }

    /// Series coefficients, flattened (orders, 4, M): which = 0 centre, 1 horizon.
    fn series<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        which: u8,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let (oh, ol) = py.detach(|| pool.install(|| api.series(h, l, which))).map_err(PyValueError::new_err)?;
        Ok((arr(py, oh), arr(py, ol)))
    }
}

/// Dense double-double LU solve of a general (n, n) system (row-major hi / lo), for tests.
#[pyfunction]
fn dd_solve<'py>(
    py: Python<'py>,
    ah: PyReadonlyArray2<'py, f64>,
    al: PyReadonlyArray2<'py, f64>,
    bh: PyReadonlyArray1<'py, f64>,
    bl: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let ahs = ah.as_slice().map_err(|_| PyValueError::new_err("contiguous"))?;
    let als = al.as_slice().map_err(|_| PyValueError::new_err("contiguous"))?;
    let n = bh.len()?;
    let ld = crate::linalg::ld_for(n);
    let mut h = vec![0.0; ld * n];
    let mut l = vec![0.0; ld * n];
    for r in 0..n {
        for c in 0..n {
            h[c * ld + r] = ahs[r * n + c];
            l[c * ld + r] = als[r * n + c];
        }
    }
    let lu = DDLu::factor(n, ld, h, l, false).map_err(PyValueError::new_err)?;
    let (b1, b2) = (slice(&bh, "bh")?, slice(&bl, "bl")?);
    let mut b: Vec<DD> = b1.iter().zip(b2).map(|(&a, &c)| DD { hi: a, lo: c }).collect();
    lu.solve(&mut b);
    Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect())))
}

/// Elementwise double-double ops on (hi, lo) arrays for tests: op 0 add, 1 mul, 2 div.
#[pyfunction]
fn dd_op<'py>(
    py: Python<'py>,
    op: u8,
    ah: PyReadonlyArray1<'py, f64>,
    al: PyReadonlyArray1<'py, f64>,
    bh: PyReadonlyArray1<'py, f64>,
    bl: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (a1, a2, b1, b2) = (slice(&ah, "ah")?, slice(&al, "al")?, slice(&bh, "bh")?, slice(&bl, "bl")?);
    let n = a1.len();
    if n % 8 != 0 {
        return Err(PyValueError::new_err("length must be a multiple of 8"));
    }
    let mut oh = vec![0.0; n];
    let mut ol = vec![0.0; n];
    for i in (0..n).step_by(8) {
        let a = DV::load(&a1[i..i + 8], &a2[i..i + 8]);
        let b = DV::load(&b1[i..i + 8], &b2[i..i + 8]);
        let r = match op {
            0 => a + b,
            1 => a * b,
            _ => a / b,
        };
        r.store(&mut oh[i..i + 8], &mut ol[i..i + 8]);
    }
    Ok((arr(py, oh), arr(py, ol)))
}

#[pyfunction]
fn isa() -> &'static str {
    V8::ISA
}

#[pymodule]
fn _hpkernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_function(wrap_pyfunction!(dd_solve, m)?)?;
    m.add_function(wrap_pyfunction!(dd_op, m)?)?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    let _ = <DV as Lane>::TANGENT;
    Ok(())
}
