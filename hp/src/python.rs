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

trait Api: Send + Sync {
    fn len(&self) -> usize;
    fn m(&self) -> usize;
    fn residual(&self, zh: &[f64], zl: &[f64]) -> Result<V, String>;
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>>;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn segs(&self) -> Vec<Vec<f64>>;
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V;
    fn series(&self, zh: &[f64], zl: &[f64], which: u8) -> Result<V, String>;
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
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>> {
        use rayon::prelude::*;
        let delta = DD { hi: zh[zh.len() - 1], lo: zl[zl.len() - 1] };
        let omega = DV::splat(DD::TWO_PI / delta);
        let m = 8 * B;
        (1..self.k())
            .into_par_iter()
            .map(|b| {
                let o = self.off_node(b - 1);
                let s = state_of::<B>(&zh[o..o + 4 * m], &zl[o..o + 4 * m]);
                model::plan_path(self, &s, self.nodes[b - 1], self.nodes[b], omega, tol, hmax)
            })
            .collect()
    }
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String> {
        if segs.len() + 1 != self.k() {
            return Err(format!("need {} segments, got {}", self.k() - 1, segs.len()));
        }
        for (b, s) in segs.iter().enumerate() {
            if s.len() < 2 || s[0] != self.nodes[b] || *s.last().unwrap() != self.nodes[b + 1] {
                return Err(format!("segment {} must run from node {} to node {}", b + 1, self.nodes[b], self.nodes[b + 1]));
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
            let lus = model::horizon_lus(self, &self.fld(zh, zl, self.off_ah()), delta, true)?;
            model::horizon_series(self, &self.fld(zh, zl, self.off_mh()), &self.fld(zh, zl, self.off_ah()), omega, &lus)
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

macro_rules! make_ctx {
    ($b:expr, $n:expr, $dh:expr, $dl:expr, $ph:expr, $pl:expr, $nodes:expr, $segs:expr, $kt:expr, $kc:expr, $ks:expr, [$($bb:literal),*]) => {
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
            }) as Box<dyn Api>, )*
            _ => return Err(PyValueError::new_err("M must be one of 8, 16, 32, 48, 64, 80, 96, 128")),
        }
    };
}

#[pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (m, n_hi, n_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc, ks, threads=0))]
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
        let mut api: Box<dyn Api> =
            make_ctx!(m / 8, n, dh, dl, ph, pl, nodes.clone(), vec![], kt, kc, ks, [1, 2, 4, 6, 8, 10, 12, 16]);
        let segs = if segs.is_empty() { (1..nodes.len()).map(|b| vec![nodes[b - 1], nodes[b]]).collect() } else { segs };
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
