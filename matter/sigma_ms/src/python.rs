//! Python bindings: `Solver(M, n_hi, n_lo, k2_hi, k2_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc,
//! ks, threads, match_node)`. Double-double arrays are (hi, lo) pairs of float64 arrays, quad-double ones four parts.
//! A lean copy of hp's python.rs: the residual, the exact Jacobian and the Newton step in double-double and
//! quad-double, the Taylor plan, the series (for their tail checks), Taylor integration, and the planted sources and
//! controls.
use crate::model::{self, Ctx};
use crate::planted::Planted;
use multidouble::dd::DD;
use multidouble::field::Fl;
use multidouble::native::{DMat, DV};
use multidouble::precision::{Double, Precision, Quad};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

type V = (Vec<f64>, Vec<f64>);
/// (dz parts, F parts, min |u_kk|, max |u_kk|)
type Step = (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64);

trait Api: Send + Sync {
    fn len(&self) -> usize;
    fn m(&self) -> usize;
    fn residual(&self, z: &[&[f64]]) -> Result<Vec<Vec<f64>>, String>;
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
    fn newton(&self, z: &[&[f64]]) -> Result<Step, String>;
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>>;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn segs(&self) -> Vec<Vec<f64>>;
    /// series coefficients (orders, 5, M), hi parts, in the scaled variable, and the scale
    fn series(&self, zh: &[f64], zl: &[f64], which: u8) -> Result<(Vec<f64>, usize, f64), String>;
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V;
    fn set_k2(&mut self, k2: DD);
    fn set_mutate(&mut self, code: u8);
    fn set_drop_col(&mut self, c: Option<usize>);
    fn set_planted(&mut self, beta: Vec<DD>, coef: Vec<Vec<Vec<Vec<DD>>>>);
    fn clear_planted(&mut self);
}

fn parts_f<P: Precision<multidouble::simd::Native>>(f: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS).map(|q| f.iter().map(|&v| P::part(v, q)).collect()).collect()
}

impl<const B: usize> Api for Ctx<B> {
    fn len(&self) -> usize {
        Ctx::<B>::len(self)
    }
    fn m(&self) -> usize {
        8 * B
    }
    fn residual(&self, z: &[&[f64]]) -> Result<Vec<Vec<f64>>, String> {
        match z.len() {
            2 => model::residual_in::<Double, B>(self, z, true).map(|f| parts_f::<Double>(&f)),
            4 => model::residual_in::<Quad, B>(self, z, true).map(|f| parts_f::<Quad>(&f)),
            p => Err(format!("z must have 2 or 4 parts, not {p}")),
        }
    }
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
        let (mut c, ld) = model::jacobian_in::<Double, B>(self, &[zh, zl], true)?;
        let jl = c.pop().expect("two parts");
        let jh = c.pop().expect("two parts");
        Ok((jh, jl, ld))
    }
    fn newton(&self, z: &[&[f64]]) -> Result<Step, String> {
        match z.len() {
            2 => {
                let s = model::newton_in::<Double, B>(self, z, true)?;
                let (pmin, pmax) = s.lu.pivot_range();
                Ok((s.dz, parts_f::<Double>(&s.f), pmin, pmax))
            }
            4 => {
                let s = model::newton_in::<Quad, B>(self, z, true)?;
                Ok((s.dz, parts_f::<Quad>(&s.f), f64::NAN, f64::NAN))
            }
            p => Err(format!("z must have 2 or 4 parts, not {p}")),
        }
    }
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>> {
        model::plan(self, zh, zl, tol, hmax)
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
    fn series(&self, zh: &[f64], zl: &[f64], which: u8) -> Result<(Vec<f64>, usize, f64), String> {
        let m = 8 * B;
        let delta = model::delta_of(zh, zl);
        let omega = DV::splat(DD::TWO_PI / delta);
        let (s, sig) = if which == 0 {
            (model::centre_series(self, &self.fld(zh, zl, 0), &self.fld(zh, zl, m), omega), model::series_scale(self.nodes[0]))
        } else {
            let o = self.off_h();
            let (mh, ah, ph) = (self.fld(zh, zl, o), self.fld(zh, zl, o + m), self.fld(zh, zl, o + 2 * m));
            let prep = model::HPrep { lus: model::horizon_lus(self, &ah, &ph, delta, true)? };
            (model::horizon_series(self, &mh, &ah, &ph, omega, &prep), model::series_scale(self.u0().hi))
        };
        let nk = s[0].len();
        let mut h = vec![0.0; nk * 5 * m];
        let mut l = vec![0.0; nk * 5 * m];
        for k in 0..nk {
            for (q, f) in s.iter().enumerate() {
                let o = (k * 5 + q) * m;
                f[k].store(&mut h[o..o + m], &mut l[o..o + m]);
            }
        }
        Ok((h, nk, sig))
    }
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V {
        let m = 8 * B;
        let s0: Vec<Fl<DV, B>> = (0..5).map(|q| Fl::load(&sh[q * m..(q + 1) * m], &sl[q * m..(q + 1) * m])).collect();
        let s = model::integrate(self, model::delta_of(zh, zl), &s0, xs);
        let mut h = vec![0.0; 5 * m];
        let mut l = vec![0.0; 5 * m];
        for (q, f) in s.iter().enumerate() {
            f.store(&mut h[q * m..(q + 1) * m], &mut l[q * m..(q + 1) * m]);
        }
        (h, l)
    }
    fn set_k2(&mut self, k2: DD) {
        self.k2 = k2;
    }
    fn set_mutate(&mut self, code: u8) {
        self.mutate = code;
    }
    fn set_drop_col(&mut self, c: Option<usize>) {
        self.drop_col = c;
    }
    fn set_planted(&mut self, beta: Vec<DD>, coef: Vec<Vec<Vec<Vec<DD>>>>) {
        self.planted = Some(Planted { beta, coef });
    }
    fn clear_planted(&mut self) {
        self.planted = None;
    }
}

macro_rules! make_ctx {
    ($b:expr, $n:expr, $k2:expr, $dh:expr, $dl:expr, $ph:expr, $pl:expr, $nodes:expr, $kt:expr, $kc:expr, $ks:expr, $mn:expr, [$($bb:literal),*]) => {
        match $b {
            $( $bb => Box::new(Ctx::<$bb> {
                n: $n,
                k2: $k2,
                d: DMat::<$bb>::from_rows($dh, $dl),
                phase: Fl::<DV, $bb>::load($ph, $pl),
                nodes: $nodes,
                segs: vec![],
                kt: $kt,
                kc: $kc,
                ks: $ks,
                match_node: $mn,
                mutate: 0,
                drop_col: None,
                planted: None,
            }) as Box<dyn Api>, )*
            _ => return Err(PyValueError::new_err("M must be one of 8, 16, 24, 32, 40, 48, 64")),
        }
    };
}

fn slice<'a>(a: &'a PyReadonlyArray1<f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

fn arr<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    v.into_pyarray(py)
}

type A1<'py> = Bound<'py, PyArray1<f64>>;

#[pyclass]
struct Solver {
    api: Box<dyn Api>,
    pool: Arc<rayon::ThreadPool>,
}

impl Solver {
    fn zparts<'a>(&self, z: &'a [PyReadonlyArray1<'a, f64>]) -> PyResult<Vec<&'a [f64]>> {
        let n = self.api.len();
        let mut out = Vec::with_capacity(z.len());
        for a in z {
            let s = slice(a, "z")?;
            if s.len() != n {
                return Err(PyValueError::new_err(format!("each part must have length {n}")));
            }
            out.push(s);
        }
        Ok(out)
    }
}

#[pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (m, n_hi, n_lo, k2_hi, k2_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc, ks, threads=0, match_node=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        m: usize,
        n_hi: f64,
        n_lo: f64,
        k2_hi: f64,
        k2_lo: f64,
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
        let mn = match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1);
        let mut api: Box<dyn Api> = make_ctx!(m / 8, DD::new(n_hi, n_lo), DD::new(k2_hi, k2_lo), dh, dl, ph, pl, nodes.clone(), kt, kc, ks, mn,
            [1, 2, 3, 4, 5, 6, 8]);
        let segs = if segs.is_empty() {
            (1..nodes.len()).map(|b| if b > mn { vec![nodes[b], nodes[b - 1]] } else { vec![nodes[b - 1], nodes[b]] }).collect()
        } else {
            segs
        };
        api.set_segs(segs).map_err(PyValueError::new_err)?;
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).stack_size(256 << 20).build().unwrap());
        Ok(Solver { api, pool })
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
        self.api.set_segs(segs).map_err(PyValueError::new_err)
    }

    /// kappa^2 = n eps (the continuation parameter)
    fn set_k2(&mut self, k2_hi: f64, k2_lo: f64) {
        self.api.set_k2(DD::new(k2_hi, k2_lo));
    }

    /// model::MUT_* (0 none): the controls
    fn set_mutate(&mut self, code: u8) {
        self.api.set_mutate(code);
    }

    /// the Jacobian column dropped by multishoot's Control::DropColumn (negative: none)
    fn set_drop_col(&mut self, c: i64) {
        self.api.set_drop_col(if c < 0 { None } else { Some(c as usize) });
    }

    /// the planted sources: beta (M, hi and lo) and coef (5, 3, degree + 1, M, hi and lo), planted.rs
    fn set_planted(
        &mut self,
        beta_hi: PyReadonlyArray1<f64>,
        beta_lo: PyReadonlyArray1<f64>,
        coef_hi: PyReadonlyArray4<f64>,
        coef_lo: PyReadonlyArray4<f64>,
    ) -> PyResult<()> {
        let m = self.api.m();
        let (bh, bl) = (slice(&beta_hi, "beta_hi")?, slice(&beta_lo, "beta_lo")?);
        if bh.len() != m || bl.len() != m {
            return Err(PyValueError::new_err("beta must have M entries"));
        }
        let (ch, cl) = (coef_hi.as_array(), coef_lo.as_array());
        let sh = ch.shape();
        if sh.len() != 4 || sh[0] != 5 || sh[1] != 3 || sh[3] != m || cl.shape() != sh {
            return Err(PyValueError::new_err("coef must be (5, 3, degree + 1, M)"));
        }
        let beta = (0..m).map(|j| DD::new(bh[j], bl[j])).collect();
        let coef = (0..5)
            .map(|f| (0..3).map(|t| (0..sh[2]).map(|d| (0..m).map(|j| DD::new(ch[[f, t, d, j]], cl[[f, t, d, j]])).collect()).collect()).collect())
            .collect();
        self.api.set_planted(beta, coef);
        Ok(())
    }

    fn clear_planted(&mut self) {
        self.api.clear_planted();
    }

    /// The residual: two parts (double-double) in, two out; four (quad-double) in, four out
    fn residual<'py>(&self, py: Python<'py>, z: Vec<PyReadonlyArray1<'py, f64>>) -> PyResult<Vec<A1<'py>>> {
        let zp = self.zparts(&z)?;
        let (api, pool) = (&self.api, &self.pool);
        let f = py.detach(|| pool.install(|| api.residual(&zp))).map_err(PyValueError::new_err)?;
        Ok(f.into_iter().map(|v| arr(py, v)).collect())
    }

    /// The exact double-double Jacobian as two (L, L) arrays (row = residual, column = unknown)
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let n = self.api.len();
        let (api, pool) = (&self.api, &self.pool);
        let (jh, jl, ld) = py.detach(|| pool.install(|| api.jacobian(h, l))).map_err(PyValueError::new_err)?;
        let to2 = |c: Vec<f64>| {
            let mut a = Array2::<f64>::zeros((n, n));
            for j in 0..n {
                for i in 0..n {
                    a[[i, j]] = c[j * ld + i];
                }
            }
            a.into_pyarray(py)
        };
        Ok((to2(jh), to2(jl)))
    }

    /// The Newton step dz = -J^-1 F in the state's precision (two parts: double-double, four: quad-double), with the
    /// exact Jacobian and a dense LU. Returns (dz parts, F parts, min |u_kk|, max |u_kk|) (pivots NaN in quad-double).
    #[allow(clippy::type_complexity)]
    fn newton<'py>(&self, py: Python<'py>, z: Vec<PyReadonlyArray1<'py, f64>>) -> PyResult<(Vec<A1<'py>>, Vec<A1<'py>>, f64, f64)> {
        let zp = self.zparts(&z)?;
        let (api, pool) = (&self.api, &self.pool);
        let (dz, f, pmin, pmax) = py.detach(|| pool.install(|| api.newton(&zp))).map_err(PyValueError::new_err)?;
        Ok((dz.into_iter().map(|v| arr(py, v)).collect(), f.into_iter().map(|v| arr(py, v)).collect(), pmin, pmax))
    }

    /// Step points for every segment from the node states of z (multishoot's planner); does not change the solver
    fn plan(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, tol: f64, hmax: f64) -> PyResult<Vec<Vec<f64>>> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        Ok(py.detach(|| pool.install(|| api.plan(h, l, tol, hmax))))
    }

    /// Series coefficients (hi parts), flattened (orders, 5, M), in the scaled variable x / sigma (u / sigma):
    /// which = 0 centre, 1 horizon. Returns (array, orders, sigma).
    fn series<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        which: u8,
    ) -> PyResult<(A1<'py>, usize, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let (v, nk, sig) = py.detach(|| pool.install(|| api.series(h, l, which))).map_err(PyValueError::new_err)?;
        Ok((arr(py, v), nk, sig))
    }

    /// Taylor-integrate a state (hi, lo; 5M) through the x points xs with Delta of z; returns the final state
    #[allow(clippy::too_many_arguments)]
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        sh: PyReadonlyArray1<'py, f64>,
        sl: PyReadonlyArray1<'py, f64>,
        xs: Vec<f64>,
    ) -> PyResult<(A1<'py>, A1<'py>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (a, b) = (slice(&sh, "sh")?, slice(&sl, "sl")?);
        if a.len() != 5 * self.api.m() || b.len() != 5 * self.api.m() {
            return Err(PyValueError::new_err("state must have 5M entries"));
        }
        let api = &self.api;
        let (oh, ol) = py.detach(|| api.integrate(h, l, a, b, &xs));
        Ok((arr(py, oh), arr(py, ol)))
    }
}

/// the SIMD backend this kernel was compiled for
#[pyfunction]
fn isa() -> &'static str {
    multidouble::native::V8::ISA
}

#[pymodule]
fn _sigmams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    Ok(())
}
