//! det M(lambda) of the double-double DSS kernel (hp/) at complex lambda, and the same for a test scalar field.
//!
//! hpkernel's `Solver.mode_det(zh, zl, lam_hi, lam_lo)` takes lambda as a real double-double (hi, lo), so the growth
//! rate can only be real there. The mode matrix here is hp's at a complex rate: multishoot's `Shooting::mode_matrix` on
//! hp's equations (`model::Ctx::shooting`, `model::Dss`) at `Rate` lr + i li, with complex tangents (multidouble's
//! `Dc`), the horizon solves of `model::psolve` (the 2M x 2M real form [[A + lr, -li], [li, A + lr]] at a complex rate)
//! and the frozen rows of the engine's exact projector (`ProjectedRows`, octo-double). So this crate holds no copy of
//! the linearisation: with li = 0 its matrix is hpkernel's `Solver.mode_matrix` bit for bit (hp/modecount_check.py).
//! On top of it:
//!
//! * multidouble's `linalg::clu_logdet`: complex double-double LU with partial pivoting (getf2 order, 8-row lane
//!   vectors): ln|det| and arg det;
//! * `pivots::clu_pivots`: the same LU with every pivot recorded (`mode_pivots`, #552);
//! * `test_field`: the test field's mode matrix (`test_det`, `test_matrix`, `test_matvec`), which holds the a and c
//!   tangents inside the recursions and so keeps its own copy of them (see there).
use hpk::dd::DD;
use hpk::dv::DV;
use hpk::field::{DMat, Fl};
use hpk::linalg::{clu_logdet, ld_for};
use hpk::model::{self as hm, Ctx};
use multishoot::Rate;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

mod pivots;
mod test_field;

use test_field::{psi_index, test_matrix_c, Cx, TEST_MUT, TEST_ON};

/// The mode matrix M(lr + i li) of hp's engine, (L-1) x (L-1) (the block rows and the unknowns before Delta, which is
/// held fixed with the phase row), column-major with leading dimension ld_for(L-1): (re_hi, re_lo, im_hi, im_lo, ld)
#[allow(clippy::type_complexity)]
pub fn mode_matrix_c<const B: usize>(ctx: &Ctx<B>, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize), String> {
    let mm = ctx.shooting().mode_matrix(&[zh, zl], Rate::new(lr, li), par)?;
    let n = ctx.len() - 1;
    if mm.n != n || mm.ld != ld_for(n) {
        return Err(format!("mode matrix of order {} (leading dimension {}), expected {n} ({})", mm.n, mm.ld, ld_for(n)));
    }
    let [rh, rl]: [Vec<f64>; 2] = mm.re.try_into().map_err(|_| "mode matrix: two real parts".to_string())?;
    let [ih, il]: [Vec<f64>; 2] = mm.im.try_into().map_err(|_| "mode matrix: two imaginary parts".to_string())?;
    Ok((rh, rl, ih, il, mm.ld))
}



// ---------------------------------------------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------------------------------------------
type CMat = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize);

trait CApi: Send + Sync {
    fn len(&self) -> usize;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn mode_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<CMat, String>;
    fn test_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, test: u8, par: bool) -> Result<CMat, String>;
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>);
}

impl<const B: usize> CApi for Ctx<B> {
    fn len(&self) -> usize {
        Ctx::<B>::len(self)
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
    fn mode_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, par: bool) -> Result<CMat, String> {
        mode_matrix_c(self, zh, zl, lr, li, par)
    }
    fn test_matrix_c(&self, zh: &[f64], zl: &[f64], lr: DD, li: DD, test: u8, par: bool) -> Result<CMat, String> {
        test_matrix_c(&Cx { ctx: self, lr, li, test }, zh, zl, par)
    }
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>) {
        psi_index(self)
    }
}

/// (Pi_low, Pi_high) from plow, which must be the projector onto the odd harmonics k <= kfreeze (model::check_freeze,
/// as hpkernel's Solver): the engine's frozen rows take only that cutoff and form their own exact projector, and the
/// test field's rows use this pair
fn freeze_pair<const B: usize>(p: &Option<(Vec<f64>, Vec<f64>)>) -> PyResult<Option<(DMat<B>, DMat<B>)>> {
    let pair = p.as_ref().map(|(h, l)| {
        let m = 8 * B;
        let lo = DMat::<B>::from_rows(h, l);
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
    });
    if let Some((lo, _)) = &pair {
        hm::check_freeze(lo).map_err(PyValueError::new_err)?;
    }
    Ok(pair)
}

macro_rules! make_ctx {
    ($b:expr, $n:expr, $dh:expr, $dl:expr, $ph:expr, $pl:expr, $nodes:expr, $kt:expr, $kc:expr, $ks:expr, $plow:expr, $mn:expr, [$($bb:literal),*]) => {
        match $b {
            $( $bb => Box::new(Ctx::<$bb> {
                n: $n,
                d: DMat::<$bb>::from_rows($dh, $dl),
                phase: Fl::<DV, $bb>::load($ph, $pl),
                nodes: $nodes,
                segs: vec![],
                kt: $kt,
                kc: $kc,
                ks: $ks,
                freeze: freeze_pair::<$bb>(&$plow)?,
                match_node: $mn,
                lam: DD::ZERO,
                mutate: 0,
                f64_dd: 0,
                horizon_alpha: 0,
            }) as Box<dyn CApi>, )*
            _ => return Err(PyValueError::new_err("M must be one of 16, 32, 40, 48, 64, 80")),
        }
    };
}

fn slice<'a>(a: &'a PyReadonlyArray1<f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

fn test_mode(mode: u8) -> PyResult<u8> {
    if mode == TEST_ON || mode == TEST_MUT {
        Ok(mode)
    } else {
        Err(PyValueError::new_err(format!("mode must be 1 (the test field) or 2 (its control), not {mode}")))
    }
}

/// (n, n) arrays (row index first) from a column-major matrix with leading dimension ld
fn to2(v: &[f64], n: usize, ld: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        for r in 0..n {
            a[[r, c]] = v[c * ld + r];
        }
    }
    a
}

/// Same constructor as hpkernel.Solver; segs must be given (take them from hpkernel's Solver.get_segs()). The layout
/// is the default one (Ctx::horizon_alpha 0, no deliberate breakage).
#[pyclass]
struct CSolver {
    api: Box<dyn CApi>,
    pool: Arc<rayon::ThreadPool>,
}

#[pymethods]
impl CSolver {
    #[new]
    #[pyo3(signature = (m, n_hi, n_lo, dmat_hi, dmat_lo, phase_hi, phase_lo, nodes, segs, kt, kc, ks, threads=1, plow_hi=None, plow_lo=None, match_node=None))]
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
        let dh = dmat_hi.as_slice().map_err(|_| PyValueError::new_err("dmat_hi must be C-contiguous"))?;
        let dl = dmat_lo.as_slice().map_err(|_| PyValueError::new_err("dmat_lo must be C-contiguous"))?;
        if m % 8 != 0 || dh.len() != m * m || dl.len() != m * m {
            return Err(PyValueError::new_err("dmat must be (M, M), M a multiple of 8"));
        }
        let ph = slice(&phase_hi, "phase_hi")?;
        let pl = slice(&phase_lo, "phase_lo")?;
        let n = DD::new(n_hi, n_lo);
        let plow: Option<(Vec<f64>, Vec<f64>)> = match (plow_hi, plow_lo) {
            (Some(a), Some(b)) => Some((
                a.as_slice().map_err(|_| PyValueError::new_err("plow"))?.to_vec(),
                b.as_slice().map_err(|_| PyValueError::new_err("plow"))?.to_vec(),
            )),
            _ => None,
        };
        let mn = match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1);
        let mut api: Box<dyn CApi> = make_ctx!(m / 8, n, dh, dl, ph, pl, nodes.clone(), kt, kc, ks, plow, mn, [2, 4, 5, 6, 8, 10]);
        api.set_segs(segs).map_err(PyValueError::new_err)?;
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap());
        Ok(CSolver { api, pool })
    }

    #[getter]
    fn size(&self) -> usize {
        self.api.len()
    }

    /// det M(lam) for lam = (lr_hi + lr_lo) + i (li_hi + li_lo): (ln|det|, arg det in (-pi, pi], min |u_kk|, max |u_kk|).
    #[pyo3(signature = (zh, zl, lr_hi, li_hi, lr_lo=0.0, li_lo=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn mode_det(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, lr_hi: f64, li_hi: f64, lr_lo: f64, li_lo: f64) -> PyResult<(f64, f64, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len() - 1;
        let (lr, li) = (DD::new(lr_hi, lr_lo), DD::new(li_hi, li_lo));
        py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let (rh, rl, ih, il, ld) = api.mode_matrix_c(h, l, lr, li, true)?;
                clu_logdet(n, ld, rh, rl, ih, il, true)
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// det M(lam) with every pivot of its LU: (ln|det|, arg det, min |u_kk|, max |u_kk|) as `mode_det` returns them,
    /// bit for bit, then ln|u_kk|, arg u_kk and the pivot row chosen at each column k, as arrays in elimination order.
    #[pyo3(signature = (zh, zl, lr_hi, li_hi, lr_lo=0.0, li_lo=0.0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn mode_pivots<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lr_hi: f64,
        li_hi: f64,
        lr_lo: f64,
        li_lo: f64,
    ) -> PyResult<(f64, f64, f64, f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len() - 1;
        let (lr, li) = (DD::new(lr_hi, lr_lo), DD::new(li_hi, li_lo));
        let (la, arg, pmin, pmax, lnu, argu, rows) = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let (rh, rl, ih, il, ld) = api.mode_matrix_c(h, l, lr, li, true)?;
                    pivots::clu_pivots(n, ld, rh, rl, ih, il, true)
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok((la, arg, pmin, pmax, lnu.into_pyarray(py), argu.into_pyarray(py), rows.into_pyarray(py)))
    }

    /// The complex mode matrix as four (L-1, L-1) arrays (re_hi, re_lo, im_hi, im_lo).
    #[allow(clippy::type_complexity)]
    fn mode_matrix<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len() - 1;
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.mode_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), true)))
            .map_err(PyValueError::new_err)?;
        Ok((to2(&rh, n, ld).into_pyarray(py), to2(&rl, n, ld).into_pyarray(py), to2(&ih, n, ld).into_pyarray(py), to2(&il, n, ld).into_pyarray(py)))
    }

    /// The test field's det M_psi(lam) (mode 1; 2 the control, a's tangent kept): (ln|det|, arg det, min |u_kk|,
    /// max |u_kk|).
    #[pyo3(signature = (zh, zl, lr_hi, li_hi, lr_lo=0.0, li_lo=0.0, mode=1))]
    #[allow(clippy::too_many_arguments)]
    fn test_det(&self, py: Python<'_>, zh: PyReadonlyArray1<'_, f64>, zl: PyReadonlyArray1<'_, f64>, lr_hi: f64, li_hi: f64, lr_lo: f64, li_lo: f64, mode: u8) -> PyResult<(f64, f64, f64, f64)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (lr, li) = (DD::new(lr_hi, lr_lo), DD::new(li_hi, li_lo));
        py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let (rh, rl, ih, il, ld) = api.test_matrix_c(h, l, lr, li, test, true)?;
                let n = rh.len() / ld;
                clu_logdet(n, ld, rh, rl, ih, il, true)
            })
        })
        .map_err(PyValueError::new_err)
    }

    /// The test field's mode matrix M_psi(lam) as four (N, N) arrays (re_hi, re_lo, im_hi, im_lo), N = 2M(K+1).
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (zh, zl, lr, li, mode=1))]
    fn test_matrix<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
        mode: u8,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.test_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), test, true)))
            .map_err(PyValueError::new_err)?;
        let n = rh.len() / ld;
        Ok((to2(&rh, n, ld).into_pyarray(py), to2(&rl, n, ld).into_pyarray(py), to2(&ih, n, ld).into_pyarray(py), to2(&il, n, ld).into_pyarray(py)))
    }

    /// M_psi(lam) v in double-double for a real double-double vector v (length N): (re_hi, re_lo, im_hi, im_lo).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    #[pyo3(signature = (zh, zl, vh, vl, lr, li, mode=1))]
    fn test_matvec<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        vh: PyReadonlyArray1<'py, f64>,
        vl: PyReadonlyArray1<'py, f64>,
        lr: f64,
        li: f64,
        mode: u8,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (vh, vl) = (slice(&vh, "vh")?.to_vec(), slice(&vl, "vl")?.to_vec());
        let test = test_mode(mode)?;
        let (api, pool) = (&self.api, &self.pool);
        let (rh, rl, ih, il, ld) = py
            .detach(|| pool.install(|| api.test_matrix_c(h, l, DD::from_f64(lr), DD::from_f64(li), test, true)))
            .map_err(PyValueError::new_err)?;
        let n = rh.len() / ld;
        if vh.len() != n || vl.len() != n {
            return Err(PyValueError::new_err(format!("v must have length {n}")));
        }
        let mut re = vec![DD::ZERO; n];
        let mut im = vec![DD::ZERO; n];
        for j in 0..n {
            let v = DD { hi: vh[j], lo: vl[j] };
            for i in 0..n {
                let k = j * ld + i;
                re[i] = re[i] + DD { hi: rh[k], lo: rl[k] } * v;
                im[i] = im[i] + DD { hi: ih[k], lo: il[k] } * v;
            }
        }
        let hi = |x: &[DD]| x.iter().map(|d| d.hi).collect::<Vec<f64>>().into_pyarray(py);
        let lo = |x: &[DD]| x.iter().map(|d| d.lo).collect::<Vec<f64>>().into_pyarray(py);
        Ok((hi(&re), lo(&re), hi(&im), lo(&im)))
    }

    /// (rows, cols): the psi rows and columns of the mode matrix (indices into its L-1 rows and columns)
    fn psi_index(&self) -> (Vec<usize>, Vec<usize>) {
        self.api.psi_index()
    }
}

/// ln|det| and arg det of a complex matrix given as four (n, n) double-double arrays (row index first): a test of
/// the complex LU against numpy / mpmath.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn cdet<'py>(rh: PyReadonlyArray2<'py, f64>, rl: PyReadonlyArray2<'py, f64>, ih: PyReadonlyArray2<'py, f64>, il: PyReadonlyArray2<'py, f64>) -> PyResult<(f64, f64, f64, f64)> {
    let a = rh.as_array();
    let n = a.shape()[0];
    let ld = ld_for(n);
    let mut v = [vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n]];
    for (q, src) in [rh.as_array(), rl.as_array(), ih.as_array(), il.as_array()].iter().enumerate() {
        for c in 0..n {
            for r in 0..n {
                v[q][c * ld + r] = src[[r, c]];
            }
        }
    }
    let [a0, a1, a2, a3] = v;
    clu_logdet(n, ld, a0, a1, a2, a3, false).map_err(PyValueError::new_err)
}

/// `cdet` with every pivot: (ln|det|, arg det, min |u_kk|, max |u_kk|, ln|u_kk|, arg u_kk, pivot rows), by the LU of
/// `mode_pivots`, for planted matrices (#552).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn cpivots<'py>(
    py: Python<'py>,
    rh: PyReadonlyArray2<'py, f64>,
    rl: PyReadonlyArray2<'py, f64>,
    ih: PyReadonlyArray2<'py, f64>,
    il: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, f64, f64, f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>)> {
    let n = rh.as_array().shape()[0];
    let ld = ld_for(n);
    let mut v = [vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n], vec![0.0; ld * n]];
    for (q, src) in [rh.as_array(), rl.as_array(), ih.as_array(), il.as_array()].iter().enumerate() {
        for c in 0..n {
            for r in 0..n {
                v[q][c * ld + r] = src[[r, c]];
            }
        }
    }
    let [a0, a1, a2, a3] = v;
    let (la, arg, pmin, pmax, lnu, argu, rows) = pivots::clu_pivots(n, ld, a0, a1, a2, a3, false).map_err(PyValueError::new_err)?;
    Ok((la, arg, pmin, pmax, lnu.into_pyarray(py), argu.into_pyarray(py), rows.into_pyarray(py)))
}

#[pymodule]
fn modecount_cx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CSolver>()?;
    m.add_function(wrap_pyfunction!(cdet, m)?)?;
    m.add_function(wrap_pyfunction!(cpivots, m)?)?;
    Ok(())
}
