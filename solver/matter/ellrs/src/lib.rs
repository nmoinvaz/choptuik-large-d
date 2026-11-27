//! ellrs: block 12 (horizon) of the l = 1 mode matrix in double-double, ell_ddm.DDEll's discrete operator in Rust
//! (#1102). Python builds the tau operators (ell_ddm.tau_operators, mpmath rounded to double-double) and their Ozaki
//! slices, the RK4 stage abscissae (numpy's exp) and, by default, the f64 LU of the periodic solves (scipy's, called
//! back), so that on one machine the result is DDEll's bit for bit. solver/matter/ell_b12rs.py is the driver.
pub mod b12;
pub mod dd;
pub mod dt;
pub mod oz;

use b12::{LuFactory, LuSolve, Mutate, NativeLu, B12};
use dd::A;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use oz::{Oz, NSLICE};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// a 2-D array's entries in row-major order and its shape
fn owned2(a: &PyReadonlyArray2<f64>) -> (Vec<f64>, usize, usize) {
    let s = a.shape();
    (a.as_array().iter().copied().collect(), s[0], s[1])
}

fn to_py<'py>(py: Python<'py>, a: &A, h: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v = if h { a.h.clone() } else { a.l.clone() };
    Ok(Array2::from_shape_vec((a.r, a.c), v).map_err(|e| err(e.to_string()))?.into_pyarray(py))
}

/// an operator's NSLICE Ozaki slices (ell_ddm.Op.sl), each n x m; harm zeroes the top odd harmonic's rows (0) or
/// columns (1) at indices k and k + mo
fn oz_from(sl: &[PyReadonlyArray2<f64>], zero: Option<(usize, usize)>) -> PyResult<Oz> {
    if sl.len() != NSLICE {
        return Err(err(format!("an operator takes {NSLICE} slices, not {}", sl.len())));
    }
    let (_, n, m) = owned2(&sl[0]);
    let mut out = Vec::with_capacity(NSLICE);
    for a in sl {
        let (mut v, r, c) = owned2(a);
        if (r, c) != (n, m) {
            return Err(err("the slices of one operator must have one shape".into()));
        }
        if let Some((axis, mo)) = zero {
            for &k in &[mo - 1, 2 * mo - 1] {
                if axis == 0 {
                    v[k * m..(k + 1) * m].iter_mut().for_each(|x| *x = 0.0);
                } else {
                    (0..n).for_each(|i| v[i * m + k] = 0.0);
                }
            }
        }
        out.push(v);
    }
    Ok(Oz::new(n, m, out))
}

struct PyLuFactory {
    f: Py<PyAny>,
}
struct PyLuSolve {
    s: Py<PyAny>,
    n: usize,
}

impl LuFactory for PyLuFactory {
    fn factor(&self, diag: &[f64]) -> Result<Box<dyn LuSolve>, String> {
        Python::attach(|py| {
            let d = PyArray1::from_slice(py, diag);
            let s = self.f.call1(py, (d,)).map_err(|e| format!("lu factory: {e}"))?;
            Ok(Box::new(PyLuSolve { s, n: diag.len() }) as Box<dyn LuSolve>)
        })
    }
}

impl LuSolve for PyLuSolve {
    fn solve(&self, rhs: &[f64], b: usize) -> Result<Vec<f64>, String> {
        Python::attach(|py| {
            let a = PyArray1::from_slice(py, rhs).reshape([b, self.n]).map_err(|e| e.to_string())?;
            let out = self.s.call1(py, (a,)).map_err(|e| format!("lu solve: {e}"))?;
            let arr: PyReadonlyArray2<f64> = out.bind(py).extract().map_err(|e| format!("lu solve result: {e}"))?;
            if arr.shape() != [b, self.n] {
                return Err(format!("lu solve returned shape {:?}, not ({b}, {})", arr.shape(), self.n));
            }
            Ok(arr.as_array().iter().copied().collect())
        })
    }
}

/// block 12 of one DSSEll: its sizes, the operators' slices and the horizon series' order
#[pyclass]
struct Block12 {
    b: B12,
    dm_h: Vec<f64>,
    pool: Arc<rayon::ThreadPool>,
}

#[pymethods]
impl Block12 {
    /// n = S.n, ll = S.L, nn = S.N, kodd = S.kodd, ks = S.Ks, u0 = S.u0; bodd, beven, sodd, seven, dm: the five slices
    /// of DDEll's B["odd"], B["even"], Sp["odd"], Sp["even"], Dm; dm_h: ops["Dm"].h (the native LU's matrix);
    /// threads: the pool's size (0: all cores); mutate: "", "nolam" or "harm" (the controls)
    #[new]
    #[pyo3(signature = (n, ll, nn, kodd, ks, u0, bodd, beven, sodd, seven, dm, dm_h, threads=0, mutate=""))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: f64,
        ll: f64,
        nn: usize,
        kodd: Vec<f64>,
        ks: usize,
        u0: f64,
        bodd: Vec<PyReadonlyArray2<f64>>,
        beven: Vec<PyReadonlyArray2<f64>>,
        sodd: Vec<PyReadonlyArray2<f64>>,
        seven: Vec<PyReadonlyArray2<f64>>,
        dm: Vec<PyReadonlyArray2<f64>>,
        dm_h: PyReadonlyArray2<f64>,
        threads: usize,
        mutate: &str,
    ) -> PyResult<Self> {
        let mut mu = Mutate::default();
        for w in mutate.split(',').filter(|w| !w.is_empty()) {
            match w {
                "nolam" => mu.nolam = true,
                "harm" => mu.harm = true,
                _ => return Err(err(format!("unknown mutation {w}: nolam or harm"))),
            }
        }
        let mo = kodd.len();
        let zo = |axis| if mu.harm { Some((axis, mo)) } else { None };
        let bodd = oz_from(&bodd, zo(0))?;
        let sodd = oz_from(&sodd, zo(1))?;
        let (beven, seven, dm) = (oz_from(&beven, None)?, oz_from(&seven, None)?, oz_from(&dm, None)?);
        let (no, ne) = (bodd.n, beven.n);
        if no != 2 * mo || bodd.m != nn || beven.m != nn || sodd.n != nn || sodd.m != no || seven.n != nn || seven.m != ne || dm.n != nn || dm.m != nn {
            return Err(err("operator shapes do not match N, n_odd and n_even".into()));
        }
        let (dmh, r, c) = owned2(&dm_h);
        if (r, c) != (nn, nn) {
            return Err(err("dm_h must be N x N".into()));
        }
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).build().map_err(|e| err(e.to_string()))?);
        Ok(Block12 { b: B12 { n, ll, nn, no, ne, kodd, ks, u0, bodd, beven, sodd, seven, dm, mutate: mu }, dm_h: dmh, pool })
    }

    #[getter]
    fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// DDEll.block(12, x) at om, lam: (tangents hi, lo (nf x blk), value hi, lo (1 x blk)). h, e: the RK4 step in y
    /// and each step's exp(y), exp(y + h/2), exp(y + h) (steps x 3); rk=False stops after the series (S.long).
    /// lu: a callable diag -> (callable R (b x N) -> solves of (om Dm_h^T + diag(diag)) x = R per row), scipy's LU
    /// in ell_b12rs.py; None for the native LU. chunks: tangent-row groups of the RK4 (0: the pool's size).
    #[pyo3(signature = (x, om, lam, h, e, lu=None, chunks=0, rk=true))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn block<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
        om: f64,
        lam: f64,
        h: f64,
        e: PyReadonlyArray2<f64>,
        lu: Option<Py<PyAny>>,
        chunks: usize,
        rk: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let xv: Vec<f64> = x.as_array().iter().copied().collect();
        let (ev, r, c) = owned2(&e);
        if c != 3 {
            return Err(err("e must be steps x 3".into()));
        }
        let steps: Vec<[f64; 3]> = (0..r).map(|i| [ev[3 * i], ev[3 * i + 1], ev[3 * i + 2]]).collect();
        let nn = self.b.nn;
        let nc = if chunks == 0 { self.pool.current_num_threads() } else { chunks };
        let out = match lu {
            Some(f) => {
                let fac = PyLuFactory { f };
                py.detach(|| self.pool.install(|| self.b.block(&xv, om, lam, if rk { Some((h, &steps)) } else { None }, &fac, nc)))
            }
            None => {
                let mut a0 = vec![0.0; nn * nn];
                for i in 0..nn {
                    for j in 0..nn {
                        a0[i * nn + j] = om * self.dm_h[j * nn + i];
                    }
                }
                let fac = NativeLu { n: nn, a0 };
                py.detach(|| self.pool.install(|| self.b.block(&xv, om, lam, if rk { Some((h, &steps)) } else { None }, &fac, nc)))
            }
        }
        .map_err(err)?;
        Ok((to_py(py, &out.t, true)?, to_py(py, &out.t, false)?, to_py(py, &out.v, true)?, to_py(py, &out.v, false)?))
    }

    /// X @ W by the Ozaki product of one operator ("bodd", "beven", "sodd", "seven", "dm"), for the checks against
    /// ell_ddm.Op.rmul
    fn ozaki<'py>(
        &self,
        py: Python<'py>,
        which: &str,
        xh: PyReadonlyArray2<f64>,
        xl: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let op = match which {
            "bodd" => &self.b.bodd,
            "beven" => &self.b.beven,
            "sodd" => &self.b.sodd,
            "seven" => &self.b.seven,
            "dm" => &self.b.dm,
            _ => return Err(err(format!("unknown operator {which}"))),
        };
        let (h, r, c) = owned2(&xh);
        let (l, _, _) = owned2(&xl);
        let x = A { h, l, r, c };
        let y = py.detach(|| self.pool.install(|| op.rmul(&x)));
        Ok((to_py(py, &y, true)?, to_py(py, &y, false)?))
    }
}

/// the instruction set of multidouble's lanes in this build
#[pyfunction]
fn isa() -> &'static str {
    multidouble::native::V8::ISA
}

#[pymodule]
fn ellrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Block12>()?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    Ok(())
}
