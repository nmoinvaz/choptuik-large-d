//! Python bindings for the DSSSpec residual kernel.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::avx;
use crate::model::{self, residual_batch, Backend, Ctx, Generic};
use crate::scalar::{Fld, Sc};
use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------------------------
fn pool(threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let m = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    g.entry(threads)
        .or_insert_with(|| Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()))
        .clone()
}

fn mat_rows<const N: usize>(a: &PyReadonlyArray2<f64>, name: &str, rows: usize, cols: usize) -> PyResult<Vec<Fld<f64, N>>> {
    let s = a.shape();
    if s[0] != rows || s[1] != cols || cols > N {
        return Err(PyValueError::new_err(format!("{name}: expected shape ({rows}, {cols}) with {cols} <= {N}, got {s:?}")));
    }
    let v = a.as_array();
    Ok((0..rows)
        .map(|i| {
            let mut f = Fld::<f64, N>([0.0; N]);
            for j in 0..cols {
                f.0[j] = v[[i, j]];
            }
            f
        })
        .collect())
}

fn transpose<const N: usize>(m: &[Fld<f64, N>], rows: usize, cols: usize) -> Vec<Fld<f64, N>> {
    (0..cols)
        .map(|j| {
            let mut f = Fld::<f64, N>([0.0; N]);
            for i in 0..rows {
                f.0[i] = m[i].0[j];
            }
            f
        })
        .collect()
}

struct Params<'a, 'py> {
    n: f64,
    nh: usize,
    d_odd: &'a PyReadonlyArray2<'py, f64>,
    t_odd: &'a PyReadonlyArray2<'py, f64>,
    t_even: &'a PyReadonlyArray2<'py, f64>,
    phys_odd: &'a PyReadonlyArray2<'py, f64>,
    phys_even: &'a PyReadonlyArray2<'py, f64>,
    spec_odd: &'a PyReadonlyArray2<'py, f64>,
    spec_even: &'a PyReadonlyArray2<'py, f64>,
    x0: f64,
    nodes: Vec<f64>,
    log_steps: usize,
    seg_steps: Vec<usize>,
    seg_u: Vec<bool>,
    u0: f64,
    kc_terms: usize,
    ks_terms: usize,
    sizes: Vec<usize>,
    phase_idx: usize,
}

fn build_ctx<const N: usize>(p: &Params) -> PyResult<Ctx<N>> {
    let nh = p.nh;
    let no = p.phys_odd.shape()[0];
    let ne = p.phys_even.shape()[0];
    let d_odd = mat_rows::<N>(p.d_odd, "d_odd", nh, nh)?;
    let spec_odd = mat_rows::<N>(p.spec_odd, "spec_odd", nh, no)?;
    let spec_even = mat_rows::<N>(p.spec_even, "spec_even", nh, ne)?;
    let kn = p.nodes.len();
    if p.seg_steps.len() != kn - 1 || p.seg_u.len() != kn - 1 {
        return Err(PyValueError::new_err("seg_steps / seg_u must have len(nodes) - 1 entries"));
    }
    let expect: Vec<usize> = {
        let mut v = vec![no, ne];
        for _ in 0..kn {
            v.extend_from_slice(&[no, no, ne, ne]);
        }
        v.extend_from_slice(&[no, ne]);
        v
    };
    if p.sizes != expect {
        return Err(PyValueError::new_err(format!("unexpected layout sizes {:?}", p.sizes)));
    }
    let mut offs = vec![0usize];
    for s in &p.sizes {
        offs.push(offs.last().unwrap() + s);
    }
    let len = offs.last().unwrap() + 1;
    Ok(Ctx {
        nh,
        n: p.n,
        d_odd,
        t_odd: mat_rows::<N>(p.t_odd, "t_odd", nh, nh)?,
        t_even: mat_rows::<N>(p.t_even, "t_even", nh, nh)?,
        phys_odd: mat_rows::<N>(p.phys_odd, "phys_odd", no, nh)?,
        phys_even: mat_rows::<N>(p.phys_even, "phys_even", ne, nh)?,
        spec_odd_t: transpose(&spec_odd, nh, no),
        spec_even_t: transpose(&spec_even, nh, ne),
        x0: p.x0,
        nodes: p.nodes.clone(),
        log_steps: p.log_steps,
        seg_steps: p.seg_steps.clone(),
        seg_u: p.seg_u.clone(),
        u0: p.u0,
        kc_terms: p.kc_terms,
        ks_terms: p.ks_terms,
        no,
        ne,
        offs,
        len,
        phase_idx: p.phase_idx,
    })
}

enum AnyCtx {
    F40(Ctx<40>),
    F72(Ctx<72>),
}

/// Residual kernel for one DSSSpec configuration (built by rustkernel.RustResidual from a Python S).
#[pyclass(frozen)]
struct Kernel {
    ctx: AnyCtx,
    len: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Which {
    Auto,
    Simd,
}

fn which(backend: &str) -> PyResult<Which> {
    match backend {
        "auto" | "generic" => Ok(Which::Auto),
        "simd" | "avx512" => {
            if simd_available() {
                Ok(Which::Simd)
            } else {
                Err(PyValueError::new_err("this build has no AVX-512 backend"))
            }
        }
        _ => Err(PyValueError::new_err("backend must be 'auto' or 'simd'")),
    }
}

fn run_f64<const N: usize>(ctx: &Ctx<N>, zb: &[f64], out: &mut [f64], threads: usize, w: Which, dedup: bool) {
    let p = if threads > 1 { Some(pool(threads)) } else { None };
    let pr = p.as_deref();
    match w {
        Which::Auto => residual_batch::<f64, Generic, N>(ctx, zb, out, pr, dedup),
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        Which::Simd => residual_batch::<f64, avx::Avx512, N>(ctx, zb, out, pr, dedup),
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        Which::Simd => unreachable!(),
    }
}

fn run_generic<T: Sc, const N: usize>(ctx: &Ctx<N>, zb: &[T], out: &mut [T], threads: usize, dedup: bool)
where
    Generic: Backend<T, N>,
{
    let p = if threads > 1 { Some(pool(threads)) } else { None };
    residual_batch::<T, Generic, N>(ctx, zb, out, p.as_deref(), dedup);
}

#[pymethods]
impl Kernel {
    #[new]
    #[pyo3(signature = (n, nh, d_odd, t_odd, t_even, phys_odd, phys_even, spec_odd, spec_even, x0, nodes,
                        log_steps, seg_steps, seg_u, u0, kc_terms, ks_terms, sizes, phase_idx))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: f64,
        nh: usize,
        d_odd: PyReadonlyArray2<f64>,
        t_odd: PyReadonlyArray2<f64>,
        t_even: PyReadonlyArray2<f64>,
        phys_odd: PyReadonlyArray2<f64>,
        phys_even: PyReadonlyArray2<f64>,
        spec_odd: PyReadonlyArray2<f64>,
        spec_even: PyReadonlyArray2<f64>,
        x0: f64,
        nodes: Vec<f64>,
        log_steps: usize,
        seg_steps: Vec<usize>,
        seg_u: Vec<bool>,
        u0: f64,
        kc_terms: usize,
        ks_terms: usize,
        sizes: Vec<usize>,
        phase_idx: usize,
    ) -> PyResult<Self> {
        let p = Params {
            n,
            nh,
            d_odd: &d_odd,
            t_odd: &t_odd,
            t_even: &t_even,
            phys_odd: &phys_odd,
            phys_even: &phys_even,
            spec_odd: &spec_odd,
            spec_even: &spec_even,
            x0,
            nodes,
            log_steps,
            seg_steps,
            seg_u,
            u0,
            kc_terms,
            ks_terms,
            sizes,
            phase_idx,
        };
        let ctx = if nh <= 40 {
            AnyCtx::F40(build_ctx::<40>(&p)?)
        } else if nh <= 72 {
            AnyCtx::F72(build_ctx::<72>(&p)?)
        } else {
            return Err(PyValueError::new_err("nh must be <= 72"));
        };
        let len = match &ctx {
            AnyCtx::F40(c) => c.len,
            AnyCtx::F72(c) => c.len,
        };
        Ok(Kernel { ctx, len })
    }

    /// Residual of a batch: zb (B, L) float64 -> (B, L) float64.
    /// threads: 1 = sequential, >1 = rayon pool over (batch row, shooting block).
    /// backend: 'simd' (explicit AVX-512) or 'auto' (plain loops, compiler-vectorised).
    /// dedup: compute each block once per distinct block input (bitwise-identical output).
    #[pyo3(signature = (zb, threads=1, backend="simd", dedup=false))]
    fn residual<'py>(&self, py: Python<'py>, zb: PyReadonlyArray2<'py, f64>, threads: usize, backend: &str, dedup: bool) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = which(backend)?;
        let sh = zb.shape();
        if sh[1] != self.len {
            return Err(PyValueError::new_err(format!("expected rows of length {}, got {}", self.len, sh[1])));
        }
        let b = sh[0];
        let z = zb.as_slice().map_err(|_| PyValueError::new_err("zb must be C-contiguous"))?;
        let mut out = vec![0.0f64; b * self.len];
        let ctx = &self.ctx;
        py.detach(|| match ctx {
            AnyCtx::F40(c) => run_f64(c, z, &mut out, threads, w, dedup),
            AnyCtx::F72(c) => run_f64(c, z, &mut out, threads, w, dedup),
        });
        Ok(Array2::from_shape_vec((b, self.len), out).unwrap().into_pyarray(py))
    }

    /// Complex residual (generic backend) for complex-step derivatives: zb (B, L) complex128.
    #[pyo3(signature = (zb, threads=1, dedup=false))]
    fn residual_complex<'py>(&self, py: Python<'py>, zb: PyReadonlyArray2<'py, Complex64>, threads: usize, dedup: bool) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let sh = zb.shape();
        if sh[1] != self.len {
            return Err(PyValueError::new_err(format!("expected rows of length {}, got {}", self.len, sh[1])));
        }
        let b = sh[0];
        let z = zb.as_slice().map_err(|_| PyValueError::new_err("zb must be C-contiguous"))?;
        let mut out = vec![Complex64::new(0.0, 0.0); b * self.len];
        let ctx = &self.ctx;
        py.detach(|| match ctx {
            AnyCtx::F40(c) => run_generic(c, z, &mut out, threads, dedup),
            AnyCtx::F72(c) => run_generic(c, z, &mut out, threads, dedup),
        });
        Ok(Array2::from_shape_vec((b, self.len), out).unwrap().into_pyarray(py))
    }

    /// Debug: horizon-series coefficients (P, M, A, C) of one row z, shape (4, Ks+1, n_tau).
    fn debug_ssh_series<'py>(&self, py: Python<'py>, z: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let zz = z.as_slice().map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
        let (v, nh) = match &self.ctx {
            AnyCtx::F40(c) => (model::ssh_series_row::<f64, Generic, 40>(c, &zz[..self.len]), c.nh),
            AnyCtx::F72(c) => (model::ssh_series_row::<f64, Generic, 72>(c, &zz[..self.len]), c.nh),
        };
        let rows = v.len() / nh;
        Ok(Array2::from_shape_vec((rows, nh), v).unwrap().into_pyarray(py))
    }

    /// Debug/profiling: seconds per call of each residual block (row 0 of zb), median of `reps`.
    #[pyo3(signature = (zb, backend="simd", reps=20))]
    fn debug_block_times(&self, zb: PyReadonlyArray2<f64>, backend: &str, reps: usize) -> PyResult<Vec<f64>> {
        let w = which(backend)?;
        let zz = zb.as_slice().map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
        fn go<const N: usize>(c: &Ctx<N>, z: &[f64], w: Which, reps: usize) -> Vec<f64> {
            let mut out = vec![0.0; c.block_len()];
            (0..c.nblocks())
                .map(|b| {
                    let mut t: Vec<f64> = (0..reps)
                        .map(|_| {
                            let t0 = std::time::Instant::now();
                            match w {
                                Which::Auto => model::block::<f64, Generic, N>(c, z, b, &mut out),
                                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                                Which::Simd => model::block::<f64, avx::Avx512, N>(c, z, b, &mut out),
                                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                                Which::Simd => unreachable!(),
                            }
                            std::hint::black_box(&out);
                            t0.elapsed().as_secs_f64()
                        })
                        .collect();
                    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    t[t.len() / 2]
                })
                .collect()
        }
        Ok(match &self.ctx {
            AnyCtx::F40(c) => go(c, &zz[..self.len], w, reps),
            AnyCtx::F72(c) => go(c, &zz[..self.len], w, reps),
        })
    }

    /// Debug/profiling: seconds per periodic solve (omega D + diag(kappa)) x = r, median of reps.
    #[pyo3(signature = (omega, kappa, backend="simd", reps=2000))]
    fn debug_solve_time(&self, omega: f64, kappa: f64, backend: &str, reps: usize) -> PyResult<f64> {
        let w = which(backend)?;
        fn go<const N: usize>(c: &Ctx<N>, omega: f64, kappa: f64, w: Which, reps: usize) -> f64 {
            let mut kap = Fld::<f64, N>([0.0; N]);
            let mut r = Fld::<f64, N>([0.0; N]);
            for j in 0..c.nh {
                kap.0[j] = kappa + 0.1 * ((j * 7 % 11) as f64);
                r.0[j] = ((j * 13 % 17) as f64) - 8.0;
            }
            let mut a = Vec::with_capacity(c.nh);
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let x = match w {
                    Which::Auto => model::solve_periodic::<f64, Generic, N>(c, &kap, &r, omega, &mut a),
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                    Which::Simd => model::solve_periodic::<f64, avx::Avx512, N>(c, &kap, &r, omega, &mut a),
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                    Which::Simd => unreachable!(),
                };
                std::hint::black_box(&x);
            }
            t0.elapsed().as_secs_f64() / reps as f64
        }
        Ok(match &self.ctx {
            AnyCtx::F40(c) => go(c, omega, kappa, w, reps),
            AnyCtx::F72(c) => go(c, omega, kappa, w, reps),
        })
    }

    #[getter]
    fn n_tau(&self) -> usize {
        match &self.ctx {
            AnyCtx::F40(c) => c.nh,
            AnyCtx::F72(c) => c.nh,
        }
    }
}

#[pyfunction]
fn simd_available() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
}

#[pyfunction]
fn build_info() -> String {
    let mut feats = vec![];
    if cfg!(target_feature = "avx512f") {
        feats.push("avx512f");
    }
    if cfg!(target_feature = "avx2") {
        feats.push("avx2");
    }
    if cfg!(target_feature = "fma") {
        feats.push("fma");
    }
    format!("target features: {}", feats.join(","))
}

#[pymodule]
fn _rustkernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Kernel>()?;
    m.add_function(wrap_pyfunction!(simd_available, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
