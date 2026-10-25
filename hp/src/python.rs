//! Python bindings: `Solver(M, n, dmat, phase, nodes, segs, kt, kc, ks, threads)`.
//! Double-double arrays are passed as (hi, lo) pairs of float64 numpy arrays.
use crate::block::BlockLu;
use crate::dd::DD;
use crate::dv::DV;
use crate::field::{DMat, Fl, State};
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
    /// Ctx::f64_dd for this Jacobian (model::HDD_* flags; 0 none)
    hdd: u8,
}

trait Api: Send + Sync {
    fn len(&self) -> usize;
    fn m(&self) -> usize;
    fn residual(&self, zh: &[f64], zl: &[f64]) -> Result<V, String>;
    /// model::residual_qdh (the horizon block in quad-double, the state in four parts); refine false: the control,
    /// the horizon solves unrefined (model::QD_MUT_NOREFINE)
    fn residual_qdh(&mut self, z: [&[f64]; 4], refine: bool) -> Result<V, String>;
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
    /// f64 Jacobian, column-major: (J, ld); `F64Jac` the lanes; mutate: Ctx::mutate for this call (0 = none)
    fn jacobian_f64(&mut self, zh: &[f64], zl: &[f64], mutate: u8, how: F64Jac) -> Result<(Vec<f64>, usize), String>;
    fn plan(&self, zh: &[f64], zl: &[f64], tol: f64, hmax: f64) -> Vec<Vec<f64>>;
    fn set_segs(&mut self, segs: Vec<Vec<f64>>) -> Result<(), String>;
    fn segs(&self) -> Vec<Vec<f64>>;
    fn integrate(&self, zh: &[f64], zl: &[f64], sh: &[f64], sl: &[f64], xs: &[f64]) -> V;
    /// series coefficients (orders, 4, M); scaled: in the scaled variable of model::series_scale (c_k sigma^k), else
    /// in x (u) as before the scaling (exact: multiplied by 1/sigma k times; it may overflow where the scaled ones do not)
    fn series(&self, zh: &[f64], zl: &[f64], which: u8, scaled: bool) -> Result<V, String>;
    /// sigma of the centre (which = 0) or horizon (1) series
    fn series_scale(&self, which: u8) -> f64;
    /// Mode matrix M(lam): the Jacobian of the residual for perturbations ~ exp(lam tau) (every tau-derivative of a
    /// perturbation gains + lam), Delta held fixed: the leading (L-1) x (L-1) block (no Delta column, no phase row).
    /// Column-major with leading dimension ld_for(L-1): (hi, lo, ld).
    fn mode_matrix(&mut self, zh: &[f64], zl: &[f64], lam: DD) -> Result<(Vec<f64>, Vec<f64>, usize), String>;
    /// The f64 Jacobian built and factored in block storage (model::jacobian_block); mutate: block::MUT_* (controls);
    /// jac_mutate: Ctx::mutate for this call (0 = none; 1, no tau derivative, is used to time the tau derivative)
    fn jacobian_block(&mut self, zh: &[f64], zl: &[f64], how: F64Jac, mutate: u8, jac_mutate: u8) -> Result<BlockLu, String>;
    /// J v in double-double and the residual (model::jvp): (Jv hi, Jv lo, F hi, F lo); cut: the series stop at
    /// double-double resolution (model::HDD_DU_CUT)
    #[allow(clippy::type_complexity)]
    fn jvp(&mut self, zh: &[f64], zl: &[f64], vh: &[f64], vl: &[f64], mutate: u8, cut: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String>;
    /// Ctx::horizon_alpha (model::HALPHA_*)
    fn set_horizon_alpha(&mut self, mode: u8);
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
    fn residual_qdh(&mut self, z: [&[f64]; 4], refine: bool) -> Result<V, String> {
        // instantiated for M = 32 and 48 only (the large-D runs and their M = 48 checks): the quad-double horizon series
        // for more sizes makes the release build need more than the cloud machine's 6.27 GB
        fn qdh<const C: usize>(ctx: &mut dyn std::any::Any, z: [&[f64]; 4], refine: bool) -> Option<Result<V, String>> {
            let c = ctx.downcast_mut::<Ctx<C>>()?;
            let keep = c.mutate;
            c.mutate = if refine { 0 } else { model::QD_MUT_NOREFINE };
            let r = model::residual_qdh(c, z, true);
            c.mutate = keep;
            Some(r)
        }
        let me: &mut dyn std::any::Any = self;
        qdh::<4>(me, z, refine)
            .or_else(|| qdh::<6>(me, z, refine))
            .unwrap_or_else(|| Err(format!("residual_qdh is built for M = 32 and 48 only, not M = {}", 8 * B)))
    }
    fn jacobian(&self, zh: &[f64], zl: &[f64]) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
        model::jacobian(self, zh, zl, true)
    }
    fn jacobian_f64(&mut self, zh: &[f64], zl: &[f64], mutate: u8, how: F64Jac) -> Result<(Vec<f64>, usize), String> {
        self.mutate = mutate;
        self.f64_dd = how.hdd;
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
        self.f64_dd = 0;
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
    fn series_scale(&self, which: u8) -> f64 {
        if which == 0 {
            model::series_scale(self.nodes[0])
        } else {
            model::series_scale(self.u0().hi)
        }
    }
    fn series(&self, zh: &[f64], zl: &[f64], which: u8, scaled: bool) -> Result<V, String> {
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
        let isig = 1.0 / self.series_scale(which);
        let mut h = vec![0.0; nk * 4 * m];
        let mut l = vec![0.0; nk * 4 * m];
        for k in 0..nk {
            for q in 0..4 {
                let o = (k * 4 + q) * m;
                let mut v = s[q][k].val();
                if !scaled {
                    for _ in 0..k {
                        v = v.mul_pwr2(isig);
                    }
                }
                v.store(&mut h[o..o + m], &mut l[o..o + m]);
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
    fn jacobian_block(&mut self, zh: &[f64], zl: &[f64], how: F64Jac, mutate: u8, jac_mutate: u8) -> Result<BlockLu, String> {
        if how.hdd != 0 {
            return Err("block storage: horizon_dd is not supported (the large-D runs use kfreeze, which block storage refuses)".into());
        }
        self.mutate = jac_mutate;
        let r = match (how.dd_base, how.tangents) {
            (false, 1) => model::jacobian_block::<crate::lane::Fu, B>(self, zh, zl, true, mutate),
            (false, 4) => model::jacobian_block::<crate::lane::Fun<4>, B>(self, zh, zl, true, mutate),
            (true, 4) => model::jacobian_block::<crate::lane::Dtn<4>, B>(self, zh, zl, true, mutate),
            (d, t) => Err(format!("block storage: tangents {t}{} not built (f64x1, f64x4, dd4 are)", if d { " with dd_base" } else { "" })),
        };
        self.mutate = 0;
        r
    }
    fn jvp(&mut self, zh: &[f64], zl: &[f64], vh: &[f64], vl: &[f64], mutate: u8, cut: bool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        self.f64_dd = if cut { model::HDD_DU_CUT } else { 0 };
        let r = model::jvp(self, zh, zl, vh, vl, true, mutate);
        self.f64_dd = 0;
        r
    }
    fn set_horizon_alpha(&mut self, mode: u8) {
        self.horizon_alpha = mode;
    }
}

/// The stored LU of the last Newton step (or lu_factor): dense (DDLu, n x n) or in block storage (BlockLu).
enum Lu {
    Dense(DDLu),
    Block(BlockLu),
}

impl Lu {
    fn solve(&self, b: &mut [DD]) {
        match self {
            Lu::Dense(l) => l.solve(b),
            Lu::Block(l) => l.solve(b),
        }
    }
    fn solve_t(&self, b: &mut [DD]) {
        match self {
            Lu::Dense(l) => l.solve_t(b),
            Lu::Block(l) => l.solve_t(b),
        }
    }
    fn solve_many(&self, bh: &mut [f64], bl: &mut [f64], k: usize, transpose: bool, par: bool) {
        match self {
            Lu::Dense(l) => l.solve_many(bh, bl, k, transpose, par),
            Lu::Block(l) => l.solve_many(bh, bl, k, transpose, par),
        }
    }
    fn n(&self) -> usize {
        match self {
            Lu::Dense(l) => l.n,
            Lu::Block(l) => l.n,
        }
    }
}

/// The largest M for which the dense n x n paths (jacobian, jacobian_f64, newton, newton_mixed with block=False,
/// lm_steps, mode_matrix, mode_det, lu_factor) run. Above it the dense double-double Jacobian would take 1.3 GB or
/// more (M = 160, 13 nodes: n = 8961), and only newton_mixed in block storage is available.
const DENSE_M_MAX: usize = 128;

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
    lu: Option<Lu>,
    /// seconds of the last newton / newton_mixed call: residual, Jacobian, factorization, solve (step_timing)
    last_t: [f64; 4],
}

impl Solver {
    fn dense_ok(&self, what: &str) -> PyResult<()> {
        if self.api.m() > DENSE_M_MAX {
            return Err(PyValueError::new_err(format!(
                "{what}: the dense n x n path is limited to M <= {DENSE_M_MAX} (memory); at M = {} use newton_mixed (block storage)",
                self.api.m()
            )));
        }
        Ok(())
    }
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
                f64_dd: 0,
                horizon_alpha: 0,
            }) as Box<dyn Api>, )*
            _ => return Err(PyValueError::new_err("M must be one of 8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512")),
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
            make_ctx!(m / 8, n, dh, dl, ph, pl, nodes.clone(), vec![], kt, kc, ks, plow, match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1), [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]);
        let mn = match_node.unwrap_or(nodes.len() - 1).min(nodes.len() - 1);
        let segs = if segs.is_empty() {
            (1..nodes.len()).map(|b| if b > mn { vec![nodes[b], nodes[b - 1]] } else { vec![nodes[b - 1], nodes[b]] }).collect()
        } else {
            segs
        };
        api.set_segs(segs).map_err(PyValueError::new_err)?;
        let t = if threads == 0 { std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) } else { threads };
        // a large stack for the workers: the field arrays live on the stack and grow with M (a state of Dtn<8> lanes
        // at M = 384 is 120 kB); the pages are only touched as needed
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(t).stack_size(256 << 20).build().unwrap());
        Ok(Solver { api, pool, lu: None, last_t: [0.0; 4] })
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

    /// How the horizon series handles a = 1 + alpha (Ctx::horizon_alpha): 0 (the default) in a, as before; 1 carries
    /// alpha = a - 1, so that a^2 - 1 and a - a^3 do not cancel (a lower residual floor at large D, where the mass mode
    /// amplifies the series' rounding; the same solution); 2 the control: alpha carried, the cancelling forms kept
    /// (bit for bit the results of 0). Drops the stored LU.
    fn set_horizon_alpha(&mut self, mode: u8) -> PyResult<()> {
        if mode > model::HALPHA_MUT {
            return Err(PyValueError::new_err(format!("horizon_alpha must be 0, 1 or 2, not {mode}")));
        }
        self.lu = None;
        self.api.set_horizon_alpha(mode);
        Ok(())
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

    /// The residual with the horizon block in quad-double (model::residual_qdh; M = 32 and 48): the state in four parts
    /// z0 + z1 + z2 + z3 (multidouble's QD; z2 = z3 = 0 for a double-double state), F as (hi, lo). refine=False is the
    /// control: the horizon solves left at the double-double LU's accuracy.
    #[pyo3(signature = (z0, z1, z2, z3, refine=true))]
    #[allow(clippy::too_many_arguments)]
    fn residual_qdh<'py>(
        &mut self,
        py: Python<'py>,
        z0: PyReadonlyArray1<'py, f64>,
        z1: PyReadonlyArray1<'py, f64>,
        z2: PyReadonlyArray1<'py, f64>,
        z3: PyReadonlyArray1<'py, f64>,
        refine: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let z = [slice(&z0, "z0")?, slice(&z1, "z1")?, slice(&z2, "z2")?, slice(&z3, "z3")?];
        if z.iter().any(|p| p.len() != self.api.len()) {
            return Err(PyValueError::new_err(format!("z0..z3 must have length {}", self.api.len())));
        }
        let (api, pool) = (&mut self.api, &self.pool);
        let (fh, fl) = py.detach(|| pool.install(|| api.residual_qdh(z, refine))).map_err(PyValueError::new_err)?;
        Ok((arr(py, fh), arr(py, fl)))
    }

    /// Full Jacobian as two (L, L) arrays (row index = residual, column = unknown).
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        self.dense_ok("jacobian")?;
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
    /// horizon_dd (default 0): parts computed in double-double and rounded to f64, bit flags (model::HDD_*): 1 the
    /// horizon series' periodic solves, 2 no early end of the horizon series, 4 the horizon block's columns (the
    /// large-D fix; notes/LARGE_D_MIXED.md), 8 the last segment's columns, 16 the horizon block on a double-double
    /// base trajectory (Dtn<4>). 0 changes nothing.
    #[pyo3(signature = (zh, zl, mutate=0, tangents=4, dd_base=false, horizon_dd=0))]
    #[allow(clippy::too_many_arguments)]
    fn jacobian_f64<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        mutate: u8,
        tangents: usize,
        dd_base: bool,
        horizon_dd: u8,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.dense_ok("jacobian_f64")?;
        let how = F64Jac { tangents, dd_base, hdd: horizon_dd };
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
    /// block: build and factor the Jacobian in block storage (model::jacobian_block, block::BlockLu: only the node
    /// blocks, never the dense n x n matrix; an f64 staircase LU with partial pivoting on a system augmented by one
    /// Delta copy per residual block; lu must be "f64"; tangents 1 or 4, dd_base only with 4). None (default): block
    /// storage when M > 128, the dense path otherwise (so M <= 128 is unchanged); notes/HIGH_M.md. block_mutate:
    /// deliberate breakage of the block path for the controls of hp/highm_check.py (1 a pivot scaled by 1 + 1e-6,
    /// 2 the Delta copies untied). jac_mutate (block path only): as jacobian_f64's mutate (1 drops the tau
    /// derivative from the f64 lanes; hp/highm_check.py profile uses it to time the tau derivative). horizon_dd: as
    /// jacobian_f64's (dense path only). The seconds of the parts are kept for step_timing.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    #[pyo3(signature = (zh, zl, tangents=4, dd_base=false, lu="f64", block=None, block_mutate=0, jac_mutate=0, horizon_dd=0))]
    fn newton_mixed<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        tangents: usize,
        dd_base: bool,
        lu: &str,
        block: Option<bool>,
        block_mutate: u8,
        jac_mutate: u8,
        horizon_dd: u8,
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
        let block = block.unwrap_or(self.api.m() > DENSE_M_MAX);
        if block && !lu_f64 {
            return Err(PyValueError::new_err("block storage factors in f64 only (lu=\"f64\")"));
        }
        if !block {
            self.dense_ok("newton_mixed with block=False")?;
        }
        let (api, pool) = (&mut self.api, &self.pool);
        let n = api.len();
        let res = py
            .detach(|| {
                pool.install(|| -> Result<_, String> {
                    let mut tm = [0.0f64; 4];
                    let t0 = std::time::Instant::now();
                    let (fh, fl) = api.residual(h, l)?;
                    tm[0] = t0.elapsed().as_secs_f64();
                    let mut b: Vec<DD> = (0..n).map(|i| -DD { hi: fh[i], lo: fl[i] }).collect();
                    let how = F64Jac { tangents, dd_base, hdd: horizon_dd };
                    let lu = if block {
                        let lu = api.jacobian_block(h, l, how, block_mutate, jac_mutate)?;
                        (tm[1], tm[2]) = (lu.t_fill, lu.t_factor);
                        let t0 = std::time::Instant::now();
                        lu.solve(&mut b);
                        tm[3] = t0.elapsed().as_secs_f64();
                        Lu::Block(lu)
                    } else {
                        let t0 = std::time::Instant::now();
                        let (jh, ld) = api.jacobian_f64(h, l, 0, how)?;
                        tm[1] = t0.elapsed().as_secs_f64();
                        let t0 = std::time::Instant::now();
                        let lu = if lu_f64 {
                            DDLu::factor_f64(n, ld, jh, true)?
                        } else {
                            let jl = vec![0.0; jh.len()];
                            DDLu::factor(n, ld, jh, jl, true)?
                        };
                        tm[2] = t0.elapsed().as_secs_f64();
                        let t0 = std::time::Instant::now();
                        lu.solve(&mut b);
                        tm[3] = t0.elapsed().as_secs_f64();
                        Lu::Dense(lu)
                    };
                    Ok((b, fh, fl, lu, tm))
                })
            })
            .map_err(PyValueError::new_err)?;
        let (b, fh, fl, lu, tm) = res;
        self.last_t = tm;
        let (pmin, pmax) = match &lu {
            Lu::Dense(l) => l.pivot_range(),
            Lu::Block(l) => l.pivot_range(),
        };
        self.lu = Some(lu);
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect()), arr(py, fh), arr(py, fl), pmin, pmax))
    }

    /// The stored LU: (bytes held by its factors, in block storage, and for block storage the seconds spent building
    /// the Jacobian's blocks and factoring them).
    fn lu_info(&self) -> PyResult<(usize, bool, Option<f64>, Option<f64>)> {
        match &self.lu {
            None => Err(PyValueError::new_err("no stored LU")),
            Some(Lu::Dense(l)) => Ok((16 * l.hi.len(), false, None, None)),
            Some(Lu::Block(l)) => Ok((l.bytes(), true, Some(l.t_fill), Some(l.t_factor))),
        }
    }

    /// Seconds spent in the parts of the last newton or newton_mixed call: (residual, Jacobian, factorization, solve).
    /// In block storage the Jacobian's blocks are built and factored in turn; their sums are reported.
    fn step_timing(&self) -> (f64, f64, f64, f64) {
        (self.last_t[0], self.last_t[1], self.last_t[2], self.last_t[3])
    }

    /// Exact Jacobian-vector product: J v in double-double (model::jvp; one forward-mode pass per block, the lanes of
    /// the double-double Jacobian, with v as the tangent), where J is the exact Jacobian at z. Returns (Jv_hi, Jv_lo),
    /// or with with_residual=True (Jv_hi, Jv_lo, F_hi, F_lo), F the residual at z (computed in the same pass).
    /// mutate: deliberate breakage for the controls (1 the Delta tangent scaled by 1 + 1e-20, 2 the horizon block's
    /// state tangent dropped, 3 the phase row dropped). cut=True: the centre and horizon series stop once the
    /// tangent terms fall below double-double resolution (2^-110 of the largest, four orders in a row) instead of
    /// running to kc and ks; at large D (kc = 420) that makes a product several times cheaper, and it agrees with the
    /// full product to double-double accuracy, but not bitwise. cut=False (default): bitwise the double-double
    /// Jacobian's column for a unit vector.
    #[pyo3(signature = (zh, zl, vh, vl, mutate=0, with_residual=false, cut=false))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn jvp<'py>(
        &mut self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        vh: PyReadonlyArray1<'py, f64>,
        vl: PyReadonlyArray1<'py, f64>,
        mutate: u8,
        with_residual: bool,
        cut: bool,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (a, b) = (slice(&vh, "vh")?, slice(&vl, "vl")?);
        let (api, pool) = (&mut self.api, &self.pool);
        let (jh, jl, fh, fl) = py.detach(|| pool.install(|| api.jvp(h, l, a, b, mutate, cut))).map_err(PyValueError::new_err)?;
        let mut out = vec![arr(py, jh), arr(py, jl)];
        if with_residual {
            out.push(arr(py, fh));
            out.push(arr(py, fl));
        }
        Ok(out)
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
        self.dense_ok("mode_matrix")?;
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
        self.dense_ok("mode_det")?;
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
        self.dense_ok("newton")?;
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let n = api.len();
        let res = py.detach(|| {
            pool.install(|| -> Result<_, String> {
                let mut tm = [0.0f64; 4];
                let t0 = std::time::Instant::now();
                let (fh, fl) = api.residual(h, l)?;
                tm[0] = t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                let (jh, jl, ld) = api.jacobian(h, l)?;
                tm[1] = t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                let lu = DDLu::factor(n, ld, jh, jl, true)?;
                tm[2] = t0.elapsed().as_secs_f64();
                let t0 = std::time::Instant::now();
                let mut b: Vec<DD> = (0..n).map(|i| -DD { hi: fh[i], lo: fl[i] }).collect();
                lu.solve(&mut b);
                tm[3] = t0.elapsed().as_secs_f64();
                Ok((b, fh, fl, lu, tm))
            })
        })
        .map_err(PyValueError::new_err)?;
        let (b, fh, fl, lu, tm) = res;
        self.last_t = tm;
        let (pmin, pmax) = lu.pivot_range();
        self.lu = Some(Lu::Dense(lu));
        Ok((arr(py, b.iter().map(|v| v.hi).collect()), arr(py, b.iter().map(|v| v.lo).collect()), arr(py, fh), arr(py, fl), pmin, pmax))
    }

    /// Factor a given (L, L) matrix (hi, and lo or None for an f64 matrix) by the double-double LU and store it as the
    /// LU for chord / lu_solve / lu_solve_t / lu_solve_many (as newton does with the Jacobian). Returns
    /// (min |u_kk|, max |u_kk|).
    /// f64: factor the hi part by the f64 LU (DDLu::factor_f64) instead.
    #[pyo3(signature = (ah, al=None, f64=false))]
    fn lu_factor(&mut self, py: Python<'_>, ah: PyReadonlyArray2<'_, f64>, al: Option<PyReadonlyArray2<'_, f64>>, f64: bool) -> PyResult<(f64, f64)> {
        self.dense_ok("lu_factor")?;
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
        self.lu = Some(Lu::Dense(lu));
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
        self.dense_ok("lm_steps")?;
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
        self.lu = Some(Lu::Dense(lu));
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
        let n = lu.n();
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

    /// Series coefficients, flattened (orders, 4, M): which = 0 centre, 1 horizon; scaled: the coefficients of the
    /// scaled variable x / sigma (u / sigma), c_k sigma^k, sigma = series_scale(which), as the kernel computes them;
    /// otherwise those of x (u), which overflow to inf where c_k is beyond the f64 range.
    #[pyo3(signature = (zh, zl, which, scaled=false))]
    fn series<'py>(
        &self,
        py: Python<'py>,
        zh: PyReadonlyArray1<'py, f64>,
        zl: PyReadonlyArray1<'py, f64>,
        which: u8,
        scaled: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (h, l) = (slice(&zh, "zh")?, slice(&zl, "zl")?);
        let (api, pool) = (&self.api, &self.pool);
        let (oh, ol) = py.detach(|| pool.install(|| api.series(h, l, which, scaled))).map_err(PyValueError::new_err)?;
        Ok((arr(py, oh), arr(py, ol)))
    }

    /// The power of two sigma by which the centre (which = 0) or horizon (1) series variable is scaled
    /// (model::series_scale of the first node, or of u0 = 1 - the last node).
    fn series_scale(&self, which: u8) -> f64 {
        self.api.series_scale(which)
    }
}

/// the SIMD backend this kernel was compiled for (multidouble's compile-time backend; the generic double-double
/// functions dd_op and dd_solve moved to multidouble's Python package)
#[pyfunction]
fn isa() -> &'static str {
    V8::ISA
}

/// (z0 + z1 + z2 + z3) + (dh + dl) entrywise in quad-double (multidouble's QD::add_dd), renormalized: the update of a
/// four-part state by a double-double step. Returns the four parts.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn qd_add_dd<'py>(
    py: Python<'py>,
    z0: PyReadonlyArray1<'py, f64>,
    z1: PyReadonlyArray1<'py, f64>,
    z2: PyReadonlyArray1<'py, f64>,
    z3: PyReadonlyArray1<'py, f64>,
    dh: PyReadonlyArray1<'py, f64>,
    dl: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let z = [slice(&z0, "z0")?, slice(&z1, "z1")?, slice(&z2, "z2")?, slice(&z3, "z3")?];
    let (h, l) = (slice(&dh, "dh")?, slice(&dl, "dl")?);
    let n = h.len();
    if l.len() != n || z.iter().any(|p| p.len() != n) {
        return Err(PyValueError::new_err("qd_add_dd: all arrays must have the same length"));
    }
    let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let r = crate::qd::QD::new(z[0][i], z[1][i], z[2][i], z[3][i]).add_dd(DD { hi: h[i], lo: l[i] });
        for (o, v) in out.iter_mut().zip(r.x) {
            o[i] = v;
        }
    }
    let [a, b, c, d] = out;
    Ok((arr(py, a), arr(py, b), arr(py, c), arr(py, d)))
}

#[pymodule]
fn _hpkernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    m.add_function(wrap_pyfunction!(qd_add_dd, m)?)?;
    let _ = <DV as crate::lane::BaseLane>::TANGENT;
    Ok(())
}
