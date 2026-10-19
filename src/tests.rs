//! Backend equivalence. Every backend performs the same operations in the same order, so each one
//! this CPU can run must reproduce the baseline plain-loop residual bit for bit.

use crate::dispatch::{self, with_exec, Exec, Isa, Target};
use crate::model::Ctx;
use crate::scalar::Fld;
use num_complex::Complex64;

/// Deterministic xorshift values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// rows x cols random entries (scaled), plus `diag` on the diagonal; lanes >= cols stay zero.
fn mat<const N: usize>(rng: &mut Rng, rows: usize, cols: usize, scale: f64, diag: f64) -> Vec<Fld<f64, N>> {
    (0..rows)
        .map(|i| {
            let mut f = Fld::<f64, N>([0.0; N]);
            for j in 0..cols {
                f.0[j] = scale * rng.next() + if i == j { diag } else { 0.0 };
            }
            f
        })
        .collect()
}

/// A configuration shaped like the real one (spectral layout, nodes, frozen step counts, centre and
/// horizon series) with random matrices. It is only good for comparing backends.
fn test_ctx<const N: usize>(nh: usize, rng: &mut Rng) -> Ctx<N> {
    let (no, ne) = (12, 13);
    let nodes = vec![0.05, 0.1, 0.2, 0.3, 0.45, 0.6];
    let kn = nodes.len();
    let mut sizes = vec![no, ne];
    for _ in 0..kn {
        sizes.extend_from_slice(&[no, no, ne, ne]);
    }
    sizes.extend_from_slice(&[no, ne]);
    let mut offs = vec![0usize];
    for s in &sizes {
        offs.push(offs.last().unwrap() + s);
    }
    let len = offs.last().unwrap() + 1;
    // Even row 0 is the constant mode, so coefficient 0 sets the level of a and c.
    let mut phys_even = mat::<N>(rng, ne, nh, 0.5, 0.0);
    phys_even[0].0[..nh].fill(1.0);
    Ctx {
        nh,
        n: 3.0,
        d_odd: mat(rng, nh, nh, 0.05, 0.0),
        t_odd: mat(rng, nh, nh, 0.02, 1.0),
        t_even: mat(rng, nh, nh, 0.02, 1.0),
        phys_odd: mat(rng, no, nh, 0.5, 0.0),
        phys_even,
        spec_odd_t: mat(rng, no, nh, 0.1, 0.0),
        spec_even_t: mat(rng, ne, nh, 0.1, 0.0),
        x0: 0.01,
        nodes,
        log_steps: 6,
        seg_steps: vec![5; kn - 1],
        seg_u: vec![false, false, false, false, true],
        u0: 0.2,
        kc_terms: 6,
        ks_terms: 8,
        no,
        ne,
        offs,
        len,
        phase_idx: 3,
    }
}

/// A batch of rows: a base point, one segment perturbed, a duplicate (for dedup) and Delta perturbed.
fn test_batch<const N: usize>(ctx: &Ctx<N>, rng: &mut Rng) -> Vec<f64> {
    let mut z: Vec<f64> = (0..ctx.len).map(|_| 0.05 * rng.next()).collect();
    let ns = ctx.offs.len() - 1;
    // Constant modes: c at the centre, a ~ 1 and c ~ 2 at every node, a ~ 1 at the horizon.
    z[ctx.offs[1]] = 2.0;
    for i in 0..ctx.nodes.len() {
        z[ctx.offs[2 + 4 * i + 2]] = 1.0;
        z[ctx.offs[2 + 4 * i + 3]] = 2.0;
    }
    z[ctx.offs[ns - 1]] = 1.0;
    z[ctx.len - 1] = 3.44;
    let mut seg = z.clone();
    seg[ctx.offs[6] + 1] += 1e-3;
    let mut delta = z.clone();
    delta[ctx.len - 1] += 1e-3;
    [z.clone(), seg, z, delta].concat()
}

fn assert_bits_eq(want: &[f64], got: &[f64], what: &str) {
    assert_eq!(want.len(), got.len());
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(w.to_bits() == g.to_bits(), "{what}: element {i} differs: {w:e} vs {g:e}");
    }
}

fn check_residual<const N: usize>(nh: usize, seed: u64) {
    let mut rng = Rng(seed);
    let ctx = test_ctx::<N>(nh, &mut rng);
    let zb = test_batch(&ctx, &mut rng);
    let base = Target::plain(Isa::Baseline).unwrap();
    let mut want = vec![0.0; zb.len()];
    dispatch::residual(base, &ctx, &zb, &mut want, None, false);
    assert!(want.iter().all(|v| v.is_finite()), "baseline residual is not finite");
    assert!(want.iter().filter(|v| v.abs() > 1e-6).count() > want.len() / 2, "baseline residual is trivial");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    for t in Target::all() {
        for (pool, dedup) in [(None, false), (None, true), (Some(&pool), false), (Some(&pool), true)] {
            let mut got = vec![0.0; zb.len()];
            dispatch::residual(t, &ctx, &zb, &mut got, pool, dedup);
            let what = format!("N={N} {} threads={} dedup={dedup}", t.name(), if pool.is_some() { 2 } else { 1 });
            assert_bits_eq(&want, &got, &what);
        }
    }
}

#[test]
fn residual_folded_matches_baseline() {
    check_residual::<40>(36, 0x9e37_79b9_7f4a_7c15);
}

#[test]
fn residual_full_matches_baseline() {
    check_residual::<72>(72, 0xd1b5_4a32_d192_ed03);
}

#[test]
fn complex_residual_matches_baseline() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let ctx = test_ctx::<40>(36, &mut rng);
    let zb: Vec<Complex64> = test_batch(&ctx, &mut rng).into_iter().map(|v| Complex64::new(v, 1e-20 * rng.next())).collect();
    let mut want = vec![Complex64::new(0.0, 0.0); zb.len()];
    dispatch::residual_plain(Target::plain(Isa::Baseline).unwrap(), &ctx, &zb, &mut want, None, false);
    assert!(want.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    for t in Target::all().into_iter().filter(|t| !t.is_simd()) {
        let mut got = vec![Complex64::new(0.0, 0.0); zb.len()];
        dispatch::residual_plain(t, &ctx, &zb, &mut got, None, true);
        let flat = |v: &[Complex64]| v.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<f64>>();
        assert_bits_eq(&flat(&want), &flat(&got), &t.name());
    }
}

/// The periodic solve on its own, with kappa small enough that LU has to pivot.
#[test]
fn solve_periodic_matches_baseline() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let ctx = test_ctx::<40>(36, &mut rng);
    let mut kappa = Fld::<f64, 40>([0.0; 40]);
    let mut r = Fld::<f64, 40>([0.0; 40]);
    for j in 0..ctx.nh {
        kappa.0[j] = 0.01 * rng.next();
        r.0[j] = rng.next();
    }
    let omega = 2.0 * std::f64::consts::PI / 3.44;
    let solve = |t: Target| {
        let mut a = Vec::new();
        with_exec!(t, e => e.solve_periodic(&ctx, &kappa, &r, omega, &mut a))
    };
    let want = solve(Target::plain(Isa::Baseline).unwrap());
    // (omega D + diag(kappa)) x = r, with column i of the matrix being omega d_odd[i] + kappa_i e_i
    let mut ax = [0.0; 40];
    for i in 0..ctx.nh {
        for j in 0..ctx.nh {
            ax[j] += want.0[i] * (omega * ctx.d_odd[i].0[j] + if i == j { kappa.0[i] } else { 0.0 });
        }
    }
    for j in 0..ctx.nh {
        assert!((ax[j] - r.0[j]).abs() < 1e-9, "solve residual {} at {j}", ax[j] - r.0[j]);
    }
    for t in Target::all() {
        assert_bits_eq(&want.0, &solve(t).0, &t.name());
    }
}

#[test]
fn backend_names() {
    assert_eq!(Target::parse("simd").ok(), Target::best_simd());
    assert_eq!(Target::parse("auto").unwrap(), Target::best_plain());
    assert!(!Target::parse("baseline").unwrap().is_simd());
    assert!(Target::parse("bogus").is_err());
    for t in Target::all() {
        assert_eq!(Target::parse(&t.name()).unwrap(), t, "{}", t.name());
    }
}
