//! y[i] = y[i] + l[i] t on 32 vectors of 8 lanes that stay in L1, per thread, on 1 and on N threads.
//! Usage: gpu_probe [N] (default 1). Prints axpy per second per thread and in total.
use multidouble::native::{DV, QV};
use multidouble::{dd::DD, qd::QD};
use std::hint::black_box;
use std::time::Instant;

fn bench<T: Copy, F: Fn(T, T, T) -> T>(mk: &impl Fn(usize) -> T, f: &F, reps: usize) -> f64 {
    let n = 32;
    let l: Vec<T> = (0..n).map(mk).collect();
    let mut y: Vec<T> = (0..n).map(|i| mk(i + 7)).collect();
    let t = mk(3);
    for _ in 0..reps / 20 {
        for i in 0..n {
            y[i] = f(y[i], l[i], t);
        }
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        for i in 0..n {
            y[i] = f(y[i], black_box(l[i]), t);
        }
        black_box(&mut y);
    }
    (reps * n * 8) as f64 / t0.elapsed().as_secs_f64()
}

fn run<T: Copy + Send, F: Fn(T, T, T) -> T + Sync>(name: &str, mk: impl Fn(usize) -> T + Sync, f: F, reps: usize, nt: usize) {
    let one = bench(&mk, &f, reps);
    let rates: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt).map(|_| s.spawn(|| bench(&mk, &f, reps))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let tot: f64 = rates.iter().sum();
    println!("{name}: 1 thread {one:.3e} axpy/s; {nt} threads {tot:.3e} axpy/s in total");
}

fn main() {
    let nt: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let s = |i: usize| 1.0 + (i as f64) * 1e-3 + 1.0 / (3.0 + i as f64);
    run("DV a+b*c", |i| DV::splat(DD::from_f64(s(i)) / DD::from_f64(7.0)), |a, b, c| a + b * c, 60000, nt);
    run("QV a+b*c", |i| QV::splat(QD::from_f64(s(i)) / QD::from_f64(7.0)), |a, b, c| a + b * c, 4000, nt);
    #[cfg(feature = "od")]
    {
        use multidouble::{native::OV, od::OD};
        run("OV a+b*c", |i| OV::splat(OD::from_f64(s(i)) / OD::from_f64(7.0)), |a, b, c| a + b * c, 800, nt);
    }
}
