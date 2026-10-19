//! Runtime instruction-set selection.
//!
//! The crate is compiled for the target's baseline, so it loads on any CPU of the architecture.
//! Each residual block runs inside a `#[target_feature]` entry point for the instruction set picked
//! at runtime, and the model code below that entry point is `#[inline(always)]`, so it is compiled
//! with the same features. Plain loops get hardware FMA and wider auto-vectorisation there, and the
//! explicit SIMD kernels inline into the block.

use crate::model::{self, Backend, Ctx};
use crate::scalar::{Fld, Sc, State};
use std::marker::PhantomData;
use std::sync::OnceLock;

/// Instruction set the kernels run with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// The compile target's baseline (SSE2 on x86_64).
    Baseline,
    /// x86_64 AVX2 and FMA.
    Avx2,
    /// x86_64 AVX-512F, with AVX2 and FMA.
    Avx512,
    /// aarch64 Advanced SIMD (part of the aarch64 baseline, so only its explicit kernels differ).
    Neon,
}

impl Isa {
    /// Instruction sets of this architecture, best first.
    pub fn candidates() -> &'static [Isa] {
        #[cfg(target_arch = "x86_64")]
        {
            &[Isa::Avx512, Isa::Avx2, Isa::Baseline]
        }
        #[cfg(target_arch = "aarch64")]
        {
            &[Isa::Neon, Isa::Baseline]
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            &[Isa::Baseline]
        }
    }

    /// Whether this CPU (and OS) supports the instruction set.
    pub fn available(self) -> bool {
        match self {
            Isa::Baseline => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => Isa::Avx2.available() && is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Whether this build has explicit SIMD kernels for the instruction set.
    pub fn has_simd_backend(self) -> bool {
        match self {
            Isa::Avx512 | Isa::Avx2 => cfg!(target_arch = "x86_64"),
            Isa::Neon => cfg!(target_arch = "aarch64"),
            Isa::Baseline => false,
        }
    }

    /// Whether the instruction set may be used: the CPU supports it, and RUSTKERNEL_ISA (e.g.
    /// "avx2" or "baseline", read once per process) does not cap it away.
    pub fn enabled(self) -> bool {
        static CAP: OnceLock<usize> = OnceLock::new();
        let cands = Isa::candidates();
        let cap = *CAP.get_or_init(|| {
            let cap = std::env::var("RUSTKERNEL_ISA").ok().and_then(|v| Isa::from_name(v.trim()));
            cap.and_then(|c| cands.iter().position(|&i| i == c)).unwrap_or(0)
        });
        cands.iter().position(|&i| i == self).is_some_and(|p| p >= cap) && self.available()
    }

    /// Best enabled instruction set.
    pub fn best() -> Isa {
        Isa::candidates().iter().copied().find(|i| i.enabled()).unwrap_or(Isa::Baseline)
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Baseline => "baseline",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Neon => "neon",
        }
    }

    pub fn from_name(name: &str) -> Option<Isa> {
        match name.to_ascii_lowercase().as_str() {
            "baseline" | "scalar" => Some(Isa::Baseline),
            "avx2" => Some(Isa::Avx2),
            "avx512" | "avx512f" => Some(Isa::Avx512),
            "neon" => Some(Isa::Neon),
            _ => None,
        }
    }
}

/// A backend choice checked against this CPU and RUSTKERNEL_ISA: explicit SIMD kernels or plain
/// loops, compiled for one instruction set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    isa: Isa,
    simd: bool,
}

impl Target {
    /// Explicit SIMD kernels for `isa`, if this build has them and `isa` is enabled.
    pub fn simd(isa: Isa) -> Option<Target> {
        (isa.has_simd_backend() && isa.enabled()).then_some(Target { isa, simd: true })
    }

    /// Plain loops (`model::Generic`) compiled for `isa`, if it is enabled.
    pub fn plain(isa: Isa) -> Option<Target> {
        isa.enabled().then_some(Target { isa, simd: false })
    }

    /// The best enabled explicit SIMD backend.
    pub fn best_simd() -> Option<Target> {
        Isa::candidates().iter().find_map(|&i| Target::simd(i))
    }

    /// Plain loops compiled for `Isa::best()`.
    pub fn best_plain() -> Target {
        Target { isa: Isa::best(), simd: false }
    }

    /// Every enabled backend, best first.
    pub fn all() -> Vec<Target> {
        let cands = Isa::candidates();
        let simd = cands.iter().filter_map(|&i| Target::simd(i));
        simd.chain(cands.iter().filter_map(|&i| Target::plain(i))).collect()
    }

    /// Parses a backend name. "simd" is the best explicit SIMD backend, "auto" (or "generic") is
    /// plain loops for the best instruction set, "auto:avx2" plain loops for a given one, "baseline"
    /// plain loops without extra features, and an instruction set name ("avx512") selects its
    /// explicit SIMD kernels.
    pub fn parse(name: &str) -> Result<Target, String> {
        match name {
            "simd" => Target::best_simd().ok_or_else(|| "no explicit SIMD backend for this CPU".to_string()),
            "auto" | "generic" => Ok(Target::best_plain()),
            "baseline" | "scalar" => Ok(Target { isa: Isa::Baseline, simd: false }),
            _ if name.starts_with("auto:") => {
                let isa = Isa::from_name(&name[5..]).ok_or_else(|| format!("unknown instruction set in '{name}'"))?;
                Target::plain(isa).ok_or_else(|| format!("'{}' is not enabled on this CPU", &name[5..]))
            }
            _ => {
                let isa = Isa::from_name(name).ok_or_else(|| {
                    format!("unknown backend '{name}' (expected 'simd', 'auto', 'baseline' or an instruction set)")
                })?;
                Target::simd(isa).ok_or_else(|| format!("the '{name}' backend is not enabled on this CPU"))
            }
        }
    }

    pub fn isa(self) -> Isa {
        self.isa
    }

    pub fn is_simd(self) -> bool {
        self.simd
    }

    /// The name `parse` accepts for this target ("avx512", "auto:avx2", "baseline").
    pub fn name(self) -> String {
        match (self.simd, self.isa) {
            (true, isa) => isa.name().to_string(),
            (false, Isa::Baseline) => "baseline".to_string(),
            (false, isa) => format!("auto:{}", isa.name()),
        }
    }
}

/// The model's entry points as the residual driver calls them: one backend under one instruction
/// set.
pub trait Exec<T: Sc, const N: usize>: Sync {
    fn block_state(&self, ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N>;
    fn block_out(&self, ctx: &Ctx<N>, z: &[T], b: usize, s: &State<T, N>, out: &mut [T]);
    fn solve_periodic(&self, ctx: &Ctx<N>, kappa: &Fld<T, N>, r: &Fld<T, N>, omega: T, a: &mut Vec<Fld<T, N>>) -> Fld<T, N>;
    fn ssh_series_row(&self, ctx: &Ctx<N>, z: &[T]) -> Vec<T>;

    /// Residual block b of one batch row (`model::block`).
    fn block(&self, ctx: &Ctx<N>, z: &[T], b: usize, out: &mut [T]) {
        let s = self.block_state(ctx, z, b);
        self.block_out(ctx, z, b, &s, out);
    }
}

/// Runs the model's entry points with one instruction set's target features enabled.
///
/// # Safety
/// The methods may only be called when `ISA.available()` is true.
pub unsafe trait Level: 'static {
    const ISA: Isa;
    unsafe fn block_state<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N>;
    unsafe fn block_out<T: Sc, B: Backend<T, N>, const N: usize>(
        ctx: &Ctx<N>,
        z: &[T],
        b: usize,
        s: &State<T, N>,
        out: &mut [T],
    );
    unsafe fn solve_periodic<T: Sc, B: Backend<T, N>, const N: usize>(
        ctx: &Ctx<N>,
        kappa: &Fld<T, N>,
        r: &Fld<T, N>,
        omega: T,
        a: &mut Vec<Fld<T, N>>,
    ) -> Fld<T, N>;
    unsafe fn ssh_series_row<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T]) -> Vec<T>;
}

/// Defines a `Level` whose entry points carry the given attributes (the target features).
macro_rules! level {
    ($(#[$feat:meta])* $name:ident, $isa:expr) => {
        pub struct $name;

        unsafe impl Level for $name {
            const ISA: Isa = $isa;

            #[inline(always)]
            unsafe fn block_state<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N> {
                $(#[$feat])*
                unsafe fn f<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N> {
                    model::block_state::<T, B, N>(ctx, z, b)
                }
                unsafe { f::<T, B, N>(ctx, z, b) }
            }

            #[inline(always)]
            unsafe fn block_out<T: Sc, B: Backend<T, N>, const N: usize>(
                ctx: &Ctx<N>,
                z: &[T],
                b: usize,
                s: &State<T, N>,
                out: &mut [T],
            ) {
                $(#[$feat])*
                unsafe fn f<T: Sc, B: Backend<T, N>, const N: usize>(
                    ctx: &Ctx<N>,
                    z: &[T],
                    b: usize,
                    s: &State<T, N>,
                    out: &mut [T],
                ) {
                    model::block_out::<T, B, N>(ctx, z, b, s, out)
                }
                unsafe { f::<T, B, N>(ctx, z, b, s, out) }
            }

            #[inline(always)]
            unsafe fn solve_periodic<T: Sc, B: Backend<T, N>, const N: usize>(
                ctx: &Ctx<N>,
                kappa: &Fld<T, N>,
                r: &Fld<T, N>,
                omega: T,
                a: &mut Vec<Fld<T, N>>,
            ) -> Fld<T, N> {
                $(#[$feat])*
                unsafe fn f<T: Sc, B: Backend<T, N>, const N: usize>(
                    ctx: &Ctx<N>,
                    kappa: &Fld<T, N>,
                    r: &Fld<T, N>,
                    omega: T,
                    a: &mut Vec<Fld<T, N>>,
                ) -> Fld<T, N> {
                    model::solve_periodic::<T, B, N>(ctx, kappa, r, omega, a)
                }
                unsafe { f::<T, B, N>(ctx, kappa, r, omega, a) }
            }

            #[inline(always)]
            unsafe fn ssh_series_row<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T]) -> Vec<T> {
                $(#[$feat])*
                unsafe fn f<T: Sc, B: Backend<T, N>, const N: usize>(ctx: &Ctx<N>, z: &[T]) -> Vec<T> {
                    model::ssh_series_row::<T, B, N>(ctx, z)
                }
                unsafe { f::<T, B, N>(ctx, z) }
            }
        }
    };
}

level!(OnBaseline, Isa::Baseline);
#[cfg(target_arch = "x86_64")]
level!(#[target_feature(enable = "avx2,fma")] OnAvx2, Isa::Avx2);
#[cfg(target_arch = "x86_64")]
level!(#[target_feature(enable = "avx512f,avx2,fma")] OnAvx512, Isa::Avx512);
#[cfg(target_arch = "aarch64")]
level!(#[target_feature(enable = "neon")] OnNeon, Isa::Neon);

/// Backend B run under level L.
pub struct On<B, L>(PhantomData<fn() -> (B, L)>);

impl<B, L: Level> On<B, L> {
    /// # Safety
    /// `L::ISA` must be available on this CPU.
    pub unsafe fn new() -> Self {
        On(PhantomData)
    }
}

impl<T: Sc, B: Backend<T, N>, L: Level, const N: usize> Exec<T, N> for On<B, L> {
    #[inline(always)]
    fn block_state(&self, ctx: &Ctx<N>, z: &[T], b: usize) -> State<T, N> {
        // SAFETY: an On<B, L> only exists once L::ISA was found available (On::new).
        unsafe { L::block_state::<T, B, N>(ctx, z, b) }
    }

    #[inline(always)]
    fn block_out(&self, ctx: &Ctx<N>, z: &[T], b: usize, s: &State<T, N>, out: &mut [T]) {
        unsafe { L::block_out::<T, B, N>(ctx, z, b, s, out) }
    }

    #[inline(always)]
    fn solve_periodic(&self, ctx: &Ctx<N>, kappa: &Fld<T, N>, r: &Fld<T, N>, omega: T, a: &mut Vec<Fld<T, N>>) -> Fld<T, N> {
        unsafe { L::solve_periodic::<T, B, N>(ctx, kappa, r, omega, a) }
    }

    #[inline(always)]
    fn ssh_series_row(&self, ctx: &Ctx<N>, z: &[T]) -> Vec<T> {
        unsafe { L::ssh_series_row::<T, B, N>(ctx, z) }
    }
}

/// Evaluates `$body` with `$e` bound to the `Exec` that the `Target` selects. The body is compiled
/// once per backend; it must work in f64, which the explicit SIMD backends are limited to.
macro_rules! with_exec {
    ($target:expr, $e:ident => $body:expr) => {{
        use $crate::dispatch::{Isa, On};
        let t: $crate::dispatch::Target = $target;
        // SAFETY (each arm): a Target only exists for enabled instruction sets.
        match (t.isa(), t.is_simd()) {
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx512, true) => {
                let $e = unsafe { On::<$crate::avx512::Avx512, $crate::dispatch::OnAvx512>::new() };
                $body
            }
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx512, false) => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnAvx512>::new() };
                $body
            }
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx2, true) => {
                let $e = unsafe { On::<$crate::simd::avx2::Avx2, $crate::dispatch::OnAvx2>::new() };
                $body
            }
            #[cfg(target_arch = "x86_64")]
            (Isa::Avx2, false) => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnAvx2>::new() };
                $body
            }
            #[cfg(target_arch = "aarch64")]
            (Isa::Neon, true) => {
                let $e = unsafe { On::<$crate::simd::neon::Neon, $crate::dispatch::OnNeon>::new() };
                $body
            }
            #[cfg(target_arch = "aarch64")]
            (Isa::Neon, false) => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnNeon>::new() };
                $body
            }
            _ => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnBaseline>::new() };
                $body
            }
        }
    }};
}

/// As `with_exec!`, but always with the plain-loop backend, so the body may use any scalar type.
macro_rules! with_plain_exec {
    ($target:expr, $e:ident => $body:expr) => {{
        use $crate::dispatch::{Isa, On};
        let t: $crate::dispatch::Target = $target;
        // SAFETY (each arm): a Target only exists for enabled instruction sets.
        match t.isa() {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnAvx512>::new() };
                $body
            }
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnAvx2>::new() };
                $body
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnNeon>::new() };
                $body
            }
            _ => {
                let $e = unsafe { On::<$crate::model::Generic, $crate::dispatch::OnBaseline>::new() };
                $body
            }
        }
    }};
}

#[allow(unused_imports)] // used by the bindings and the tests
pub(crate) use {with_exec, with_plain_exec};

/// Full residual of a batch (`model::residual_batch`) with the backend `t` selects.
pub fn residual<const N: usize>(
    t: Target,
    ctx: &Ctx<N>,
    zb: &[f64],
    out: &mut [f64],
    pool: Option<&rayon::ThreadPool>,
    dedup: bool,
) {
    with_exec!(t, e => model::residual_batch(ctx, zb, out, pool, dedup, &e))
}

/// As `residual`, with plain loops for `t`'s instruction set, for any scalar type (Complex64 gives
/// complex-step derivatives).
pub fn residual_plain<T: Sc, const N: usize>(
    t: Target,
    ctx: &Ctx<N>,
    zb: &[T],
    out: &mut [T],
    pool: Option<&rayon::ThreadPool>,
    dedup: bool,
) {
    with_plain_exec!(t, e => model::residual_batch(ctx, zb, out, pool, dedup, &e))
}
