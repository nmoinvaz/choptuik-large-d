//! 8-lane f64 vector. With AVX-512F enabled at compile time (`-C target-cpu=native` on such a CPU) it is one zmm
//! register and every operation is one intrinsic; with AVX2 + FMA it is two ymm registers; otherwise a plain array
//! (`f64::mul_add` for the fused operations, which the double-double algorithms need to be exact).
//!
//! All operations are IEEE round-to-nearest with no reassociation, as the error-free transformations in `dv`
//! require.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod imp {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct V8(pub __m512d);

    #[allow(unused_unsafe)]
    impl V8 {
        pub const ISA: &'static str = "avx512";
        #[inline(always)]
        pub fn splat(x: f64) -> Self {
            unsafe { V8(_mm512_set1_pd(x)) }
        }
        #[inline(always)]
        pub fn zero() -> Self {
            unsafe { V8(_mm512_setzero_pd()) }
        }
        #[inline(always)]
        pub fn load(s: &[f64]) -> Self {
            assert!(s.len() >= 8);
            unsafe { V8(_mm512_loadu_pd(s.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            unsafe { _mm512_storeu_pd(s.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            unsafe { V8(_mm512_add_pd(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            unsafe { V8(_mm512_sub_pd(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            unsafe { V8(_mm512_mul_pd(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            unsafe { V8(_mm512_div_pd(self.0, o.0)) }
        }
        /// self * b + c with one rounding
        #[inline(always)]
        pub fn fma(self, b: Self, c: Self) -> Self {
            unsafe { V8(_mm512_fmadd_pd(self.0, b.0, c.0)) }
        }
        /// self * b - c with one rounding
        #[inline(always)]
        pub fn fms(self, b: Self, c: Self) -> Self {
            unsafe { V8(_mm512_fmsub_pd(self.0, b.0, c.0)) }
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe { V8(_mm512_abs_pd(self.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { V8(_mm512_max_pd(self.0, o.0)) }
        }
        /// all lanes = lane j
        #[inline(always)]
        pub fn bcast(self, j: usize) -> Self {
            unsafe { V8(_mm512_permutexvar_pd(_mm512_set1_epi64(j as i64), self.0)) }
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f"), target_feature = "avx2", target_feature = "fma"))]
mod imp {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy, Debug)]
    pub struct V8(pub [__m256d; 2]);

    macro_rules! two {
        ($f:ident, $a:expr, $b:expr) => {
            unsafe { V8([$f($a.0[0], $b.0[0]), $f($a.0[1], $b.0[1])]) }
        };
    }

    #[allow(unused_unsafe)]
    impl V8 {
        pub const ISA: &'static str = "avx2";
        #[inline(always)]
        pub fn splat(x: f64) -> Self {
            unsafe { V8([_mm256_set1_pd(x), _mm256_set1_pd(x)]) }
        }
        #[inline(always)]
        pub fn zero() -> Self {
            unsafe { V8([_mm256_setzero_pd(), _mm256_setzero_pd()]) }
        }
        #[inline(always)]
        pub fn load(s: &[f64]) -> Self {
            assert!(s.len() >= 8);
            unsafe { V8([_mm256_loadu_pd(s.as_ptr()), _mm256_loadu_pd(s.as_ptr().add(4))]) }
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            unsafe {
                _mm256_storeu_pd(s.as_mut_ptr(), self.0[0]);
                _mm256_storeu_pd(s.as_mut_ptr().add(4), self.0[1]);
            }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            two!(_mm256_add_pd, self, o)
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            two!(_mm256_sub_pd, self, o)
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            two!(_mm256_mul_pd, self, o)
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            two!(_mm256_div_pd, self, o)
        }
        #[inline(always)]
        pub fn fma(self, b: Self, c: Self) -> Self {
            unsafe { V8([_mm256_fmadd_pd(self.0[0], b.0[0], c.0[0]), _mm256_fmadd_pd(self.0[1], b.0[1], c.0[1])]) }
        }
        #[inline(always)]
        pub fn fms(self, b: Self, c: Self) -> Self {
            unsafe { V8([_mm256_fmsub_pd(self.0[0], b.0[0], c.0[0]), _mm256_fmsub_pd(self.0[1], b.0[1], c.0[1])]) }
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe {
                let m = _mm256_set1_pd(-0.0);
                V8([_mm256_andnot_pd(m, self.0[0]), _mm256_andnot_pd(m, self.0[1])])
            }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            two!(_mm256_max_pd, self, o)
        }
        #[inline(always)]
        pub fn bcast(self, j: usize) -> Self {
            let mut a = [0.0; 8];
            self.store(&mut a);
            Self::splat(a[j])
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
)))]
mod imp {
    #[derive(Clone, Copy, Debug)]
    pub struct V8(pub [f64; 8]);

    macro_rules! lanes {
        ($a:expr, $b:expr, $op:tt) => {{
            let mut r = [0.0; 8];
            for i in 0..8 {
                r[i] = $a.0[i] $op $b.0[i];
            }
            V8(r)
        }};
    }

    impl V8 {
        pub const ISA: &'static str = "portable";
        #[inline(always)]
        pub fn splat(x: f64) -> Self {
            V8([x; 8])
        }
        #[inline(always)]
        pub fn zero() -> Self {
            V8([0.0; 8])
        }
        #[inline(always)]
        pub fn load(s: &[f64]) -> Self {
            let mut r = [0.0; 8];
            r.copy_from_slice(&s[..8]);
            V8(r)
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f64]) {
            s[..8].copy_from_slice(&self.0);
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            lanes!(self, o, +)
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            lanes!(self, o, -)
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            lanes!(self, o, *)
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            lanes!(self, o, /)
        }
        #[inline(always)]
        pub fn fma(self, b: Self, c: Self) -> Self {
            let mut r = [0.0; 8];
            for i in 0..8 {
                r[i] = self.0[i].mul_add(b.0[i], c.0[i]);
            }
            V8(r)
        }
        #[inline(always)]
        pub fn fms(self, b: Self, c: Self) -> Self {
            let mut r = [0.0; 8];
            for i in 0..8 {
                r[i] = self.0[i].mul_add(b.0[i], -c.0[i]);
            }
            V8(r)
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            V8(self.0.map(f64::abs))
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            let mut r = [0.0; 8];
            for i in 0..8 {
                r[i] = self.0[i].max(o.0[i]);
            }
            V8(r)
        }
        #[inline(always)]
        pub fn bcast(self, j: usize) -> Self {
            V8([self.0[j]; 8])
        }
    }
}

pub use imp::V8;

impl V8 {
    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let mut a = [0.0; 8];
        self.store(&mut a);
        a
    }
    #[inline(always)]
    pub fn from_array(a: [f64; 8]) -> Self {
        Self::load(&a)
    }
    #[inline(always)]
    pub fn hmax(self) -> f64 {
        self.to_array().iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v))
    }
}
