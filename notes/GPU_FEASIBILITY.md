# GPUs for multidouble's arithmetic: a feasibility study for the solver (2026-11-05)

A study for the coordinator, with small experiments on the Mac only. Nothing in hp or in multidouble was changed. The
probes are in `hp/gpu_probe/` (section 8), and every measured number below is printed by one of them.

## Recommendation

**Do not pursue GPU work on the project's three machines.** None of them has a GPU with fast float64.

- **The Mac's GPU has no float64 at all.** Metal rejects `double` ("'double' is not supported in Metal", measured here),
  and its OpenCL 1.2 device lists no `cl_khr_fp64`. Emulating binary64 with 64-bit integers there is bitwise exact
  (0 of 2,097,152 results differ from the CPU) but runs a double-double axpy at 0.12–0.16 times the Mac's own CPU and a
  quad-double axpy at 0.20–0.28 times. Float-float on its float32 units is fast, but it carries 48 bits in float32's
  exponent range (down to about 10⁻³⁸, 10⁻⁴⁵ subnormal), which cannot hold the solver's residuals of 10⁻³⁰–10⁻⁴⁵. It
  is also a different arithmetic, so every bit-identity reference would have to be replaced.
- **win's Iris Xe (i7-1185G7) has no float64 hardware**, and Intel's driver emulates it only on Linux (unverified,
  section 7).
- **win2's graphics are unknown** (commands in section 1). Its integrated GPU is small. A consumer NVIDIA card runs
  float64 at 1/64 of its float32 rate and an RDNA3 card at 1/32, which puts even an RTX 4090 at about 1–3 times win2's
  own CPU on quad-double work (estimate, section 5). That does not repay a port of the kernel.

**What would change it.** A rented datacenter GPU with full-rate float64 (A100, H100/H200, B200, MI300X) could run the
quad-double Jacobian (`jacobian_q`, 769 independent dual columns at M 32) and the quad-double jet series
(`residual_jet_q`) of the large-D chain. The estimate there is 20–80 times win2's CPU for the Jacobian and 7–40 times
for the jet series, whose parallel width per dependent stage is small. The cost is a CUDA port of hp's quad-double
residual path and the multidouble pieces it uses, by estimate 4000–6000 lines of device code and 2–4 months with its
bit-identity harness. Bitwise agreement with the CPU is attainable on CUDA's float64 (add, multiply and fma are
correctly rounded) with contraction off and multidouble's operation order kept everywhere (section 6). That trade pays
only if the large-D chain's wall time, rather than its precision, becomes the limit. Its walls so far have been
precision and basin walls (QUAD_DOUBLE.md 13.19–13.22), which a GPU does not move, and a step costs minutes on win.
NVIDIA's newest part points the other way. The B300 (GB300) has about 1.4 TFLOPS of float64 per GPU against the
B200's 40.

**Cheaper steps first.** The CPU-side items of notes/LOW_D_SPEEDUP.md stay ahead of any GPU work. Two observations from
this study, both untested as speedups:

- `residual_jet_q` runs in parallel over node blocks only (`hp/src/qsolve.rs:147–148`), and at M 32 with 769 unknowns
  there are six (5 nodes, `Ctx::len` = 4M(K+1)+1). The jet series, the chain's largest cost, can then keep at most six
  threads busy on win's 8 and win2's 20.
- On NEON a quad-double lane multiply spills. Its 468 vector arithmetic instructions come with 202 loads and stores
  (section 3), where the double-double multiply has only its 12 operand loads and stores. A QV value is 16 of NEON's 32
  registers and 4 of AVX-512's 32. Whether a narrower lane layout would close part of the Mac's quad-double gap to win
  is not known.

## 1. The machines' GPUs, and what to ask the Windows runners

| machine | GPU | float64 | source |
|---|---|---|---|
| Mac (MacBook Air, Apple M5, 4 + 6 cores, 32 GB) | Apple M5, 10 cores, Metal 4, unified memory | none (Metal has no `double`, OpenCL 1.2 has no `cl_khr_fp64`) | measured here (section 4) |
| win (Surface Laptop 4, i7-1185G7, 8 threads, 31.8 GB) | Intel Iris Xe (Tiger Lake, Xe-LP), expected | no hardware float64, no emulation on Windows | Intel forum (unverified) |
| win2 (Core Ultra 7 265K, 20 cores, 95.6 GB) | Intel graphics (4 Xe-LPG cores) on the CPU, a discrete card unknown | the iGPU's float64 executes on port 1 of each vector engine, rate unpublished | Chips and Cheese (read) |

The coordinator can ask both Windows runners to run these in PowerShell.

```powershell
Get-CimInstance Win32_VideoController | Format-List Name, VideoProcessor, AdapterCompatibility, DriverVersion, AdapterRAM
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,driver_version --format=csv
```

`AdapterRAM` is a 32-bit field and reads at most 4 GB, so the memory comes from `nvidia-smi`. A "not recognized" error
from `nvidia-smi` means no NVIDIA driver, which is itself the answer.

| answer | float64 rate | what it changes |
|---|---|---|
| only Intel graphics (Iris Xe on win, Intel Graphics on win2) | none or small | nothing, the GPU route is closed on that machine |
| NVIDIA GeForce or RTX workstation card, `compute_cap` 7.5 (1/32), 8.6, 8.9 or 12.0 | 1/64 of float32 | at most 1–3 times win2's CPU on quad-double for a 4090 or 5090, below 1 for smaller cards. Run the CPU probe on win2 first (below). A port is not justified unless it shows win2's CPU far below the estimate |
| NVIDIA `compute_cap` 6.0, 7.0, 8.0, 9.0 or 10.0 (P100, V100 or Titan V, A100, H100, B200, not the B300) | 1/2 of float32 | a pilot becomes worth it, QDLu on CUDA first as a one-week bit-identity test, then the quad-double residual path |
| AMD Radeon RX or PRO 7000 (RDNA3) | 1/32 | about parity with the CPU, and ROCm on Windows is limited, so closed |
| AMD Radeon VII or Instinct | 1/4 or more | as the full-rate NVIDIA row, through HIP |

The ratios in this table are from section 2 for Ada, RDNA3 and the datacenter parts, and from memory for Turing,
Ampere's consumer parts and the Radeon VII (unverified).

If an NVIDIA card turns up, win2 should first measure its CPU anchor, from a checkout that has `hp/gpu_probe`.

```powershell
cd hp\gpu_probe
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release -- 20
```

It prints double-double and quad-double axpy per second on 1 and on 20 threads, the numbers section 5 sets against the
card. It builds multidouble 2.2.0 from crates.io and writes only `hp/gpu_probe/target`.

## 2. The hardware facts

One float64 operation below is one add, multiply or fused multiply-add on one value. Vendors count an FMA as two flops,
so the operation rate is half the quoted TFLOPS. Double-double and quad-double code is mostly adds (section 3), so the
operation rate is what bounds it.

| GPU | float64 TFLOPS (vector / matrix) | float32 TFLOPS | ratio | memory GB/s | source |
|---|---|---|---|---|---|
| Apple M5, 10 cores (the Mac) | none | 3.3–3.4 measured (1.67–1.69×10¹² FMA/s) | — | unified | section 4 |
| GeForce RTX 4090 (AD102) | 1.29 | 82.6 | 1/64 | 1008 | Ada whitepaper (read), Ozaki II Table 1 (read) |
| GeForce RTX 5090 (GB202) | 1.64 | 104.8 | 1/64 | — | TechPowerUp (not primary, unverified) |
| Radeon PRO W7900 (RDNA3) | 1.92 | 61.3 | 1/32 | — | AMD datasheet as indexed (unverified) |
| A100 SXM4 | 9.7 / 19.5 | 19.5 | 1/2 | — | Ozaki II Table 1 (read) |
| H100 / H200 SXM5 | 34 / 67 | 67 | 1/2 | 3350 (H100) | Ozaki II Table 1 (read), bandwidth unverified |
| B200 | 40 / 40 | 80 | 1/2 | — | Ozaki II Table 1 (read) |
| B300 (GB300 NVL72, per GPU) | about 1.4 | — | — | — | Mukunoki 2025 Table 1, 100 TFLOPS for 72 GPUs (read) |
| Instinct MI300X | 81.7 / 163.4 | 163.4 | 1/2 | 5300 | AMD ROCm documentation (read) |

Measured rather than peak, cuBLAS DGEMM runs at 0.61–0.62 TFLOPS on the RTX 4090 (Ozaki II Table 4, read), 48% of
its peak.

**Emulation routes on hardware without fast float64.**

- *Float-float and longer float32 expansions* (Thall 2006, CAMPARY). Precision 48 bits per pair, 96 per quadruple, at
  float32's exponent range. The Mac measurement (section 4) shows the speed. A five-part expansion of DD's 106 bits
  spans about 2¹⁰⁶, so its lowest part leaves the normal range for values below about 10⁻⁶ (10⁻¹³ with subnormals),
  and float32 cannot represent 10⁻⁴⁵ at all. The bits differ from multidouble's.
- *Integer emulation of binary64.* Exact and reproducible by construction. On the Mac it is 4–8 times slower than the
  CPU for the solver's arithmetic (section 4). wgpu exposes 64-bit integers on Metal (`SHADER_INT64`), so this is the
  only float64-exact route there.
- *An integer-mantissa format* (32-bit limbs and an integer exponent). Exact, so a CPU and a GPU implementation agree
  bit for bit, but it is a third arithmetic beside DD and QD, with new references. The Mac's GPU does about 3×10¹¹
  32-bit multiply-high pairs per second (section 4). A 256-bit product of 8 limbs needs 64 of them, so a rough count
  gives a few 10⁹ axpy per second, perhaps 5–10 times the Mac's CPU on quad-double (an estimate, not built).
- *Ozaki-type schemes* (exact integer or low-precision tensor-core products, then an exact sum). They emulate GEMM only,
  are deterministic, and on an RTX 4090 reach 7.4–9.8 TFLOPS of float64-equivalent DGEMM against the native 0.62
  (Ozaki II, read). The solver has no GEMM-dominated hot path in quad-double. The low-D τ-derivative is GEMM-shaped in
  float64 (section 5), but an Ozaki product rounds differently from `DMat::apply`, so it would change the bits.

## 3. What multidouble's operations cost

The counts come from the disassembly of `hp/gpu_probe` (multidouble 2.2.0, NEON, release build with fat LTO) by
`count.py`, which follows calls. The lane code is branch-free, so the static count is the executed one. On NEON one
8-lane operation is four 2-lane instructions, and the table gives operations per value.

| operation | add/sub | mul | FMA | neg | compare | select/logic | arithmetic | all |
|---|---|---|---|---|---|---|---|---|
| DD add (`DV +`) | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 20 |
| DD mul (`DV *`) | 3 | 1 | 3 | 1 | 0 | 0 | 7 | 8 |
| DD axpy a + b c | 23 | 1 | 3 | 1 | 0 | 0 | 27 | 28 |
| QD add (`QV +`, QD's sloppy add) | 84 | 0 | 0 | 1 | 4 | 36 | 84 | 125 |
| QD mul (`QV *`, sloppy) | 101 | 7 | 9 | 7 | 4 | 36 | 117 | 164 |
| QD axpy a + b c | 185 | 7 | 9 | 8 | 8 | 72 | 201 | 289 |

The selects are the lane form of the renormalization's zero tests. A scalar GPU thread takes those branches instead, so
it executes about 201 arithmetic operations plus a few compares per quad-double axpy. The two-product is one multiply
and one FMA. QD multiplication is two-products and two-sums, and two-sum is six adds, which is why adds dominate. The
octo-double lanes loop, so their static count is not the executed one. Measured time per octo-double axpy is 5.0–5.2
times a quad-double one (below).

The next table gives the Mac's throughput of the axpy y + l t on 32 vectors of 8 lanes that stay in L1 (`gpu_probe`).

| lane axpy | P-core, 1 thread | E-core, 1 thread | 10 threads (load 6.4) | operations per cycle, P-core |
|---|---|---|---|---|
| double-double (DV) | 0.87–1.13×10⁹ /s | 2.5×10⁸ /s | 4.75×10⁹ /s | about 7 of 8 |
| quad-double (QV) | 5.4–7.0×10⁷ /s | 2.0×10⁷ /s | 3.90×10⁸ /s | about 4.6 of 8 |
| octo-double (OV) | 1.05–1.40×10⁷ /s | 3.5×10⁶ /s | 7.92×10⁷ /s | — |

"Of 8" is four 128-bit pipes of two lanes at an assumed 4.4 GHz (unverified clock). The 10-thread run shared the CPU
with other jobs (load average 6.4), so it is a lower bound. The single-thread rates were taken at load averages of
2.3–3. At 7–9 the same command gave 0.40–0.44×10⁹ (DV) and 3.3–3.5×10⁷ (QV), so they depend on the machine's load.
Octo-double was built from multidouble's `feat/octo-double` branch at f7c85d3. The task brief's QDLu timings (194 ms at
n 256 and 5.9 s at n 800, one thread) are 35 ns per update of the n³/3, 2.2 times the in-L1 quad-double axpy. The rest
is memory traffic and the multiplier column.

## 4. Measurements on the Mac's GPU

The kernels are in `hp/gpu_probe/metal` (`bench.swift` and `kernels.metal`). They are compiled at run time with
`MTLMathMode.safe` and run on 2¹⁸ threads, best of 5. Ranges are over three runs.

| kernel | rate | note |
|---|---|---|
| `double` in a kernel | compile error | "'double' is not supported in Metal" |
| float32 FMA, 16 chains per thread | 1.67–1.69×10¹² FMA/s | 3.3–3.4 TFLOPS |
| float-float axpy (multidouble's DD algorithms on float, 27 operations) | 7.0–7.4×10¹⁰ /s | 48-bit significand |
| binary64 multiply from 64-bit integers | 2.5–3.0×10¹⁰ /s | round to nearest even, normal range only |
| binary64 add from 64-bit integers | 1.6–2.2×10¹⁰ /s | same |
| 32-bit integer mul + mulhi + add | 2.5–3.3×10¹¹ /s of each | about 5 FMA slots per triple |

Both arithmetics agree with the CPU bit for bit. Float-float add and multiply differ in 0 of 4,194,304 results from
the same algorithms in Swift with fused `addingProduct`. The integer binary64 add and multiply differ in 0 of 2,097,152
from the CPU's own float64, on random normal inputs of which a third cancel and a third have large alignment shifts.
The control breaks the references (`bench . mutate`), and the checks then report 1,283,614 and 1,048,576 differences
and the program exits 1.

From the integer rates, with an FMA costed as one multiply and one add (a lower bound on its cost, and no special cases
for zeros, infinities or subnormals), the GPU would run multidouble's DD axpy at 5.6–7.7×10⁸ per second and its QD axpy
at 0.8–1.1×10⁸, which is 0.12–0.16 and 0.20–0.28 times the CPU's 10-thread rates. The float-float rate is 15 times the
CPU's double-double rate, but for 48 bits. A float32 expansion of DD's 106 bits needs five parts and costs at least
what QD does, 210–289 operations against float-float's 27. That is about a tenth of the float-float rate, some
7×10⁹ per second or 1.5 times the CPU's double-double rate, for less range and different bits.

## 5. The hot paths

The estimates below assume these quad-double axpy rates. They are estimates except where marked, and the GPUs' are
taken at 30–60% of their float64 peak with 210 operations per axpy (section 3).

| machine | QD axpy/s | basis |
|---|---|---|
| Mac CPU, 10 threads | 3.9×10⁸ | measured (section 3) |
| win2 CPU, 20 threads | 0.6–1.2×10⁹ | estimate, 1.5–3 times the Mac (8 P-cores and 12 E-cores, AVX2), unmeasured |
| Mac GPU, integer binary64 | 0.8–1.1×10⁸ | derived from measured rates (section 4) |
| RTX 4090 | 0.9–1.8×10⁹ | 6.45×10¹¹ float64 op/s peak |
| H100 SXM | 2.4–4.9×10¹⁰ | 1.7×10¹³ float64 op/s peak |

| path | CPU cost now | parallel width per dependent stage | Mac GPU | RTX 4090 against win2's CPU | H100 against win2's CPU |
|---|---|---|---|---|---|
| QD jet series, `residual_jet_q` (M 32, N 24) | about 230 s a step on win (brief), 770 s on the Mac with its LU (13.20) | 6 node blocks on the CPU, and on a GPU about 6 × 4 fields × 32 τ values × 24 coefficients ≈ 2×10⁴ | 0.2–0.3 × the Mac's CPU | 0.8–2.5 | 7–40 (the width fills a fraction of the H100) |
| QD Jacobian, `jacobian_q` (769 dual columns) | 17 s with the LU on two Mac threads (13.19) | 769 columns × the residual's width, above 10⁵ | 0.2–0.3 × | 1–3 | 20–80 |
| QDLu (n 769, inside the Newton step) | about 5 s on one Mac thread, 0.85 s on ten (estimate) | (n − k)² updates at pivot k, 769 pivots | 1.9 s | 0.1–0.15 s | under 0.01 s |
| low-D Jacobian, float64 `Fun<4>` lanes (M 320–384) | 2035–2088 s a call at M 320 and 4026–4150 s at 384, 4 threads on win2 (LOW_D_SPEEDUP A.1.1) | (4M + 1) columns of one block; the blocks run in sequence | 0.1–0.2 × (emulated float64) | 1–2 | 10–30 |

Path by path, the estimates read as follows.

- **The jet series** is quad-double arithmetic in truncated power series, compute-bound (QD's 210 operations per
  32-byte value moved). Its dependent stages are the jet orders (24), the Taylor steps of each segment and the Taylor
  orders within a step, about 10⁴ per series. Between stages a GPU needs a block or grid synchronization, a few
  microseconds, so the stage count costs little. The limit is width. About 2×10⁴ values per stage keep a 4090's 256
  float64 units busy (2 per SM, Ada whitepaper) but only part of an H100's several thousand (64 per SM, unverified).
- **The quad-double Jacobian** is 769 independent dual evaluations, the shape GPUs do best. Its output, 769² QD values
  (19 MB), crosses PCIe in about a millisecond.
- **QDLu** is compute-bound everywhere (4.5 operations per byte streamed by the unblocked update, against a 4090's ridge
  of 0.64). It takes at most a few seconds of a 10–40 s Newton step, so moving it alone gains at most 10–20% of a step.
  It is the right first test of bitwise agreement on a GPU, not a speedup.
- **The low-D Jacobian** is float64 already, so a GPU needs no emulation. Its τ-derivative (43–57%) is a GEMM-shaped
  product and the rest is Cauchy sums that are memory-bound on the CPU (LOW_D_SPEEDUP A.1.3). A 4090's float64 GEMM
  (0.62 TFLOPS measured by Ozaki II) is about what win2's CPU does (an estimate), and its 1 TB/s against dual-channel
  DDR5's 0.1 TB/s (an estimate) helps only the Cauchy sums. A datacenter GPU gains on both.

Every path needs the same port, of hp's model (`model.rs` 1404 lines, `qsolve.rs` 575) and the multidouble arithmetic it
runs on (scalar QD, the `Qu` and `QJet` lanes, `Fl`, `DMat`, the horizon block's double-double and quad-double solves),
written as device code that keeps every operation in multidouble's order. Only QDLu ports cheaply (about 300 lines
through cudarc), and it gains little.

## 6. Software routes from Rust, and bit identity

| route | float64 | Rust side | bit identity with multidouble |
|---|---|---|---|
| CUDA C kernels through cudarc (NVRTC, driver API, CUDA 11.4–13.3) | NVIDIA only | maintained (0.19.9 on docs.rs, as indexed), dynamic loading | attainable. CUDA's float64 add, mul, div and fma round to nearest even with 0 ulp error (Programming Guide 13.3, section 5.5). The guide allows contraction and re-association by default, so a port needs `--fmad=false`, explicit `fma()` where multidouble calls `mul_add`, no `--use_fast_math`, `-ftz=false`, and reductions in the CPU's index order |
| Rust kernels through Rust-CUDA (rustc_codegen_nvvm, cust) | NVIDIA only | "still in early development" (its README) | the best fit in principle, since multidouble's scalar `QD` code would compile from the same source in the same order. The toolchain's maturity is the risk |
| wgpu (WGSL, naga) | `SHADER_F64` is Vulkan-only and native-only, and Metal has none | mature | poor. No control of contraction across naga and the driver compilers |
| Metal (objc2-metal, since metal-rs is deprecated) | none | mature | only by integer emulation (section 4) |
| OpenCL (opencl3) | `cl_khr_fp64` on NVIDIA and AMD, not on Apple (measured) or Intel Xe-LP by default | maintained | attainable with `#pragma OPENCL FP_CONTRACT OFF` (OpenCL C, unverified here). Apple's OpenCL is deprecated at 1.2 |
| HIP (AMD) | full rate on Instinct | not surveyed here | as CUDA, with contraction off |

The regression checks need from any route the same IEEE operations in the same order per value, round to nearest
even, subnormals kept, and no transcendental functions on the device (a computed plan already differs between macOS and
Windows through libm's `pow`, HIGH_M.md 9.5). Divergent branches (the renormalization's zero tests) cost time on a GPU,
not bits. GQD's README makes the same point for its port of QD ("highly suggested to disable FMAD ... to generate more
consistent results as the CPU-based QD library") and does not guarantee equal results across GPU architectures.

## 7. Prior work

| work | arithmetic, hardware | measured | against | read |
|---|---|---|---|---|
| Thall 2006, SIGGRAPH poster | float-float (48 bits) and float quadruples (96), on GPUs without float64 | — | — | listing only (unverified) |
| Lu, He, Luo 2010, DaMoN '10, 19–26, doi:10.1145/1869389.1869392 (GQD) | QD's DD and QD algorithms ported to CUDA | not read | — | repository README only, measurements unverified |
| Mukunoki and Takahashi, PARA 2010, LNCS 7133 (2012), doi:10.1007/978-3-642-28151-8_25 | DD BLAS, Tesla C1060 and C2050 | C2050 DD GEMM 22 GDDFlops, 86% of its DD peak, 21× two quad-core Xeon E5630 running MBLAS, 13× slower than DGEMM, AXPY 6×, GEMV 15× | an 8-core CPU, scalar DD | the authors' 2012 ICL slides; chapter not opened |
| Verschelde and Yoffe, arXiv:1210.0800 | complex DD and QD Gram-Schmidt, Tesla C2050 | 7.0× (DD) and 7.3× (QD) at n 32, almost 10× (QD) at n 48 | one core of a Xeon X5690 | read |
| Verschelde and Yu, arXiv:1501.06625 | DD and QD path tracking, Tesla K20C | double-digit speedups past dimension about 65 | one CPU core | read |
| Joldes, Muller, Popescu, Tucker, ICMS 2016, LNCS 9725, 232–240, doi:10.1007/978-3-319-42432-3_29 (CAMPARY) | n-part float expansions in CUDA | — | — | HAL and Springer blocked; bibliographic data as indexed (unverified) |
| Verschelde, arXiv:2110.08375 | DD, QD and octo-double Householder QR (CAMPARY code), V100, P100, RTX 2080 | V100 at n 1024, kernel time, DD 2.30, QD 3.21 and OD 4.10 TFLOPS of double operations (peak 7.9), RTX 2080 Max-Q QD 0.28 | — | read |
| Verschelde, arXiv:2101.10881 | power series in multiple double, GPUs | teraflop performance in deca-double at degree 152 | — | abstract only |
| Mukunoki, Ozaki, Ogita, Imamura, ISC 2020, LNCS 12151, 230–248, doi:10.1007/978-3-030-50743-5_12 | DGEMM on tensor cores by the Ozaki scheme, reproducible versions | — | — | as indexed and as cited by Ozaki II (unverified) |
| Ozaki, Uchino, Imamura, arXiv:2504.08009 (v3), Ozaki scheme II | DGEMM from INT8 tensor cores by the CRT | RTX 4090 7.4–9.8 TFLOPS against native 0.61–0.62; GH200 56.6–80.2 against 61 | cuBLAS DGEMM | read |
| Mukunoki, arXiv:2508.00441 (v3) | Ozaki scheme on FP8 tensor cores and float64 emulation, RTX 5060 Ti | — | — | read in part (its Table 1) |
| NVIDIA blog, "Unlocking tensor core performance with floating point emulation in cuBLAS" | cuBLAS 13.0 Update 2 float64 GEMM emulation (Ozaki scheme), Blackwell | up to 2.3× on GB200 and 13× on RTX PRO 6000 Blackwell Server Edition | native DGEMM | read |

Two readings. The large speedups of the 2010–2015 papers are against one CPU core, or eight cores without SIMD, on
Tesla parts with fast float64 (the C2050 at 515 GFLOPS, in Mukunoki's slides). multidouble's lanes use the CPU's
SIMD, and today's consumer GPUs run float64 at 1/64, so those ratios do not carry over. What does carry over is
Verschelde's finding that the higher the precision, the closer a GPU kernel gets to its peak (QD at 41% and OD at 52%
of a V100), since the arithmetic per byte grows with the precision.

Hardware sources. NVIDIA, "NVIDIA Ada GPU Architecture" whitepaper (read, "The FP64 TFLOP rate is 1/64th the TFLOP rate
of FP32 operations", 288 FP64 cores on AD102, RTX 4090 at 82.6 TFLOPS and 1008 GB/s). AMD ROCm documentation, "AMD
Instinct MI300 Series microarchitecture" (read). Chips and Cheese, "Intel's Ambitious Meteor Lake iGPU", 2024 (read,
"Rare operations like FP64 math are also offloaded to Port 1"). Intel community forum, staff answer that Tiger Lake's
Iris Xe has no native float64 and that emulation is Linux-only (as indexed, the page returned 404, unverified). NVIDIA
CUDA C++ Programming Guide 13.3, section 5.5 (read). wgpu `Features` documentation (read). The READMEs of metal-rs,
Rust-CUDA and gpuprec (GQD), read through the GitHub API.

Links for the entries above and the hardware sources follow, all opened on 2026-11-05 unless marked.

- [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [AMD Instinct MI300 Series microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html)
- [Chips and Cheese, Intel's Ambitious Meteor Lake iGPU](https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu)
- [CUDA Programming Guide 13.3, floating-point computation](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [wgpu Features](https://wgpu.rs/doc/wgpu/struct.Features.html)
- [NVIDIA blog, floating point emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- arXiv [1210.0800](https://arxiv.org/abs/1210.0800), [1501.06625](https://arxiv.org/abs/1501.06625),
  [2110.08375](https://arxiv.org/abs/2110.08375), [2101.10881](https://arxiv.org/abs/2101.10881),
  [2504.08009](https://arxiv.org/abs/2504.08009), [2508.00441](https://arxiv.org/abs/2508.00441)
- [Mukunoki, Quadruple precision BLAS subroutines on GPUs, ICL slides 2012](https://icl.utk.edu/newsletter/presentations/2012/Mukunoki-Quadruple-Precision-BLAS-Subroutines-on-GPUs-2012-02-03.pdf)
- [gpuprec (GQD)](https://github.com/lumianph/gpuprec), [Rust-CUDA](https://github.com/Rust-GPU/rust-cuda),
  [metal-rs](https://github.com/gfx-rs/metal-rs), [cudarc](https://github.com/chelsea0x3b/cudarc)
- Not opened, as indexed: [TechPowerUp, RTX 5090](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216),
  [AMD Radeon PRO W7900 datasheet](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/amd-radeon-pro-w7900-dual-slot-datasheet-competitive.pdf),
  [CAMPARY on HAL](https://hal.archives-ouvertes.fr/hal-01312858/)

## 8. How to rerun the probes

The CPU probe runs from `hp/gpu_probe` on any machine with Rust, and takes multidouble 2.2.0 from crates.io.

```
cargo run --release -- 10
```

It prints DV and QV axpy per second on 1 and on 10 threads. On x86, set `RUSTFLAGS="-C target-cpu=native"` first, so
that the lanes use AVX2 or AVX-512. Octo-double needs the unreleased branch, through
`cargo run --release --features od --config 'patch.crates-io.multidouble.path="<checkout of feat/octo-double>"' -- 10`.
On the Mac, `taskpolicy -c background target/release/gpu_probe` runs it on an E-core.

The instruction counts are for aarch64 only.

```
cargo build --release
objdump -d --no-show-raw-insn target/release/libgpu_probe.dylib > "$TMPDIR/all.s"
python3 count.py "$TMPDIR/all.s" k_dv_add k_dv_mul k_dv_axpy k_qv_add k_qv_mul k_qv_axpy
```

The Metal measurements, from `hp/gpu_probe/metal`, need Xcode's Swift compiler and no Metal toolchain download, since
the kernels compile at run time.

```
xcrun -sdk macosx swiftc -O bench.swift -o "$TMPDIR/gpu_bench" && "$TMPDIR/gpu_bench" .
"$TMPDIR/gpu_bench" . mutate
```

The first exits 0 when both bitwise checks pass. The second must report differences and exit 1. The OpenCL query of
section 1 is `clq.c` in the same folder. On the Mac it prints an OpenCL 1.2 device without `cl_khr_fp64` and a
`CL_DEVICE_DOUBLE_FP_CONFIG` of 0.

```
xcrun -sdk macosx clang -O1 clq.c -framework OpenCL -o "$TMPDIR/clq" && "$TMPDIR/clq"
```

## 9. Open items and discrepancies

- QUAD_DOUBLE.md 13.21 has the Mac's quad-double LU at 40.6 s against win's 8.6 s. The brief's QDLu timing and this
  study's axpy rate put the factorization at n 769 near 5 s on one Mac thread. So that 40.6 s is either not the
  factorization alone or a loaded machine. Not resolved here.
- The brief's QDLu timings and the 230 s per jet step on win are not in a file of this repository. They are quoted as
  the brief's.
- The probes' printed output is quoted in this note. The runs' logs were not pushed to the raw-data repository.
- win2's CPU rate on the quad-double axpy is an estimate. The command in section 1 measures it.
