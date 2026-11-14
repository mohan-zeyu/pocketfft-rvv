# pocketfft-RVV: solution design document

| | |
|:---|:---|
| Project | RVV 1.0 acceleration of pocketfft (C++ header-only FFT library vendored by NumPy, PyTorch and ITK) |
| Version | 2026-11-08, repository `/home/user/fft-rvv`, commit `d1a003c` (complex passes radix 2/3/4/5/8, real passes radix 2/3/4/5, vectorised twiddle generation, RVV copy/pack/unpack paths) |
| Licence | BSD-3-Clause, the same as pocketfft (`pocketfft_hdronly.h` header comment) |
| Companion documents | `docs/correctness-report.md`, `docs/performance-report.md`, `docs/integration.md` (consumer call chains), `docs/hardware-run-guide.md` (board protocol), `docs/related-work.md` (prior art), `tests/README.md` (harness) |

Provenance marks: numbers are quoted with the file they come from; every performance number in this document is a
**QEMU dynamic instruction count** (a trend indicator, not a time) unless it says otherwise. **[V]** verified
against source or our own experiment, **[L]** likely, **[U]** unverified.

**Abstract.** pocketfft is the FFT behind `numpy.fft`, `torch.fft` on CPU (whenever MKL is absent, i.e. on every
RISC-V build) and ITK's FFT filters. On RISC-V it is scalar: upstream only vectorises batched/multi-dimensional
lines through a compiler-generated 256-bit generic-vector type (PR #31, 2026-05), NumPy and PyTorch pin commits
that predate even that, and FFTW has no merged RVV backend at all. This project adds a vector-length-agnostic,
runtime-dispatched RVV 1.0 intrinsics backend for the single 1-D transforms that every `np.fft.fft(x)` /
`torch.fft.rfft(x)` call and every residual line of a batched call executes: the complex radix-2/3/4/5/8 passes, the
real radix-2/3/4/5 passes, twiddle generation, the buffer copies and the half-complex repacking. It changes no API,
compiles to the unchanged scalar code on every other ISA and when V is absent, and produces results that differ from
the scalar build only by floating-point rounding (relative error about 1e-17 double / 1e-8 float against the scalar
build, 1e-16 / 1e-7 against NumPy and FFTW). Under QEMU the same binary executes 2.2-2.6x fewer instructions per
transform at VLEN 128, 4-5x at VLEN 256 and 6.5-10x at VLEN 512 than the scalar build for complex transforms of
256 to 2^20 points, and 3.7-4x fewer than the x86-64 AVX2 build of the same library; real transforms gain more.
Hardware timing on a RISC-V board is the remaining step of the three-tier protocol.

## 0. Repository layout

```
pocketfft_hdronly.h        upstream pocketfft cpp @ 4d865cb + the dispatch hooks (diff: dist/pocketfft_hdronly.rvv.patch)
pocketfft_rvv.h            RVV helper layer, complex passes, twiddle generation, copy/scale/pack helpers   (562 lines)
pocketfft_rvv_real.h       RVV real (half-complex, FFTPACK layout) passes radf2/3/4/5, radb2/3/4/5         (658 lines)
dist/                      single-file header pocketfft_hdronly_rvv.h (scripts/make_single_header.py) and the patch
tests/                     fft_check.cpp (FFTW oracle, dumps), fftdump.cpp (NumPy oracle, instruction counting),
                           compare_numpy.py, README.md (harness documentation)
bench/fft_bench.cpp        timing / instruction-count driver (ns per transform natively, counts under QEMU)
scripts/                   build.sh, common.sh, insncount.sh, measure_all.sh, run_correctness.sh, run_bench_qemu.sh,
                           run_bench_xisa.sh, make_report_tables.py, make_single_header.py
results/                   raw measurements (*.txt, final_insns.csv) and the correctness dumps (results/dumps)
docs/                      this document, correctness-report.md, performance-report.md (+ gen_tables.py, data/),
                           integration.md, hardware-run-guide.md, related-work.md, design-doc-skeleton.md
work/, work-real/          development artefacts: original header, assembly listings, the real-FFT agent's
                           stand-alone test (test_real.cpp) and per-pass benchmark, FFTW reference builds
build/                     cross-compiled binaries used for the numbers in results/ (see docs/performance-report.md section 10)
```

## 1. Background and problem statement

### 1.1 Who computes FFTs on RISC-V servers through pocketfft

`docs/integration.md` traces the call chains in the consumers' `main` branches (checked 2026-11-07/08) **[V]**:

| consumer | how pocketfft is reached | what executes on RISC-V today |
|:---|:---|:---|
| NumPy `numpy.fft` (`numpy/fft/_pocketfft_umath.cpp`, submodule `pocketfft @ 33ae5dc`, 2024-03, pre-RVV) | per-row `get_plan<pocketfft_c<T>>` + `plan->exec` for 1-D calls, padded calls and batches smaller than `VLEN`; `pocketfft::c2c/r2c/c2r` for larger contiguous batches; `POCKETFFT_NO_MULTITHREADING`; no plan cache (`POCKETFFT_CACHE_SIZE` = 0) | scalar `cfftp::pass_all` / `rfftp::exec` for every call (the pinned header has no RISC-V `VLEN` entry, so even the batched path is scalar) |
| PyTorch CPU `torch.fft` (`aten/src/ATen/native/mkl/SpectralOps.cpp`, `third_party/pocketfft @ 0fa0ef5`, 2024-11, pre-RVV; active whenever MKL is absent, i.e. on all riscv64 builds) | `pocketfft::c2c/r2c/c2r` with the whole tensor shape, `nthreads` omitted (= 1), no plan cache | scalar passes for 1-D tensors and for the residual lines of batches |
| ITK FFT module (`Modules/ThirdParty/PocketFFT`, upstream `c90e55b`, 2026-06, post-PR-#31) | `c2c/r2c/c2r` over all image dimensions, multi-threaded; 1-D filters per line | scalar for 1-D filters and residual lines; batched lines use the compiler-generated generic-vector path if the toolchain default enables V (Ubuntu RVA23) |
| SciPy `scipy.fft` | ducc0, not pocketfft; `std::experimental::simd` is width 1 on RISC-V | scalar; out of scope here, same algorithm family (future work, section 8) |

### 1.2 Status quo on RISC-V (`docs/related-work.md`)

* FFTW 3.3.11 and `master` have no RVV backend; three PRs (#279, #377 "r5v", #408) are unmerged, fixed-VLEN and
  validated mostly under QEMU; Debian/Ubuntu/Fedora ship FFTW scalar on riscv64 **[V]**.
* pocketfft PR #31 (merged 2026-05-01) adds `VLEN<float>=8, VLEN<double>=4` (256-bit GCC `vector_size` types):
  only batched lines are vectorised, in lock-step by the compiler; **a single 1-D transform stays scalar**; the
  author measured 1.11x on a 1-D 4M-point transform and the maintainer "didn't expect any noticeable speedup on 1D".
  Built with `-mrvv-vector-bits=zvl` the binary is VLEN-specific (crashes at other VLENs under QEMU); GCC 13 emits
  no RVV at all for those types **[V]**.
* PFFFT merged an RVV backend (2026-10, `SIMD_SZ` fixed at 4, real transforms 2-3x, complex 1.3x), juFFTe (Fortran,
  SPIRAL-generated VLA kernels) and NMSIS-DSP exist; none is consumable by NumPy/PyTorch/ITK **[V]**.
* The most concrete published RVV-FFT advice is camel-cdr's review in FFTW issue #371 (2026-06): nf=2 segment
  loads/stores to split interleaved complex data, no `vrgather` (quadratic in LMUL on the SpacemiT X60), shuffle-free
  complex multiply with `vfmul + vfnmsac` / `vfmul + vfmacc`, conjugation by sign manipulation **[V]**. The X60
  microbenchmarks (rvv-bench) add: strided loads cost ~4 cycles vs ~1.25-2.4 for unit-stride, LMUL > 1 is executed
  as LMUL m1-sized pieces, 32 architectural vector registers **[V]**.

### 1.3 Problem statement

> Provide a VLEN-agnostic, runtime-dispatched RVV 1.0 backend inside pocketfft that accelerates single 1-D (and
> residual-line) c2c, r2c and c2r transforms without changing results beyond rounding, consumable by NumPy, PyTorch
> and ITK through a header update and by nothing else.

### 1.4 Goals and non-goals

Goals: (1) no API or ABI change, no new dependency, header-only; (2) one binary for every VLEN >= 128 (128/256/512
tested, 1024 by construction); (3) graceful behaviour without V (compile-time gate when the ISA string lacks V,
runtime `AT_HWCAP` check otherwise); (4) results equal to the scalar build up to FMA rounding; (5) the batched
generic-vector path, threading, Bluestein, DCT/DST, `long double` and all other ISAs untouched; (6) both GCC >= 14
and clang >= 17 (frozen intrinsics API `__riscv_v_intrinsic >= 12000`).

Non-goals (this round): the batched/multi-D lock-step path (compiler-generated today; section 7), radix 7/11 and
generic-radix passes, Bluestein's chirp multiplications, RVV 0.7.1 (`xtheadvector`) parts, multithreading changes,
function-level `target("arch=+v")` dispatch for `rv64gc` distro builds (section 8).

### 1.5 Success criteria (competition rubric)

| rubric item | criterion | where it is met |
|:---|:---|:---|
| functional correctness vs mainstream libraries | max error / scale <= 1e-5 | 1e-16 (double) / 1e-7 (float) vs NumPy 2.5.3 and FFTW 3.3.11; RVV vs scalar build 1e-17 / 1e-8; every VLEN, both compilers - `docs/correctness-report.md` |
| FFT performance >= +10 % | measured speed-up | 2.2-2.6x (VLEN 128), 4-5x (VLEN 256), 6.5-10x (VLEN 512) fewer instructions per transform under QEMU; hardware timing to follow the protocol of `docs/hardware-run-guide.md` - `docs/performance-report.md` |
| interface adaptation to >= 2 frameworks | consumers reached without code changes | NumPy, PyTorch, ITK vendor this header (`docs/integration.md`); SciPy via ducc0 is future work |
| x86 / ARM comparison | same library, same measure | kernel: 3.7-4x fewer instructions than the x86-64 AVX2 build and ~3.5x fewer than the aarch64 NEON build at VLEN 256; public API: 2.7-3.5x (c2c) and 3.4-6.4x (r2c); native x86 timing recorded (35.6 us for n = 4096 double c2c, where FFTW 3.3.11 AVX2 is 3.5x faster than pocketfft's scalar path) - `performance-report.md` section 6 |
| QEMU = correctness + trend, real machines final | three-tier protocol | QEMU counts in this repository; board procedure, pinning, counters and run skeleton in `docs/hardware-run-guide.md` |

## 2. Architecture

### 2.1 Where RVV plugs into pocketfft

Line numbers refer to `pocketfft_hdronly.h` at commit `d1a003c` (`git diff --no-index work/pocketfft_hdronly.orig.h
pocketfft_hdronly.h` shows the complete change, 190 lines; `dist/pocketfft_hdronly.rvv.patch` is the same diff).

```
 NumPy _pocketfft_umath.cpp           PyTorch SpectralOps.cpp            ITK itkPocketFFT*.hxx
 per row: get_plan + plan->exec,      pocketfft::c2c/r2c/c2r,            pocketfft::c2c/r2c/c2r,
 or pocketfft::c2c/r2c/c2r            nthreads = 1, no plan cache        nthreads = ITK threads
          |                                    |                                  |
          +------------------------------------+----------------------------------+
                                               v
  pocketfft::c2c / r2c / c2r  ........................................  [public API, 3700 ff., unchanged]
        |                 |                  |
   general_nd        general_r2c         general_c2r  ...............  [3477 / 3632 / 3696, per 1-D line]
        |                 |                  |
   copy_input / copy_output  --RVV-->  rvv::copy_bytes (contiguous lines)        [3435, 3463]
        |                 +--RVV-->  rvv::unpack_halfcomplex (r2c output)        [3676-3680]
        |                                    +--RVV-->  rvv::pack_halfcomplex    [3747-3751]
   get_plan<pocketfft_c<T>> / get_plan<pocketfft_r<T>>  .............  [3032; cache size 0 in NumPy/PyTorch]
        |
   pocketfft_c<T>::exec  ->  cfftp<T0>::pass_all<fwd>  ..............  [2804 / 1691]
        |                      |  ip = 2,3,4,5,8 : passN_d  --RVV-->  rvv::passN  (bf2..bf8 through drive)   [1668-1687]
        |                      |  ip = 7, 11, generic       :  scalar pass7 / pass11 / passg (unchanged)
        |                      +- scale_copy_d  --RVV-->  rvv::scale_copy                                     [1728, 1734]
        |                      plan construction: comp_twiddle -> sincos_2pibyn::fill --RVV--> rvv::fill_twiddles [1779, 430]
        +- fftblue (Bluestein): chirp multiplications scalar, the embedded cfftp transforms use the RVV passes  [2702]
   pocketfft_r<T>::exec  ->  rfftp<T0>::exec  ........................  [2841 / 2572]
                               |  ip = 2,3,4,5 : radfN_d / radbN_d  --RVV-->  rvv::radfN / radbN (rf2..rb5 through rpass) [2484-2552]
                               |  other factors : scalar radfg / radbg (unchanged)
                               +- copy_and_norm_d  --RVV-->  rvv::real_scale_copy                             [2549, 2617]
                               plan construction: comp_twiddle -> sincos_2pibyn::fill (same vector path)      [2655]
```

Dispatch is by **overload resolution on the element type**, not by `if constexpr` in the templates: `pass_all` is
instantiated with `T = cmplx<T0>` for single lines and `T = cmplx<vtype_t<T0>>` for the batched generic-vector lines
(`integration.md` section 2). `passN_d` exists as a generic member template that forwards to the unchanged scalar
`passN`, and - under `#ifdef POCKETFFT_RVV` - as a non-template overload taking `cmplx<rvv::real_t<T0>> *`
(1677-1687). `rvv::real_t<T>` is `T` for `float`/`double` and the tag type `unsupported_type` (deleted constructor)
otherwise (`pocketfft_rvv.h` 64-71), so the RVV overload is viable exactly for plain `float`/`double` lines: the
batched vector type and the `long double` instantiations that NumPy builds (`npy_longdouble` loops) take the generic
template and compile to the old code. The real passes use the same guard (`radfN_d`/`radbN_d`, 2484-2552), and the
pack/unpack helpers have `unsupported_type` overloads that compile to nothing (`pocketfft_rvv.h` 510-511).

Include placement: `pocketfft_rvv.h` is included *inside* `namespace pocketfft::detail` at line 349-351
(`POCKETFFT_RVV_INSIDE_DETAIL` makes it open only `namespace rvv`) so that `cmplx<T>` is visible to the helper
layer; `pocketfft_rvv_real.h` is included at lines 1815-1816 after `cfftp` (the namespace is closed and reopened
around it) because it needs the complex helper layer but is used by `rfftp`. The single-file distribution
(`dist/pocketfft_hdronly_rvv.h`) inlines both at those points.

### 2.2 Compile-time and run-time gates

```c++
#if defined(__riscv_vector) && defined(__riscv_v_intrinsic) && (__riscv_v_intrinsic >= 12000) && !defined(POCKETFFT_NO_RVV)
#define POCKETFFT_RVV 1            // pocketfft_rvv.h 14-16
```

* `__riscv_vector` is defined when the ISA string contains V (`-march=rv64gcv...`, or the toolchain default on
  Ubuntu 25.10+/26.04 whose baseline is RVA23; Debian's `rv64gc` default leaves it undefined and the header compiles
  to the unchanged scalar code, `integration.md` section 6).
* `__riscv_v_intrinsic >= 12000`: the frozen v0.12/v1.0 intrinsics API; GCC 14/15 define 12000, GCC 16+ and
  clang 20+ define 1000000, GCC 13 defines 11000 and is excluded (its `riscv_vector.h` also refuses to compile
  without global V) **[V]**.
* `POCKETFFT_NO_RVV` disables the paths at compile time (the A/B baseline of every measurement).
* `rvv::enabled()` (`pocketfft_rvv.h` 36-48): `getauxval(AT_HWCAP) & (1 << ('V'-'A'))` (`COMPAT_HWCAP_ISA_V`,
  Linux >= 6.5), evaluated once into a function-local static; every dispatch wrapper tests it. In a
  `-march=rv64gcv` build the compiler may already have used V before `main`, so the check matters for the
  `target("arch=+v")` build mode of section 8 rather than today's whole-program builds; it costs one predictable
  branch per pass.
* Tuning macros: `POCKETFFT_RVV_LMUL{2,3,4,5,8}` and `POCKETFFT_RVV_RLMUL{2,3,4,5,_EDGE}` (LMUL per radix),
  `POCKETFFT_RVV_SPLIT_STORE` (0, section 3), `POCKETFFT_RVV_NOSCHED` (GCC attribute, section 3),
  `POCKETFFT_RVV_INLINE` (`always_inline`).

Build modes: (a) whole program with `-march=rv64gcv` - what all numbers in this repository use; (b) `rv64gc`
baseline plus `__attribute__((target("arch=+v")))` on the RVV functions with `enabled()` deciding at run time -
supported by GCC 14+/clang 19+, planned (section 8), the `enabled()` plumbing is already in place; (c) the upstream
generic-vector batched path is **left as is**: with GCC 15/clang 23 in the default scalable mode the 256-bit
`vector_size` types lower to RVV safely at any VLEN (`/home/user/research/notes/orchestrator-calibration.md`), the
VLEN-specific crash only happens with `-mrvv-vector-bits=zvl`, which we do not use.

### 2.3 The helper layer (`pocketfft_rvv.h`)

```
 enabled()                        run-time V check (once)
 real_t<T>                        dispatch guard: float/double -> T, otherwise unsupported_type
 traits<T,LMUL>   (double,float) x (m1,m2,m4), macro-generated wrappers over the intrinsics:
                  vsetvl/vlmax, vle/vlse/vse/vsse, vlseg2e/vlsseg2e/vsseg2e/vssseg2e (nf=2 segments),
                  vget/vcreate (tuple <-> registers), vfadd/vfsub/vfmul(.vv,.vf), vfmacc/vfnmsac/vfmsac(.vv,.vf), vfneg, vfmv.v.f
 ops<T,LMUL>      complex vectors = (re, im) register tuples vfloat{32,64}m{1,2,4}x2_t:
                  load/loads/store/stores (segment, unit or strided), add, sub, mulf, fmaf, fnmsf, neg, mul,
                  twmul (pocketfft's special_mul<fwd>: a*conj(w) or a*w), rot90 (ROTX90), mul_if (i*f*t)
 ctx<T0,LMUL,TWM,SI,SO>   access context of one loop shape: strided (SI) or unit (not SI) input, strided/unit output (SO),
                  twiddles none (TWM=0) / vectors from memory (1) / broadcast scalars (2); ld, st, tw, twm
 bf2 bf3 bf4 bf5 bf8      one butterfly on vl lanes; always_inline; arguments = base pointers and j-strides
 drive<fwd,T0,LMUL,BF>    the loop shapes (section 2.4); vsetvl per chunk, no scalar tail
 pass2 .. pass8           entry points, LMUL per radix (4/2/2/1/1)
 fill_twiddles            vectorised sincos_2pibyn::fill (section 2.7)
 copy_bytes, scale_copy, pack_halfcomplex, unpack_halfcomplex
 ---- pocketfft_rvv_real.h ----
 rops<T0,LMUL>            extra complex helpers (a + conj b, a - conj b, conj(a - b), ...)
 racc<T0,LMUL,AK>         real-pass access policy: lanes across the pair index (AK=false) or across k (AK=true)
 rf2..rf5, rb2..rb5       main-loop butterflies and edge loops of radf2/3/4/5 and radb2/3/4/5
 rdrive / rpass           mode selection and loop driver; radfN / radbN entry points; real_scale_copy
```

Why this shape: RVV vector types are *sizeless* - they cannot be struct members, cannot be captured by value in
generic lambdas portably and are a non-deduced context for template argument deduction. The helper layer therefore
consists of static member functions that take the active vector length `vl` explicitly, and the "complex vector" is
the intrinsics' own 2-member tuple type returned by `vlseg2e` (element 0 = real parts, element 1 = imaginary parts).
A butterfly is written once per radix against `ops<T0,LMUL>` and `ctx<>`, and the compiler instantiates it for
float/double, every LMUL and every access pattern (12 instantiations of `bf4::run` in a double build). All helpers
are `always_inline` (`POCKETFFT_RVV_INLINE`): GCC's size estimate for functions taking tuple types is far too large,
and without the attribute it emits `ops::add` out of line - one call per vector addition (section 3, step 0).

### 2.4 Data layout and the loop shapes of the complex passes

pocketfft keeps FFTPACK's layout. For a pass with radix `ip`, `l1` butterfly groups and `ido` elements per group
(`n = ip * l1 * ido`), with complex arrays `cc` (input) and `ch` (output) and twiddles `wa`:

```
 CC(i,j,k) = cc[i + ido*(j + ip*k)]        i < ido, j < ip, k < l1   (input)
 CH(i,k,j) = ch[i + ido*(k + l1*j)]                                   (output)
 WA(x,i)   = wa[(i-1) + x*(ido-1)]         i = 1..ido-1, x = 0..ip-2  (twiddles; i = 0 needs none)
```

A butterfly combines the `ip` inputs `CC(i,0..ip-1,k)` (stride `ido` between them) into the `ip` outputs
`CH(i,k,0..ip-1)` (stride `ido*l1`), multiplying output `x+1` by `WA(x,i)` (conjugated for forward transforms). The
RVV passes vectorise **across butterflies** - the lanes of a vector are different `(i,k)` pairs with the same
arithmetic - and choose, per pass, which index runs across the lanes (`drive`, `pocketfft_rvv.h` 338-402):

| loop shape | when | lanes | input access | output access | twiddles |
|:---|:---|:---|:---|:---|:---|
| **whole pass, `ido == 1`** | the last pass of a power-of-two plan and every pass of a plan factor list that ends in it | consecutive `k` | strided segment load, stride `ip` complex elements | **unit-stride** segment store (`chk = ido = 1`) | none |
| **`i = 0` column** | always (before the twiddled columns) | consecutive `k` | strided segment load, stride `ido*ip` | strided segment store, stride `ido` | none |
| **mode A** | `ido - 1 >= (vlmax + 1) / 2` (at least half a vector of consecutive `i` exists; `vlmax` read at run time for the pass's LMUL) | consecutive `i`, fixed `k` | **unit-stride** segment loads of `CC(i..i+vl-1, j, k)` | unit-stride segment stores | **vectors** `WA(x, i..i+vl-1)` loaded from memory (they are consecutive in `wa`) |
| **mode B** | otherwise (small `ido`: the last two or three passes) | consecutive `k`, fixed `i` | strided segment loads, stride `ido*ip` | strided segment stores, stride `ido` | **broadcast** `vfmv.v.f` of `WA(x,i).r/.i`, the same for every lane |

Each chunk is `vl = vsetvl(remaining)` lanes long, so partial vectors at the end of every row are handled by the
hardware's vector-length register and there is no scalar remainder loop and no VLEN constant anywhere. Mode A is
preferred because unit-stride segment loads are the cheapest way to de-interleave complex data (X60: ~1.25-2.4
cycles per `vle` against ~4 for a strided `vlse`, `related-work.md` section 7) and because the twiddle vectors come
straight from memory; mode B accepts strided accesses when `ido` is too small for a vector but then pays nothing
for twiddles. The access policy is a compile-time template parameter of `ctx` (`SI`, `SO`, `TWM`), so the
butterfly bodies contain no run-time stride tests, and the butterfly receives precomputed base pointers and
j-strides (`cc0, ccj, ch0, chj`) instead of `(i, k, ido, l1)`.

**Split re/im in registers, interleaved in memory.** The register representation is planar (one vector of real
parts, one of imaginary parts per complex vector), produced by the nf=2 segment loads and consumed by nf=2 segment
stores - exactly camel-cdr's recommendation (section 1.2). Consequences: a complex add/sub is two vector ops, a
complex multiply is four (`vfmul`, `vfnmsac`, `vfmul`, `vfmacc`), multiplication by `+-i` is a register swap plus
one `vfneg` (`rot90`), multiplication by `exp(+-i pi/4)` in the radix-8 butterfly is two adds and two `.vf`
multiplies, conjugation needs no instruction at all because `twmul` selects the `fmacc`/`fnmsac` arrangement at
compile time from `fwd`. There is no `vrgather`, no `vslide`, no mask-controlled permutation in any pass; the
instruction stream of a pass is 77 % vector instructions, of which 78 % are floating-point arithmetic
(`performance-report.md` section 4).

### 2.5 Vector-length agnosticism

Every loop is strip-mined with `vsetvl`, the only VLEN-dependent decision (mode A vs B) reads `vsetvlmax` at run
time, twiddle generation and the copy helpers are strip-mined the same way, and no code depends on `__riscv_v_fixed_vlen`
or `-mrvv-vector-bits`. Evidence: one binary passes the complete correctness matrix at VLEN 128, 256 and 512 under
QEMU (`correctness-report.md`), and its instruction count halves each time VLEN doubles (`performance-report.md`
section 3: n = 65,536 kernel 2.83 M -> 1.42 M -> 0.72 M). VLEN 1024 (the SpacemiT K3's AI cores) is covered by
construction but was not run.

### 2.6 LMUL choices

| radix | LMUL (`POCKETFFT_RVV_LMUL*`) | register budget (32 vector registers) |
|:---|:---|:---|
| 2 | 4 | 2 inputs + 1 twiddle = 3 complex tuples = 24 registers; outputs reuse inputs |
| 3 | 2 | 3 inputs + temporaries + twiddle ~ 7 tuples = 28 registers |
| 4 | 2 | 4 inputs + 4 temporaries + twiddle ~ 10 tuples; fits with reuse |
| 5 | 1 | 5 inputs, 4 sums/differences, 2 partial products, twiddle: ~ 20 registers |
| 8 | 1 | ~ 20 registers live (comment in `pocketfft_rvv.h` 513) |
| real passes | 4 / 2 / 2 / 1 for radix 2/3/4/5, edge loops at 4 | the edge loops (element 0 and ido-1 of each column) carry no twiddles and little state |

A larger LMUL means fewer instructions (each vector instruction does LMUL times the work, fewer loop iterations,
fewer `vsetvl`) but LMUL times the register footprint; the per-radix values are the largest that keep every
instantiation spill-free in both compilers. On the X60 LMUL > 1 instructions are executed as LMUL m1-sized pieces
(no throughput gain, only fewer instructions), on the C920v2 (VLEN 128) LMUL 2 is the natural way to get 256-bit
work items. The LMUL sweep in `results/lmul_experiment.txt` (kernel measure, clang 23, `performance-report.md`
section 4) shows the trade-off: LMUL 1 for every radix costs +10 % at n = 65,536 (3.11 M vs 2.83 M at VLEN 128),
LMUL 2 for every radix or for radix 8/5 costs +21-24 % on powers of two (radix 8 spills), while radix 5 at LMUL 2
*gains* 15-21 % on n = 1000 at VLEN 128/512 and loses 3 % at VLEN 256; radix 4 at LMUL 4 is neutral. The defaults
are the register-budget maximum that never spills; radix 5 at LMUL 2 is a candidate for per-core tuning on hardware
(section 7).

### 2.7 Twiddle generation (`sincos_2pibyn::fill`, `rvv::fill_twiddles`)

NumPy and PyTorch build a plan per call (section 1.1), so for them twiddle computation is part of every transform.
pocketfft's `sincos_2pibyn` computes `exp(2 pi i idx / N)` from two tables (`v1[idx & mask]`, `v2[idx >> shift]`,
both `cmplx<double>` for float and double plans) as the product `v1 * v2`, with the symmetry `idx -> N - idx` and a
conjugated result for `2 idx > N` (`pocketfft_hdronly.h` 419-428). `comp_twiddle` fills the per-pass twiddle arrays
`fact[k].tw[(j-1)*(ido-1) + i-1] = twiddle[j*l1*i]`, i.e. arithmetic index sequences `start + t*step`. The new
`fill(dst, count, start, step)` method (430-443) hands such a sequence to `rvv::fill_twiddles` (`pocketfft_rvv.h`
430-469): indices by `vid * step + start` (e64m2), the symmetry mask `lo = 2*idx <= N`, `idx2 = lo ? idx : N-idx`
by `vmerge`, byte offsets `(idx2 & mask) << 4` and `(idx2 >> shift) << 4`, **indexed segment loads**
`vluxseg2ei64` gathering the (re, im) pairs of both tables, the four-instruction complex product, the imaginary
sign restored by a masked `vmerge` with the negated vector, and an interleaving segment store (`vsseg2e64`, or
`vfncvt` to float followed by `vsseg2e32`). `long double` plans keep the scalar loop (the generic `fill_twiddles`
overload returns `false`). Effect (`performance-report.md` section 5, API without plan cache, clang, VLEN 256):
n = 4096 216,793 -> 129,748 instructions (-40 %), n = 256 23,017 -> 18,230. The scalar `operator[]` is still used
for the `tws` extra twiddles of generic-radix factors (`ip > 11`), which are few.

### 2.8 Real transforms (`pocketfft_rvv_real.h`)

`rfftp` works on FFTPACK's half-complex layout: in a pass, element 0 of each column is real, element `ido-1` (even
`ido`) is the Nyquist term, and the remaining elements form pairs `(i-1, i)` and their mirror pairs `(ic-1, ic)`,
`ic = ido - i`, which are complex numbers stored in natural address order - the mirror pairs run **backwards** in
memory. The scalar `radf*`/`radb*` therefore have three parts (file comment, lines 36-57): the edge loop over `k`
for element 0 (and `ido-1` of the outputs), the Nyquist loop for even `ido`, and the main loop over `k` and
`i = 2, 4, ..., ido-1`. The RVV version vectorises the two edge loops **across k** with strided scalar-element loads
and stores at LMUL 4 (no twiddles, little register pressure; `rf4::edge` is a direct transcription of
pocketfft's `radf4` edge code), and the main loop either **across the pair index m** (`racc<..., AK=false>`:
unit-stride segment loads for the `(i-1,i)` pairs, segment loads/stores with stride **-2 elements** for the mirror
pairs, twiddle vectors `wa[x*(ido-1) + 2m, 2m+1]` from memory) when `nvec = (ido-1)/2 >= (vlmax+1)/2` or
`nvec >= l1`, or **across k** (`AK=true`: strided segment accesses, broadcast twiddles) when `ido` is small. The
butterfly bodies (`rf2..rb5::run`) are transcriptions of the scalar passes onto `rops<T0,LMUL>` (which adds
`a + conj(b)`, `a - conj(b)`, `conj(a - b)` to `ops`). Dispatch mirrors the complex side: `radfN_d`/`radbN_d`
overloads on `rvv::real_t<T0>*`, `copy_and_norm_d` -> `rvv::real_scale_copy`. Radix 2/3/4/5 cover every length
of the form `2^a 3^b 5^c`; `radfg`/`radbg` (factors 7, 11, 13, ...) stay scalar.

The public r2c/c2r functions additionally repack between the half-complex array and the interleaved complex
output/input (`general_r2c` 3660-3690, `general_c2r` 3736-3760); for contiguous lines this is now one
`rvv::unpack_halfcomplex` / `pack_halfcomplex` (a unit-stride segment copy with an optional sign flip of the
imaginary parts for the conjugate direction, `pocketfft_rvv.h` 481-511) instead of a scalar loop, and `copy_input`
/ `copy_output` use `rvv::copy_bytes` (e8m8 vector copy) for contiguous lines.

### 2.9 Threading, memory, alignment

Unchanged: `thread_count` / `thread_map` are not touched (NumPy compiles with `POCKETFFT_NO_MULTITHREADING`,
PyTorch passes no `nthreads`); per-call temporaries are pocketfft's own `arr<T>` buffers (aligned allocation);
user buffers are accessed through the same pointers as before. RVV segment loads accept any element-aligned
address; whether misaligned *vector* accesses are fast is per-core (`RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF`,
`hardware-run-guide.md` section 2) and all accesses here are naturally aligned to the element size.

## 3. Optimisation strategy and the iterations we went through

All numbers: dynamic instructions per transform under QEMU, double, c2c forward, n = 4096, VLEN 256 unless stated;
files in `results/` and `work/` (`performance-report.md` sections 4-5 hold the full tables).

| step | change | evidence and effect |
|:---|:---|:---|
| 0 | **First version** (GCC 15 only): generic helper layer with ordinary `inline` members, butterflies written as functions of `(i, k, ido, l1)` that recomputed all `CC`/`CH` addresses and reloaded the context on every call, run-time stride tests | GCC left the helpers and the butterflies **out of line**: one `bf4::run` call compiled to ~400 instructions, 19 `vsetvli`, 266 scalar instructions, 25 whole-register moves and `csrr vlenb` for spill slots (`work/p4_gcc.s`, dated 11:32). The RVV build executed *more* instructions than the auto-vectorised scalar build: 451,291 vs 413,313 at VLEN 256, 770,976 at VLEN 128 (`results/c2c_insns.txt`). |
| 1 | **Second compiler**: the same source built with clang 23 | clang inlined everything by itself (`work/p4_clang.s`: no calls, 9 `vsetvli`) and was already 2.2x better than scalar (164,861 vs 358,692, API with plan cache, `results/c2c_insns_cache16.txt`), which located the problem in GCC's code generation, not in the algorithm. |
| 2 | **Forced inlining** (`always_inline` on every helper), **precomputed base pointers and j-strides** passed to the butterflies, **static access selection** (`ctx<T0,LM,TWM,SI,SO>` template parameters instead of run-time `if (stride == 1)`), unit-stride stores for the `ido == 1` pass | kernel count roughly halved for both compilers: clang 69,501, GCC 140,544 (`results/kernel_insns.txt`, 11:55); API with cache 164,861 -> 74,013 (`results/api_cache16_insns.txt`). |
| 3 | **Split-store experiment** (`POCKETFFT_RVV_SPLIT_STORE=1`: two strided `vsse` per complex store to relieve GCC's adjacent-register constraint of `vsseg2e`) | **negative**: +5 % GCC, +2.5 % clang (`results/c2c_insns_v2.txt`); segment stores kept, macro left in for reference. |
| 4 | **LMUL experiments** (lower LMUL per radix, `*_low` builds; later the sweep in `results/lmul_experiment.txt`) | early: neutral (0 % clang, -2 % GCC, `results/c2c_insns_cache16.txt`); sweep: all-LMUL-1 +10 % at n = 65,536, LMUL 2 for radix 8 +21-24 % (spills), radix 5 at LMUL 2 -15..-21 % on n = 1000 at VLEN 128/512 but +3 % at VLEN 256; defaults kept (section 2.6). |
| 5 | **GCC flags** (`-O2`, `-fno-schedule-insns` globally, `-fno-tree-vectorize`) | within 1 % of the baseline (`build/g_*`, `performance-report.md` section 4). |
| 6 | **Vectorised twiddle generation** (section 2.7) | API without plan cache: 216,793 -> 129,748 (-40 %), n = 65,536: 3.47 M -> 2.07 M (`results/api_nocache_insns.txt` -> `_v2`). |
| 7 | **RVV `copy_input`/`copy_output` for contiguous lines** | 129,748 -> 111,348 (`_v2` -> `_v3`); with plan cache the API converges to the kernel count (74,013 vs 69,501). |
| 8 | **Real passes** radix 2/3/4/5 (section 2.8), merged from the real-FFT agent's branch together with the GCC attribute below | r2c n = 65,536 with plan cache (GCC): 4.45 M -> 1.55 M / 1.02 M / 0.76 M at VLEN 128/256/512; per pass `radf4` in mode A 508 k -> 62 k (clang, 8.5x, `work-real/pass_bench.cpp`). |
| 9 | **GCC `optimize("no-schedule-insns")` per entry point** (`POCKETFFT_RVV_NOSCHED`): GCC's pre-register-allocation scheduler hoisted all loads of a butterfly to the loop top and then spilled | GCC kernel 140,544 -> 96,171 (-32 %); the two compilers now execute the same 42,081 vector FP instructions per transform and GCC's remaining excess is `vsetvl` and register moves (`performance-report.md` section 4). |
| 10 | **RVV pack/unpack of the half-complex layout** in `general_r2c`/`general_c2r` | r2c without plan cache n = 65,536 (clang): 5.51 M -> 0.70 M at VLEN 256 (`results/final_insns.csv`). |

Design rules distilled from these iterations: (1) measure both compilers from the first day - GCC's RVV tuple
handling is not representative of the algorithm; (2) make every access pattern a compile-time property; (3) never
let a helper that takes an RVV tuple type be a call; (4) a vector instruction count that goes *down* when a store is
replaced by two is a sign of register-allocation trouble, not of a better memory pattern; (5) the plan-construction
and copy costs are as important as the passes for the API that NumPy/PyTorch actually call.

## 4. Verification process

Full details and the measured error tables are in `docs/correctness-report.md`; this is the design of the process.

* **Reference oracles.** (a) NumPy 2.5.3 `numpy.fft` on the x86 host (`tests/compare_numpy.py`) over binary dumps
  written by `tests/fftdump.cpp` under QEMU; (b) FFTW 3.3.11 in double precision, built scalar for riscv64 and
  linked statically into `tests/fft_check.cpp` (same binary for scalar and RVV builds); (c) the **scalar build of
  the same header** (`-DPOCKETFFT_NO_RVV`), compared element-wise on bit-identical inputs (`--dump` /
  `--compare-scalar`, and `compare_numpy.py`'s second prefix); (d) an exact long-double / double-double reference
  (naive DFT for n <= 4096, long-double pocketfft above) for the real passes (`work-real/test_real.cpp`).
* **Metric.** `max |out - ref| / max |ref|` per transform, the rubric's quantity, pass if <= 1e-5; also the relative
  L2 error and the RVV-vs-scalar difference (pass thresholds 1e-12 double / 1e-4 float, four orders of magnitude
  above the expected rounding differences, so any wrong lane, wrong twiddle or wrong store stride fails loudly).
* **Matrix.** types {float, double} x kinds {c2c fwd, c2c bwd, r2c, c2r} x lengths: powers of two 1..2^20, mixed
  radix `2^a 3^b 5^c`, radix-7/11/13 and generic factors (49, 77, 121, 143, 169, 231, 352, 448, 1001, 1331, 2401),
  primes (Bluestein: 1009, 10007, 65537, ...), odd/non-aligned lengths (3, 5, 7, 9, 15, 33, 65, 129, 1023, 1025)
  x VLEN {128, 256, 512} x compilers {clang 23, GCC 15}; plus batched 2-D shapes {5,n} and {9,n} (the generic-vector
  path with residual lines), strided input/output (3/2), in-place c2c, and the real-FFT agent's 226-length sweep.
* **What a failure would catch.** Wrong mode selection or stride (every length between 1 and 65536 with every factor
  mix exercises `ido`/`l1` combinations around `vlmax` at three VLENs), wrong twiddle index or conjugation (forward
  and backward, float and double), wrong tail handling (`vsetvl` with lengths that are not multiples of any VLEN),
  wrong dispatch (long double and the batched type compile and run the scalar code), wrong half-complex pairing
  (odd and even `ido`, the Nyquist term), scaling (`fct = 1/n` on backward transforms).
* **Reproducibility.** Inputs are generated from integer LCGs seeded from the case parameters - identical on every
  machine and compiler; compiler versions and flags are recorded in `build/*/*.log` and `tests/README.md`; the
  scripts `scripts/run_correctness.sh`, `tests/compare_numpy.py`, `work-real/test_real.cpp` are the entry points.
* **Not done yet.** Running NumPy's `numpy/fft/tests/test_pocketfft.py` and PyTorch's `test_spectral_ops.py` on a
  RISC-V system (requires a system-mode QEMU image or a board with the Python stacks); `-cpu rv64,v=false` fallback
  test of the dispatch (trivial to add to `run_correctness.sh`); sanitizer builds under QEMU.

## 5. Performance evaluation method

Designed as the three-tier protocol the rubric asks for; details and tables in `docs/performance-report.md`.

1. **QEMU trend (done).** Dynamic instruction counts per transform with the `libinsn` TCG plugin:
   `(insns(11 reps) - insns(1 rep)) / 10`, so process start-up, input generation and the first transform cancel
   (`scripts/insncount.sh`). Three measures: *kernel* (`plan->exec` on a cached plan, in place - NumPy's per-row
   path), *API without plan cache* (`pocketfft::c2c/r2c/c2r` with `POCKETFFT_CACHE_SIZE=0`, what NumPy and
   PyTorch do: includes plan construction and copies) and *API with plan reused* (`POCKETFFT_CACHE_SIZE=16`). The
   scalar baseline is the same header built for `rv64gc` with `POCKETFFT_NO_RVV` (no V in the ISA string - the
   lowest-count scalar build we could produce; GCC's auto-vectorised build and clang's are reported too). VLEN 128,
   256 and 512 with one binary; clang 23 and GCC 15. The instruction-class split (`vsetvl`, vector loads/stores,
   vector FP, scalar) comes from a custom plugin (`librvvclass`).
2. **Hardware (to do, procedure ready).** `bench/fft_bench` (ns per transform, median of >= 5 launches of >= 0.5 s,
   pinned core, performance governor, thermal log) and `perf stat` cycles/instructions on N and 2N repetitions,
   on a SpacemiT K1 (X60, VLEN 256) and an SG2044 (C920v2, VLEN 128) or K3 (X100), per
   `docs/hardware-run-guide.md`; the `perf` instruction counts validate the QEMU counts, the cycle ratio is the
   speed-up. Metrics: ns/transform, GFLOP/s by `5 n log2 n` / `2.5 n log2 n`, cycles per point, flop/cycle vs the
   core's peak.
3. **x86 / ARM counterparts (counts done, timing partial).** The same `fftdump` kernel measure compiled with
   GCC 13 for `x86-64-v3` (AVX2) and `armv8.2-a` (NEON), counted under `qemu-x86_64`/`qemu-aarch64`, and native
   wall time on the x86 build host; pocketfft's 1-D path is scalar on every ISA, so this is "pocketfft as shipped"
   per ISA.

## 6. Results (summary; tables in the two reports)

* Kernel, double c2c forward, clang 23 (`results/kernel_insns.txt`): n = 4096 297,934 -> 136,819 / 69,501 / 35,842
  (2.2x / 4.3x / 8.3x at VLEN 128 / 256 / 512); n = 65,536 6,896,666 -> 2,832,190 / 1,424,978 / 721,372
  (2.4x / 4.8x / 9.6x); n = 2^20 143.6 M -> 66.9 M / 37.5 M / 22.0 M (2.1x / 3.8x / 6.5x).
* Public API without plan cache, clang, VLEN 256 (`results/api_nocache_insns_v3.txt`): n = 256 2.0x, 1024 3.4x,
  4096 4.1x, 65,536 5.1x, 2^20 4.2x; VLEN 512 6.5-9.8x, VLEN 128 2.2-2.6x. float gains about twice as much as double
  at the same VLEN (n = 65,536 at VLEN 256: 8.7x, `results/final_insns.csv`).
* Real transforms, no plan cache, clang (`results/final_insns.csv`): r2c double n = 65,536 5.51 M -> 1.30 M /
  0.70 M / 0.39 M (4.2x / 7.9x / 14.2x); r2c n = 4096 3.4x / 5.2x / 7.1x.
* GCC 15.3: correct everywhere; ~1.4x the instructions of clang on the complex kernel after the no-schedule
  attribute (2.0x before), but on par or better than clang at the public-API level for r2c/c2r and small n (its scalar
  code is 5-15 % leaner); recommendation: clang >= 19 for complex-heavy use, either compiler for real transforms.
* Cross-ISA (`results/xisa_insns.txt`, `results/xisa_api_insns.txt`): x86-64 GCC 13 AVX2 build 270,489 and aarch64
  NEON build 241,206 instructions for the n = 4096 kernel (scalar 1-D path on both) vs 69,501 for RISC-V RVV at
  VLEN 256 (3.9x / 3.5x fewer) and 136,819 at VLEN 128 (2.0x / 1.8x fewer); at the public-API level (plan built
  per call) 297,233 (x86) vs 111,346 (RVV, 2.7x), r2c 194,533 vs 57,580 (3.4x). Native x86 time 35.6 us (n = 4096),
  1.16 ms (n = 65,536) per double c2c (`results/bench_xisa.csv`); FFTW 3.3.11 AVX2 is 3.5-3.6x faster than pocketfft's
  scalar path on that host (`results/x86_native_fftw_vs_pocketfft.txt`) - the gap a vectorised 1-D path closes.
* Correctness (`docs/correctness-report.md`): every case passes at 1e-5; worst double error vs NumPy 1.5e-16, vs
  FFTW 1.3e-15 (Bluestein n = 65537); worst float 6.4e-8 (vs NumPy) / 3.6e-7 (vs FFTW, float rounding of
  pocketfft itself); RVV vs scalar build 8e-17 double / 5e-8 float; real passes vs exact reference 3.8e-16 L2 double.
* Where the gain stops: n <= 64 (not vectorised profitably, fallback to scalar is an open item), the n = 2^20 kernel
  (3.8x instead of 4.8x at VLEN 256, cause under investigation), radix 7/11/generic and Bluestein chirp loops
  (scalar), and on hardware the memory-bound regime.

## 7. Risks, limitations and mitigations

| risk / limitation | impact | mitigation / status |
|:---|:---|:---|
| **QEMU is not timing.** Count ratios can over- or understate cycle ratios: segment and strided loads cost several cycles on the X60, LMUL > 1 is cracked, large n is memory-bound. | the >= 10 % bar must be shown on silicon | board protocol ready (`hardware-run-guide.md`), margins of 2-10x; the kernel's instruction mix (77 % vector, strided accesses limited to `ido == 1` / `i = 0` / mode B) is favourable; `perf stat` on the board validates the counts. |
| **GCC code generation** (register moves, `vsetvl`, spills around tuple types) | GCC's complex kernels are ~1.4x clang's count (2x before the attribute; 2.8x on radix-5 lengths); real transforms and API-level small sizes are on par or better with GCC | correct in all cases; `POCKETFFT_RVV_NOSCHED` attribute; clang recommended for complex-heavy use; remaining gap (radix 5 first) to be reported upstream to GCC with the `work/p4_gcc.s` reproducer. |
| **Batched / multi-D path untouched** (compiler-generated 256-bit generic vectors; `VLEN<double>=4`) | 2-D/batched transforms on RISC-V gain only what GCC 15/clang 23 make of the generic vectors (PR #31 numbers: 1.06-1.9x); under-uses VLEN 512; residual lines use our kernels | by design this round; the intrinsics kernels can be instantiated for the batched buffers later (section 8); ITK is the only consumer that compiles that path today. |
| **Radix 7, 11, generic passes and Bluestein chirp loops scalar** | lengths with large prime factors (7, 11, 13, primes) gain only through the embedded power-of-two FFT (Bluestein) and the radix-2/3/4/5/8 passes they also contain | correct via the scalar code; radix 7/11 butterflies are mechanical additions to `pocketfft_rvv.h` (open item). |
| **Real passes: `radfg`/`radbg` scalar; DCT/DST not separately vectorised** | real lengths with factors other than 2/3/5 partly scalar; DCT/DST inherit the radix 2/3/4/5 gains through `rfftp::exec` only | correct; same open item as above. |
| **Tiny transforms** (n <= 64) | per-pass fixed cost not amortised; the first version was at parity or worse for n = 64 | measure on hardware; add a length threshold in the dispatch (one comparison). |
| **Numerical differences vs scalar** (FMA contraction and summation order differ) | results are not bit-identical to the scalar build | documented tolerance; measured 8e-17 (double) / 5e-8 (float) relative, below the reference libraries' own disagreement; forward/backward round-trips unaffected. |
| **Register pressure at higher radices / VLEN-specific tuning** | radix 8 at LMUL 1 only; LMUL values tuned for instruction count, not for a specific core | macros `POCKETFFT_RVV_LMUL*`; re-tune on K1/SG2044 with cycles. |
| **`rv64gc` distro builds** (Debian: V not in the default ISA string) | header compiles to scalar code there | `target("arch=+v")` + `enabled()` dispatch (GCC 14+/clang 19+), planned (section 8); Ubuntu 25.10+/26.04 (RVA23 baseline) gets the RVV paths with no change. |
| **RVV 0.7.1 parts** (SG2042, TH1520) | RVV 1.0 binaries trap there | excluded by design; `enabled()` relies on the kernel's `HWCAP` V bit which those kernels do not set for 0.7.1. |
| **Misaligned vector accesses** | slow or unsupported on some cores (`hwprobe MISALIGNED_VECTOR_PERF`) | all accesses are element-aligned; note in the hardware guide to record the hwprobe value. |
| **Upstream acceptance** (maintainer prefers portable code) | the PR could stall | opt-in macro, zero effect on other ISAs, header-only, `POCKETFFT_NAMESPACE`/`POCKETFFT_NO_EXCEPTIONS` compatible, QEMU CI precedent (PFFFT #110), hardware numbers attached. |
| Concurrent development artefacts | `build/` binaries and `results/*.txt` were produced at different commits | provenance table in `performance-report.md` section 10; the final matrix (`results/final_insns.csv`) and the independent rebuild (section 9 there) are at fixed commits. |

## 8. Upstreaming and adoption plan

1. **pocketfft `cpp` branch PR** (maintainer merges within days in 2026; PR #31 precedent): the three-file form
   (`pocketfft_hdronly.h` hooks + `pocketfft_rvv.h` + `pocketfft_rvv_real.h`) or the single-file form
   (`dist/pocketfft_hdronly_rvv.h`, generated by `scripts/make_single_header.py`; `dist/pocketfft_hdronly.rvv.patch`
   is the diff against upstream `4d865cb`), with `tests/fft_check.cpp` as the QEMU CI job (VLEN 128/256/512 and
   `v=false`) and the hardware numbers. Suggested companion fix: make the upstream RISC-V `VLEN` entry conditional
   on `__riscv_v_fixed_vlen` so `-mrvv-vector-bits=zvl` builds cannot be VLEN-specific by accident
   (`integration.md` section 6a).
2. **NumPy**: submodule bump `numpy/fft/pocketfft` 33ae5dc -> merged commit; no glue change needed (NumPy's
   per-row path inherits the 1-D kernels, its `vlen`-gated batched branch becomes live). Recommend defining
   `POCKETFFT_CACHE_SIZE` (e.g. 16) in `_pocketfft_umath.cpp`: with the plan reused the n = 4096 call costs 74,013
   instead of 111,348 instructions (RVV) and 329,034 instead of 453,387 (scalar) - a 25-35 % saving independent of
   RVV, free for the repeated same-length calls that dominate real workloads (`results/api_cache16_insns.txt` vs
   `api_nocache_insns_v3.txt`). The NumPy-side `copy_input`/`copy_output` for non-contiguous rows is a second
   follow-up.
3. **PyTorch**: `third_party/pocketfft` 0fa0ef5 -> merged commit; `SpectralOps.cpp` unchanged (public API only).
   Same `POCKETFFT_CACHE_SIZE` recommendation, plus passing `at::get_num_threads()` as `nthreads` (today 1).
4. **ITK**: `UpdateFromUpstream.sh` to the merged commit (ITK already carries the post-PR-#31 header).
5. **Distributions**: Ubuntu 25.10+/26.04 riscv64 (RVA23 baseline) picks the RVV paths up at the next NumPy/PyTorch
   rebuild; for Debian/Fedora (`rv64gc` baseline) implement build mode (b) of section 2.2: `target("arch=+v")` on the
   `rvv::` functions (GCC 14+/clang 19+) with the existing `enabled()` check - a mechanical change confined to the
   two RVV headers.
6. **ducc0 / SciPy** (`integration.md` section 5): port `ops<T,LMUL>` into ducc0's `vtp<T,len>` SIMD class and the
   1-D passes into its `cfftp`/`rfftp` (same FFTPACK ancestry) - 2-3 weeks, then SciPy refreshes its vendored copy.
7. **Toolchain follow-ups**: GCC bug report on RVV tuple register allocation with the `work/p4_gcc.s` reproducer;
   `-mcpu=spacemit-x60` / `spacemit-x100` tuning tables once hardware numbers exist.
8. **Maintenance**: the RVV headers are self-contained and gated; new radices, LMUL tables per core and the batched
   instantiation are additive; CI = `scripts/run_correctness.sh` under QEMU at three VLENs plus `v=false`.

## 9. Appendices

### A. Build and run commands used for every number in this repository

```sh
# clang 23 (recommended), VLEN-agnostic RVV build, static (runs under qemu-riscv64 and on any rv64gcv Linux board)
clang++-23 --target=riscv64-linux-gnu --gcc-toolchain=/usr -isystem /usr/riscv64-linux-gnu/include \
  -isystem /usr/riscv64-linux-gnu/include/c++/13 -isystem /usr/riscv64-linux-gnu/include/c++/13/riscv64-linux-gnu \
  -fuse-ld=lld -static -O3 -std=c++17 -march=rv64gcv_zba_zbb_zbs -I. tests/fftdump.cpp -o build/c0_rvv_clang
# scalar baseline: same, with -march=rv64gc_zba_zbb_zbs -DPOCKETFFT_NO_RVV
# GCC 15.3 (bootlin): /opt/riscv-gcc-bootlin/bin/riscv64-linux-g++ -static -O3 -std=c++17 -march=rv64gcv_zba_zbb_zbs ...
# QEMU, one VLEN:
/opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true build/c0_rvv_clang c2c double 1000 1 /tmp/x
python3 tests/compare_numpy.py c2c double 1000 1 /tmp/x
# instruction count per transform (reps 11 minus reps 1, divided by 10):
scripts/insncount.sh build/c16_rvv_clang 256 x2c double 4096
```

### B. Raw data

`results/*.txt`, `results/final_insns.csv` (final matrix, `scripts/measure_all.sh`), `results/dumps/` (binary
input/output dumps of the c2c sweep), `work-real/build/*.log` and `bench*_final_65536.txt` (real passes),
`work/p4_gcc.s`, `work/p4_clang.s`, `work/p8_clang.s` (assembly of the passes), `docs/data/` (independent
re-measurement). `docs/gen_tables.py` turns them into the tables of the performance report.

### C. Instruction histogram of one kernel transform

n = 4096, double, VLEN 256, `librvvclass` plugin (`performance-report.md` section 4): clang RVV 69,490 instructions =
42,081 vector FP + 6,936 vector loads + 4,248 vector stores + 534 `vsetvl` + 10,793 scalar ALU + 4,165 compressed
scalar + 688 branches + 45 other; scalar build 297,934 = 228,728 scalar FP + 38,426 ALU + 28,551 compressed +
2,194 branches + 35 loads/stores.

### D. Glossary

VLEN: bits per vector register (128 on C920v2/SG2044, 256 on SpacemiT X60/X100, 512 on SiFive X280). SEW: element
width (32 float, 64 double). LMUL: register-group multiplier (m1..m8; a vector instruction at LMUL 2 works on two
registers' worth of elements). `vsetvl`: sets the active vector length `vl <= VLMAX = VLEN/SEW*LMUL`; the loops here
request `vsetvl(remaining)` per chunk. Segment load `vlseg2e64`: loads `vl` pairs of 64-bit elements into two
registers, de-interleaving them (field 0 = real parts). `vluxseg2ei64`: the indexed (gather) form. `ido`, `l1`,
`ip`: FFTPACK pass geometry (section 2.4). Half-complex: FFTPACK's packed real-transform layout
`r0, r1, i1, r2, i2, ..., (r_{n/2})`. Mode A / mode B: lanes across `i` / across `k` (section 2.4).

### E. References

`docs/related-work.md` (FFTW PRs #279/#377/#408 and issue #371, pocketfft PR #31, PFFFT #110, juFFTe, camel-cdr's
rvv-bench X60 numbers, compiler support matrix) and `docs/integration.md` (consumer source references with line
numbers); the survey `/home/user/research/survey/fft.md` and `/home/user/research/notes/orchestrator-calibration.md`.
