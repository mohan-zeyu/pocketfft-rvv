# Related work: FFT on RISC-V Vector (for the design document)

Status: 2026-11-08. Every claim carries a link or a path; **[V]** = checked against the primary source or our own
experiment (details in `/home/user/research/survey/fft.md`), **[L]** = secondary source, **[U]** = unverified.

## 1. FFTW: no RVV upstream, three unmerged attempts

* FFTW 3.3.11 (NEWS dated 2026-04-18) and `master` (a7b679c, 2026-10-17) have SIMD backends for SSE2/AVX/AVX2/AVX-512/KCVI/
  AltiVec/VSX/NEON/SVE/LSX/LASX only; `simd-support/` contains no RVV file and the only RISC-V code is the cycle
  counter in `kernel/cycle.h` (`/home/user/research/src/fftw3`) **[V]**. Debian/Ubuntu/Fedora build FFTW scalar on riscv64 **[V]**.
* [PR #279](https://github.com/FFTW/fftw3/pull/279) (sh-zheng, opened 2022-03-29): first "RISC-V vector spec v1.0" port;
  SiFive built it for customers (2023); superseded **[V]**.
* [PR #377](https://github.com/FFTW/fftw3/pull/377) (R. Dolbeau, 2025-02-05, open): "r5v" masked multi-width backend
  (`simd-r5v{128..16384}.h`), official intrinsics, tested on Banana Pi F3/K1, "faster overall". Maintainer M. Frigo asked
  for a practical VLEN set, macro-based twiddles instead of generated `vtw.h`, and possibly a separate `dynsimd/`
  codelet family for dynamic VL; the thread stalled in 2026-09 on whether AI-assisted code is acceptable **[V]**.
  Follow-up [rdolbeau/fftw3#1](https://github.com/rdolbeau/fftw3/pull/1) (hugomeiland, 2026-09-12): `--enable-r5v`
  with 128/256/512/1024 packs (~4.1 MiB), `vtw.h` generator removed, `make check` passes on Orange Pi RV2 (K1) with GCC 14.3 **[V]**.
* [PR #408](https://github.com/FFTW/fftw3/pull/408) (sh-zheng, 2026-06-26, open, no reviews): `--enable-rvv`, one header
  per VLEN 128..65536, `vslide1up/down` instead of `vrgather`, validated under QEMU only **[V]**.
* [Issue #371](https://github.com/FFTW/fftw3/issues/371) (2024-10-08 ff.): LMUL=1-only codelets, register pressure,
  K1 `vrgather` cost quadratic in LMUL (256 cycles at LMUL=8), no FCMLA-style complex multiply, GCC beats LLVM, one user
  (fengyuentau, 2025-05-29) measured **no** 2-D FFT gain from the RVV branch. **camel-cdr's review (2026-06-03)** is the
  most concrete RVV-FFT advice published so far: use **nf=2 segment loads/stores** (`vlseg2e`/`vsseg2e`) to split
  interleaved complex data into re/im vectors instead of in-register shuffles; implement VDUPL/VDUPH/FLIP_RI with a single
  **masked `vslideup`/`vslidedown`** rather than `vrgather`; use **`vfsgnjn`** for conjugation / multiply-by-i sign flips;
  write the complex multiply shuffle-free as `vfmul` + `vfnmsac` (real part) and `vfmul` + `vfmacc` (imaginary part);
  consider Zvzip once hardware exists **[V]**. All of this is what the pocketfft-RVV design adopts.
* Pre-RVV-1.0 FFTW work on SG2042 (RVV 0.7.1): EPCC ISC23 slides
  ([Zhang, "FFT on SG2042"](http://riscv.epcc.ed.ac.uk/assets/files/isc23/zhang_isc23.pdf); split re/im vectors,
  register groups for small radices) **[V, partial]**; [arXiv 2506.08653](https://arxiv.org/abs/2506.08653) (2025-06-10):
  FFTW OpenMP/MPI on the 64-core SG2042 vs a 64-core EPYC 7742, ~8x gap for double 2-D FFT **[V, abstract]**.

## 2. pocketfft (NumPy / PyTorch / ITK) - PR #31

* [mreineck/pocketfft PR #31](https://github.com/mreineck/pocketfft/pull/31) (qiurui144, "Enable RISC-V Vector (RVV)
  auto-vectorization", commit 862d906 2026-04-30, merged 5f27d5a **2026-05-01**): adds
  `#elif (defined(__riscv_vector) || defined(__riscv_v_intrinsic))` with `VLEN<float>=8`, `VLEN<double>=4`
  (fixed 256-bit GCC `vector_size` types). Only batched / multi-dimensional transforms are vectorised (independent lines
  in lock-step); a single 1-D transform stays scalar. Author's numbers on SpacemiT K3 (X100, VLEN 256, GCC 15.2,
  `-O3 -march=rv64gcv_zvl256b`): **2-D c2c float 1.42-1.93x, double 1.06-1.29x, 1-D (4M points) 1.11x**; the maintainer
  "didn't expect any noticeable speedup on 1D" **[V]**.
* Our QEMU experiment (clang 18, `-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl`): the generic-vector path does become
  RVV (`vfadd.vv` 1201, `vfsub.vv` 1111, `vfmul.vf` 921, `vfmacc.vf` 829, `vlseg2e64.v` 154 ... static), results
  bit-identical to scalar, **but the binary is VLEN-specific** (crashes at VLEN 128 and 512) and **GCC 13 emits no RVV at
  all** for `vector_size` types (`/home/user/research/survey/fft.md` §1, §4) **[V]**.
* Consumers pin older commits: NumPy `numpy/fft/pocketfft @ 33ae5dc` (2024-03-22), PyTorch `third_party/pocketfft @ 0fa0ef5`
  (2024-11-30) - both pre-RVV; ITK vendors `c90e55b` (2026-06-30, post-RVV). Checked against the `main` branches on
  2026-11-07/08 (`docs/integration.md` §1) **[V]**.

## 3. PFFFT RVV backend (merged 2026-10-25)

[marton78/pffft PR #110](https://github.com/marton78/pffft/pull/110) (supersedes [#100](https://github.com/marton78/pffft/pull/100),
M. Neuling): `src/simd/pf_rvv_float.h` (`v4sf`, 128-bit `vector_size`) and `pf_rvv_double.h` (256-bit), shuffles via
`__builtin_shufflevector`, `SIMD_SZ` fixed at 4, compile-time VLEN probe, QEMU CI (`-cpu rv64,v=true,vlen=256`).
Author's SpacemiT X100 numbers vs scalar: **real FFT float 2.60-2.96x, double 2.04-2.32x; complex only ~1.3-1.4x; clang
10-25% slower than GCC** (shuffle lowering). No runtime dispatch, no LMUL scaling **[V]**. Lesson for us: real transforms
gain most; shuffles are the bottleneck of the generic-vector approach.

## 4. juFFTe (FZJ-JSC, 2026)

[arXiv 2608.28076](https://arxiv.org/html/2608.28076) (2026-08-28, ISC26 RISC-V HPC workshop; code
[FZJ-JSC/juFFTe](https://github.com/FZJ-JSC/juFFTe), Apache-2.0): Fortran library with SPIRAL-generated,
vector-length-agnostic RVV intrinsic kernels and an FFTW-compatible layer ("only one plan live at a time"). Baseline
is Dolbeau's FFTW `riscv-v-clean` branch. Single-core juFFTe/FFTW-R5V: **1.53x on SiFive X280 (VLEN 512), 0.90x on K3
X100 (VLEN 256), 0.73x on SG2044 C920v2 (VLEN 128)**; multi-core **1.09x / 1.91x / 3.03x**. Findings we reuse:
segmented loads (`vlsseg`/`vsseg`) pay off on X280 but not on X100; only 32 vector registers, so split re/im + segments
double register pressure (radix <= 16 on X100 to avoid spills); large N is memory-bound everywhere, worst on SG2044;
RISC-V cores reach ~15% of a Zen 5 core single-threaded **[V]**.

## 5. Other RVV FFT implementations and papers

* **NMSIS-DSP** (Nuclei, Apache-2.0): CMSIS-style `riscv_cfft_f32.c` with an RVV 1.0 path
  (`RISCV_MATH_VECTOR_ZVE32F`, `__riscv_vlseg2e32_v_f32m2x2`) - an embedded-oriented reference for segment-load
  butterflies ([source](https://github.com/Nuclei-Software/NMSIS/blob/master/NMSIS/DSP/Source/TransformFunctions/riscv_cfft_f32.c)) **[V]**.
* **EPCC SG2044 evaluation** ([arXiv 2508.13840](https://arxiv.org/html/2508.13840v1), N. Brown, 2025-08-19): 64x C920v2,
  RVV 1.0/VLEN 128, GCC 15.2; NAS FT gains only modestly from vectorisation (1023.8 vs 982.9 Mop/s single core); SG2044
  support in mainline Linux 6.16 **[V]**. Earlier: [arXiv 2309.00381](https://arxiv.org/html/2309.00381v2) (SG2042, 2023).
* **Peak numbers** for the cores we target come from Nassyr et al., [arXiv 2608.28097](https://arxiv.org/abs/2608.28097)
  (GEMM paper, Table 1): X60 1.6 GHz VLEN 256 -> 12.8 GFLOPS FP64 / 25.6 FP32 per core; X100 2.2 GHz -> 17.6 FP64/core;
  SG2044 C920v2 2.6 GHz VLEN 128 -> 20.8 FP64/core (`/home/user/research/survey/gemm.md`) **[V]**.
* **Vector-length-agnostic FFT convolutions** (Chalmers MSc thesis, 2023,
  [link](https://odr.chalmers.se/items/6824a91f-24ea-4c00-96dd-41d15132cc50)): VLA FFT code for RVV and SVE, 3-11x over
  GEMM/Winograd on small kernels in simulation **[L]**. **BSC long-vector FFT** (NEC SX-Aurora + RVV, Concurrency and
  Computation, [listing](https://bsc.es/ca/printpdf/research-and-development/publications/acceleration-long-vector-architectures-implementation-and)) **[L]**.
  **Zoozve** (LCTES 2025, [link](https://pldi25.sigplan.org/details/LCTES-2025-main/6/Zoozve-A-Strip-Mining-Free-RISC-V-Vector-Extension-with-Arbitrary-Register-Grouping-)):
  ISA proposal; FFT dynamic instruction count -10x - shows how much strip-mining/LMUL limits cost today **[L]**.
* **KissFFT**: no RISC-V code (`USE_SIMD` = SSE/LSX only) **[V]**; **Google Highway**: no FFT **[V]**; **SciPy** uses
  ducc0 whose `infra/simd.h` has no RVV path and whose `std::experimental::simd` fallback is width 1 on RISC-V
  (our QEMU test, `integration.md` §5) **[V]**; **OpenCV `cv::dft`** RVV status **[U]**.

## 6. Compiler support that bounds the design space

* GCC 13 (Ubuntu 24.04 cross default): intrinsics v0.11 (`__riscv_v_intrinsic` 11000), no `-mrvv-vector-bits`, no RVV
  lowering of `vector_size` types, `riscv_vector.h` refuses to compile without global V **[V]**.
* GCC 14 (2024-05): intrinsics v0.12 API (`12000`; release notes call it the v1.0 spec), loop/SLP vectoriser on for RVV,
  `-mrvv-vector-bits=zvl` + `riscv_rvv_vector_bits`, **`target` attribute** incl. RVV intrinsics inside
  `target("arch=+v")` functions ([GCC 14 changes](https://gcc.gnu.org/gcc-14/changes.html); gcc-14 branch sources) **[V]**.
  GCC 15: same macro value; GCC 16/17 (master) bump to `1000000` (v1.0) and add `-mcpu=spacemit-x100` **[V]**.
* clang 17-19: `12000`; clang 19 adds per-function `target("arch=+v")` for RVV intrinsics
  ([llvm#83674](https://github.com/llvm/llvm-project/pull/83674)); clang 20+: `1000000` **[V]**.
* Runtime detection: `getauxval(AT_HWCAP)` bit 21 or `riscv_hwprobe` (`RISCV_HWPROBE_IMA_V`; glibc 2.40 `__riscv_hwprobe`),
  xtheadvector via `RISCV_HWPROBE_KEY_VENDOR_EXT_THEAD_0` ([kernel docs](https://docs.kernel.org/arch/riscv/hwprobe.html)) **[V]**.

## 7. Microarchitectural facts that drive the design (SpacemiT X60 = K1)

Source: camel-cdr's rvv-bench results for the Banana Pi BPI-F3
([bpi_f3 page](https://camel-cdr.github.io/rvv-bench-results/bpi_f3/index.html), 8x X60 @ 1.6 GHz, RVV 1.0 + Zvfh,
VLEN 256; the page notes the vector unit "seems to be exactly the same as the one from the XuanTie C908" and links
P. Reames' [bp3-microarch notes](https://github.com/preames/public-notes/blob/master/bp3-microarch.md)); summary also in
`/home/user/research/survey/gemm.md` (E17) **[V]**:

| Instruction (cycles per instruction, X60) | m1 | m2 | m4 | m8 | Design consequence |
|---|---|---|---|---|---|
| `vfmacc.vv` / `vfadd.vv` e32/e64 | 1.00 | 2 | 4 | 8 | Throughput scales linearly with LMUL: **LMUL>1 executes as LMUL m1-sized pieces** ("cracked"); it saves instructions/loop overhead but adds no throughput. Peak = one 256-bit FMA per cycle. |
| `vle32.v` unit-stride | ~1.25-1.6 | | | | Loads are ~half the FMA rate: **FMA:load ratio matters**; radix-4/8 butterflies with register-resident data are the goal. |
| `vle64.v` unit-stride | ~2.1-2.4 | ~4.2 | ~8.5 | ~16.5 | Same. |
| **`vlse32.v` / `vlse64.v` strided** | **4.00** | 8.00 | 16 | 32 | **Strided loads cost ~2-3x a unit-stride load** -> prefer `vlseg2e` (nf=2 segment) loads for re/im de-interleave and unit-stride access patterns; use strides only for genuine transposes. |
| `vrgather.vv` e8 | 4 | 16 | 64 | 256 | **Quadratic in LMUL: avoid `vrgather`** (also reported in FFTW #371). |
| `vslideup.vx` / `vslidedown.vx` e8 | 2 / 3 | 4 / 5 | 8 / 9 | 16 / 17 | Masked slides are the cheap in-register permutation (camel-cdr's VDUP/FLIP_RI advice). |

Further constraints: only **32 architectural vector registers** (RVV spec), which with LMUL=2 means 16 register
groups - the reason juFFTe caps radix at 16 on X100 and why our passes keep LMUL=1 for high radices; the X60 is an 8-stage
in-order dual-issue core (RT-RK GCC scheduler work, `gemm.md`) so instruction scheduling (interleaving loads with FMAs,
`-fno-schedule-insns` experiments as in FFTW) matters more than on the out-of-order C920v2. For SG2044 (VLEN 128, OoO,
2 MB L2 per 4 cores, 64 MB L3) the published evidence says large transforms are memory-bound and segmented loads are
less of a win than on X280 (juFFTe); LMUL=2 is the natural way to get 256-bit work units there.

## 8. Gap this project fills

No shipped library gives NumPy/PyTorch users an RVV-accelerated **single 1-D** FFT today: FFTW's ports are unmerged
and fixed-VLEN, pocketfft upstream vectorises only batched lines with a VLEN-specific generic-vector type that GCC 13
ignores and that crashes on other VLENs, PFFFT covers only `SIMD_SZ=4` real transforms, juFFTe is Fortran with a
one-plan limitation. A VLEN-agnostic, runtime-dispatched intrinsics backend inside pocketfft (reached by NumPy,
PyTorch and ITK through the call chains in `integration.md`) is therefore both novel and immediately consumable.
