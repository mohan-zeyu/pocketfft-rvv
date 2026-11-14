# pocketfft-RVV: functional correctness test report

Status: 2026-11-08. Code under test: `pocketfft_hdronly.h` + `pocketfft_rvv.h` + `pocketfft_rvv_real.h` of
`/home/user/fft-rvv` (commits `3b58513` complex passes, `2e99710` real passes and framework fixes, `d1a003c`
half-complex pack/unpack); the exact header state behind each table is named with it. Companion documents:
`docs/design.md` section 4 (design of the verification process), `docs/performance-report.md`, `tests/README.md`.

**Verdict.** Every configuration passes the competition criterion (max |out - ref| / max |ref| <= 1e-5) with a
margin of ten orders of magnitude in double precision and two in single precision: the worst double-precision
error of the RVV build is 1.4e-15 against FFTW 3.3.11 and 2.7e-16 against NumPy 2.5.3, the worst single-precision
error 3.7e-7 / 1.2e-7 - the same values the unmodified scalar build produces. The RVV and scalar builds differ
from each other by at most 8.8e-16 (double) / 4.0e-7 (float) relative to the output scale (the pathological
Bluestein length 65537), typically 1e-17 / 1e-8, and the RVV results are **bit-identical across VLEN 128, 256 and
512** in every case. No failure, crash or non-finite output was observed in 8 x 1080 + 5 x 1080 FFTW cases, 348 + 2,072 + 1,554
NumPy cases, 6 x 226 x 4 exact-reference real-transform cases and the real-FFT agent's multi-D check.

## 1. Method

### 1.1 Oracles and comparisons

| oracle | what it checks | tool | notes |
|:---|:---|:---|:---|
| **NumPy 2.5.3** `numpy.fft.fft/ifft/rfft/irfft` (x86 host, pocketfft C++ 33ae5dc inside, computed in complex128/float64) | the public `pocketfft::c2c/r2c/c2r` results of the RISC-V binaries, dumped under QEMU | `tests/fftdump.cpp` writes `<prefix>.in/.out` (binary, native element type); `tests/compare_numpy.py <kind> <type> <n> <fwd> <prefix> [scalar-prefix]` | the rubric's "mainstream library"; the reference is computed from the exact float/double inputs the binary used |
| **FFTW 3.3.11** double precision, built scalar for riscv64 (`-march=rv64gc_zba_zbb_zbs`, no V) and linked statically into the test binary | 1080 cases per configuration: c2c fwd/bwd, r2c, c2r, float/double, 1-D, batched 2-D (shape {5,n}/{9,n}: the generic-vector path with residual lines), strided (3/2), in place | `tests/fft_check.cpp` (`--csv`, `--dump`, `--compare-scalar`) | a second independent reference, runnable on the board; the FFTW build does not contain any of the code under test |
| **Scalar build of the same header** (`-DPOCKETFFT_NO_RVV`, `-march=rv64gc`) | element-wise difference RVV vs scalar on bit-identical inputs | `fft_check --dump` (scalar) + `--compare-scalar` (RVV); `compare_numpy.py` second prefix | isolates what the RVV code changes: only FMA-contraction / summation-order rounding is expected |
| **Exact reference** (naive O(n^2) DFT in long double for n <= 4096, long-double pocketfft above, stored as double-double) | the real passes (r2c/c2r) over 226 lengths 1..65536, float and double | `work-real/test_real.cpp genref / check` | written by the real-FFT agent; also compares RVV with the scalar build and runs a multi-D (generic-vector path) check |

### 1.2 Metrics and pass criteria

Per transform: `max_err_norm = max_i |out_i - ref_i| / max_i |ref_i|` (the rubric's quantity; **pass iff <= 1e-5**),
`l2_rel = ||out - ref|| / ||ref||`, and for the scalar comparison `vs_dump = max_i |out_i - out_scalar_i| / max_i |ref_i|`
(pass thresholds 1e-12 double / 1e-4 float, i.e. four orders of magnitude above the expected rounding differences,
so a wrong lane, stride, twiddle or sign fails loudly instead of hiding in the tolerance). `compare_numpy.py`
prints the same quantities as `rel_to_scale`, `l2rel`, `rvv_vs_scalar`.

Scaling convention: `fct = 1` forward, `1/n` backward in `fft_check` (NumPy's convention; both the scaled and the
unscaled code paths are exercised); `fftdump` uses `fct = 1` and `compare_numpy.py` undoes NumPy's `1/n`.

### 1.3 Inputs

All inputs are pseudo-random in [-1, 1) from 64-bit integer LCGs seeded from the case parameters (`fft_check`:
seed, type, kind, direction, n, batch, strides; `fftdump`: a fixed seed), generated in integer arithmetic, so the
same values are produced on every machine, compiler and ISA; float inputs are the double samples rounded to float
and the reference is computed from exactly those values. c2r inputs have the imaginary parts of the DC and Nyquist
bins zeroed so that every library interprets the Hermitian input identically.

### 1.4 VLEN sweep and compilers

One RVV binary per compiler is run under `qemu-riscv64 -cpu rv64,v=true,vlen=V,elen=64,zba=true,zbb=true,zbs=true`
with V = 128, 256, 512 (`vext_spec` v1.0, QEMU 11.1.3). Because every loop is strip-mined with `vsetvl`, a VLEN
change alters the chunking of every pass and the mode-A/mode-B decision (`design.md` section 2.4); identical
results across VLENs are therefore a strong check of the vector-length-agnostic code paths. Both toolchains are
tested because their code generation differs substantially (`performance-report.md` section 4).

## 2. Environment

| component | version / setting |
|:---|:---|
| Host | x86-64, Ubuntu 24.04.5 LTS, Intel Xeon 2.1 GHz (4 cores, container), Python 3.13.18, **NumPy 2.5.3** (`/usr/local/lib/python3.13/site-packages/numpy`) |
| RISC-V compilers | **clang 23.1.6** (Ubuntu clang version 23.1.6, `--target=riscv64-linux-gnu`, Ubuntu riscv64 sysroot with libstdc++ 13, `-fuse-ld=lld -static`); **GCC 15.3.0** (toolchains.bootlin.com 2026.08-1, `/opt/riscv-gcc-bootlin`, static) |
| Flags | `-O3 -std=c++17`; RVV: `-march=rv64gcv_zba_zbb_zbs`; scalar: `-march=rv64gc_zba_zbb_zbs -DPOCKETFFT_NO_RVV` (`scripts/common.sh`, `scripts/build.sh`) |
| Emulator | **QEMU 11.1.3** user mode, built from source with TCG plugins (`/opt/qemu/bin/qemu-riscv64`), CPU `rv64,v=true,vlen={128,256,512},elen=64,zba,zbb,zbs` |
| References | NumPy 2.5.3 (host); FFTW 3.3.11 static, riscv64 `rv64gc_zba_zbb_zbs` build in `work/fftw-riscv64` (`tests/README.md`); long-double naive DFT (`work-real/test_real.cpp`) |
| Code states | (A) archived dumps: header of 2026-11-08 11:22 UTC (complex passes, before the twiddle vectorisation; `build/fftdump_rvv` = GCC 15 build, `build/fftdump_scalar`); (B, C) snapshot of commit `2e99710` rebuilt from source for this report (`docs/data/SNAPSHOT_2e99710.txt` has the md5 sums); (D) the real-FFT agent's branch and its integration build (`work-real/build/integ2`); (E) commit `d1a003c` (HEAD) |

## 3. Test matrix

| id | code state | tool / oracle | kinds | types | lengths | VLEN | compilers | cases | result |
|:---|:---|:---|:---|:---|:---|:---|:---|---:|:---|
| A | 11:22 header | `fftdump` vs NumPy + scalar dump (`results/dumps/`, re-checked for this report) | c2c fwd, c2c bwd | float, double | 29: 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 24, 25, 30, 32, 60, 64, 100, 120, 128, 243, 256, 360, 625, 1000, 1024, 1536, 4096, 10007 (prime, Bluestein), 65536 | 128, 256, 512 | GCC 15 | 348 | all pass |
| B | `2e99710` | `fft_check` vs FFTW (+ scalar dump for the RVV builds) | c2c fwd/bwd, r2c, c2r; 1-D, batched {5,n} {9,n}, strided 3/2, in place | float, double | 85 lengths: powers of two 1..2^20, 2^a 3^b 5^c mixes, radix 7/11/13 and generic factors (49, 77, 121, 143, 169, 231, 352, 448, 1001, 1331, 2401), primes 13..65537 (Bluestein), odd/non-aligned (3, 5, 7, 9, 15, 33, 65, 129, 1023, 1025) | 128, 256, 512 (+ scalar) | clang 23, GCC 15 | 8 x 1080 | all pass |
| C | `2e99710` | `fftdump` vs NumPy + scalar dump | c2c fwd, c2c bwd, r2c, c2r | float, double | 37: 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 24, 25, 30, 32, 49, 60, 64, 77, 100, 120, 121, 128, 243, 256, 360, 625, 1000, 1024, 1331, 1536, 4096, 10007, 65536 | 128, 256, 512 (+ scalar) | clang 23, GCC 15 | 7 x 296 = 2,072 | all pass |
| D | real-FFT branch (`work-real`) and its integration build | `test_real.cpp` vs exact long-double reference + cross-build (RVV vs scalar) + multi-D check | r2c, c2r (+ 2-D generic-vector path) | float, double | 226 lengths 1..65536 (all 2^a 3^b 5^c up to 65536, primes 509, 1009, 1021, 2003, 4093, 8191, 65537-free set, 7-smooth and 11-smooth lengths, ...) | 128, 256, 512 | clang 23, GCC 15 (`-O3`; also `-O0` builds) | 6 x 904 + 2 integration runs | ALL PASSED |
| E | `d1a003c` (HEAD, adds pack/unpack) | `fftdump` vs NumPy + scalar dump | r2c, c2r, c2c fwd | float, double | the 37 lengths of C | 128, 256, 512 | clang 23, GCC 15 | 7 x 222 = 1,554 | all pass (section 4.5) |
| F | `13bfcab+` | orchestrator's sweep: `fftdump` vs NumPy (`scripts/numpy_sweep.sh`, log `results/numpy_sweep_23len.log`) | c2c, r2c, c2r | float, double | 23 lengths (2..65536 incl. 10007) | 128, 256, 512 | clang 23, GCC 15 | - | 828 comparisons, 828 pass, 0 fail; worst rel-to-scale error double 2.72e-16, float 7.41e-08 |
| G | `d1a003c` (`build/riscv64-*`) | test agent's run of `scripts/run_correctness.sh`: `fft_check` vs FFTW (+ scalar dump) | as B | float, double | as B | 128, 256, 512 (+ scalar, + x86-64 native) | clang 23 (+ GCC 13 on x86-64) | 5 x 1080 | all pass - `results/correctness_summary.md`, `results/correctness_*.csv` |
| H | `d1a003c` | `tests/framework_callsites.cpp`: NumPy's per-row route (`get_plan` + `exec`) and the NumPy-batched / PyTorch whole-tensor route (`pocketfft::c2c` with byte strides, axes {1}) on the same data, cross-checked against each other and a direct 1-D call, outputs dumped for NumPy | c2c (routes), r2c, c2r | float, double | per test | 128, 256, 512 | clang 23 | - | interface-adaptation evidence (`tests/README.md`) |

## 4. Results

### 4.1 Matrix A - archived c2c dumps vs NumPy (29 lengths, re-checked)

All 348 comparisons pass (`python3 tests/compare_numpy.py c2c <type> <n> <fwd> results/dumps/r<V>_<type>_<n>_<fwd> results/dumps/s_<type>_<n>_<fwd>`):

| VLEN | type | cases | pass | worst max err / scale (case) | worst L2 rel | worst RVV vs scalar (case) | bit-identical to scalar |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 128 | double | 58 | 58 | 1.49e-16 (n = 3, bwd) | 6.09e-16 (n = 10007) | 8.1e-17 (n = 10007) | 4 cases |
| 256 | double | 58 | 58 | 1.49e-16 (n = 3, bwd) | 6.09e-16 | 8.1e-17 (n = 10007) | 4 |
| 512 | double | 58 | 58 | 1.49e-16 (n = 3, bwd) | 6.09e-16 | 8.1e-17 (n = 10007) | 4 |
| 128 | float | 58 | 58 | 6.39e-08 (n = 1536) | 2.43e-07 (n = 10007) | 4.65e-08 (n = 10007) | 6 |
| 256 | float | 58 | 58 | 6.39e-08 (n = 1536) | 2.43e-07 | 4.65e-08 (n = 10007) | 6 |
| 512 | float | 58 | 58 | 6.39e-08 (n = 1536) | 2.43e-07 | 4.65e-08 (n = 10007) | 6 |

Representative individual cases (VLEN 256; the numbers are identical at 128 and 512):

| kind | type | n | dir | max abs err | max err / scale | L2 rel | RVV vs scalar |
|:---|:---|---:|:---|---:|---:|---:|---:|
| c2c | double | 16 | fwd | 0 | 0 | 0 | 1.7e-17 |
| c2c | double | 120 | fwd | 3.8e-15 | 2.2e-17 | 8.2e-17 | 2.3e-17 |
| c2c | double | 243 | bwd | 2.8e-14 | 8.2e-17 | 1.6e-16 | 2.1e-17 |
| c2c | double | 1000 | fwd | 3.5e-14 | 2.5e-17 | 1.4e-16 | 1.6e-17 |
| c2c | double | 1024 | bwd | 2.8e-14 | 2.0e-17 | 1.2e-16 | 2.0e-17 |
| c2c | double | 4096 | fwd | 4.5e-14 | 7.8e-18 | 1.3e-16 | 7.5e-18 |
| c2c | double | 10007 (Bluestein) | fwd | 1.8e-12 | 1.3e-16 | 6.1e-16 | 8.1e-17 |
| c2c | double | 65536 | fwd | 2.7e-13 | 2.9e-18 | 1.6e-16 | 2.6e-18 |
| c2c | float | 16 | fwd | 8.4e-07 | 3.3e-08 | 5.6e-08 | 9.3e-09 |
| c2c | float | 243 | fwd | 1.7e-05 | 5.0e-08 | 9.2e-08 | 1.3e-08 |
| c2c | float | 1000 | fwd | 6.6e-05 | 4.6e-08 | 1.0e-07 | 7.2e-09 |
| c2c | float | 4096 | fwd | 2.1e-04 | 3.6e-08 | 1.1e-07 | 4.1e-09 |
| c2c | float | 10007 (Bluestein) | fwd | 5.5e-04 | 3.9e-08 | 2.4e-07 | 4.7e-08 |
| c2c | float | 65536 | fwd | 3.0e-03 | 3.3e-08 | 1.2e-07 | 1.4e-09 |

(Absolute errors grow with n because the output scale grows like sqrt(n); the normalised error does not.)

### 4.2 Matrix B - `fft_check` vs FFTW, 1080 cases per configuration (commit `2e99710`)

Binaries built from the snapshot for this report; logs and CSVs in the documentation pass's scratch directory,
summary lines reproduced here:

| configuration | cases | pass | fail | worst float max err / scale (case) | worst double (case) | worst float L2 rel | worst double L2 rel | RVV vs scalar float (bit-identical) | RVV vs scalar double (bit-identical) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clang 23 scalar (rv64gc) | 1080 | 1080 | 0 | 3.77e-07 (r2c fwd n = 65537) | 1.32e-15 (c2c bwd n = 65537) | 3.12e-07 | 1.15e-15 | - | - |
| clang 23 RVV, VLEN 128 | 1080 | 1080 | 0 | 3.65e-07 (r2c bwd n = 65537) | 1.36e-15 (r2c fwd n = 65537) | 3.10e-07 | 1.18e-15 | 4.05e-07 (161 / 540) | 8.76e-16 (92 / 540) |
| clang 23 RVV, VLEN 256 | 1080 | 1080 | 0 | 3.65e-07 | 1.36e-15 | 3.10e-07 | 1.18e-15 | 4.05e-07 (161 / 540) | 8.76e-16 (92 / 540) |
| clang 23 RVV, VLEN 512 | 1080 | 1080 | 0 | 3.65e-07 | 1.36e-15 | 3.10e-07 | 1.18e-15 | 4.05e-07 (161 / 540) | 8.76e-16 (92 / 540) |
| GCC 15.3 scalar (rv64gc) | 1080 | 1080 | 0 | 3.23e-07 (r2c fwd n = 65537) | 1.39e-15 (r2c fwd n = 65537) | 3.01e-07 | 1.14e-15 | - | - |
| GCC 15.3 RVV, VLEN 128 | 1080 | 1080 | 0 | 3.63e-07 (c2c fwd n = 65537) | 1.36e-15 (r2c fwd n = 65537) | 3.11e-07 | 1.18e-15 | 3.85e-07 (147 / 540) | 7.79e-16 (79 / 540) |
| GCC 15.3 RVV, VLEN 256 | 1080 | 1080 | 0 | 3.63e-07 | 1.36e-15 | 3.11e-07 | 1.18e-15 | 3.85e-07 (147 / 540) | 7.79e-16 (79 / 540) |
| GCC 15.3 RVV, VLEN 512 | 1080 | 1080 | 0 | 3.63e-07 | 1.36e-15 | 3.11e-07 | 1.18e-15 | 3.85e-07 (147 / 540) | 7.79e-16 (79 / 540) |

Every worst case is the prime length 65537 (Bluestein: a chirp multiplication, a 2^17-point FFT pair and a second
multiplication, in which both libraries accumulate the most rounding); the scalar build has the same worst cases
with the same magnitudes, so the RVV code does not degrade accuracy anywhere. The "bit-identical" counts are the
cases whose RVV output equals the scalar output exactly (short lengths and the lengths whose passes contain no
FMA-contracted products).

### 4.2b Matrix G - the test agent's committed `fft_check` run (`results/correctness_summary.md`)

Independently of the run above, the test agent ran `scripts/run_correctness.sh` on the `build/riscv64-*` binaries of
commit `d1a003c` and committed the per-case CSVs. Its worst values are identical to the table above (clang 23 RVV at
VLEN 128/256/512: float 3.645e-07 and double 1.360e-15 vs FFTW, RVV-vs-scalar 4.048e-07 / 8.755e-16, all at
n = 65537; clang scalar 3.770e-07 / 1.319e-15; and the x86-64 GCC 13 native build 3.226e-07 / 1.340e-15), i.e. the
RISC-V builds are as accurate as the x86 build of the same library. 5 x 1080 cases, all pass.

### 4.3 Matrix C - fresh `fftdump` vs NumPy, c2c / r2c / c2r, 37 lengths (commit `2e99710`)

<!-- fresh-numpy-table -->
2072 comparisons (scalar build vs NumPy, and each RVV build vs NumPy and vs the scalar build), 2072 PASS, 0 FAIL, 0 other lines. Per (build, kind, type): cases, passes, number of distinct lengths, worst `max |err| / max |ref|` with its case, worst relative L2 error, worst RVV-vs-scalar difference with its length, and the number of cases whose RVV output is bit-identical to the scalar output.

| build / VLEN | kind | type | cases | pass | lengths | worst max err / scale (n, dir) | worst L2 rel | worst RVV vs scalar (n) | bit-identical cases |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| scalar (clang, rv64gc) | c2c | double | 74 | 74 | 37 (1..65536) | 1.49e-16 (n=3, bwd) | 6.09e-16 | - | - |
| scalar (clang, rv64gc) | c2r | double | 37 | 37 | 37 (1..65536) | 2.59e-16 (n=100, bwd) | 4.58e-16 | - | - |
| scalar (clang, rv64gc) | r2c | double | 37 | 37 | 37 (1..65536) | 1.83e-16 (n=10007, fwd) | 4.74e-16 | - | - |
| scalar (clang, rv64gc) | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.41e-07 | - | - |
| scalar (clang, rv64gc) | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.79e-07 | - | - |
| scalar (clang, rv64gc) | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.82e-07 | - | - |
| clang 23 RVV, VLEN 128 | c2c | double | 74 | 74 | 37 (1..65536) | 2.59e-16 (n=10007, bwd) | 6.06e-16 | 1.30e-16 (n=10007) | 16 |
| clang 23 RVV, VLEN 128 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 128 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 128 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 27 |
| clang 23 RVV, VLEN 128 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 128 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 128 | c2c | double | 74 | 74 | 37 (1..65536) | 1.49e-16 (n=3, bwd) | 5.82e-16 | 1.30e-16 (n=10007) | 14 |
| GCC 15.3 RVV, VLEN 128 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 128 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 128 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.69e-08 (n=10007) | 19 |
| GCC 15.3 RVV, VLEN 128 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 128 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 256 | c2c | double | 74 | 74 | 37 (1..65536) | 2.59e-16 (n=10007, bwd) | 6.06e-16 | 1.30e-16 (n=10007) | 16 |
| clang 23 RVV, VLEN 256 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 256 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 256 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 27 |
| clang 23 RVV, VLEN 256 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| clang 23 RVV, VLEN 256 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| GCC 15.3 RVV, VLEN 256 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 256 | c2c | double | 74 | 74 | 37 (1..65536) | 1.49e-16 (n=3, bwd) | 5.82e-16 | 1.30e-16 (n=10007) | 14 |
| GCC 15.3 RVV, VLEN 256 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 256 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.69e-08 (n=10007) | 19 |
| GCC 15.3 RVV, VLEN 256 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |
| GCC 15.3 RVV, VLEN 256 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |
| clang 23 RVV, VLEN 512 | c2c | double | 74 | 74 | 37 (1..65536) | 2.59e-16 (n=10007, bwd) | 6.06e-16 | 1.30e-16 (n=10007) | 16 |
| clang 23 RVV, VLEN 512 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 512 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 512 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 512 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 27 |
| clang 23 RVV, VLEN 512 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 512 | c2c | double | 74 | 74 | 37 (1..65536) | 1.49e-16 (n=3, bwd) | 5.82e-16 | 1.30e-16 (n=10007) | 14 |
| GCC 15.3 RVV, VLEN 512 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 512 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 512 | c2c | float | 74 | 74 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.69e-08 (n=10007) | 19 |
| GCC 15.3 RVV, VLEN 512 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |
| GCC 15.3 RVV, VLEN 512 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |

### 4.4 Matrix D - real passes vs exact reference (real-FFT agent, `work-real/test_real.cpp`)

226 lengths 1..65536, float and double, r2c and c2r, against the long-double reference, plus the cross-build
comparison with the scalar binary and a multi-D check of the generic-vector path (`work-real/build/run7_*.log`,
`integ2/{c,g}.log`; the two numbers per cell are the worst relative L2 error `||err|| / ||ref||` and the worst `max |err| / max |ref|` over the sweep, as printed by `test_real.cpp`):

| build | VLEN | double r2c (L2 rel / max err / scale) | double c2r | float r2c | float c2r | RVV vs scalar (worst, case) | result |
|:---|:---|---:|---:|---:|---:|---:|:---|
| clang 23 | 128 / 256 / 512 (identical) | 3.64e-16 / 1.82e-16 | 3.66e-16 / 9.99e-16 | 2.32e-07 / 1.72e-07 | 1.98e-07 / 4.77e-07 | 1.69e-07 (n = 1009, float r2c) | ALL PASSED, multi-D ok |
| GCC 15.3 | 128 / 256 / 512 (identical) | 3.83e-16 / 2.25e-16 | 3.54e-16 / 8.88e-16 | 2.29e-07 / 1.72e-07 | 1.95e-07 / 4.17e-07 | 1.2e-07 (n = 59049, float c2r) | ALL PASSED, multi-D ok |
| integration build (merged header), clang VLEN 512 / GCC VLEN 256 | | 3.74e-16 / 1.82e-16 | 3.55e-16 / 7.77e-16 | 1.89e-07 / 1.23e-07 | 2.0e-07 / 4.17e-07 | 2.2e-07 (n = 4093, float r2c) | ALL PASSED, multi-D ok |
| GCC `-O0` (unoptimised build of the same code) | 128 / 512 | 3.8e-16 / 1.82e-16 | 3.79e-16 / 8.88e-16 | 2.4e-07 / 1.72e-07 | 1.97e-07 / 5.37e-07 | 1.94e-07 (n = 1009, float r2c) | ALL PASSED |

### 4.5 Matrix E - HEAD (`d1a003c`) re-check of r2c / c2r / c2c against NumPy

The half-complex pack/unpack added in `d1a003c` is the newest code; it was re-checked with binaries built from
HEAD:

<!-- head-check-table -->
Binaries built from a snapshot of `31e33be` (headers identical to `d1a003c`), the 37 lengths of matrix C, r2c (forward), c2r (backward) and c2c forward, float and double: 1554 comparisons, 1554 PASS, 0 FAIL, 0 other lines. The pack/unpack path is taken for every length > 2 of these runs (contiguous 1-D data).

| build / VLEN | kind | type | cases | pass | lengths | worst max err / scale (n, dir) | worst L2 rel | worst RVV vs scalar (n) | bit-identical cases |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| scalar (clang, rv64gc) | r2c | double | 37 | 37 | 37 (1..65536) | 1.83e-16 (n=10007, fwd) | 4.74e-16 | - | - |
| scalar (clang, rv64gc) | c2r | double | 37 | 37 | 37 (1..65536) | 2.59e-16 (n=100, bwd) | 4.58e-16 | - | - |
| scalar (clang, rv64gc) | c2c | double | 37 | 37 | 37 (1..65536) | 8.90e-17 (n=7, fwd) | 5.93e-16 | - | - |
| scalar (clang, rv64gc) | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.82e-07 | - | - |
| scalar (clang, rv64gc) | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.79e-07 | - | - |
| scalar (clang, rv64gc) | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.41e-07 | - | - |
| clang 23 RVV, VLEN 128 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 128 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 128 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.92e-16 | 1.30e-16 (n=10007) | 7 |
| clang 23 RVV, VLEN 128 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 128 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 14 |
| clang 23 RVV, VLEN 128 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 128 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 128 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 128 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.82e-16 | 1.30e-16 (n=10007) | 6 |
| GCC 15.3 RVV, VLEN 128 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 128 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |
| GCC 15.3 RVV, VLEN 128 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.57e-08 (n=10007) | 10 |
| clang 23 RVV, VLEN 256 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 256 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 256 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.92e-16 | 1.30e-16 (n=10007) | 7 |
| clang 23 RVV, VLEN 256 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 256 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| clang 23 RVV, VLEN 256 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 14 |
| GCC 15.3 RVV, VLEN 256 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 256 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 256 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.82e-16 | 1.30e-16 (n=10007) | 6 |
| GCC 15.3 RVV, VLEN 256 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |
| GCC 15.3 RVV, VLEN 256 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.57e-08 (n=10007) | 10 |
| GCC 15.3 RVV, VLEN 256 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |
| clang 23 RVV, VLEN 512 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.24e-16 | 1.83e-16 (n=10007) | 9 |
| clang 23 RVV, VLEN 512 | c2r | double | 37 | 37 | 37 (1..65536) | 2.72e-16 (n=10007, bwd) | 4.59e-16 | 2.09e-16 (n=5) | 10 |
| clang 23 RVV, VLEN 512 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.92e-16 | 1.30e-16 (n=10007) | 7 |
| clang 23 RVV, VLEN 512 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.86e-07 | 3.39e-08 (n=10007) | 15 |
| clang 23 RVV, VLEN 512 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.83e-07 | 1.12e-07 (n=5) | 15 |
| clang 23 RVV, VLEN 512 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.44e-07 | 6.43e-08 (n=10007) | 14 |
| GCC 15.3 RVV, VLEN 512 | r2c | double | 37 | 37 | 37 (1..65536) | 6.93e-17 (n=13, fwd) | 4.25e-16 | 1.83e-16 (n=10007) | 9 |
| GCC 15.3 RVV, VLEN 512 | c2r | double | 37 | 37 | 37 (1..65536) | 1.81e-16 (n=10007, bwd) | 4.51e-16 | 2.09e-16 (n=5) | 10 |
| GCC 15.3 RVV, VLEN 512 | c2c | double | 37 | 37 | 37 (1..65536) | 1.30e-16 (n=10007, fwd) | 5.82e-16 | 1.30e-16 (n=10007) | 6 |
| GCC 15.3 RVV, VLEN 512 | c2r | float | 37 | 37 | 37 (1..65536) | 1.16e-07 (n=32, bwd) | 1.78e-07 | 1.12e-07 (n=5) | 15 |
| GCC 15.3 RVV, VLEN 512 | c2c | float | 37 | 37 | 37 (1..65536) | 6.39e-08 (n=1536, fwd) | 2.43e-07 | 3.57e-08 (n=10007) | 10 |
| GCC 15.3 RVV, VLEN 512 | r2c | float | 37 | 37 | 37 (1..65536) | 8.86e-08 (n=120, fwd) | 1.85e-07 | 3.31e-08 (n=10007) | 15 |

## 5. Observations

1. **Accuracy is unchanged by vectorisation.** Against both references the RVV builds show the same worst cases
   and magnitudes as the scalar build of the same compiler (1.3-1.4e-15 double, 3.2-3.8e-7 float vs FFTW;
   1.5-2.7e-16 / 0.6-1.2e-7 vs NumPy). The RVV-vs-scalar differences are 1e-17..1e-16 (double) and
   1e-9..5e-8 (float) for ordinary lengths and reach 8.8e-16 / 4e-7 only for the 65537-point Bluestein transform.
   They come from FMA contraction and summation order (`twmul` evaluates `a*w` as `vfmul` + `vfmacc`/`vfnmsac`,
   the scalar code lets the compiler contract differently), not from any algorithmic change.
2. **Results are bit-identical across VLEN 128, 256 and 512** in every case of every matrix (the worst-case rows
   agree to all printed digits and the per-case CSVs match), because each lane executes the same sequence of
   operations irrespective of how many lanes a chunk has. This also confirms that the mode-A/mode-B selection,
   which does depend on VLEN, produces the same arithmetic per element.
3. **Both compilers are correct**; their RVV outputs differ slightly from each other (different contraction in the
   scalar code they compare against), which is visible as different "bit-identical" counts (92 vs 79 of 540 double
   cases).
4. **Single precision** errors (1e-7) are the float32 rounding of pocketfft's own float arithmetic (identical for
   the scalar build), measured against double-precision references; they are two orders of magnitude inside the
   1e-5 criterion.
5. **Coverage of the dispatch**: `long double` instantiations (which NumPy builds) compile to the scalar code
   through the `real_t<T>` guard; the batched {5,n}/{9,n} cases exercise the generic-vector path plus residual
   RVV lines (both the FFTW matrix and the real-FFT agent's multi-D check pass); strided and in-place cases take
   the scalar `copy_input/copy_output` paths around the RVV kernels.

## 6. Rubric mapping

| requirement | evidence |
|:---|:---|
| Functional correctness against mainstream libraries, error <= 1e-5 | NumPy 2.5.3: worst 2.7e-16 (double), 1.2e-7 (float); FFTW 3.3.11: 1.4e-15 / 3.7e-7 - all `max |err| / max |ref|`, every case passes; 13 x 1080 FFTW cases + 348 + 2,072 + 1,554 NumPy cases + 6 x 904 exact-reference cases |
| Several transform kinds, sizes and precisions | c2c forward/backward, r2c, c2r; float and double; lengths 1..2^20 including primes (Bluestein), radix 7/11/13 and generic factors, odd/non-aligned lengths; batched, strided, in-place |
| Portability of one binary across vector lengths | identical results at VLEN 128/256/512 under QEMU; scalar fallback build identical to upstream behaviour |
| Toolchain independence | clang 23 and GCC 15 |
| Reproducibility | deterministic integer-seeded inputs, scripts and commands in section 7, raw dumps in `results/dumps/`, CSV logs per configuration |

## 7. How to reproduce

```sh
# A: archived dumps vs NumPy (host python3 with NumPy)
for v in 128 256 512; do for t in double float; do for d in 1 0; do for f in results/dumps/r${v}_${t}_*_${d}.in; do
  n=$(basename $f .in); n=${n#r${v}_${t}_}; n=${n%_${d}}
  python3 tests/compare_numpy.py c2c $t $n $d results/dumps/r${v}_${t}_${n}_${d} results/dumps/s_${t}_${n}_${d}
done; done; done; done
# B: FFTW matrix for every built variant (scalar dumps first, then RVV at VLEN 128/256/512 with --compare-scalar)
scripts/build.sh && scripts/run_correctness.sh            # --quick for n <= 4096
# C / E: one fresh dump under QEMU and its check against NumPy (RVV build, VLEN 256) and against the scalar build
/opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true build/c0_rvv_clang r2c double 1000 1 /tmp/r
/opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true build/c0_scalar_clang_rv64gc r2c double 1000 1 /tmp/s
python3 tests/compare_numpy.py r2c double 1000 1 /tmp/r /tmp/s
# D: real passes vs exact reference (reference generated once natively, checked under QEMU)
g++ -O2 -std=c++17 -I. work-real/test_real.cpp -o /tmp/test_real_x86 && /tmp/test_real_x86 genref /tmp/ref.bin
/opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true work-real/build/integ2/integ_clang check /tmp/ref.bin
```

## 8. Not yet done / open items

* NumPy's own `numpy/fft/tests/test_pocketfft.py` and PyTorch's `test/test_spectral_ops.py` on a RISC-V Python
  stack (needs a system-mode QEMU image or a board); the kernels they exercise are the ones covered above.
* An explicit `-cpu rv64,v=false` run of the RVV binary to demonstrate the `enabled()` fallback (the binaries are
  built with `-march=rv64gcv`, so the compiler may emit V instructions outside the dispatched kernels; the test
  belongs to the `target("arch=+v")` build mode planned in `design.md` section 8).
* Sanitizer builds (ASan/UBSan) under QEMU; valgrind has no RVV support.
* Hardware run of `fft_check` on a board (the correctness gate of `docs/hardware-run-guide.md` section 4, item 6).
