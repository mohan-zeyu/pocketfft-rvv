# pocketfft RVV - correctness oracle and measurement harness

Everything here is independent of the optimisation work itself: it only consumes
`../pocketfft_hdronly.h` (+ `../pocketfft_rvv.h`) and never modifies them.

```
tests/fft_check.cpp        correctness oracle (pocketfft vs FFTW3 double-precision reference, + scalar-vs-RVV dumps)
bench/fft_bench.cpp        timing / instruction-count driver (one transform configuration per invocation)
scripts/common.sh          shared paths, toolchains, QEMU helpers (sourced by the other scripts)
scripts/build.sh           builds both programs for every variant into build/<variant>/
scripts/run_correctness.sh fft_check natively (x86), under qemu-aarch64 and under qemu-riscv64 (VLEN 128/256/512)
scripts/run_bench_qemu.sh  per-transform dynamic instruction counts, scalar vs RVV, + vector share -> results/bench_qemu.csv
scripts/run_bench_xisa.sh  cross-ISA sweep (x86 native ns + qemu insns, aarch64, riscv64 scalar/RVV) -> results/bench_xisa.csv
scripts/make_report_tables.py  CSVs -> results/report.md (Markdown tables) + results/chart_*.png (matplotlib)
work/fftw-{x86,riscv64,aarch64}/   FFTW 3.3.11 static builds used as the reference
results/                   CSVs, logs, report.md, charts
```

## Prerequisites (as found on the development container)

* native: gcc 13 (`g++`), clang 23, python3 (+ `pip install matplotlib` for the charts)
* riscv64 cross: bootlin GCC 15.3 at `/opt/riscv-gcc-bootlin/bin/riscv64-linux-g++` (static binaries, no `-L`
  needed under QEMU) and `clang++-23 --target=riscv64-linux-gnu ... -fuse-ld=lld -static` (flags in `common.sh`)
* aarch64 cross: `aarch64-linux-gnu-g++` (gcc 13)
* QEMU 11.1 user mode with TCG plugins: `/opt/qemu/bin/qemu-{riscv64,aarch64,x86_64}`,
  `libinsn.so` (total dynamic instructions) and the custom `librvvclass.so` (RISC-V instruction classes:
  vsetvl / vector load / vector store / OPIVV / OPFVV / OPMVV / scalar classes, total and vector share)
* FFTW 3.3.11 reference libraries (already built; to rebuild, from an extracted `fftw-3.3.11` tarball):

```sh
# native x86-64
../fftw-3.3.11/configure --prefix=$ROOT/work/fftw-x86 --enable-static --disable-shared --disable-fortran --disable-doc CFLAGS=-O2
make -j4 && make install
# riscv64 (plain rv64gc_zba_zbb_zbs, no V: the reference must not depend on the code under test)
../fftw-3.3.11/configure --host=riscv64-linux-gnu CC=/opt/riscv-gcc-bootlin/bin/riscv64-linux-gcc \
    AR=/opt/riscv-gcc-bootlin/bin/riscv64-linux-ar RANLIB=/opt/riscv-gcc-bootlin/bin/riscv64-linux-ranlib \
    --prefix=$ROOT/work/fftw-riscv64 --enable-static --disable-shared --disable-fortran --disable-doc \
    CFLAGS="-O2 -march=rv64gc_zba_zbb_zbs -mabi=lp64d"
make -j4 && make install
# aarch64
../fftw-3.3.11/configure --host=aarch64-linux-gnu CC=aarch64-linux-gnu-gcc --prefix=$ROOT/work/fftw-aarch64 \
    --enable-static --disable-shared --disable-fortran --disable-doc CFLAGS="-O2 -march=armv8.2-a"
```

(Each cross build takes about 3 minutes; FFTW's double-precision library is used as the reference for both the
float and the double pocketfft transforms.)

## 1. Build

```sh
scripts/build.sh                      # all variants, in parallel
scripts/build.sh riscv64-gcc15-rvv    # one variant
EXTRA_CXXFLAGS="-DPOCKETFFT_NO_VECTORS" scripts/build.sh x86-64-gcc13
```

| variant | compiler / flags | what it measures |
|---|---|---|
| `x86-64-gcc13` | g++ 13 `-O3 -march=x86-64-v3` | native reference point (NumPy-like build; generic GCC-vector path for batched transforms) |
| `aarch64-gcc13` | aarch64 g++ 13 `-O3 -march=armv8.2-a -static` | NEON reference point (under qemu-aarch64) |
| `riscv64-gcc15-rvv` | GCC 15.3 `-O3 -march=rv64gcv_zba_zbb_zbs -static` | the hand-written RVV path (`__riscv_vector` defined) |
| `riscv64-gcc15-scalar` | `-march=rv64gc_zba_zbb_zbs -DPOCKETFFT_NO_RVV` | pure scalar baseline (no V in the ISA string at all) |
| `riscv64-gcc15-autovec` | `-march=rv64gcv_zba_zbb_zbs -DPOCKETFFT_NO_RVV` | what the compiler's auto-vectoriser alone achieves |
| `riscv64-clang23-{rvv,scalar,autovec}` | clang 23, same three configurations | second compiler |

All compiles use `-O3 -std=c++17`, one translation unit per program. Build logs: `build/<variant>/<program>.log`.
If the header is mid-edit and fails to compile, `build.sh` prints the first errors and exits non-zero; the other
variants are still built.

## 2. Correctness

```sh
scripts/run_correctness.sh --quick    # n <= 4096, ~1-2 min
scripts/run_correctness.sh            # full default list up to n = 2^20 (about 3 min per QEMU configuration,
                                      # configurations run in parallel: JOBS=nproc)
```

What a run does:

1. `x86-64-gcc13` natively and `aarch64-gcc13` under qemu-aarch64;
2. the riscv64 **scalar** builds under qemu-riscv64, each also writing a binary dump of all their outputs
   (`work/dump_<variant>.bin`);
3. the riscv64 **RVV** and **autovec** builds at VLEN 128, 256 and 512, each checked against FFTW *and*
   compared element-wise against the scalar dump of the same compiler (`--compare-scalar`).

Output per configuration: `results/correctness_<arch>_<variant>[-vlen<V>].csv` (one row per transform) and the
matching `.log`; a summary table on stdout; exit status 1 if any case failed. `VARIANTS="..."` selects the
riscv64 builds, `SKIP_X86=1` / `SKIP_AARCH64=1` drop the native / aarch64 runs, `JOBS` limits parallel QEMU runs.

`scripts/summarize_correctness.py results/correctness_*.csv` prints the per-class breakdown (precision x kind x
direction: worst max|err|/max|ref| and where, the same restricted to the Bluestein lengths, worst L2 error,
worst RVV-vs-scalar difference); the same table is part of `results/report.md`.

`fft_check` itself (run `build/<variant>/fft_check --help`):

* default case list (1080 cases): lengths 1..2^20 (powers of two), mixed radix 2^k 3^m 5^n (6, 12, 24, 30, ... 32400),
  radix-7/11 and generic-radix factors (49, 77, 121, 143, 169, 231, 352 = 11*32, 448 = 7*64, 1001 = 7*11*13, 1331,
  2401), primes (13, 17, 19, 23, 31, 47, 53, 97, 101, 257, 1009, 10007, 65537 - the larger ones use Bluestein),
  odd/non-aligned lengths (3, 5, 7, 9, 15, 33, 65, 129, 1023, 1025); for each length float and double,
  forward and backward, c2c, r2c and c2r; plus extras: batched 2-D transforms (shape {5,n} and {9,n} along axis 1,
  which is the path the vector code takes), strided input/output (3/2) and in-place c2c.
* inputs: 64-bit LCG seeded from (seed, type, kind, direction, n, batch, strides) in integer arithmetic ->
  bit-identical data on every machine; floats are the double samples rounded to float, and the reference is
  computed from exactly those values.
* scaling: `fct = 1` for forward, `1/n` for backward transforms (both scaled and unscaled paths are exercised).
* conventions: pocketfft's backward r2c is `conj(forward r2c)`, its "forward" c2r conjugates the input first;
  the reference does the same, so FFTW's r2c/c2r can be used directly.
* metrics per case: `max_abs_err`, `max_err_norm` = max|err| / max|ref| (**pass iff <= 1e-5**, the rubric metric),
  `l2_rel` = ||err||/||ref||, and with `--compare-scalar` the `vs_dump_norm` = max|out - scalar out| / max|ref|
  (pass thresholds 1e-12 double / 1e-4 float, i.e. several orders of magnitude above the expected ~1e-16 / ~1e-7).
* typical values: double ~1e-16 ... 1.5e-15 (worst: Bluestein n = 65537), float ~1e-7 ... 3.6e-7.
* useful options: `--max-n N`, `--lengths 1000,4096*3,65537`, `--type float`, `--kind r2c`, `--batch 8`
  (runs the given lengths as batched transforms), `--ref naive` (long-double O(n^2) DFT instead of FFTW, for
  cross-checking the reference itself), `--dump FILE` / `--compare-scalar FILE`, `--quiet`.

Manual QEMU invocation (static binaries need no `-L`):

```sh
/opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true \
    build/riscv64-gcc15-rvv/fft_check --max-n 65536 --label gcc15-rvv-256 --csv /tmp/c.csv
```

## 3. Benchmarks

`fft_bench --type double|float --kind c2c-fwd|c2c-bwd|r2c|c2r --n N [--batch B] [--reps R] [--trials T] [--mode api|plan]`

* natively it prints `ns_per_transform` (minimum over `--trials` of the mean over `--reps`), an FFTW-style
  `mflops` figure (5 n log2 n per complex transform, half for real ones), a checksum of the output (scalar and
  RVV builds must agree to rounding), the code path compiled in (`path=rvv|riscv-scalar|generic-vector`) and,
  on RISC-V with V, the hardware VLEN read from the `vlenb` CSR;
* `--mode api` (default) goes through `pocketfft::c2c/r2c/c2r` with `POCKETFFT_CACHE_SIZE=16` (the plan is
  created once and reused, as in SciPy); `--mode plan` calls `pocketfft::detail::pocketfft_c/r<T>::exec()`
  directly on a contiguous buffer;
* `--batch B` transforms a {B, n} array along axis 1 - this is where pocketfft's generic vector path (and a
  multi-row RVV path) applies; B = 1 is the plain 1-D transform.
* default scaling mirrors NumPy (`fct = 1` for c2c-fwd/r2c, `1/n` for c2c-bwd/c2r); `--fct` overrides.

### QEMU instruction counts (`run_bench_qemu.sh`)

QEMU cannot measure time, so we count dynamic instructions with the `libinsn`/`librvvclass` TCG plugins. Each
configuration is run twice, `--reps N` and `--reps 2N`; everything except the extra N transforms is identical
between the two runs (same input generation, plan creation and warm-up call), so

    insns_per_transform = (insns(2N) - insns(N)) / N

and the vector share is computed on the same difference. `N` is chosen so that each run executes about
10^6 transform elements (`bench_reps` in `common.sh`).

```sh
scripts/run_bench_qemu.sh --quick       # 2 sizes, double c2c-fwd, ~1 min
scripts/run_bench_qemu.sh               # default sweep: 4 variants x 3 VLEN x 2 types x 3 kinds x 6 sizes x 2 batches
VARIANTS="riscv64-gcc15-scalar riscv64-gcc15-rvv" SIZES="1024 65536 1048576" BATCHES=1 scripts/run_bench_qemu.sh
```

`results/bench_qemu.csv` columns: `variant,compiler,vlen,type,kind,n,batch,insns_per_transform,vector_share,
reps,vec_insns_per_transform,insns_N,insns_2N`. The scalar build is run at every VLEN as well (its count varies
very slightly with VLEN because glibc's ifunc-selected `memcpy` uses RVV when QEMU advertises V).

Caveats: an instruction-count ratio is not a cycle ratio. A vector instruction processes VLEN/SEW elements per
LMUL register group, so on a core with a wide datapath the real speedup is larger than the count ratio; on a
core that splits vector instructions into many uops (or has slow segment loads `vlseg2e64`) it is smaller.
Only a real board gives cycles (section 5).

### Native x86: pocketfft vs FFTW (`run_bench_x86_fftw.sh`)

`bench/fftw_bench.cpp` is the FFTW counterpart of `fft_bench` (same arguments, same LCG inputs, same output
line; `--planner estimate|measure|patient`). It needs an AVX2-enabled double **and** float FFTW in
`work/fftw-x86-simd` (built with `--enable-sse2 --enable-avx --enable-avx2`, plus `--enable-float --enable-sse`
for the float library; `build.sh x86-64-gcc13` then also produces `build/x86-64-gcc13/fftw_bench`).

```sh
scripts/run_bench_x86_fftw.sh      # sizes 1024 4096 65536 1048576, double+float, c2c-fwd+r2c -> results/bench_x86_fftw.csv
```

Columns: `type,kind,n,batch,pocketfft_ns,fftw_estimate_ns,fftw_measure_ns,pocketfft_mflops,fftw_measure_mflops,
pocketfft_over_fftw_measure` (last = pocketfft time / FFTW-measure time, > 1 means FFTW is faster). Both programs
are pinned to one core (`CPU_PIN`, default 1) and report the minimum over `TRIALS` (default 7).

### Cross-ISA (`run_bench_xisa.sh`)

Same sizes; rows for x86-64 (native ns/transform on the build host - no PMU in the container - plus
qemu-x86_64 `-cpu max` instruction counts of the AVX2 build), aarch64 (qemu-aarch64 counts of the NEON build)
and riscv64 scalar/RVV (VLEN 256 by default). `results/bench_xisa.csv` columns:
`isa,variant,compiler,vlen,type,kind,n,batch,insns_per_transform,ns_per_transform,vector_share`.

## 4. Report

```sh
scripts/make_report_tables.py            # -> results/report.md + results/chart_*.png
scripts/make_report_tables.py --no-charts
```

Tables: correctness summary per configuration (cases, pass/fail, worst float/double error, worst RVV-vs-scalar
difference); per (compiler, type, kind, batch) instruction counts for scalar / autovec / RVV at each VLEN with the
speedup and vector share, plus geometric-mean speedups; cross-ISA tables. Charts: speedup vs n per VLEN,
vector share vs n, instructions relative to x86-64, error vs n against the 1e-5 threshold.

## 5. Running on a real RISC-V board

The static riscv64 binaries run unchanged on any rv64gcv Linux board (glibc static, no dependencies):

```sh
scp build/riscv64-gcc15-{rvv,scalar}/fft_{check,bench} board:~/fft/
```

or rebuild natively on the board with the same flags (`-O3 -std=c++17 -march=rv64gcv_zba_zbb_zbs`; drop the
`zb*` extensions if the core lacks them). Then:

```sh
# 1. fixed frequency, isolated core
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq          # record it
# 2. correctness on the hardware (vector unit behaves like QEMU? - it should: identical results to ~1e-16)
taskset -c 2 ./fft_check --label board-rvv --csv board_correctness.csv
taskset -c 2 ./fft_check_scalar --dump board_scalar.bin --quiet && taskset -c 2 ./fft_check --compare-scalar board_scalar.bin
# 3. cycles / instructions per transform: perf stat on the N and 2N runs, difference / N
for b in fft_bench_scalar fft_bench_rvv; do for r in 200 400; do
  perf stat -e cycles,instructions,task-clock -r 5 taskset -c 2 ./$b --type double --kind c2c-fwd --n 4096 --reps $r --trials 1
done; done
# 4. wall-clock per transform directly from fft_bench (pinned, several trials)
taskset -c 2 ./fft_bench_rvv --type float --kind r2c --n 65536 --trials 11
```

Record for each measurement: board/SoC and core type, kernel version, `isa` line of `/proc/cpuinfo`, VLEN (the
`vlen=` field fft_bench prints, from the `vlenb` CSR), the governor and the actual frequency, the compiler version
and flags, the binary (scalar / RVV), type, kind, n, batch, `ns_per_transform`, and from `perf stat`:
cycles, instructions, IPC and task-clock for the N and 2N runs (report `(cycles(2N) - cycles(N)) / N`). Compare
the instruction counts with the QEMU numbers for the same configuration (they should match to within the
noise of the libc / kernel overhead) - that validates the QEMU methodology, and the cycle counts then give the
real speedup. If `perf` lacks hardware events on the board, `ns_per_transform` x frequency is the fallback.

## Notes / caveats

* Under QEMU, wall-clock time is meaningless (TCG emulates each vector instruction with a per-element helper
  loop: the RVV binary is *slower* in wall time than the scalar one); only instruction counts are used.
* The reference is FFTW in double precision for both float and double. For float the reported error is
  dominated by pocketfft's float rounding (~1e-7), for double both libraries contribute (~1e-16 ... 1e-15).
* `fft_check` ignores the imaginary parts of the DC (and Nyquist) bins in c2r inputs by zeroing them in the
  generated data, so that all libraries interpret the Hermitian input identically.
* All numbers in this directory were obtained with the header version noted in `build/BUILD_INFO.txt`.
