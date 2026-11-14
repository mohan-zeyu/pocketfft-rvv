# Reproducing the pocketfft-RVV benchmarks on real RISC-V hardware

Status: groundwork, 2026-11-08. Companion to `docs/integration.md`, `docs/related-work.md` and the
benchmark harness `/home/user/fft-rvv/bench/fft_bench` (written by the harness agent; CLI:
`fft_bench <float|double> <c2c-fwd|c2c-bwd|r2c|c2r> <length> --reps N --batch B`).
Legend: **[V]** verified from a primary source or our own experiment, **[L]** likely (secondary source),
**[U]** unverified. QEMU numbers are only trend indicators; every number in the final report must come
from this procedure on silicon.

## 1. Candidate machines

| Class | SoC / core | RVV | VLEN | Boards (examples) | Notes |
|---|---|---|---|---|---|
| Mass-market SBC | **SpacemiT K1 / M1**, 8x X60 @ 1.6 GHz, in-order dual-issue | **1.0** (+Zvfh) | **256** | Banana Pi BPI-F3, Milk-V Jupiter, Orange Pi RV2, LicheePi 3A, MUSE Pi | Best-documented RVV microarchitecture (camel-cdr's instruction timings, see `related-work.md` §7). Example `/proc/cpuinfo` ISA line on a BPI-F3 vendor kernel: `rv64imafdcv_sscofpmf_sstc_svpbmt_zicbom_zicboz_zicbop_zihintpause`, model name `Spacemit(R) X60` ([openbenchmarking cpuinfo](https://openbenchmarking.org/s/Spacemit+X60)) **[V]**. Sscofpmf present, so `perf stat` hardware counters are possible (§4). |
| New SBC / mini-ITX (RVA23) | **SpacemiT K3**, 8x X100 @ up to 2.4-2.5 GHz + 8x A100 "AI cores" (VLEN 1024) | 1.0 | **256** (X100) | SpacemiT K3 Pico-ITX + K3-CoM260 (launched 2026-05-11), Sipeed K3 SBCs, DeepComputing DC-ROMA Mainboard III (Framework 13, pre-orders June 2026), Milk-V Jupiter2 (RVA23 mini-ITX, reported K3-based) | X100 is the core used for the pocketfft PR #31 and PFFFT RVV numbers. Sources: [linuxgizmos K3](https://linuxgizmos.com/spacemit-k3-integrates-8-core-risc-v-cpu-cluster-and-60-tops-ai-engine/), [cnx-software K3 sysinfo](https://www.cnx-software.com/2026/01/23/spacemit-k3-16-core-risc-v-soc-system-information-and-early-benchmarks/), [dev.to K3 X100 vs A100 vlen](https://dev.to/gounthar/benchmarking-llamacpp-on-spacemit-k3-risc-v-ai-cores-vs-standard-rvv-part-4-10mc), [Phoronix: GCC 17 `-mcpu=spacemit-x100`](https://www.phoronix.com/news/GCC-17-SpacemiT-X100), [Hackster Jupiter2](https://www.hackster.io/news/milk-v-offers-risc-v-rva23-compatibility-with-the-new-multi-core-mini-itx-jupter2-sbc-0169d0a8e5c4) **[V]** (Jupiter2 SoC identity **[L]**). **Pin the benchmark to an X100 core** (the A100 cores have VLEN 1024 and a different pipeline); identify cores via `/proc/cpuinfo` (`uarch`/`model name`) or `lscpu -e` before choosing the `taskset` mask. |
| Data-center | **Sophgo SG2044**, 64x XuanTie C920v2 (12-stage OoO) in 16 clusters of 4, ~2.6 GHz (unofficial), 64 KB L1I/D, 2 MB L2 per cluster, 64 MB L3, 32x DDR5-4266 channels, PCIe Gen5 | **1.0** | **128** | "SG2044 EVB" (RISC-V Devboards Program update, March 2026); no retail board confirmed as of 2026-11-08 | Mainline Linux support since 6.16; openEuler + GCC 15.2 used in the EPCC evaluation ([arXiv 2508.13840, N. Brown, 2025-08-19](https://arxiv.org/html/2508.13840v1)) **[V]**. The 64-core/single-NUMA part is the "data-center" story for the report; VLEN 128 means LMUL=2 is attractive there. |
| **Do not use for RVV 1.0** | Sophgo **SG2042** (C920 v1, Milk-V Pioneer), T-Head **TH1520** (C910; LicheePi 4A, BeagleV-Ahead, Milk-V Meles) | **0.7.1 (xtheadvector)** | 128 | Milk-V Pioneer, LicheePi 4A | Different encodings (`vsetvli`, loads/stores, ...): RVV 1.0 binaries trap with SIGILL. Only `-march=..._xtheadvector` builds (GCC 14+) run there, and the kernel reports it through `RISCV_HWPROBE_KEY_VENDOR_EXT_THEAD_0` / `RISCV_HWPROBE_VENDOR_EXT_XTHEADVECTOR` ([kernel hwprobe docs](https://docs.kernel.org/arch/riscv/hwprobe.html)) **[V]**. [Milk-V Pioneer overview](https://milkv.io/docs/pioneer/overview) confirms SG2042 "RVV 0.71". |
| No vector unit | SiFive HiFive Premier P550 / Milk-V Megrez (P550), StarFive VisionFive 2 (JH7110, U74), UltraRISC UR-DP1000 (Milk-V Titan) | none | - | - | Useful only to prove the scalar fallback / runtime dispatch path (`pocketfft::detail::rvv::enabled()` must return false, no SIGILL). UR-DP1000 vector status **[U]**, P550/U74 **[V]**. |
| Wide VLEN | SiFive X280 (VLEN 512) | 1.0 | 512 | partner dev systems only (juFFTe paper) | Not purchasable; cover VLEN 512/1024 with QEMU only. |

Recommended minimum for the competition: one **K1 board** (cheap, best-known microarchitecture, VLEN 256) **and**
one **SG2044** or **K3** system (VLEN 128 vs 256 contrast; SG2044 adds the 64-core data-center angle).

## 2. Kernel, firmware and user-space prerequisites

* **Kernel >= 6.5** for user-space V support (vector context save/restore), **>= 6.4** for the `riscv_hwprobe`
  syscall; newer is better (6.16 adds SG2044 support). Vendor BSP kernels (SpacemiT Bianbu 6.6, Sophgo openEuler)
  already include both. Check: `uname -r`.
* V must be enabled for user space: `/proc/sys/abi/riscv_v_default_allow` (default 1) and the per-process
  `prctl(PR_RISCV_V_SET_CONTROL)` (Linux `Documentation/arch/riscv/vector.rst`) **[V]**.
* **ISA string**: `grep -m1 isa /proc/cpuinfo` must contain the letter `v` in the single-letter part (e.g.
  `rv64imafdcv...`), and upstream kernels also list `zvl128b`/`zvl256b`, `zve64d`, `zvfh` etc.
  Note `/proc/cpuinfo` on vendor kernels may omit sub-extensions.
* **hwprobe** (authoritative, per-CPU): key `RISCV_HWPROBE_KEY_IMA_EXT_0` (=4), bit `RISCV_HWPROBE_IMA_V`
  ("the V extension, per version 1.0 of the RISC-V Vector manual"), also `RISCV_HWPROBE_EXT_ZVE32X/ZVE64D`,
  `RISCV_HWPROBE_EXT_ZVFH`, `RISCV_HWPROBE_EXT_ZVBB`, and `RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF`
  (SLOW/FAST/UNSUPPORTED - record it; it decides whether unaligned `vle` is acceptable). glibc >= 2.40 exports
  `__riscv_hwprobe()` (`<sys/hwprobe.h>`); otherwise call `syscall(__NR_riscv_hwprobe, ...)` with
  `<asm/hwprobe.h>`. hwprobe has **no VLEN key**: read VLEN in-process with `csrr vlenb` (VLEN = vlenb*8) or
  `__riscv_vsetvlmax_e8m1()*8` ([kernel hwprobe docs](https://docs.kernel.org/arch/riscv/hwprobe.html)) **[V]**.
* `getauxval(AT_HWCAP) & (1UL << ('V'-'A'))` (bit 21, `COMPAT_HWCAP_ISA_V`) is the cheapest check and what
  `pocketfft_rvv.h::enabled()` uses.

Minimal probe program (also prints VLEN); build with `gcc -O2 -march=rv64gcv probe.c`:

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/auxv.h>
#include <asm/hwprobe.h>
int main(void){
  struct riscv_hwprobe p[3] = {{RISCV_HWPROBE_KEY_IMA_EXT_0,0},
                               {RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF,0},
                               {RISCV_HWPROBE_KEY_VENDOR_EXT_THEAD_0,0}};
  long rc = syscall(__NR_riscv_hwprobe, p, 3, 0, NULL, 0);
  unsigned long vlenb = 0;
  if (getauxval(AT_HWCAP) & (1UL<<21)) __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
  printf("hwprobe rc=%ld IMA_EXT_0=%#llx V=%d ZVFH=%d misaligned_vec_perf=%llu thead_ext=%#llx VLEN=%lu\n",
         rc, (unsigned long long)p[0].value, !!(p[0].value & RISCV_HWPROBE_IMA_V),
         !!(p[0].value & RISCV_HWPROBE_EXT_ZVFH), (unsigned long long)p[1].value,
         (unsigned long long)p[2].value, vlenb*8);
  return 0;
}
```

## 3. Toolchain and build flags

| Need | Minimum | Recommended | Evidence |
|---|---|---|---|
| Frozen RVV intrinsics API (`__riscv_v_intrinsic >= 12000`, required by `pocketfft_rvv.h`) | **GCC 14** (defines 12000; GCC 13 defines 11000), **clang 17** (12000) | GCC 15.x, clang 19/20 | `gcc/config/riscv/riscv-c.cc` on branches gcc-13/14/15 (`riscv_ext_version_value (0, 11)` / `(0, 12)` / `(0, 12)`; master: `(1, 0)` = 1000000) and `clang/lib/Basic/Targets/RISCV.cpp` (17.x-19.x: `getVersionValue(0, 12)`; 20.x+: `(1, 0)`), fetched 2026-11-08 **[V]** |
| `__attribute__((target("arch=+v")))` on functions that use intrinsics (needed for runtime dispatch in rv64gc builds) | **GCC 14** (the `#error "Vector intrinsics require the vector extension."` guard was removed from `riscv_vector.h` and the test `gcc.target/riscv/rvv/base/target_attribute_v_with_intrinsic-1.c` exists on the gcc-14 branch), **clang 19** (PR [llvm/llvm-project#83674](https://github.com/llvm/llvm-project/pull/83674), merged 2024-03-27) | GCC 15 / clang 20 | our test on this machine: GCC 13.3 cross and clang 18 both fail with the `#error` above; see `docs/integration.md` §6 **[V]** |
| `-mrvv-vector-bits=zvl` / `riscv_rvv_vector_bits` (only for the upstream generic-vector path) | GCC 14, clang 17 | - | [GCC 14 release notes](https://gcc.gnu.org/gcc-14/changes.html) **[V]** |
| Profile names in `-march` (`rva23u64`) | not in GCC 13 / clang 18 (both reject the string, tested) | GCC 14+ / clang 19+ **[L]**; otherwise spell the ISA string out | local test 2026-11-08 |

Flags used for the measurements (record them verbatim in the report):

* RISC-V, VLEN-agnostic binary (runs on K1/K3/SG2044): `-O3 -march=rv64gcv -fno-plt` (V implies `zvl128b`;
  all our loops use `vsetvl`, so one binary serves every VLEN). Add `-mcpu=spacemit-x60` /
  `-mcpu=spacemit-x100` (GCC 17 for X100) or `-mtune=...` only as a documented extra experiment.
* RISC-V, VLEN-specialised reference build (for the "how much does `zvl256b` buy" question):
  `-O3 -march=rv64gcv_zvl256b` - **this binary must not be run on VLEN 128 hardware**.
* Scalar baseline: same flags plus `-DPOCKETFFT_NO_RVV` (keeps the compiler's own auto-vectorisation; if you want
  a strictly scalar baseline add `-fno-tree-vectorize` and say so).
* Upstream generic-vector path (pocketfft PR #31 behaviour): `-O3 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl`
  without `POCKETFFT_RVV` (note: VLEN-specific, crashes at other VLENs - see `/home/user/research/survey/fft.md` §4).
* Always `-ffp-contract=fast` or `off` consistently across baseline and RVV builds (FMA contraction changes
  rounding and the error analysis).
* x86 / ARM counterparts: `-O3 -march=native` (pocketfft then picks AVX-512/AVX/SSE2 or NEON `VLEN`).

Distribution toolchains: Ubuntu 25.10/26.04 riscv64 ship GCC 15 with an RVA23 default ISA (V on by default);
Debian trixie/forky riscv64 ship GCC 14/15 with an `rv64gc` default (V off by default) - see `integration.md` §6.

## 4. Making the measurement stable

1. **CPU frequency / governor** (root): 
   `for c in /sys/devices/system/cpu/cpu*/cpufreq; do echo performance > $c/scaling_governor; cat $c/scaling_max_freq > $c/scaling_min_freq; done`
   (or `cpupower frequency-set -g performance`). Record `scaling_cur_freq` / `cpuinfo_cur_freq` before and after
   each sweep; K1 boards run 1.6 GHz, K3 up to 2.4-2.5 GHz, SG2044 ~2.6 GHz. If no cpufreq driver is exposed, say so.
2. **Thermals**: use the vendor heatsink/fan; log `/sys/class/thermal/thermal_zone*/temp` during the run; discard
   runs whose frequency dropped.
3. **Isolation**: `taskset -c <core>` on a core other than 0 (IRQs), no desktop session, `systemctl stop` of
   background services if possible; for multi-threaded runs pin with `OMP_PLACES`/`taskset` masks and report the mask.
   On K3 choose X100 cores explicitly; on SG2044 stay inside one 4-core cluster for single-thread and record the L3 sharing.
4. **Repetition**: `--reps` so that every measurement lasts >= 0.5 s; >= 5 independent process launches; report the
   **median** (and min) and the spread; warm up once (first call also builds the plan - `POCKETFFT_CACHE_SIZE` is 0 in
   NumPy/PyTorch builds, see `integration.md` §2.4).
5. **Counters**: `perf stat -e cycles,instructions,task-clock` works when the kernel's SBI PMU driver finds counters
   (`dmesg | grep -i pmu`, `perf list hw`). The K1 ISA string advertises Sscofpmf **[V]**; whether a given firmware
   exposes them must be checked on the board (`<not supported>` means OpenSBI/firmware lacks PMU descriptors) **[U]**.
   The sysctl `kernel.perf_user_access` controls direct `rdcycle`/`rdinstret`: default **1** = counters readable only
   through perf (direct `rdcycle` traps with SIGILL), **2** = deprecated legacy mode with direct `cycle`/`instret`
   access, **0** = disabled; `rdtime` is always readable ([kernel sysctl docs](https://docs.kernel.org/admin-guide/sysctl/kernel.html)) **[V]**.
   Therefore: time with `clock_gettime(CLOCK_MONOTONIC)` (= `rdtime`), and get cycles/instructions from `perf stat`
   (or `perf_event_open` + mmap'ed `rdpmc`-style reads); do not hard-code `rdcycle` in the harness.
6. **Correctness gate before timing**: run the harness self-check (max relative L2 error vs. the scalar build /
   double-precision reference) on the same binary; a timing without a passing check is not reportable.

## 5. What to record for the competition report

* Machine: board + SoC, firmware/OpenSBI version (`dmesg | head`), `uname -a`, distro, `/proc/cpuinfo` ISA line,
  hwprobe output and VLEN (program above), cache sizes (`lscpu`, `getconf -a | grep CACHE`), memory, governor and
  measured frequency, temperature range, `taskset` mask.
* Software: compiler `--version`, exact flags, `pocketfft_hdronly.h`/`pocketfft_rvv.h` git hash of `/home/user/fft-rvv`,
  harness version, `POCKETFFT_*` macros, libc version (`ldd --version`).
* Raw data: one CSV row per launch (type, kind, N, batch, reps, median ns/transform, min, GFLOP/s using
  `5 N log2 N` (c2c) or `2.5 N log2 N` (r2c/c2r), cycles, instructions, IPC), plus the `perf stat -x,` files.
* Derived: speedup RVV vs scalar per (type, kind, N, batch); cycles per point; flop/cycle vs the core's peak
  (X60: 1 FMA vreg/cycle at VLEN 256 -> 8 FP32 FMA/cycle = 16 flop/cycle, see `related-work.md` §7).
* Cross-ISA: the same CSV from an x86 (AVX2/AVX-512) and an ARM (NEON/SVE) machine built with `-O3 -march=native`,
  presented as absolute time **and** as cycles-per-point / flop-per-cycle so the comparison is per-core efficiency,
  not clock speed. State clearly that upstream pocketfft's x86/ARM 1-D path is scalar as well (only batched
  transforms are vectorised there), so the fair cross-ISA comparison is "pocketfft as shipped" on each ISA.
* QEMU trend data (dynamic instruction counts per VLEN) belong in a separate table, never mixed with hardware time.

### x86 / ARM counterpart procedure (same script, §6)

1. Build the identical harness natively: `g++ -O3 -march=native -std=c++17 ...` (pocketfft selects `VLEN` from
   `__AVX512F__`/`__AVX__`/`__SSE2__`/`__ARM_NEON`; record which).
2. Pin: `taskset -c <core>`; governor `performance`; disable turbo/boost
   (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo` or `echo 0 > /sys/devices/system/cpu/cpufreq/boost`);
   on ARM servers check `cpupower frequency-info`.
3. `perf stat -e cycles,instructions` is available on all of them; record `lscpu` and `/proc/cpuinfo` flags.
4. Optional external references on the same machine: FFTW 3 (`fftw-wisdom`/`bench`) and, on x86, MKL - only as
   context; the required comparison is the same library across ISAs.

## 6. Ready-to-paste run skeleton

```bash
#!/usr/bin/env bash
# run_fft_bench.sh - sweep fft_bench on one pinned core, capture environment, timings and perf counters.
# Usage: sudo CPU=2 RUNS=5 REPS=200 ./run_fft_bench.sh   (root only needed for governor/perf; otherwise omit sudo)
set -euo pipefail
BENCH=${BENCH:-/home/user/fft-rvv/bench/fft_bench}               # RVV-enabled build
BENCH_BASE=${BENCH_BASE:-/home/user/fft-rvv/bench/fft_bench_scalar} # same flags + -DPOCKETFFT_NO_RVV
CPU=${CPU:-2}; RUNS=${RUNS:-5}; REPS=${REPS:-200}
TYPES=${TYPES:-"float double"}
KINDS=${KINDS:-"c2c-fwd c2c-bwd r2c c2r"}
# powers of two, 2^a 3^b 5^c 7 11 mixes, and a prime (Bluestein)
LENGTHS=${LENGTHS:-"64 256 1024 4096 16384 65536 262144 1048576 1000 1200 1536 1331 4725 100003"}
BATCHES=${BATCHES:-"1 8 64"}
OUT=${OUT:-/home/user/fft-rvv/results/$(hostname)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

# --- 1. environment snapshot -------------------------------------------------
{
  echo "date: $(date -Is)"; uname -a; cat /etc/os-release 2>/dev/null | head -3
  grep -m1 -E '^isa' /proc/cpuinfo || true; grep -m1 -E '^(uarch|model name)' /proc/cpuinfo || true
  echo "nproc: $(nproc)"; lscpu 2>/dev/null | grep -iE 'model|cache|mhz' || true
  for f in scaling_governor scaling_cur_freq cpuinfo_cur_freq; do
    printf '%s: ' "$f"; cat /sys/devices/system/cpu/cpu$CPU/cpufreq/$f 2>/dev/null || echo n/a; done
  echo "temp(mC): $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo n/a)"
  echo "perf_user_access: $(cat /proc/sys/kernel/perf_user_access 2>/dev/null || echo n/a)"
  echo "riscv_v_default_allow: $(cat /proc/sys/abi/riscv_v_default_allow 2>/dev/null || echo n/a)"
  ${CXX:-g++} --version | head -1; ldd --version | head -1
  echo "fft-rvv commit: $(git -C /home/user/fft-rvv rev-parse HEAD 2>/dev/null || echo n/a)"
  [ -x ./probe ] && ./probe || true          # hwprobe/VLEN program from section 2, if built
} > "$OUT/env.txt" 2>&1

# --- 2. governor (best effort, needs root) -----------------------------------
if [ -w /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor ]; then
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo performance > "$c/scaling_governor" || true
    cat "$c/scaling_max_freq" > "$c/scaling_min_freq" 2>/dev/null || true
  done
fi

# --- 3. correctness gate (harness self-check, adapt flag name to the harness) -
taskset -c "$CPU" "$BENCH" double c2c-fwd 1000 --reps 1 --batch 1 > "$OUT/selfcheck.txt" 2>&1 || {
  echo "self-check failed, aborting"; exit 1; }

# --- 4. sweep ----------------------------------------------------------------
echo "bin,type,kind,n,batch,run,raw" > "$OUT/timings.csv"
for bin in "$BENCH_BASE" "$BENCH"; do
  tag=$(basename "$bin")
  for type in $TYPES; do for kind in $KINDS; do for n in $LENGTHS; do for b in $BATCHES; do
    for run in $(seq "$RUNS"); do
      raw=$(taskset -c "$CPU" "$bin" "$type" "$kind" "$n" --reps "$REPS" --batch "$b" 2>>"$OUT/stderr.log" | tr '\n' ' ')
      echo "$tag,$type,$kind,$n,$b,$run,\"$raw\"" >> "$OUT/timings.csv"
    done
    # hardware counters: 3 perf-internal repeats, CSV output
    taskset -c "$CPU" perf stat -x, -r 3 -e cycles,instructions,task-clock \
      -o "$OUT/perf-$tag-$type-$kind-n$n-b$b.csv" -- \
      "$bin" "$type" "$kind" "$n" --reps "$REPS" --batch "$b" > /dev/null 2>>"$OUT/stderr.log" || true
    printf 'temp %s %s %s n=%s b=%s: %s\n' "$tag" "$type" "$kind" "$n" "$b" \
      "$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo n/a)" >> "$OUT/thermal.log"
  done; done; done; done
done
echo "done -> $OUT"
```

Post-processing (to be written next to the harness once its output format is fixed): parse `raw` into
ns/transform, compute medians per (bin,type,kind,n,batch), the RVV/scalar ratio, GFLOP/s and cycles-per-point from
the `perf-*.csv` files, and emit the Markdown tables for the design document.

## 7. QEMU cross-check (for the "optimisation trend", not for timing)

`qemu-riscv64 -cpu rv64,v=true,vlen=128|256|512|1024,vext_spec=v1.0 ./fft_bench ...` with the harness built by
the same compiler; count dynamic instructions with a TCG plugin (`-plugin libinsn.so`) and compare the vector
instruction share across VLENs (QEMU 8.2.2 is installed here; `-d plugin` requires a QEMU build with plugins).
Static histograms: `llvm-objdump --mattr=+v -d fft_bench | grep -cE '^\s+[0-9a-f]+:.*\sv[a-z]'` (GNU objdump
2.42 prints RVV as `.insn`). This sweep also proves that one binary runs at every VLEN (the upstream
`-mrvv-vector-bits=zvl` build does not).
