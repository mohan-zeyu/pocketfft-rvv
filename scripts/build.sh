#!/bin/bash
# build.sh [variant ...]  - builds tests/fft_check and bench/fft_bench for every build variant
# (default: all of them) into build/<variant>/.  Compiles run in parallel (JOBS, default nproc).
#
# variants:
#   x86-64-gcc13            g++ -O3 -march=x86-64-v3                     (native; generic-vector path for batches)
#   aarch64-gcc13           aarch64-linux-gnu-g++ -O3 -march=armv8.2-a   (static, run under qemu-aarch64)
#   riscv64-gcc15-rvv       bootlin GCC 15.3 -O3 -march=rv64gcv_zba_zbb_zbs       (RVV intrinsics path)
#   riscv64-gcc15-scalar    bootlin GCC 15.3 -O3 -march=rv64gc_zba_zbb_zbs  -DPOCKETFFT_NO_RVV (pure scalar baseline)
#   riscv64-gcc15-autovec   bootlin GCC 15.3 -O3 -march=rv64gcv_zba_zbb_zbs -DPOCKETFFT_NO_RVV (compiler auto-vec only)
#   riscv64-clang23-{rvv,scalar,autovec}   same three with clang 23 (lld, static, Ubuntu riscv64 sysroot)
#
# env: EXTRA_CXXFLAGS (appended to every compile), JOBS, NO_FFTW=1 (build fft_check with the naive reference)
set -u
. "$(dirname "$0")/common.sh"

VARIANTS=${*:-$ALL_VARIANTS}
COMMON="-O3 -std=c++17 -I$ROOT ${EXTRA_CXXFLAGS:-}"

compiler_cmd() { # prints the compiler invocation prefix for a variant
  case $1 in
    x86-64-gcc13)           echo "$X86_GXX -march=x86-64-v3";;
    aarch64-gcc13)          echo "$A64_GXX -march=armv8.2-a -static";;
    riscv64-gcc15-rvv)      echo "$RV_GXX15 -march=$RV_MARCH_V -static";;
    riscv64-gcc15-scalar)   echo "$RV_GXX15 -march=$RV_MARCH_S -static -DPOCKETFFT_NO_RVV";;
    riscv64-gcc15-autovec)  echo "$RV_GXX15 -march=$RV_MARCH_V -static -DPOCKETFFT_NO_RVV";;
    riscv64-clang23-rvv)    echo "$CLANGXX $CLANG_RV_FLAGS -march=$RV_MARCH_V";;
    riscv64-clang23-scalar) echo "$CLANGXX $CLANG_RV_FLAGS -march=$RV_MARCH_S -DPOCKETFFT_NO_RVV";;
    riscv64-clang23-autovec) echo "$CLANGXX $CLANG_RV_FLAGS -march=$RV_MARCH_V -DPOCKETFFT_NO_RVV";;
    *) return 1;;
  esac
}
fftw_dir() { case $(variant_arch "$1") in x86-64) echo "$FFTW_X86";; riscv64) echo "$FFTW_RV";; aarch64) echo "$FFTW_A64";; esac; }

build_one() { # variant program
  local v=$1 prog=$2 cc src out log fftw="" rc
  cc=$(compiler_cmd "$v") || { echo "unknown variant $v" >&2; return 1; }
  mkdir -p "$BUILD/$v"
  out=$BUILD/$v/$prog; log=$BUILD/$v/$prog.log
  if [ "$prog" = fft_check ]; then
    src=$ROOT/tests/fft_check.cpp
    local fd; fd=$(fftw_dir "$v")
    if [ -z "${NO_FFTW:-}" ] && [ -f "$fd/lib/libfftw3.a" ]; then fftw="-DHAVE_FFTW -I$fd/include -L$fd/lib -lfftw3"
    else echo "  note: $v/fft_check built WITHOUT FFTW (naive long-double reference only)" >&2; fi
  elif [ "$prog" = fftw_bench ]; then
    # FFTW timing counterpart (x86 only): needs the SIMD double+float FFTW build in work/fftw-x86-simd
    src=$ROOT/bench/fftw_bench.cpp
    local sd=$ROOT/work/fftw-x86-simd
    if [ -f "$sd/lib/libfftw3.a" ] && [ -f "$sd/lib/libfftw3f.a" ]; then fftw="-I$sd/include -L$sd/lib -lfftw3 -lfftw3f"
    else echo "  skip $v/fftw_bench: $sd/lib/libfftw3{,f}.a not found (see tests/README.md)" >&2; return 0; fi
  else
    src=$ROOT/bench/fft_bench.cpp
  fi
  rm -f "$out"
  # shellcheck disable=SC2086
  { echo "+ $cc $COMMON $src $fftw -lm -lpthread -o $out"; $cc $COMMON "$src" $fftw -lm -lpthread -o "$out"; } >"$log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then echo "  built $v/$prog"; else echo "  FAILED $v/$prog (see $log):"; grep -m5 -E "error|Error" "$log" | sed 's/^/     /'; fi
  return $rc
}

export -f build_one compiler_cmd fftw_dir variant_arch
export ROOT BUILD COMMON FFTW_X86 FFTW_RV FFTW_A64 X86_GXX A64_GXX RV_GXX15 CLANGXX CLANG_RV_FLAGS RV_MARCH_V RV_MARCH_S NO_FFTW

echo "building: $VARIANTS  (jobs=$JOBS, flags: $COMMON)"
start=$(date +%s)
fail=0
for v in $VARIANTS; do
  progs="fft_check fft_bench"; [ "$(variant_arch "$v")" = x86-64 ] && progs="$progs fftw_bench"
  for p in $progs; do echo "$v $p"; done
done |
  xargs -P "$JOBS" -L 1 bash -c 'build_one "$0" "$1" || echo "BUILD-FAILED $0/$1" >> "$BUILD/.failed"'
if [ -f "$BUILD/.failed" ]; then cat "$BUILD/.failed"; rm -f "$BUILD/.failed"; fail=1; fi
echo "done in $(( $(date +%s) - start )) s"
# record the header version that was built (for the report)
{ echo "built: $(date -u +%FT%TZ)"; echo "header sha256: $(sha256sum "$ROOT/pocketfft_hdronly.h" | cut -c1-16)  rvv header: $(sha256sum "$ROOT/pocketfft_rvv.h" 2>/dev/null | cut -c1-16)"; } > "$BUILD/BUILD_INFO.txt"
exit $fail
