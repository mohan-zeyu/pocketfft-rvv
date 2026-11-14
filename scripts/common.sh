#!/bin/bash
# common.sh - shared settings/helpers for the fft-rvv scripts (sourced, not executed).
# Paths below match the container this harness was developed in; override via environment if needed.

ROOT=${FFT_RVV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BUILD=$ROOT/build
RESULTS=$ROOT/results
WORK=$ROOT/work
JOBS=${JOBS:-$(nproc)}

# --- toolchains
RV_GXX15=${RV_GXX15:-/opt/riscv-gcc-bootlin/bin/riscv64-linux-g++}
CLANGXX=${CLANGXX:-clang++-23}
CLANG_RV_FLAGS=${CLANG_RV_FLAGS:-"--target=riscv64-linux-gnu --gcc-toolchain=/usr -isystem /usr/riscv64-linux-gnu/include -isystem /usr/riscv64-linux-gnu/include/c++/13 -isystem /usr/riscv64-linux-gnu/include/c++/13/riscv64-linux-gnu -fuse-ld=lld -static"}
A64_GXX=${A64_GXX:-aarch64-linux-gnu-g++}
X86_GXX=${X86_GXX:-g++}
RV_MARCH_V=${RV_MARCH_V:-rv64gcv_zba_zbb_zbs}
RV_MARCH_S=${RV_MARCH_S:-rv64gc_zba_zbb_zbs}

# --- FFTW reference installs (see tests/README.md for how they were built)
FFTW_X86=$WORK/fftw-x86
FFTW_RV=$WORK/fftw-riscv64
FFTW_A64=$WORK/fftw-aarch64

# --- QEMU + plugins
QEMU_RV=${QEMU_RV:-/opt/qemu/bin/qemu-riscv64}
QEMU_A64=${QEMU_A64:-/opt/qemu/bin/qemu-aarch64}
QEMU_X86=${QEMU_X86:-/opt/qemu/bin/qemu-x86_64}
PLUGIN_INSN=${PLUGIN_INSN:-/home/user/research/work/qemu/build/tests/tcg/plugins/libinsn.so}
PLUGIN_RVVCLASS=${PLUGIN_RVVCLASS:-/home/user/harness/plugins/librvvclass.so}
RV_CPU_BASE=${RV_CPU_BASE:-"rv64,v=true,elen=64,zba=true,zbb=true,zbs=true"}
rv_cpu() { echo "$RV_CPU_BASE,vlen=$1"; }

# --- build variants (directory names under build/)
ALL_VARIANTS="x86-64-gcc13 aarch64-gcc13 riscv64-gcc15-rvv riscv64-gcc15-scalar riscv64-gcc15-autovec riscv64-clang23-rvv riscv64-clang23-scalar riscv64-clang23-autovec"
variant_arch()     { case $1 in riscv64-*) echo riscv64;; aarch64-*) echo aarch64;; x86-64-*) echo x86-64;; *) echo unknown;; esac; }
variant_compiler() { case $1 in *gcc15*) echo gcc15;; *clang23*) echo clang23;; *gcc13*) echo gcc13;; *) echo unknown;; esac; }
variant_kind()     { case $1 in *-rvv) echo rvv;; *-scalar) echo scalar;; *-autovec) echo autovec;; *) echo native;; esac; }
variant_uses_vlen() { case $1 in *-rvv|*-autovec) return 0;; *) return 1;; esac; }

# qemu_run <variant> <vlen> <binary> [args...]   (vlen ignored for non-riscv variants)
qemu_run() {
  local variant=$1 vlen=$2 bin=$3; shift 3
  case $(variant_arch "$variant") in
    riscv64) "$QEMU_RV" -cpu "$(rv_cpu "$vlen")" "$bin" "$@";;
    aarch64) "$QEMU_A64" "$bin" "$@";;
    x86-64)  "$QEMU_X86" -cpu max "$bin" "$@";;
  esac
}

# count_insns <variant> <vlen> <binary> [args...]  -> prints "TOTAL VECTOR" (VECTOR empty unless rvvclass)
# riscv64 uses librvvclass (gives the total AND the vector-instruction count); other ISAs libinsn.
count_insns() {
  local variant=$1 vlen=$2 bin=$3; shift 3
  local tmp; tmp=$(mktemp "${TMPDIR:-/tmp}/insn.XXXXXX")
  local rc=0
  case $(variant_arch "$variant") in
    riscv64)
      local plug=$PLUGIN_RVVCLASS; [ -f "$plug" ] || plug=$PLUGIN_INSN
      "$QEMU_RV" -cpu "$(rv_cpu "$vlen")" -plugin "$plug" -d plugin -D "$tmp" "$bin" "$@" >/dev/null 2>&1 || rc=$?;;
    aarch64) "$QEMU_A64" -plugin "$PLUGIN_INSN" -d plugin -D "$tmp" "$bin" "$@" >/dev/null 2>&1 || rc=$?;;
    x86-64)  "$QEMU_X86" -cpu max -plugin "$PLUGIN_INSN" -d plugin -D "$tmp" "$bin" "$@" >/dev/null 2>&1 || rc=$?;;
  esac
  local total vec
  total=$(grep -oE 'total insns: *[0-9]+' "$tmp" | grep -oE '[0-9]+' | tail -1)
  vec=$(grep -oE 'vector insns: *[0-9]+' "$tmp" | grep -oE '[0-9]+' | tail -1)
  rm -f "$tmp"
  [ "$rc" = 0 ] || { echo "count_insns: run failed rc=$rc ($variant vlen=$vlen $bin $*)" >&2; return 1; }
  [ -n "$total" ] || { echo "count_insns: no plugin output ($variant $bin $*)" >&2; return 1; }
  echo "$total ${vec:-}"
}

# reps for the instruction-count differencing: roughly 1e6 transform-elements per run, at least 2
bench_reps() { local n=$1 b=${2:-1}; local r=$(( 1000000 / (n * b) )); [ "$r" -lt 2 ] && r=2; [ "$r" -gt 4000 ] && r=4000; echo $r; }

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
