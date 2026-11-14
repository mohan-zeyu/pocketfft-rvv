#!/bin/bash
# run_correctness.sh [--quick]  - runs tests/fft_check for every built variant:
#   * natively on x86-64 (and aarch64 under qemu-aarch64 if built)
#   * riscv64 scalar builds under QEMU (also writing a binary dump of their outputs)
#   * riscv64 RVV / autovec builds under QEMU at VLEN 128, 256 and 512, each ALSO compared
#     element-wise against the same compiler's scalar dump (--compare-scalar)
# Results: results/correctness_<arch>_<variant>[-vlen<V>].csv (+ .log), summary on stdout,
# exit status non-zero if any case failed.
#
# env: VLENS="128 256 512"  MAXN=<n> (skip lengths above n)  EXTRA_ARGS="..." (passed to fft_check)
#      JOBS (parallel QEMU runs, default nproc)  VARIANTS="riscv64-gcc15-rvv ..." (riscv64 variants to run)
#      --quick  == MAXN=4096 (a full run with the 2^18..2^20 lengths takes ~3 min per QEMU configuration)
set -u
. "$(dirname "$0")/common.sh"

[ "${1:-}" = "--quick" ] && MAXN=${MAXN:-4096}
VLENS=${VLENS:-"128 256 512"}
EXTRA_ARGS=${EXTRA_ARGS:-}
[ -n "${MAXN:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --max-n $MAXN"
RV_VARIANTS=${VARIANTS:-"riscv64-gcc15-scalar riscv64-gcc15-rvv riscv64-gcc15-autovec riscv64-clang23-scalar riscv64-clang23-rvv riscv64-clang23-autovec"}
mkdir -p "$RESULTS" "$WORK"

# run_check <variant> <vlen|-> <label> [extra fft_check args]
run_check() {
  local v=$1 vlen=$2 label=$3; shift 3
  local bin=$BUILD/$v/fft_check arch; arch=$(variant_arch "$v")
  local csv=$RESULTS/correctness_${arch}_${label#"$arch"-}.csv
  local log=${csv%.csv}.log
  [ -x "$bin" ] || { log "skip $label: $bin not built"; return 0; }
  rm -f "$csv"
  local t0; t0=$(date +%s)
  # shellcheck disable=SC2086
  case $arch in
    x86-64)  "$bin" --label "$label" --csv "$csv" $EXTRA_ARGS "$@" > "$log" 2>&1;;
    *)       qemu_run "$v" "$vlen" "$bin" --label "$label" --csv "$csv" $EXTRA_ARGS "$@" > "$log" 2>&1;;
  esac
  local rc=$?
  printf "%-36s rc=%d %4ds  %s\n" "$label" "$rc" "$(( $(date +%s) - t0 ))" "$(grep -E '^SUMMARY' "$log" | sed 's/SUMMARY label=[^:]*: //')" >> "$RESULTS/.corr_status"
  grep -E 'worst ' "$log" | sed "s/^/    /" >> "$RESULTS/.corr_status"
  return $rc
}
export -f run_check qemu_run variant_arch rv_cpu log
export BUILD RESULTS EXTRA_ARGS QEMU_RV QEMU_A64 QEMU_X86 RV_CPU_BASE

rm -f "$RESULTS/.corr_status"
start=$(date +%s)
log "phase 1: native x86-64, aarch64 (qemu) and riscv64 scalar builds (the latter write dumps)"
{
  [ -n "${SKIP_X86:-}" ] || echo "x86-64-gcc13 - x86-64-gcc13"
  [ -n "${SKIP_AARCH64:-}" ] || echo "aarch64-gcc13 - aarch64-gcc13"
  for v in $RV_VARIANTS; do
    variant_uses_vlen "$v" && continue
    echo "$v 256 $v --dump $WORK/dump_$v.bin"
  done
} | xargs -P "$JOBS" -L 1 bash -c 'run_check "$@"' _

log "phase 2: riscv64 RVV/autovec builds at VLEN $VLENS (+ comparison with the scalar dump)"
{
  for v in $RV_VARIANTS; do
    variant_uses_vlen "$v" || continue
    dump=$WORK/dump_${v%-*}-scalar.bin
    for vl in $VLENS; do
      if [ -f "$dump" ]; then echo "$v $vl $v-vlen$vl --compare-scalar $dump"; else echo "$v $vl $v-vlen$vl"; fi
    done
  done
} | xargs -P "$JOBS" -L 1 bash -c 'run_check "$@"' _

echo
echo "=== correctness summary ($(( $(date +%s) - start )) s) ==="
cat "$RESULTS/.corr_status" 2>/dev/null
fails=$(grep -cE '^[^ ].* rc=[1-9]' "$RESULTS/.corr_status" 2>/dev/null || true)
rm -f "$RESULTS/.corr_status"
if [ "${fails:-0}" != 0 ]; then echo "RESULT: $fails configuration(s) FAILED"; exit 1; fi
echo "RESULT: all configurations passed"
