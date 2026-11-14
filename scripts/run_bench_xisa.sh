#!/bin/bash
# run_bench_xisa.sh [--quick]  - cross-ISA comparison on the same size sweep:
#   x86-64   : native ns/transform (std::chrono, min over trials)  +  qemu-x86_64 (-cpu max) instruction counts
#   aarch64  : qemu-aarch64 instruction counts
#   riscv64  : qemu-riscv64 instruction counts for the scalar and the RVV build (VLEN, default 256) + vector share
# Instruction counts use the same (2N - N)/N differencing as run_bench_qemu.sh.
#
# Output: results/bench_xisa.csv with
#   isa,variant,compiler,vlen,type,kind,n,batch,insns_per_transform,ns_per_transform,vector_share
# (ns_per_transform is only filled for the native x86-64 rows; vector_share only for riscv64 rows)
#
# env: TYPES="double float" KINDS="c2c-fwd r2c c2r" SIZES="64 256 1024 4096 16384 65536" BATCHES="1 8"
#      VLEN=256 RV_VARIANTS="riscv64-gcc15-scalar riscv64-gcc15-rvv" MODE=api OUT=results/bench_xisa.csv
#      CPU_PIN=<core for native timing, default 1> JOBS
set -u
. "$(dirname "$0")/common.sh"

if [ "${1:-}" = "--one" ]; then
  shift
  v=$1 vlen=$2 type=$3 kind=$4 n=$5 batch=$6 mode=$7 out=$8
  bin=$BUILD/$v/fft_bench; arch=$(variant_arch "$v")
  reps=$(bench_reps "$n" "$batch")
  r1=$(count_insns "$v" "$vlen" "$bin" --type "$type" --kind "$kind" --n "$n" --batch "$batch" --reps "$reps" --trials 1 --mode "$mode" --quiet) || exit 1
  r2=$(count_insns "$v" "$vlen" "$bin" --type "$type" --kind "$kind" --n "$n" --batch "$batch" --reps $((2*reps)) --trials 1 --mode "$mode" --quiet) || exit 1
  set -- $r1; t1=$1; v1=${2:-0}; set -- $r2; t2=$1; v2=${2:-0}
  ns=""
  if [ "$arch" = x86-64 ]; then
    pin=""; command -v taskset >/dev/null && pin="taskset -c ${CPU_PIN:-1}"
    ns=$($pin "$bin" --type "$type" --kind "$kind" --n "$n" --batch "$batch" --mode "$mode" --trials 7 | grep -oE 'ns_per_transform=[0-9.]+' | cut -d= -f2)
  fi
  vs=""; [ "$arch" = riscv64 ] && vs=$(awk -v d=$((t2-t1)) -v dv=$((v2-v1)) 'BEGIN{printf "%.4f", (d>0? dv/d : 0)}')
  echo "$arch,$(variant_kind "$v"),$(variant_compiler "$v"),$([ "$arch" = riscv64 ] && echo "$vlen" || echo -),$type,$kind,$n,$batch,$(awk -v d=$((t2-t1)) -v r="$reps" 'BEGIN{printf "%.1f", d/r}'),$ns,$vs" >> "$out"
  exit 0
fi

if [ "${1:-}" = "--quick" ]; then SIZES=${SIZES:-"256 4096"}; TYPES=${TYPES:-double}; KINDS=${KINDS:-c2c-fwd}; BATCHES=${BATCHES:-1}; fi
TYPES=${TYPES:-"double float"}
KINDS=${KINDS:-"c2c-fwd r2c c2r"}
SIZES=${SIZES:-"64 256 1024 4096 16384 65536"}
BATCHES=${BATCHES:-"1 8"}
VLEN=${VLEN:-256}
RV_VARIANTS=${RV_VARIANTS-"riscv64-gcc15-scalar riscv64-gcc15-rvv"}   # RV_VARIANTS="" skips riscv64 entirely
MODE=${MODE:-api}
OUT=${OUT:-$RESULTS/bench_xisa.csv}
mkdir -p "$RESULTS"
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/benchx.XXXXXX")
i=0
for v in x86-64-gcc13 aarch64-gcc13 $RV_VARIANTS; do
  [ -x "$BUILD/$v/fft_bench" ] || { log "skip $v (not built)"; continue; }
  for type in $TYPES; do for kind in $KINDS; do for n in $SIZES; do for b in $BATCHES; do
    i=$((i+1)); printf "%s %s %s %s %s %s %s %s/%06d.csv\n" "$v" "$VLEN" "$type" "$kind" "$n" "$b" "$MODE" "$TMPD" "$i"
  done; done; done; done
done > "$TMPD/jobs"
log "run_bench_xisa: $(wc -l < "$TMPD/jobs") configurations, jobs=$JOBS -> $OUT"
start=$(date +%s)
# native timings are sensitive to load: run the x86 rows alone first, then everything else in parallel
grep '^x86-64' "$TMPD/jobs" | xargs -P 1 -L 1 "$0" --one
grep -v '^x86-64' "$TMPD/jobs" | xargs -P "$JOBS" -L 1 "$0" --one
{ echo "isa,variant,compiler,vlen,type,kind,n,batch,insns_per_transform,ns_per_transform,vector_share"; cat "$TMPD"/*.csv 2>/dev/null; } > "$OUT"
rm -rf "$TMPD"
log "done in $(( $(date +%s) - start )) s: $(( $(wc -l < "$OUT") - 1 )) rows in $OUT"
python3 - "$OUT" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keys = sorted({(r['type'], r['kind'], int(r['n']), int(r['batch'])) for r in rows})
print("%-6s %-8s %-8s %-3s | %12s %10s | %12s | %12s %12s %8s" % ("type", "kind", "n", "B", "x86 insns", "x86 ns", "a64 insns", "rv64 scalar", "rv64 rvv", "rvv gain"))
for t, k, n, b in keys:
    sel = {(r['isa'], r['variant']): r for r in rows if r['type'] == t and r['kind'] == k and int(r['n']) == n and int(r['batch']) == b}
    g = lambda isa, var, f='insns_per_transform': sel.get((isa, var), {}).get(f, '')
    s, v = g('riscv64', 'scalar'), g('riscv64', 'rvv')
    gain = "%.2fx" % (float(s) / float(v)) if s and v else "-"
    print("%-6s %-8s %-8d %-3d | %12s %10s | %12s | %12s %12s %8s" % (t, k, n, b, g('x86-64', 'native'), g('x86-64', 'native', 'ns_per_transform'), g('aarch64', 'native'), s, v, gain))
EOF
