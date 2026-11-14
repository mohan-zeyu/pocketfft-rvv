#!/bin/bash
# run_bench_qemu.sh [--quick]  - dynamic instruction counts per transform under QEMU for the riscv64
# builds (scalar vs RVV, VLEN 128/256/512) plus the vector-instruction share (librvvclass plugin).
#
# Method: fft_bench is run twice per configuration with --reps N and --reps 2N (same seed, same
# warm-up, same plan creation); insns_per_transform = (insns(2N) - insns(N)) / N.  The vector share
# is computed on the same difference, so it refers to the transform only, not to process start-up.
#
# Output: results/bench_qemu.csv with
#   variant,compiler,vlen,type,kind,n,batch,insns_per_transform,vector_share,reps,vec_insns_per_transform,insns_N,insns_2N
#
# env: VARIANTS="riscv64-gcc15-scalar riscv64-gcc15-rvv riscv64-clang23-scalar riscv64-clang23-rvv"
#      VLENS="128 256 512"  TYPES="double float"  KINDS="c2c-fwd r2c c2r"  SIZES="64 256 1024 4096 16384 65536"
#      BATCHES="1 8"  MODE=api|plan  OUT=results/bench_qemu.csv  JOBS=<parallel qemu runs>
#      --quick: SIZES="256 4096" TYPES=double KINDS=c2c-fwd BATCHES=1
set -u
. "$(dirname "$0")/common.sh"

if [ "${1:-}" = "--one" ]; then # internal: one configuration -> one CSV line
  shift
  v=$1 vlen=$2 type=$3 kind=$4 n=$5 batch=$6 mode=$7 out=$8
  bin=$BUILD/$v/fft_bench
  reps=$(bench_reps "$n" "$batch")
  r1=$(count_insns "$v" "$vlen" "$bin" --type "$type" --kind "$kind" --n "$n" --batch "$batch" --reps "$reps" --trials 1 --mode "$mode" --quiet) || exit 1
  r2=$(count_insns "$v" "$vlen" "$bin" --type "$type" --kind "$kind" --n "$n" --batch "$batch" --reps $((2*reps)) --trials 1 --mode "$mode" --quiet) || exit 1
  set -- $r1; t1=$1; v1=${2:-0}; set -- $r2; t2=$1; v2=${2:-0}
  awk -v var="$(variant_kind "$v")" -v comp="$(variant_compiler "$v")" -v vlen="$vlen" -v type="$type" -v kind="$kind" \
      -v n="$n" -v b="$batch" -v reps="$reps" -v t1="$t1" -v t2="$t2" -v v1="$v1" -v v2="$v2" 'BEGIN {
        d = t2 - t1; dv = v2 - v1;
        printf "%s,%s,%s,%s,%s,%s,%s,%.1f,%.4f,%d,%.1f,%d,%d\n", var, comp, vlen, type, kind, n, b, d / reps, (d > 0 ? dv / d : 0), reps, dv / reps, t1, t2 }' >> "$out"
  exit 0
fi

if [ "${1:-}" = "--quick" ]; then SIZES=${SIZES:-"256 4096"}; TYPES=${TYPES:-double}; KINDS=${KINDS:-c2c-fwd}; BATCHES=${BATCHES:-1}; fi
VARIANTS=${VARIANTS:-"riscv64-gcc15-scalar riscv64-gcc15-rvv riscv64-clang23-scalar riscv64-clang23-rvv"}
VLENS=${VLENS:-"128 256 512"}
TYPES=${TYPES:-"double float"}
KINDS=${KINDS:-"c2c-fwd r2c c2r"}
SIZES=${SIZES:-"64 256 1024 4096 16384 65536"}
BATCHES=${BATCHES:-"1 8"}
MODE=${MODE:-api}
OUT=${OUT:-$RESULTS/bench_qemu.csv}
mkdir -p "$RESULTS"
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/benchq.XXXXXX")

jobs_file=$TMPD/jobs
i=0
for v in $VARIANTS; do
  [ -x "$BUILD/$v/fft_bench" ] || { log "skip $v (not built)"; continue; }
  for vlen in $VLENS; do
    for type in $TYPES; do for kind in $KINDS; do for n in $SIZES; do for b in $BATCHES; do
      i=$((i+1)); printf "%s %s %s %s %s %s %s %s/%06d.csv\n" "$v" "$vlen" "$type" "$kind" "$n" "$b" "$MODE" "$TMPD" "$i"
    done; done; done; done
  done
done > "$jobs_file"
total=$(wc -l < "$jobs_file")
log "run_bench_qemu: $total configurations x 2 runs, jobs=$JOBS, mode=$MODE -> $OUT"
start=$(date +%s)
xargs -P "$JOBS" -L 1 "$0" --one < "$jobs_file"
{
  echo "variant,compiler,vlen,type,kind,n,batch,insns_per_transform,vector_share,reps,vec_insns_per_transform,insns_N,insns_2N"
  cat "$TMPD"/*.csv 2>/dev/null
} > "$OUT"
rm -rf "$TMPD"
log "done in $(( $(date +%s) - start )) s: $(( $(wc -l < "$OUT") - 1 )) rows in $OUT"
# quick human summary: scalar vs rvv per (compiler,type,kind,batch) at each vlen
python3 - "$OUT" <<'EOF'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
sc = {(r['compiler'], r['vlen'], r['type'], r['kind'], r['n'], r['batch']): float(r['insns_per_transform']) for r in rows if r['variant'] == 'scalar'}
print("%-8s %-5s %-6s %-8s %-8s %-3s %14s %14s %8s %8s" % ("comp", "vlen", "type", "kind", "n", "B", "scalar", "rvv", "speedup", "vshare"))
for r in rows:
    if r['variant'] != 'rvv': continue
    k = (r['compiler'], r['vlen'], r['type'], r['kind'], r['n'], r['batch'])
    s = sc.get(k); v = float(r['insns_per_transform'])
    print("%-8s %-5s %-6s %-8s %-8s %-3s %14s %14.0f %8s %7.1f%%" % (r['compiler'], r['vlen'], r['type'], r['kind'], r['n'], r['batch'], "%.0f" % s if s else "-", v, "%.2fx" % (s / v) if s and v else "-", 100 * float(r['vector_share'])))
EOF
