#!/bin/bash
# run_bench_x86_fftw.sh  - native x86-64 timing of pocketfft (build/x86-64-gcc13/fft_bench, public API with
# plan cache) against FFTW 3.3.11 (build/x86-64-gcc13/fftw_bench, AVX2 build, FFTW_ESTIMATE and FFTW_MEASURE
# plans) on identical inputs.  Minimum over TRIALS of the mean over the automatically chosen number of reps,
# pinned to one core.
#
# Output: results/bench_x86_fftw.csv with
#   type,kind,n,batch,pocketfft_ns,fftw_estimate_ns,fftw_measure_ns,pocketfft_mflops,fftw_measure_mflops,pocketfft_over_fftw_measure
# (last column = pocketfft_ns / fftw_measure_ns: 1.0 = same speed, 2.0 = FFTW twice as fast)
#
# env: SIZES="1024 4096 65536 1048576" TYPES="double float" KINDS="c2c-fwd r2c" BATCHES="1" TRIALS=7 CPU_PIN=1
#      OUT=results/bench_x86_fftw.csv
set -u
. "$(dirname "$0")/common.sh"
SIZES=${SIZES:-"1024 4096 65536 1048576"}
TYPES=${TYPES:-"double float"}
KINDS=${KINDS:-"c2c-fwd r2c"}
BATCHES=${BATCHES:-"1"}
TRIALS=${TRIALS:-7}
OUT=${OUT:-$RESULTS/bench_x86_fftw.csv}
PF=$BUILD/x86-64-gcc13/fft_bench
FB=$BUILD/x86-64-gcc13/fftw_bench
[ -x "$PF" ] && [ -x "$FB" ] || { echo "need $PF and $FB (scripts/build.sh x86-64-gcc13 builds both)" >&2; exit 1; }
pin=""; command -v taskset >/dev/null && pin="taskset -c ${CPU_PIN:-1}"
mkdir -p "$RESULTS"
field() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }
echo "type,kind,n,batch,pocketfft_ns,fftw_estimate_ns,fftw_measure_ns,pocketfft_mflops,fftw_measure_mflops,pocketfft_over_fftw_measure" > "$OUT"
log "pocketfft vs FFTW native x86 timing -> $OUT (load average now: $(cut -d' ' -f1-3 /proc/loadavg))"
for type in $TYPES; do for kind in $KINDS; do for n in $SIZES; do for b in $BATCHES; do
  p=$($pin "$PF" --type "$type" --kind "$kind" --n "$n" --batch "$b" --trials "$TRIALS")
  e=$($pin "$FB" --type "$type" --kind "$kind" --n "$n" --batch "$b" --trials "$TRIALS" --planner estimate)
  m=$($pin "$FB" --type "$type" --kind "$kind" --n "$n" --batch "$b" --trials "$TRIALS" --planner measure)
  pns=$(echo "$p" | field ns_per_transform); ens=$(echo "$e" | field ns_per_transform); mns=$(echo "$m" | field ns_per_transform)
  pmf=$(echo "$p" | field mflops); mmf=$(echo "$m" | field mflops)
  ratio=$(awk -v a="$pns" -v b="$mns" 'BEGIN{ if (b>0) printf "%.3f", a/b; else print "" }')
  echo "$type,$kind,$n,$b,$pns,$ens,$mns,$pmf,$mmf,$ratio" >> "$OUT"
  printf "%-6s %-8s %-8s B=%-3s pocketfft %10.0f ns   fftw(estimate) %10.0f ns   fftw(measure) %10.0f ns   pocketfft/fftw = %s\n" "$type" "$kind" "$n" "$b" "$pns" "$ens" "$mns" "$ratio"
done; done; done; done
log "done"
