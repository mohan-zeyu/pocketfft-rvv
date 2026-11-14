#!/bin/bash
# Remainder of the final matrix, run 3 jobs in parallel: cache=16 API rows (sizes 1k/4k/64k; 1M for clang c2c only),
# kernel (x2c) rows, and the 3 missing cache=0 rows. Each job writes one CSV row file; rows are appended at the end.
cd /home/user/fft-rvv; P=results/final_parts
job(){ # variant compiler cache kind type n vlen bin
  local out="$P/$1_$2_$3_$4_$5_$6_$7.row"; [ -s "$out" ] && return
  local fwd=1; [ "$4" = c2r ] && fwd=0; local v=$7; [ "$v" = "-" ] && v=256
  echo "$1,$2,$3,$4,$5,$6,$7,$(scripts/insncount.sh $8 $v $4 $5 $6 $fwd)" > "$out"
}
export -f job
{
for v in 128 256 512; do echo "rvv gcc 0 c2r float 1048576 $v build/c0_rvv_gcc"; done
for cc in clang gcc; do for kind in c2c r2c c2r; do for typ in double float; do for n in 1024 4096 65536; do
  echo "scalar-rv64gc $cc 16 $kind $typ $n - build/c16_scalar_${cc}_rv64gc"
  for v in 128 256 512; do echo "rvv $cc 16 $kind $typ $n $v build/c16_rvv_$cc"; done
done; done; done; done
echo "scalar-rv64gc clang 16 c2c double 1048576 - build/c16_scalar_clang_rv64gc"; for v in 128 256 512; do echo "rvv clang 16 c2c double 1048576 $v build/c16_rvv_clang"; done
for cc in clang gcc; do for n in 1024 4096 65536 1048576; do
  echo "kernel-scalar-rv64gc $cc 16 x2c double $n - build/c16_scalar_${cc}_rv64gc"
  for v in 128 256 512; do echo "kernel-rvv $cc 16 x2c double $n $v build/c16_rvv_$cc"; done
done; done
} | xargs -P 3 -L 1 bash -c 'job "$@"' _
cat $P/*.row >> results/final_insns.csv
echo DONE >> results/final_measure.log
