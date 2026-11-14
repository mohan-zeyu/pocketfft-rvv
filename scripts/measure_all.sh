#!/bin/bash
# Final instruction-count matrix -> results/final_insns.csv (variant,compiler,cache,kind,type,n,vlen,insns)
cd /home/user/fft-rvv; OUT=results/final_insns.csv; echo "variant,compiler,cache,kind,type,n,vlen,insns_per_transform" > $OUT
SIZES="1024 4096 65536 1048576"
for cache in 0 16; do for cc in clang gcc; do
  pre=c${cache}; sc=build/${pre}_scalar_${cc}_rv64gc; rv=build/${pre}_rvv_${cc}
  for kind in c2c r2c c2r; do for typ in double float; do for n in $SIZES; do
    fwd=1; [ $kind = c2r ] && fwd=0
    [ -x $sc ] && echo "scalar-rv64gc,$cc,$cache,$kind,$typ,$n,-,$(scripts/insncount.sh $sc 256 $kind $typ $n $fwd)" >> $OUT
    for v in 128 256 512; do echo "rvv,$cc,$cache,$kind,$typ,$n,$v,$(scripts/insncount.sh $rv $v $kind $typ $n $fwd)" >> $OUT; done
  done; done; done
done; done
# kernel-only (plan cached, exec in place) c2c double
for cc in clang gcc; do for n in $SIZES; do echo "kernel-scalar-rv64gc,$cc,16,x2c,double,$n,-,$(scripts/insncount.sh build/c16_scalar_${cc}_rv64gc 256 x2c double $n)" >> $OUT; for v in 128 256 512; do echo "kernel-rvv,$cc,16,x2c,double,$n,$v,$(scripts/insncount.sh build/c16_rvv_${cc} $v x2c double $n)" >> $OUT; done; done; done
echo DONE >> results/final_measure.log
