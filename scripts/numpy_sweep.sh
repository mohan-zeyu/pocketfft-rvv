#!/bin/bash
# NumPy-oracle sweep of the current RVV builds (clang + GCC) : c2c/r2c/c2r x float/double x 23 lengths x VLEN 128/256/512
cd /home/user/fft-rvv; LOG=results/numpy_sweep_23len.log; : > $LOG
echo "# $(date -u) header commit $(git rev-parse --short HEAD); binaries build/c0_rvv_clang build/c0_rvv_gcc" >> $LOG
pass=0; fail=0
for b in c0_rvv_clang c0_rvv_gcc; do for kind in c2c r2c c2r; do for typ in double float; do for n in 2 3 4 5 6 7 8 9 12 16 25 30 64 100 128 243 360 625 1000 1024 4096 10007 65536; do for v in 128 256 512; do
  fwd=1; [ $kind = c2r ] && fwd=0
  /opt/qemu/bin/qemu-riscv64 -cpu rv64,v=true,vlen=$v ./build/$b $kind $typ $n $fwd /tmp/ns_$$
  line=$(python3 tests/compare_numpy.py $kind $typ $n $fwd /tmp/ns_$$); rc=$?
  echo "$b vlen=$v $line" >> $LOG; [ $rc -eq 0 ] && pass=$((pass+1)) || fail=$((fail+1))
done; done; done; done; done
echo "# TOTAL pass=$pass fail=$fail" >> $LOG
