#!/bin/bash
# xcount.sh <name> <source.c> [extra cflags...]
# Builds one C source for x86-64 (native), aarch64 and riscv64 (gcc -O3), runs each under QEMU user-mode with the libinsn
# plugin and prints dynamic instruction counts + native x86 wall time. For cross-ISA comparison tables.
. /home/user/harness/env.sh
NAME=$1; SRC=$2; shift 2; EXTRA="$@"
OUT=/home/user/harness/out; mkdir -p $OUT
set -e
gcc -O3 -march=x86-64-v3 $EXTRA -o $OUT/$NAME.x86 $SRC -lm
$A64_GCC -O3 -march=armv8.2-a $EXTRA -o $OUT/$NAME.a64 $SRC -lm
$RV_GCC15 -O3 -march=$RV_MARCH_V $EXTRA -o $OUT/$NAME.rv64v $SRC -lm
$RV_GCC15 -O3 -march=$RV_MARCH_BASE $EXTRA -o $OUT/$NAME.rv64s $SRC -lm
set +e
count() { local q=$1 sys=$2 bin=$3 cpu=$4; local t=$(mktemp); local s=$(date +%s.%N); $q -L $sys ${cpu:+-cpu $cpu} -plugin $QEMU_PLUGINS/libinsn.so -d plugin -D $t $bin >/dev/null; local e=$(date +%s.%N); local n=$(grep -oE 'total insns: *[0-9]+' $t | grep -oE '[0-9]+'); rm -f $t; echo "$n $(echo "$e - $s" | bc)"; }
printf "%-10s %-28s %14s %10s\n" "ISA" "config" "dyn-insns" "qemu-wall(s)"
r=$(count $QEMU_X86 / $OUT/$NAME.x86 max); printf "%-10s %-28s %14s %10s\n" x86-64 "v3 (AVX2), qemu-x86_64" $r
s=$(date +%s.%N); $OUT/$NAME.x86 >/dev/null; e=$(date +%s.%N); printf "%-10s %-28s %14s %10s\n" x86-64 "native wall (no PMU)" "-" "$(echo "$e - $s" | bc)"
r=$(count $QEMU_AARCH64 $A64_SYSROOT $OUT/$NAME.a64 max); printf "%-10s %-28s %14s %10s\n" aarch64 "armv8.2 NEON, qemu-aarch64" $r
r=$(count $QEMU_RV $RV_SYSROOT15 $OUT/$NAME.rv64s "rv64,zba=true,zbb=true,zbs=true"); printf "%-10s %-28s %14s %10s\n" riscv64 "scalar rv64gc_zba_zbb_zbs" $r
for v in 128 256 512; do r=$(count $QEMU_RV $RV_SYSROOT15 $OUT/$NAME.rv64v "$RV_QEMU_CPU_FULL,vlen=$v"); printf "%-10s %-28s %14s %10s\n" riscv64 "rvv vlen=$v" $r; done
