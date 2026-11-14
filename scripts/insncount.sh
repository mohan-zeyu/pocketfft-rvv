#!/bin/bash
# insncount.sh <binary> <vlen> <kind> <type> <n> [fwd]  -> per-transform dynamic instruction count (reps 11 minus reps 1, /10)
Q=/opt/qemu/bin/qemu-riscv64; P=/home/user/research/work/qemu/build/tests/tcg/plugins/libinsn.so
bin=$1; v=$2; kind=$3; typ=$4; n=$5; fwd=${6:-1}
count(){ $Q -cpu rv64,v=true,vlen=$v,zba=true,zbb=true,zbs=true -plugin $P -d plugin -D /tmp/ic_$$.txt $bin $kind $typ $n $fwd /tmp/icd_$$ $1 >/dev/null; grep -oE 'total insns: *[0-9]+' /tmp/ic_$$.txt | grep -oE '[0-9]+'; }
a=$(count 1); b=$(count 11); rm -f /tmp/ic_$$.txt /tmp/icd_$$.*; echo $(( (b-a)/10 ))
