# Source this file: . /home/user/harness/env.sh
# Toolchains
export RV_GCC=riscv64-linux-gnu-gcc
export RV_GXX=riscv64-linux-gnu-g++
export RV_GFORTRAN=riscv64-linux-gnu-gfortran
export RV_SYSROOT=/usr/riscv64-linux-gnu
# clang (prefer newest installed)
for v in 23 22 21 20 19 18; do if command -v clang-$v >/dev/null 2>&1; then export RV_CLANG=clang-$v; export RV_CLANGXX=clang++-$v; break; fi; done
export RV_CLANG_FLAGS="--target=riscv64-linux-gnu --gcc-toolchain=/usr -isystem $RV_SYSROOT/include -fuse-ld=lld"
# bootlin GCC 15.3 toolchain (newer RVV intrinsics incl. bf16)
if [ -x /opt/riscv-gcc-bootlin/bin/riscv64-linux-gcc ]; then export RV_GCC15=/opt/riscv-gcc-bootlin/bin/riscv64-linux-gcc; export RV_GXX15=/opt/riscv-gcc-bootlin/bin/riscv64-linux-g++; export RV_GFORTRAN15=/opt/riscv-gcc-bootlin/bin/riscv64-linux-gfortran; export RV_SYSROOT15=/opt/riscv-gcc-bootlin/riscv64-buildroot-linux-gnu/sysroot; fi
# aarch64 cross (for cross-ISA comparison)
export A64_GCC=aarch64-linux-gnu-gcc; export A64_GXX=aarch64-linux-gnu-g++; export A64_SYSROOT=/usr/aarch64-linux-gnu
export RVVCLASS_PLUGIN=/home/user/harness/plugins/librvvclass.so
# QEMU (prefer our build with plugins)
if [ -x /opt/qemu/bin/qemu-riscv64 ]; then export QEMU_RV=/opt/qemu/bin/qemu-riscv64; export QEMU_AARCH64=/opt/qemu/bin/qemu-aarch64; export QEMU_X86=/opt/qemu/bin/qemu-x86_64; export QEMU_PLUGINS=/home/user/research/work/qemu/build/tests/tcg/plugins; export QEMU_CONTRIB_PLUGINS=/home/user/research/work/qemu/build/contrib/plugins; else export QEMU_RV=qemu-riscv64; fi
# Default march strings
export RV_MARCH_BASE=rv64gc_zba_zbb_zbs
export RV_MARCH_V=rv64gcv_zba_zbb_zbs
export RV_MARCH_VFULL=rv64gcv_zba_zbb_zbs_zbc_zvbb_zvbc_zvfh
# QEMU cpu strings (VLEN substituted by qrun.sh)
export RV_QEMU_CPU_BASE='rv64,v=true,elen=64,zba=true,zbb=true,zbs=true'
export RV_QEMU_CPU_FULL='rv64,v=true,elen=64,zba=true,zbb=true,zbs=true,zbc=true,zvbb=true,zvbc=true,zvkb=true,zvfh=true,zvfhmin=true,zfh=true,zicond=true'
