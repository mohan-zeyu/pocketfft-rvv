QEMU TCG plugin and runner scripts used for the instruction-count measurements.
- rvvclass.c: plugin classifying executed RISC-V instructions (vsetvl, vector load/store, vector int/fp, scalar classes).
  Build: gcc -O2 -shared -fPIC -I<qemu>/include $(pkg-config --cflags glib-2.0) rvvclass.c -o librvvclass.so
  (needs qemu-plugin.h from QEMU >= 11, built with --enable-plugins; libinsn.so comes from the QEMU build tree).
- qrun.sh / xcount.sh / env.sh: run a riscv64 binary under qemu-riscv64 at a chosen VLEN with a plugin, and the
  cross-ISA (x86-64 / aarch64 / riscv64) instruction-count helper.
