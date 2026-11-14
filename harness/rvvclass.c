/* rvvclass: QEMU TCG plugin that classifies executed RISC-V instructions
 * (vector load/store/arith/vsetvl vs scalar classes) by decoding the raw encoding.
 * Usage: -plugin /path/librvvclass.so -d plugin  */
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <qemu-plugin.h>
QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
enum { C_VSET, C_VLOAD, C_VSTORE, C_VINT, C_VFP, C_VMASK_MUL, C_SLOAD, C_SSTORE, C_SFP, C_SALU, C_BRANCH, C_ATOMIC, C_SYSTEM, C_COMPRESSED, C_OTHER, NCLASS };
static const char *names[NCLASS] = {"vsetvl","vload","vstore","vint(OPIV*)","vfp(OPFV*)","vmask/mul(OPMV*)","scalar-load","scalar-store","scalar-fp","scalar-alu","branch/jump","atomic","system/fence","compressed16","other"};
typedef struct { uint64_t c[NCLASS]; } Counts;
static struct qemu_plugin_scoreboard *score;
static int classify(uint32_t d){
  if ((d & 3) != 3) return C_COMPRESSED;
  uint32_t op = d & 0x7f, f3 = (d >> 12) & 7;
  switch (op) {
    case 0x57: if (f3 == 7) return C_VSET; if (f3 == 1 || f3 == 5) return C_VFP; if (f3 == 2 || f3 == 6) return C_VMASK_MUL; return C_VINT;
    case 0x07: return (f3 == 0 || f3 >= 5) ? C_VLOAD : C_SFP;   /* LOAD-FP: widths 0,5,6,7 are vector */
    case 0x27: return (f3 == 0 || f3 >= 5) ? C_VSTORE : C_SFP;  /* STORE-FP */
    case 0x03: return C_SLOAD; case 0x23: return C_SSTORE;
    case 0x13: case 0x1b: case 0x33: case 0x3b: case 0x37: case 0x17: return C_SALU;
    case 0x53: case 0x43: case 0x47: case 0x4b: case 0x4f: return C_SFP;
    case 0x63: case 0x6f: case 0x67: return C_BRANCH;
    case 0x2f: return C_ATOMIC; case 0x73: case 0x0f: return C_SYSTEM;
    default: return C_OTHER;
  }
}
static void vcpu_tb_trans(struct qemu_plugin_tb *tb, void *udata){
  size_t n = qemu_plugin_tb_n_insns(tb);
  for (size_t i = 0; i < n; i++) {
    struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
    uint32_t d = 0; qemu_plugin_insn_data(insn, &d, sizeof d);
    int cls = classify(d);
    qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(insn, QEMU_PLUGIN_INLINE_ADD_U64, qemu_plugin_scoreboard_u64_in_struct(score, Counts, c[cls]), 1);
  }
}
static void plugin_exit(void *p){
  g_autoptr(GString) out = g_string_new("rvvclass summary\n");
  uint64_t tot = 0, vec = 0, sums[NCLASS];
  for (int k = 0; k < NCLASS; k++) { sums[k] = qemu_plugin_u64_sum(qemu_plugin_scoreboard_u64_in_struct(score, Counts, c[k])); tot += sums[k]; if (k <= C_VMASK_MUL) vec += sums[k]; }
  for (int k = 0; k < NCLASS; k++) g_string_append_printf(out, "  %-18s %12" PRIu64 "  %6.2f%%\n", names[k], sums[k], tot ? 100.0 * sums[k] / tot : 0.0);
  g_string_append_printf(out, "  total insns: %" PRIu64 "\n  vector insns: %" PRIu64 " (%.2f%%)\n", tot, vec, tot ? 100.0 * vec / tot : 0.0);
  qemu_plugin_outs(out->str);
  qemu_plugin_scoreboard_free(score);
}
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv){
  if (strcmp(info->target_name, "riscv64") && strcmp(info->target_name, "riscv32")) { fprintf(stderr, "rvvclass: riscv only (target %s)\n", info->target_name); return -1; }
  score = qemu_plugin_scoreboard_new(sizeof(Counts));
  qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans, NULL);
  qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
  return 0;
}
