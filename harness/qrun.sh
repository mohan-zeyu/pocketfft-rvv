#!/bin/bash
# qrun.sh [-v VLEN] [-c cpu-extra] [-p insn|howvec|none] [-o out.csv] [-t tag] -- <riscv64 binary> [args...]
# Runs a riscv64 binary under QEMU user-mode; optionally collects TCG plugin instruction statistics.
. /home/user/harness/env.sh
VLEN=256; CPUX=""; PLUGIN=insn; OUT=""; TAG=""
while [ $# -gt 0 ]; do case "$1" in
  -v) VLEN=$2; shift 2;; -c) CPUX=",$2"; shift 2;; -p) PLUGIN=$2; shift 2;; -o) OUT=$2; shift 2;; -t) TAG=$2; shift 2;; --) shift; break;; *) break;; esac; done
BIN=$1; shift
CPU="${RV_QEMU_CPU_FULL},vlen=${VLEN}${CPUX}"
SYSROOT=${RV_SYSROOT}
case "$(file -b "$BIN" 2>/dev/null)" in *"interpreter /lib/ld-linux-riscv64"*) : ;; esac
PLUGARGS=()
TMP=$(mktemp)
PLUGSO=""
for d in "$QEMU_PLUGINS" "$QEMU_CONTRIB_PLUGINS"; do [ -n "$d" ] && [ -f "$d/lib${PLUGIN%%,*}.so" ] && PLUGSO="$d/lib${PLUGIN%%,*}.so" && break; done
if [ "$PLUGIN" != "none" ] && [ -n "$PLUGSO" ]; then
  PARGS=""; case "$PLUGIN" in *,*) PARGS=",${PLUGIN#*,}";; esac
  PLUGARGS=(-plugin "$PLUGSO$PARGS" -d plugin -D "$TMP")
fi
START=$(date +%s.%N)
"$QEMU_RV" -L "$SYSROOT" -cpu "$CPU" "${PLUGARGS[@]}" "$BIN" "$@"
RC=$?
END=$(date +%s.%N)
WALL=$(echo "$END - $START" | bc)
if [ -s "$TMP" ]; then
  if [ "$PLUGIN" = insn ]; then
    INSNS=$(grep -oE 'total insns: *[0-9]+' "$TMP" | grep -oE '[0-9]+' | tail -1)
    [ -z "$INSNS" ] && INSNS=$(grep -oE 'insns: *[0-9]+' "$TMP" | grep -oE '[0-9]+' | paste -sd+ | bc)
    echo "[qrun] tag=$TAG vlen=$VLEN insns=$INSNS wall=${WALL}s rc=$RC" >&2
    [ -n "$OUT" ] && echo "$TAG,$VLEN,$INSNS,$WALL,$RC" >> "$OUT"
  else
    echo "[qrun] tag=$TAG vlen=$VLEN wall=${WALL}s rc=$RC plugin output:" >&2; cat "$TMP" >&2
    [ -n "$OUT" ] && { echo "### $TAG vlen=$VLEN" >> "$OUT"; cat "$TMP" >> "$OUT"; }
  fi
else
  echo "[qrun] tag=$TAG vlen=$VLEN wall=${WALL}s rc=$RC (no plugin data)" >&2
fi
rm -f "$TMP"
exit $RC
