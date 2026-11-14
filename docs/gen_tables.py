#!/usr/bin/env python3
"""gen_tables.py - regenerate the Markdown tables of docs/performance-report.md from the raw measurement files.

    python3 docs/gen_tables.py            # rewrite the <!-- gen:NAME --> ... <!-- /gen --> blocks in place
    python3 docs/gen_tables.py --print    # print every table to stdout
    python3 docs/gen_tables.py --print --only final-summary,kernel
    python3 docs/gen_tables.py --check    # exit 1 if the report is out of date

Primary input (written by scripts/measure_all.sh, the final measurement matrix):

  results/final_insns.csv   variant,compiler,cache,kind,type,n,vlen,insns_per_transform
                            variant  = scalar-rv64gc (vlen "-") | rvv
                            compiler = clang | gcc
                            cache    = 0 (POCKETFFT_CACHE_SIZE=0, NumPy/PyTorch default) | 16 (plan reused)
                            kind     = c2c | r2c | c2r (public API)  | x2c (kernel only: cached plan, in-place exec)
                            type     = double | float;  n = transform length;  vlen = 128 | 256 | 512

Older per-step files (kept for the optimisation history; used as fall-back when the CSV lacks a table):

  results/kernel_insns.txt            variant n vlen128 vlen256 vlen512          kernel (x2c), double c2c fwd
  results/api_nocache_insns_v3.txt    n scalar-gc rvv128 rvv256 rvv512           pocketfft::c2c, no plan cache (clang 23)
  results/api_cache16_insns.txt       n scalar-gc rvv128 rvv256 rvv512           pocketfft::c2c, POCKETFFT_CACHE_SIZE=16 (clang 23)
  results/api_nocache_insns.txt       n=.. rv64gc-scalar .. rvv64gcv-scalar .. rvv ..   VLEN 256, before twiddle vectorisation
  results/api_nocache_insns_v2.txt    n=.. rv64gc-scalar .. rvv ..                       VLEN 256, after twiddle vectorisation
  results/c2c_insns.txt               type n scalar rvv128 rvv256 rvv512 x128 x256 x512  first RVV version, GCC 15, no cache
  results/c2c_insns_cache16.txt       variant n vlen128 vlen256 vlen512          early GCC-vs-clang / LMUL experiment (API, cache 16)
  results/c2c_insns_v2.txt            variant kind n vlen128 vlen256 vlen512     split-store experiment (API c2c and kernel x2c)
  results/xisa_insns.txt              ISA build n=1024 n=4096 n=65536            x86-64 / aarch64 counts (kernel measure)
  results/real_*.txt, work-real/build/bench*_final_*.txt
                                      build dir vlen insns_1 insns_11 per_transform   r2c / c2r before the final matrix
  docs/data/*.txt                     bin kind type n vlen128 vlen256 vlen512    independent re-measurement at a fixed commit

Every number is a dynamic instruction count per transform under qemu-riscv64 with the libinsn TCG plugin:
(insns(11 reps) - insns(1 rep)) / 10; see docs/performance-report.md section 1.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "docs", "data")
REPORT = os.path.join(ROOT, "docs", "performance-report.md")
FINAL = os.path.join(RESULTS, "final_insns.csv")
VLENS = (128, 256, 512)
COMP = {"clang": "clang 23", "gcc": "GCC 15.3"}
KINDLABEL = {"c2c": "`pocketfft::c2c`", "r2c": "`pocketfft::r2c`", "c2r": "`pocketfft::c2r`",
             "x2c": "kernel (`pocketfft_c<T>::exec`, cached plan, in place)"}
CACHELABEL = {"0": "no plan cache (`POCKETFFT_CACHE_SIZE=0`, NumPy/PyTorch default)", "16": "plan reused (`POCKETFFT_CACHE_SIZE=16`)"}


# ----------------------------------------------------------------------------------------- helpers
def rel(path):
    return os.path.relpath(path, ROOT)


def read_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                rows.append(line.split())
    return rows


def num(tok):
    try:
        return int(str(tok).replace(",", ""))
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return None


def fi(v):
    return "-" if v is None else f"{int(round(v)):,}"


def fx(s, v):
    return "-" if not s or not v else f"{s / v:.2f}x"


def gmean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def table(headers, rows, align=None):
    align = align or (["l"] + ["r"] * (len(headers) - 1))
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def placeholder(what, path=None):
    return f"[to be added: {what}" + (f" - expected in `{rel(path)}`]" if path else "]")


def source(path):
    return f"\nSource: `{rel(path)}`."


# ------------------------------------------------------------------------------------ final CSV
class Final:
    """results/final_insns.csv -> scalar(compiler,cache,kind,type,n) and rvv(...) -> {vlen: insns}"""

    def __init__(self, path=FINAL):
        self.path = path
        self.sc, self.rv, self.ok = {}, {}, False
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    key = (r["compiler"].strip(), r["cache"].strip(), r["kind"].strip(), r["type"].strip(), int(r["n"]))
                    v = num(r["insns_per_transform"])
                except (KeyError, ValueError, AttributeError):
                    continue
                if v is None:
                    continue
                if "scalar" in r["variant"].strip():   # scalar-rv64gc / kernel-scalar-rv64gc
                    self.sc[key] = v
                else:
                    vl = num(r["vlen"])
                    if vl:
                        self.rv.setdefault(key, {})[int(vl)] = v
        self.ok = bool(self.rv)

    def complete(self):
        """True when scripts/measure_all.sh has finished (results/final_measure.log ends with DONE)"""
        for log in glob.glob(os.path.join(RESULTS, "final_measure*.log")):
            tail = open(log).read().strip().splitlines()
            if tail and tail[-1].strip().upper().startswith("DONE"):
                return True
        return False

    def keys(self, kind=None, cache=None, compiler=None, typ=None):
        ks = set(self.sc) | set(self.rv)
        return sorted(k for k in ks if (kind is None or k[2] == kind) and (cache is None or k[1] == cache)
                      and (compiler is None or k[0] == compiler) and (typ is None or k[3] == typ))

    def row(self, key):
        sc = self.sc.get(key)
        rv = self.rv.get(key, {})
        return [fi(sc)] + [fi(rv.get(v)) for v in VLENS] + [fx(sc, rv.get(v)) for v in VLENS]

    def status_note(self):
        n = len(self.rv)
        state = "complete" if self.complete() else "**measurement still running, table partial**"
        return f"\nSource: `{rel(self.path)}` ({n} RVV rows, {state}; method: `scripts/measure_all.sh`)."


def kind_cache_table(kind, cache, compilers=("clang", "gcc"), types=("double", "float")):
    F = Final()
    if not F.ok:
        return placeholder(f"{kind} ({CACHELABEL.get(cache, cache)}) counts", FINAL)
    out = []
    for comp in compilers:
        keys = [k for k in F.keys(kind=kind, compiler=comp) if (kind == "x2c" or k[1] == cache) and k[3] in types]
        keys.sort(key=lambda k: (k[3] != "double", k[4]))
        if not keys:
            out.append(f"{COMP[comp]}: [to be added: {kind} cache={cache} rows not yet in `{rel(FINAL)}`]")
            continue
        rows = [[k[3], f"{k[4]:,}"] + F.row(k) for k in keys]
        out.append(f"{COMP[comp]}" + (" (recommended toolchain)" if comp == "clang" else "") + "\n\n" +
                   table(["type", "n", "scalar (rv64gc)", "RVV @128", "@256", "@512", "speed-up @128", "@256", "@512"], rows,
                         ["l", "r"] + ["r"] * 7))
    return "\n\n".join(out) + F.status_note()


def t_final_summary():
    F = Final()
    if not F.ok:
        return placeholder("summary of the final matrix", FINAL)
    rows = []
    combos = sorted({(k[2], k[1], k[3]) for k in F.keys()}, key=lambda c: (["x2c", "c2c", "r2c", "c2r"].index(c[0]) if c[0] in ("x2c", "c2c", "r2c", "c2r") else 9, c[1], c[2] != "double"))
    for kind, cache, typ in combos:
        row = [KINDLABEL.get(kind, kind).split(" (")[0], "-" if kind == "x2c" else ("no" if cache == "0" else "yes"), typ]
        for comp in ("clang", "gcc"):
            keys = F.keys(kind=kind, cache=cache, compiler=comp, typ=typ)
            ns = [k[4] for k in keys]
            row.append("n=" + ",".join(f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n) for n in ns) if ns else "-")
            for v in VLENS:
                sp = [F.sc[k] / F.rv[k][v] for k in keys if k in F.sc and v in F.rv.get(k, {})]
                row.append(f"{gmean(sp):.2f}x ({min(sp):.1f}-{max(sp):.1f})" if sp else "-")
        rows.append(row)
    hdr = ["kind", "plan cached", "type", "clang: sizes", "clang @128", "@256", "@512", "GCC: sizes", "GCC @128", "@256", "@512"]
    return table(hdr, rows, ["l", "l", "l"] + ["r"] * 8) + F.status_note()


def t_gcc_vs_clang():
    F = Final()
    if not F.ok:
        return placeholder("GCC vs clang comparison", FINAL)
    rows = []
    for k in F.keys(compiler="clang"):
        g = ("gcc",) + k[1:]
        if g not in F.rv and g not in F.sc:
            continue
        rc, rg = F.rv.get(k, {}).get(256), F.rv.get(g, {}).get(256)
        rows.append([KINDLABEL.get(k[2], k[2]).split(" (")[0], "-" if k[2] == "x2c" else k[1], k[3], f"{k[4]:,}",
                     fi(F.sc.get(k)), fi(F.sc.get(g)), fi(rc), fi(rg), fx(rg, rc) if rg and rc else "-"])
    if not rows:
        return placeholder("GCC rows of the final matrix", FINAL)
    hdr = ["kind", "cache", "type", "n", "scalar clang", "scalar GCC", "RVV@256 clang", "RVV@256 GCC", "GCC / clang"]
    return table(hdr, rows, ["l", "l", "l", "r", "r", "r", "r", "r", "r"]) + F.status_note()


# ------------------------------------------------------------------------------------ txt parsers
def parse_variant_table(path, with_kind=False):
    d = {}
    for r in read_rows(path):
        if r[0] == "variant":
            continue
        if with_kind:
            if len(r) < 6:
                continue
            key = (r[0], r[1], int(r[2])); vals = r[3:6]
        else:
            if len(r) < 5:
                continue
            key = (r[0], int(r[1])); vals = r[2:5]
        d[key] = {v: num(x) for v, x in zip(VLENS, vals)}
    return d


def parse_api_v3(path):
    d = {}
    for r in read_rows(path):
        if r[0] == "n" or len(r) < 5:
            continue
        d[int(r[0])] = (num(r[1]), {v: num(x) for v, x in zip(VLENS, r[2:5])})
    return d


def parse_kv(path):
    d = {}
    for r in read_rows(path):
        if not r[0].startswith("n="):
            continue
        d[int(r[0][2:])] = {r[i]: num(r[i + 1]) for i in range(1, len(r) - 1, 2)}
    return d


def parse_first(path):
    d = {}
    for r in read_rows(path):
        if r[0] == "type" or len(r) < 6:
            continue
        d[(r[0], int(r[1]))] = (num(r[2]), {v: num(x) for v, x in zip(VLENS, r[3:6])})
    return d


def parse_xisa(path):
    ns, d = [], {}
    for r in read_rows(path):
        if r[0] == "ISA":
            ns = [int(x[2:]) for x in r[2:]]
            continue
        if len(r) >= 3:
            d[(r[0], r[1])] = [num(x) for x in r[2:]]
    return ns, d


def parse_real(path):
    d = {}
    for r in read_rows(path):
        if r[0] == "build" or len(r) < 6:
            continue
        d[(r[0], r[1], r[2])] = num(r[5])
    return d


def parse_data(path):
    d = {}
    for r in read_rows(path):
        if r[0] == "bin" or len(r) < 7:
            continue
        d[(r[0], r[1], r[2], int(r[3]))] = {v: num(x) for v, x in zip(VLENS, r[4:7])}
    return d


# ------------------------------------------------------------------------------------ txt tables
def t_kernel():
    path = os.path.join(RESULTS, "kernel_insns.txt")
    d = parse_variant_table(path)
    if not d:
        return placeholder("kernel instruction counts", path)
    rows = []
    for n in sorted({k[1] for k in d}):
        sc = d.get(("c16_scalar_clang_rv64gc", n), {}).get(256)
        sg = d.get(("c16_scalar_gcc", n), {}).get(256)
        rc = d.get(("c16_rvv_clang", n), {}); rg = d.get(("c16_rvv_gcc", n), {})
        rows.append([f"{n:,}", fi(sc), fi(sg)] + [fi(rc.get(v)) for v in VLENS] + [fx(sc, rc.get(v)) for v in VLENS] + [fi(rg.get(v)) for v in VLENS])
    hdr = ["n", "scalar clang (rv64gc)", "scalar GCC (rv64gcv, auto-vec)", "RVV clang @128", "@256", "@512",
           "speed-up @128", "@256", "@512", "RVV GCC @128 (pre no-schedule fix)", "@256", "@512"]
    return table(hdr, rows) + source(path)


def t_api(fname, what):
    path = os.path.join(RESULTS, fname)
    d = parse_api_v3(path)
    if not d:
        return placeholder(what, path)
    rows = [[f"{n:,}", fi(sc)] + [fi(rv.get(v)) for v in VLENS] + [fx(sc, rv.get(v)) for v in VLENS] for n, (sc, rv) in sorted(d.items())]
    return table(["n", "scalar (rv64gc)", "RVV @128", "@256", "@512", "speed-up @128", "@256", "@512"], rows) + source(path)


def t_trend():
    p1, p2, p3 = (os.path.join(RESULTS, f) for f in ("api_nocache_insns.txt", "api_nocache_insns_v2.txt", "api_nocache_insns_v3.txt"))
    a, b, c = parse_kv(p1), parse_kv(p2), parse_api_v3(p3)
    if not (a or b or c):
        return placeholder("API optimisation trend", p1)
    rows = []
    for n in sorted(set(a) | set(b) | set(c)):
        sc = (c.get(n, (None,))[0]) or a.get(n, {}).get("rv64gc-scalar") or b.get(n, {}).get("rv64gc-scalar")
        v1 = a.get(n, {}).get("rvv"); v2 = b.get(n, {}).get("rvv"); v3 = c.get(n, (None, {}))[1].get(256)
        rows.append([f"{n:,}", fi(sc), fi(a.get(n, {}).get("rvv64gcv-scalar")), fi(v1), fx(sc, v1), fi(v2), fx(sc, v2), fi(v3), fx(sc, v3)])
    hdr = ["n", "scalar rv64gc", "scalar rv64gcv (clang auto-vec)", "RVV passes (inlined)", "x", "+ vector twiddle fill", "x",
           "+ RVV copy_input/output", "x"]
    return table(hdr, rows) + f"\nSources: `{rel(p1)}`, `{rel(p2)}`, `{rel(p3)}` (VLEN 256, clang 23, double c2c forward, no plan cache)."


def t_first():
    path = os.path.join(RESULTS, "c2c_insns.txt")
    d = parse_first(path)
    if not d:
        return placeholder("first-version counts", path)
    rows = [[t, f"{n:,}", fi(sc)] + [fi(rv.get(v)) for v in VLENS] + [fx(sc, rv.get(v)) for v in VLENS]
            for (t, n), (sc, rv) in sorted(d.items(), key=lambda kv: (kv[0][0] != "double", kv[0][1]))]
    hdr = ["type", "n", "scalar (GCC 15 rv64gcv, auto-vec)", "RVV v0 @128", "@256", "@512", "ratio @128", "@256", "@512"]
    return table(hdr, rows, ["l", "r"] + ["r"] * 7) + source(path)


def t_early():
    path = os.path.join(RESULTS, "c2c_insns_cache16.txt")
    d = parse_variant_table(path)
    if not d:
        return placeholder("early GCC/clang/LMUL experiment", path)
    rows = [[var, f"{n:,}"] + [fi(d[(var, n)].get(v)) for v in VLENS]
            for n in sorted({k[1] for k in d})
            for var in ("c16_scalar_gcc", "c16_rvv_gcc", "c16_rvv_gcc_low", "c16_scalar_clang", "c16_rvv_clang", "c16_rvv_clang_low") if (var, n) in d]
    return table(["build", "n", "VLEN 128", "VLEN 256", "VLEN 512"], rows, ["l", "r", "r", "r", "r"]) + source(path)


def t_split():
    path = os.path.join(RESULTS, "c2c_insns_v2.txt")
    d = parse_variant_table(path, with_kind=True)
    if not d:
        return placeholder("split-store experiment", path)
    rows = [[var, {"c2c": "API (cache 16)", "x2c": "kernel"}.get(kind, kind), f"{n:,}"] + [fi(d[(var, kind, n)].get(v)) for v in VLENS]
            for (var, kind, n) in sorted(d, key=lambda k: (k[1], k[2], k[0]))]
    return table(["build", "measure", "n", "VLEN 128", "VLEN 256", "VLEN 512"], rows, ["l", "l", "r", "r", "r", "r"]) + source(path)


def t_xisa():
    path = os.path.join(RESULTS, "xisa_insns.txt")
    ns, d = parse_xisa(path)
    kpath = os.path.join(RESULTS, "kernel_insns.txt")
    k = parse_variant_table(kpath)
    if not d:
        return placeholder("cross-ISA counts", path)
    labels = {("x86-64", "gcc13-v3"): "x86-64, GCC 13 `-O3 -march=x86-64-v3` (AVX2 build)",
              ("x86-64", "clang23-v3"): "x86-64, clang 23 `-O3 -march=x86-64-v3`",
              ("aarch64", "gcc13-armv8.2"): "aarch64, GCC 13 `-O3 -march=armv8.2-a` (NEON build)"}
    rows = [[labels.get(key, " ".join(key))] + [fi(v) for v in vals] for key, vals in d.items()]
    rows.append(["riscv64, clang 23 `-march=rv64gc` scalar (this header, `POCKETFFT_NO_RVV`)"] + [fi(k.get(("c16_scalar_clang_rv64gc", n), {}).get(256)) for n in ns])
    for vl in VLENS:
        rows.append([f"riscv64, clang 23 RVV, VLEN {vl}"] + [fi(k.get(("c16_rvv_clang", n), {}).get(vl)) for n in ns])
    x = d.get(("x86-64", "gcc13-v3"))
    if x:
        for vl in VLENS:
            rows.append([f"ratio x86-64 (GCC) / riscv64 RVV @{vl}"] + [fx(xv, k.get(("c16_rvv_clang", n), {}).get(vl)) for xv, n in zip(x, ns)])
    return table(["ISA / build (kernel measure: cached plan, in-place exec, double)"] + [f"n={n:,}" for n in ns], rows) + \
        f"\nSources: `{rel(path)}` (x86-64 under qemu-x86_64 `-cpu max`, aarch64 under qemu-aarch64), `{rel(kpath)}` (riscv64)."


def t_real():
    files = sorted(glob.glob(os.path.join(RESULTS, "real_*.txt"))) + sorted(glob.glob(os.path.join(ROOT, "work-real", "build", "bench*_final_*.txt")))
    if not files:
        return placeholder("r2c/c2r results", os.path.join(RESULTS, "real_insns.txt"))
    out = []
    for path in files:
        d = parse_real(path)
        if not d:
            continue
        m = re.search(r"_(\d+)\.txt$", path)
        n = int(m.group(1)) if m else 0
        cache = "plan reused (`POCKETFFT_CACHE_SIZE=16`)" if "cache" in os.path.basename(path) else "no plan cache (NumPy/PyTorch default)"
        rows = []
        for dir_ in ("r2c", "c2r"):
            for comp in ("gcc", "clang"):
                sc = next((v for (b, dd, vl), v in d.items() if dd == dir_ and comp in b and "sca" in b), None)
                rv = {int(vl): v for (b, dd, vl), v in d.items() if dd == dir_ and comp in b and "rvv" in b and vl != "-"}
                rows.append([dir_, COMP[comp], fi(sc)] + [fi(rv.get(v)) for v in VLENS] + [fx(sc, rv.get(v)) for v in VLENS])
        out.append(f"**n = {n:,}, double, {cache}** (real-FFT agent's measurement before the final framework fixes)\n\n" +
                   table(["kind", "compiler", "scalar (rv64gc)", "RVV @128", "@256", "@512", "speed-up @128", "@256", "@512"], rows, ["l", "l"] + ["r"] * 7) + source(path))
    return "\n\n".join(out) if out else placeholder("r2c/c2r results")


def t_remeasured(kinds, label, bins_scalar, bins_rvv, types, fname):
    path = os.path.join(DATA, fname)
    d = parse_data(path)
    if not d:
        return placeholder(f"{label} re-measurement", path)
    rows = []
    for kind, typ, n in sorted({(k[1], k[2], k[3]) for k in d if k[1] in kinds and k[2] in types}, key=lambda k: (kinds.index(k[0]), k[1] != "double", k[2])):
        sc = next((d[(b, kind, typ, n)].get(256) for b in bins_scalar if (b, kind, typ, n) in d), None)
        row = [kind, typ, f"{n:,}", fi(sc)]
        for b in bins_rvv:
            rv = d.get((b, kind, typ, n), {})
            row += [fi(rv.get(v)) for v in VLENS] + [fx(sc, rv.get(v)) for v in VLENS]
        rows.append(row)
    hdr = ["kind", "type", "n", "scalar clang rv64gc"]
    for b in bins_rvv:
        hdr += [f"RVV {'clang 23' if 'clang' in b else 'GCC 15.3'} @128", "@256", "@512", "x @128", "@256", "@512"]
    return table(hdr, rows, ["l", "l", "r"] + ["r"] * (len(hdr) - 3)) + source(path)


def t_lmul():
    path = os.path.join(RESULTS, "lmul_experiment.txt")
    rows_in = [r for r in read_rows(path) if r[0] != "lmul-config" and len(r) >= 5]
    if not rows_in:
        return placeholder("LMUL experiment", path)
    labels = {"default_4_2_2_1_1": "default (radix 2/3/4/5/8 at LMUL 4/2/2/1/1)", "all1": "LMUL 1 for every radix",
              "all2": "LMUL 2 for every radix", "p8m2_p5m2": "radix 8 and 5 at LMUL 2 (others default)", "p4m4": "radix 4 at LMUL 4 (others default)"}
    base = {int(r[1]): {v: num(x) for v, x in zip(VLENS, r[2:5])} for r in rows_in if r[0].startswith("default")}
    rows = []
    for r in sorted(rows_in, key=lambda r: (list(labels).index(r[0]) if r[0] in labels else 9, int(r[1]))):
        n = int(r[1]); vals = {v: num(x) for v, x in zip(VLENS, r[2:5])}
        rows.append([labels.get(r[0], r[0]), f"{n:,}"] + [fi(vals[v]) for v in VLENS] +
                    [("-" if r[0].startswith("default") or not base.get(n, {}).get(v) else f"{100 * (vals[v] / base[n][v] - 1):+.0f} %") for v in VLENS])
    return table(["LMUL configuration", "n", "VLEN 128", "VLEN 256", "VLEN 512", "vs default @128", "@256", "@512"], rows, ["l", "r"] + ["r"] * 6) + \
        source(path) + " Kernel measure (plan cached), double c2c forward, clang 23."


def t_xisa_api():
    path = os.path.join(RESULTS, "xisa_api_insns.txt")
    rows_in = read_rows(path)
    if not rows_in:
        return placeholder("API-level cross-ISA counts", path)
    ns = [int(x[2:]) for x in rows_in[0][2:]] if rows_in[0][0] == "ISA" else []
    d = {(r[0], r[1]): [num(x) for x in r[2:]] for r in rows_in if r[0] != "ISA"}
    F = Final()
    out = []
    for kind in ("c2c", "r2c"):
        rows = []
        for isa, label in (("x86-64", "x86-64, GCC 13 `-march=x86-64-v3` (AVX2 build)"), ("aarch64", "aarch64, GCC 13 armv8.2 (NEON build)")):
            if (isa, kind) in d:
                rows.append([label] + [fi(v) for v in d[(isa, kind)]])
        sc = [F.sc.get(("clang", "0", kind, "double", n)) for n in ns]
        rows.append(["riscv64, clang 23 `rv64gc` scalar (this header, `POCKETFFT_NO_RVV`)"] + [fi(v) for v in sc])
        for vl in VLENS:
            rows.append([f"riscv64, clang 23 RVV, VLEN {vl}"] + [fi(F.rv.get(("clang", "0", kind, "double", n), {}).get(vl)) for n in ns])
        x = d.get(("x86-64", kind))
        if x:
            for vl in VLENS:
                rows.append([f"ratio x86-64 / riscv64 RVV @{vl}"] + [fx(xv, F.rv.get(("clang", "0", kind, "double", n), {}).get(vl)) for xv, n in zip(x, ns)])
        out.append(f"`pocketfft::{kind}`, double, no plan cache\n\n" + table(["ISA / build"] + [f"n={n:,}" for n in ns], rows))
    return "\n\n".join(out) + f"\nSources: `{rel(path)}` (x86-64 under qemu-x86_64 `-cpu max`, aarch64 under qemu-aarch64, same `fftdump` API measure), `{rel(FINAL)}` (riscv64)."


def t_x86_native():
    path = os.path.join(RESULTS, "bench_xisa.csv")
    if not os.path.exists(path):
        return placeholder("native x86 timing", path)
    rows = []
    with open(path, newline="") as f:
        rs = [r for r in csv.DictReader(f) if r["isa"] == "x86-64" and r.get("ns_per_transform")]
    for r in sorted(rs, key=lambda r: (r["type"] != "double", r["kind"], int(r["n"]))):
        ns, n = float(r["ns_per_transform"]), int(r["n"])
        flops = (5.0 if r["kind"].startswith("c2c") else 2.5) * n * math.log2(n)
        rows.append([r["type"], r["kind"], f"{n:,}", f"{ns / 1000:,.1f}", f"{flops / ns:,.2f}", fi(float(r["insns_per_transform"]))])
    return table(["type", "kind", "n", "us per transform", "GFLOP/s (5 n log2 n / 2.5 n log2 n)", "instructions per transform (qemu-x86_64)"], rows, ["l", "l", "r", "r", "r", "r"]) + \
        source(path) + " `bench/fft_bench` on the x86-64 build host (Xeon 2.1 GHz, 4 shared cores, `taskset -c 1`, minimum over 7 trials, `pocketfft::c2c/r2c` with plan cache 16)."


def t_x86_fftw():
    path = os.path.join(RESULTS, "x86_native_fftw_vs_pocketfft.txt")
    rows_in = [r for r in read_rows(path) if r[0] in ("c2c", "r2c")]
    if not rows_in:
        return placeholder("x86 FFTW comparison", path)
    rows = [[r[0], f"{int(r[1]):,}", f"{num(r[2]) / 1000:,.1f}", f"{num(r[3]) / 1000:,.1f}", f"{num(r[4]):.2f}x", r[5].strip("()").replace("max|diff|=", "") if len(r) > 5 else "-"] for r in rows_in]
    return table(["kind", "n", "pocketfft (scalar 1-D path, plan cached) us", "FFTW 3.3.11 AVX2, `FFTW_MEASURE` us", "pocketfft / FFTW", "max abs difference"], rows, ["l", "r", "r", "r", "r", "r"]) + \
        source(path) + " `bench/fftw_vs_pocketfft_x86.cpp`, same host, double precision."


TABLES = {
    "final-summary": t_final_summary,
    "final-kernel": lambda: kind_cache_table("x2c", "16"),
    "final-c2c-0": lambda: kind_cache_table("c2c", "0"),
    "final-c2c-16": lambda: kind_cache_table("c2c", "16"),
    "final-r2c-0": lambda: kind_cache_table("r2c", "0"),
    "final-r2c-16": lambda: kind_cache_table("r2c", "16"),
    "final-c2r-0": lambda: kind_cache_table("c2r", "0"),
    "final-c2r-16": lambda: kind_cache_table("c2r", "16"),
    "final-gcc-vs-clang": t_gcc_vs_clang,
    "kernel": t_kernel,
    "api-nocache": lambda: t_api("api_nocache_insns_v3.txt", "API (no cache) counts"),
    "api-cache16": lambda: t_api("api_cache16_insns.txt", "API (cache 16) counts"),
    "trend": t_trend,
    "first": t_first,
    "early": t_early,
    "split": t_split,
    "xisa": t_xisa,
    "xisa-api": t_xisa_api,
    "x86-native": t_x86_native,
    "x86-fftw": t_x86_fftw,
    "lmul": t_lmul,
    "real": t_real,
    "re-kernel": lambda: t_remeasured(["x2c"], "kernel", ["c16_scalar_clang_rv64gc"], ["c16_rvv_clang", "c16_rvv_gcc"], ("double", "float"), "kernel_insns_remeasured.txt"),
    "re-api": lambda: t_remeasured(["c2c", "r2c", "c2r"], "API", ["c0_scalar_clang_rv64gc"], ["c0_rvv_clang", "c0_rvv_gcc"], ("double",), "api_insns_remeasured.txt"),
}


def update(report, check=False):
    if not os.path.exists(report):
        print(f"{report} does not exist; use --print", file=sys.stderr)
        return 2
    text = open(report).read()
    pat = re.compile(r"(<!-- gen:([a-z0-9-]+) -->\n)(.*?)(<!-- /gen -->)", re.S)
    unknown = []

    def sub(m):
        name = m.group(2)
        if name not in TABLES:
            unknown.append(name)
            return m.group(0)
        return m.group(1) + TABLES[name]() + "\n" + m.group(4)

    new = pat.sub(sub, text)
    for n in unknown:
        print(f"warning: unknown table '{n}' left untouched", file=sys.stderr)
    if check:
        print("report is up to date" if new == text else "report is out of date (run docs/gen_tables.py)")
        return 0 if new == text else 1
    if new != text:
        open(report, "w").write(new)
        print(f"updated {rel(report)}")
    else:
        print("no changes")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print tables to stdout")
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/performance-report.md is out of date")
    ap.add_argument("--only", help="comma-separated table names for --print")
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    if a.print:
        for n in (a.only.split(",") if a.only else list(TABLES)):
            print(f"### {n}\n\n{TABLES[n]()}\n")
        return 0
    return update(a.report, a.check)


if __name__ == "__main__":
    sys.exit(main())
