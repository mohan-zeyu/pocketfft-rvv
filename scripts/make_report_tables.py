#!/usr/bin/env python3
"""make_report_tables.py [--results DIR] [--no-charts]

Turns the CSVs written by run_correctness.sh / run_bench_qemu.sh / run_bench_xisa.sh into
  results/report.md                      Markdown tables (correctness, QEMU scalar-vs-RVV, cross-ISA)
  results/chart_qemu_speedup_<comp>_b<B>.png   instruction-count speedup RVV vs scalar per VLEN
  results/chart_qemu_vshare_<comp>_b<B>.png    vector-instruction share per VLEN
  results/chart_xisa_<type>_b<B>.png           dynamic instructions relative to the x86-64 (AVX2) build
  results/chart_correctness_<label>.png        max|err|/max|ref| vs n against the 1e-5 pass threshold
Charts need matplotlib (pip install matplotlib); without it only the Markdown is produced.
"""
import argparse
import csv
import glob
import math
import os
import sys
from collections import OrderedDict, defaultdict

# --- palette (validated default from the dataviz method: categorical slots in fixed order) -------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"


def fnum(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def md_table(headers, rows, align=None):
    align = align or ["r"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def fmt_i(v):
    return "-" if v is None else f"{v:,.0f}"


def fmt_x(v):
    return "-" if v is None else f"{v:.2f}x"


def fmt_pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def fmt_e(v):
    return "-" if v is None or v < 0 else f"{v:.2e}"


def gmean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


KIND_ORDER = ["c2c-fwd", "c2c-bwd", "r2c", "c2r"]
TYPE_ORDER = ["double", "float"]


def sort_key_kind(k):
    return KIND_ORDER.index(k) if k in KIND_ORDER else 99


# ------------------------------------------------------------------------------------------------
# correctness
# ------------------------------------------------------------------------------------------------
def correctness_section(results, charts):
    files = sorted(glob.glob(os.path.join(results, "correctness_*.csv")))
    if not files:
        return "", {}
    md = ["## Correctness (pocketfft vs FFTW3 double-precision reference)\n",
          "Pass criterion: max|err| / max|ref| <= 1e-5 per transform (also L2 relative error reported in the CSVs). "
          "`vs scalar` is the element-wise difference between the RVV build and the scalar build of the same compiler "
          "on identical inputs (dump comparison).\n"]
    rows = []
    data = {}
    for f in files:
        rs = read_csv(f)
        if not rs:
            continue
        label = rs[0]["label"]
        data[label] = rs
        n = len(rs)
        npass = sum(1 for r in rs if r["status"] == "PASS")
        nfail = sum(1 for r in rs if r["status"] == "FAIL")
        nnoref = sum(1 for r in rs if r["status"] == "NOREF")
        worst = {}
        for t in TYPE_ORDER:
            sub = [(fnum(r["max_err_norm"], -1), r) for r in rs if r["type"] == t]
            sub = [s for s in sub if s[0] is not None and s[0] >= 0]
            if sub:
                v, r = max(sub, key=lambda s: s[0])
                worst[t] = (v, f"{r['kind']} {r['dir']} n={r['n']}" + (f" B={r['batch']}" if r["batch"] != "1" else ""))
            dsub = [fnum(r["vs_dump_norm"], -1) for r in rs if r["type"] == t]
            dsub = [d for d in dsub if d is not None and d >= 0]
            worst[t + "_dump"] = max(dsub) if dsub else None
        rows.append([label, n, npass, nfail, nnoref,
                     fmt_e(worst.get("float", (None,))[0]) + (f" ({worst['float'][1]})" if "float" in worst else ""),
                     fmt_e(worst.get("double", (None,))[0]) + (f" ({worst['double'][1]})" if "double" in worst else ""),
                     fmt_e(worst.get("float_dump")), fmt_e(worst.get("double_dump")),
                     "**FAIL**" if nfail else "pass"])
    md.append(md_table(["configuration", "cases", "pass", "fail", "noref", "worst float err (case)", "worst double err (case)",
                        "float vs scalar", "double vs scalar", "result"], rows, ["l"] + ["r"] * 8 + ["l"]))
    # per-class breakdown (precision x kind x direction, with the Bluestein lengths singled out)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from summarize_correctness import md_per_class
        md.append("\n### Per-class maximum errors\n")
        md.append("Bluestein primes = lengths 101, 257, 1009, 10007, 65537. `max vs scalar` = largest element-wise difference "
                  "between this build and the scalar build on identical inputs, normalised by max|ref|.\n")
        for label, rs in data.items():
            md.append(md_per_class(rs, label=label))
    except ImportError:
        pass
    if charts:
        for label, rs in data.items():
            p = chart_correctness(results, label, rs)
            if p:
                md.append(f"\n![correctness {label}]({os.path.basename(p)})\n")
                break  # one chart is enough for the report; the others are produced on request
    return "\n".join(md) + "\n", data


def chart_correctness(results, label, rs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    style_axes(fig, ax)
    floor = 1e-18
    for i, t in enumerate(TYPE_ORDER):
        pts = [(int(r["n"]), fnum(r["max_err_norm"])) for r in rs if r["type"] == t and fnum(r["max_err_norm"], -1) >= 0]
        xs = [n for n, e in pts if e > 0]
        ys = [e for n, e in pts if e > 0]
        ax.scatter(xs, ys, s=12, color=SERIES[i], label=t, alpha=0.85, linewidths=0)
        zx = sorted({n for n, e in pts if e == 0})
        if zx:  # exact results (error 0) cannot be shown on a log axis: hollow markers on the floor line
            ax.scatter(zx, [floor] * len(zx), s=22, facecolors="none", edgecolors=SERIES[i], linewidths=0.8,
                       label=f"{t}, exact (err = 0)")
    ax.axhline(1e-5, color=CRITICAL, linewidth=1, linestyle="--")
    ax.text(1.05, 1e-5, "pass threshold 1e-5", color=CRITICAL, fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("transform length n")
    ax.set_ylabel("max|err| / max|ref|")
    ax.set_title(f"pocketfft vs FFTW3: {label} (all kinds, directions, batches)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, title="precision", title_fontsize=8)
    path = os.path.join(results, f"chart_correctness_{label}.png")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------
# QEMU scalar vs RVV
# ------------------------------------------------------------------------------------------------
def qemu_section(results, charts):
    path = os.path.join(results, "bench_qemu.csv")
    if not os.path.exists(path):
        return ""
    rs = read_csv(path)
    if not rs:
        return ""
    vlens = sorted({int(r["vlen"]) for r in rs})
    variants = [v for v in ["scalar", "autovec", "rvv"] if any(r["variant"] == v for r in rs)]
    # index: (compiler,type,kind,batch,n) -> variant -> vlen -> (insns, vshare)
    idx = defaultdict(lambda: defaultdict(dict))
    for r in rs:
        key = (r["compiler"], r["type"], r["kind"], int(r["batch"]), int(r["n"]))
        idx[key][r["variant"]][int(r["vlen"])] = (fnum(r["insns_per_transform"]), fnum(r["vector_share"]))
    md = ["## QEMU dynamic instruction counts: scalar vs RVV (riscv64)\n",
          "Instructions per transform = (insns(2N reps) - insns(N reps)) / N under qemu-riscv64 "
          "(`-cpu rv64,v=true,vlen=<V>,zba,zbb,zbs`), i.e. plan creation and process start-up are excluded. "
          "Speedup = scalar insns / RVV insns (a count ratio, not cycles: one vector instruction does VLEN/SEW "
          "lanes of work, so this understates the gain on hardware with wide datapaths and overstates it where "
          "vector instructions are split into many uops). `vec%` = share of vector instructions (vsetvl, vector "
          "loads/stores, OPIVV/OPFVV/OPMVV) in the transform's instruction stream (librvvclass plugin). The scalar "
          "build is compiled without the V extension; it is run with the same CPU model, so glibc's ifunc-selected "
          "RVV memcpy accounts for a few vector instructions per transform in the scalar rows too.\n"]
    groups = sorted({k[:4] for k in idx}, key=lambda g: (g[0], TYPE_ORDER.index(g[1]) if g[1] in TYPE_ORDER else 9, sort_key_kind(g[2]), g[3]))
    # summary: geometric-mean speedup per group per vlen
    srows = []
    for g in groups:
        ns = sorted(k[4] for k in idx if k[:4] == g)
        row = [g[0], g[1], g[2], g[3]]
        for vl in vlens:
            sp = []
            for n in ns:
                d = idx[g + (n,)]
                s = d.get("scalar", {}).get(vl, (None,))[0]
                v = d.get("rvv", {}).get(vl, (None,))[0]
                if s and v:
                    sp.append(s / v)
            row.append(fmt_x(gmean(sp)))
        srows.append(row)
    md.append("### Geometric-mean instruction-count speedup (RVV vs scalar) over the size sweep\n")
    md.append(md_table(["compiler", "type", "kind", "batch"] + [f"VLEN {vl}" for vl in vlens], srows, ["l", "l", "l", "r"] + ["r"] * len(vlens)))
    for g in groups:
        ns = sorted(k[4] for k in idx if k[:4] == g)
        md.append(f"\n### {g[0]} {g[1]} {g[2]} batch={g[3]}\n")
        hdr = ["n", "scalar"]
        if "autovec" in variants:
            hdr += [f"autovec@{vl}" for vl in vlens]
        hdr += [f"rvv@{vl}" for vl in vlens] + [f"speedup@{vl}" for vl in vlens] + [f"vec%@{vl}" for vl in vlens]
        rows = []
        for n in ns:
            d = idx[g + (n,)]
            sc = [d.get("scalar", {}).get(vl, (None,))[0] for vl in vlens]
            s0 = next((s for s in sc if s), None)
            row = [n, fmt_i(s0)]
            if "autovec" in variants:
                row += [fmt_i(d.get("autovec", {}).get(vl, (None,))[0]) for vl in vlens]
            rv = [d.get("rvv", {}).get(vl, (None, None)) for vl in vlens]
            row += [fmt_i(x[0]) for x in rv]
            row += [fmt_x(d.get("scalar", {}).get(vl, (None,))[0] / x[0]) if x[0] and d.get("scalar", {}).get(vl, (None,))[0] else "-" for vl, x in zip(vlens, rv)]
            row += [fmt_pct(x[1]) if x[0] else "-" for x in rv]
            rows.append(row)
        md.append(md_table(hdr, rows))
    if charts:
        for comp in sorted({g[0] for g in groups}):
            for b in sorted({g[3] for g in groups if g[0] == comp}):
                for metric in ("speedup", "vshare"):
                    p = chart_qemu(results, idx, comp, b, vlens, metric)
                    if p:
                        md.append(f"\n![{metric} {comp} batch {b}]({os.path.basename(p)})\n")
    return "\n".join(md) + "\n"


def style_axes(fig, ax):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_color(INK)


def chart_qemu(results, idx, comp, batch, vlens, metric):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    types = [t for t in TYPE_ORDER if any(k[0] == comp and k[1] == t and k[3] == batch for k in idx)]
    kinds = sorted({k[2] for k in idx if k[0] == comp and k[3] == batch}, key=sort_key_kind)
    if not types or not kinds:
        return None
    fig, axes = plt.subplots(len(kinds), len(types), figsize=(4.2 * len(types), 2.6 * len(kinds)), dpi=150, squeeze=False)
    for i, kind in enumerate(kinds):
        for j, t in enumerate(types):
            ax = axes[i][j]
            style_axes(fig, ax)
            ns = sorted(k[4] for k in idx if k[:4] == (comp, t, kind, batch))
            for vi, vl in enumerate(vlens):
                xs, ys = [], []
                for n in ns:
                    d = idx[(comp, t, kind, batch, n)]
                    s = d.get("scalar", {}).get(vl, (None,))[0]
                    v, share = d.get("rvv", {}).get(vl, (None, None))
                    if metric == "speedup" and s and v:
                        xs.append(n); ys.append(s / v)
                    elif metric == "vshare" and v and share is not None:
                        xs.append(n); ys.append(100 * share)
                if xs:
                    ax.plot(xs, ys, color=SERIES[vi], linewidth=1.0, marker="o", markersize=4, label=f"VLEN {vl}")
            if metric == "speedup":
                ax.axhline(1.0, color=AXIS, linewidth=0.8, linestyle="--")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{kind}, {t}", fontsize=9, loc="left")
            if i == len(kinds) - 1:
                ax.set_xlabel("n")
            if j == 0:
                ax.set_ylabel("scalar insns / RVV insns" if metric == "speedup" else "vector instruction share (%)")
            if metric == "vshare":
                ax.set_ylim(0, 100)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"pocketfft on riscv64 ({comp}, batch={batch}): " + ("instruction-count speedup of the RVV path" if metric == "speedup" else "vector share of the RVV path"),
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(results, f"chart_qemu_{metric}_{comp}_b{batch}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------
# cross-ISA
# ------------------------------------------------------------------------------------------------
def xisa_section(results, charts):
    path = os.path.join(results, "bench_xisa.csv")
    if not os.path.exists(path):
        return ""
    rs = read_csv(path)
    if not rs:
        return ""
    idx = defaultdict(dict)  # (type,kind,batch,n) -> (isa,variant) -> row
    for r in rs:
        idx[(r["type"], r["kind"], int(r["batch"]), int(r["n"]))][(r["isa"], r["variant"])] = r
    cols = []
    for c in [("x86-64", "native"), ("aarch64", "native"), ("riscv64", "scalar"), ("riscv64", "autovec"), ("riscv64", "rvv")]:
        if any(c in d for d in idx.values()):
            cols.append(c)
    vlen = next((r["vlen"] for r in rs if r["isa"] == "riscv64"), "?")
    md = ["## Cross-ISA comparison (same sizes, same source)\n",
          f"x86-64: gcc 13 `-O3 -march=x86-64-v3` (native ns/transform on the build host, instruction counts under "
          f"qemu-x86_64 -cpu max); aarch64: gcc 13 `-O3 -march=armv8.2-a` under qemu-aarch64; riscv64: gcc 15 "
          f"scalar and RVV builds under qemu-riscv64 (VLEN {vlen}). Instruction counts are per transform "
          f"(2N-N differencing). Different ISAs do different amounts of work per instruction, so treat the counts "
          f"as a proxy for code path efficiency, not for wall-clock time.\n"]
    groups = sorted({k[:3] for k in idx}, key=lambda g: (TYPE_ORDER.index(g[0]) if g[0] in TYPE_ORDER else 9, sort_key_kind(g[1]), g[2]))
    for g in groups:
        ns = sorted(k[3] for k in idx if k[:3] == g)
        md.append(f"\n### {g[0]} {g[1]} batch={g[2]}\n")
        hdr = ["n", "x86-64 ns/transform"] + [f"{c[0]} {c[1]} insns" if c[1] != "native" else f"{c[0]} insns" for c in cols]
        hdr += ["rv64 rvv / x86 insns", "rv64 rvv / rv64 scalar", "rvv vec%"]
        rows = []
        for n in ns:
            d = idx[g + (n,)]
            x86 = d.get(("x86-64", "native"), {})
            row = [n, ("%.0f" % fnum(x86.get("ns_per_transform"))) if fnum(x86.get("ns_per_transform")) else "-"]
            vals = {c: fnum(d.get(c, {}).get("insns_per_transform")) for c in cols}
            row += [fmt_i(vals[c]) for c in cols]
            xi, sc, rv = vals.get(("x86-64", "native")), vals.get(("riscv64", "scalar")), vals.get(("riscv64", "rvv"))
            row += [fmt_x(rv / xi) if rv and xi else "-", fmt_x(sc / rv) if sc and rv else "-",
                    fmt_pct(fnum(d.get(("riscv64", "rvv"), {}).get("vector_share")))]
            rows.append(row)
        md.append(md_table(hdr, rows))
    if charts:
        for t in sorted({g[0] for g in groups}, key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else 9):
            for b in sorted({g[2] for g in groups if g[0] == t}):
                p = chart_xisa(results, idx, t, b, cols)
                if p:
                    md.append(f"\n![cross-ISA {t} batch {b}]({os.path.basename(p)})\n")
    return "\n".join(md) + "\n"


def chart_xisa(results, idx, t, batch, cols):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    kinds = sorted({k[1] for k in idx if k[0] == t and k[2] == batch}, key=sort_key_kind)
    series = [c for c in cols if c != ("x86-64", "native")]
    if not kinds or not series:
        return None
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.4 * len(kinds), 3.2), dpi=150, squeeze=False)
    names = {("aarch64", "native"): "aarch64 NEON (gcc13)", ("riscv64", "scalar"): "riscv64 scalar (gcc15)",
             ("riscv64", "autovec"): "riscv64 auto-vec (gcc15)", ("riscv64", "rvv"): "riscv64 RVV (gcc15)"}
    for ai, kind in enumerate(kinds):
        ax = axes[0][ai]
        style_axes(fig, ax)
        ns = sorted(k[3] for k in idx if k[:3] == (t, kind, batch))
        x = np.arange(len(ns))
        w = 0.8 / len(series)
        for si, c in enumerate(series):
            ys = []
            for n in ns:
                d = idx[(t, kind, batch, n)]
                xi = fnum(d.get(("x86-64", "native"), {}).get("insns_per_transform"))
                v = fnum(d.get(c, {}).get("insns_per_transform"))
                ys.append(v / xi if v and xi else float("nan"))
            ax.bar(x + (si - (len(series) - 1) / 2) * w, ys, width=w * 0.9, color=SERIES[si], label=names.get(c, str(c)), linewidth=0)
        ax.axhline(1.0, color=AXIS, linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7)
        ax.set_title(f"{kind}, {t}", fontsize=9, loc="left")
        ax.set_xlabel("n")
        if ai == 0:
            ax.set_ylabel("dynamic instructions relative to x86-64 AVX2 build")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Cross-ISA instruction counts per transform, batch={batch} (x86-64 = 1.0)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(results, f"chart_xisa_{t}_b{batch}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------
# native x86: pocketfft vs FFTW
# ------------------------------------------------------------------------------------------------
def x86_fftw_section(results, charts):
    path = os.path.join(results, "bench_x86_fftw.csv")
    if not os.path.exists(path):
        return ""
    rs = read_csv(path)
    if not rs:
        return ""
    md = ["## Native x86-64: pocketfft vs FFTW 3.3.11\n",
          "Same inputs, same machine, pinned to one core, minimum over trials of the mean over many repetitions. "
          "pocketfft: gcc 13 `-O3 -march=x86-64-v3`, public API with plan cache (`POCKETFFT_CACHE_SIZE=16`). "
          "FFTW: SSE2/AVX/AVX2 build, `FFTW_ESTIMATE` (one-shot caller) and `FFTW_MEASURE` (tuned plan) planners, "
          "planning excluded. Ratio = pocketfft time / FFTW(measure) time (> 1: FFTW faster). "
          "Measured on a shared 4-core container without a PMU; treat differences below ~10% as noise.\n"]
    hdr = ["type", "kind", "n", "batch", "pocketfft ns", "FFTW estimate ns", "FFTW measure ns", "pocketfft MFLOP/s", "FFTW measure MFLOP/s", "pocketfft / FFTW"]
    rows = []
    for r in sorted(rs, key=lambda r: (TYPE_ORDER.index(r["type"]) if r["type"] in TYPE_ORDER else 9, sort_key_kind(r["kind"]), int(r["n"]), int(r["batch"]))):
        rows.append([r["type"], r["kind"], r["n"], r["batch"], fmt_i(fnum(r["pocketfft_ns"])), fmt_i(fnum(r["fftw_estimate_ns"])),
                     fmt_i(fnum(r["fftw_measure_ns"])), fmt_i(fnum(r["pocketfft_mflops"])), fmt_i(fnum(r["fftw_measure_mflops"])),
                     fmt_x(fnum(r["pocketfft_over_fftw_measure"]))])
    md.append(md_table(hdr, rows, ["l", "l"] + ["r"] * 8))
    if charts:
        p = chart_x86_fftw(results, rs)
        if p:
            md.append(f"\n![pocketfft vs FFTW]({os.path.basename(p)})\n")
    return "\n".join(md) + "\n"


def chart_x86_fftw(results, rs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    kinds = sorted({r["kind"] for r in rs}, key=sort_key_kind)
    types = [t for t in TYPE_ORDER if any(r["type"] == t for r in rs)]
    fig, axes = plt.subplots(1, len(kinds) * len(types), figsize=(3.6 * len(kinds) * len(types), 3.2), dpi=150, squeeze=False)
    ai = 0
    series = [("pocketfft_mflops", "pocketfft"), ("fftw_measure_mflops", "FFTW (measure)")]
    for t in types:
        for k in kinds:
            ax = axes[0][ai]; ai += 1
            style_axes(fig, ax)
            sel = sorted([r for r in rs if r["type"] == t and r["kind"] == k and r["batch"] == "1"], key=lambda r: int(r["n"]))
            for si, (col, name) in enumerate(series):
                xs = [int(r["n"]) for r in sel if fnum(r[col])]
                ys = [fnum(r[col]) for r in sel if fnum(r[col])]
                ax.plot(xs, ys, color=SERIES[si], linewidth=1.0, marker="o", markersize=4, label=name)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{k}, {t}", fontsize=9, loc="left")
            ax.set_xlabel("n")
            if ai == 1:
                ax.set_ylabel("MFLOP/s (5 n log2 n; 2.5 for real)")
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Native x86-64: pocketfft vs FFTW 3.3.11 (higher is better)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(results, "chart_x86_fftw.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--out", default=None, help="markdown output (default results/report.md)")
    a = ap.parse_args()
    results = os.path.abspath(a.results)
    charts = not a.no_charts
    if charts:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib not available: writing tables only (pip install matplotlib for charts)", file=sys.stderr)
            charts = False
    parts = ["# pocketfft RVV: correctness and benchmark report\n"]
    info = os.path.join(results, "..", "build", "BUILD_INFO.txt")
    if os.path.exists(info):
        parts.append("```\n" + open(info).read().strip() + "\n```\n")
    corr, _ = correctness_section(results, charts)
    parts.append(corr)
    parts.append(x86_fftw_section(results, charts))
    parts.append(qemu_section(results, charts))
    parts.append(xisa_section(results, charts))
    out = a.out or os.path.join(results, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(p for p in parts if p))
    print(f"wrote {out}")
    if charts:
        for p in sorted(glob.glob(os.path.join(results, "chart_*.png"))):
            print(f"  {p}")


if __name__ == "__main__":
    main()
