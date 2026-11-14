#!/usr/bin/env python3
"""summarize_correctness.py results/correctness_*.csv

Per-class summary of fft_check CSVs: for every configuration (label) and every class
(precision x kind x direction) the maximum of max|err|/max|ref| over all lengths (with the length where it
occurs), the maximum L2 relative error, the number of cases / failures, and the same restricted to the
Bluestein lengths (large primes).  Also used by make_report_tables.py.
"""
import csv
import math
import sys
from collections import OrderedDict

BLUESTEIN = {101, 257, 1009, 10007, 65537}   # lengths where pocketfft picks the Bluestein algorithm
KINDS = ["c2c", "r2c", "c2r"]
DIRS = ["fwd", "bwd"]
TYPES = ["double", "float"]


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def fmt(v):
    return "-" if v is None else f"{v:.2e}"


def per_class(rows):
    """returns OrderedDict[(type,kind,dir)] -> dict(cases, fails, worst, worst_n, l2, blue, blue_n, dump)"""
    out = OrderedDict()
    for t in TYPES:
        for k in KINDS:
            for d in DIRS:
                sel = [r for r in rows if r["type"] == t and r["kind"] == k and r["dir"] == d]
                if not sel:
                    continue
                e = [(f(r["max_err_norm"]), int(r["n"]), r) for r in sel if f(r["max_err_norm"]) is not None and f(r["max_err_norm"]) >= 0]
                l2 = [f(r["l2_rel"]) for r in sel if f(r["l2_rel"]) is not None and f(r["l2_rel"]) >= 0]
                bl = [x for x in e if x[1] in BLUESTEIN]
                dmp = [f(r["vs_dump_norm"]) for r in sel if f(r["vs_dump_norm"]) is not None and f(r["vs_dump_norm"]) >= 0]
                w = max(e, key=lambda x: x[0]) if e else None
                wb = max(bl, key=lambda x: x[0]) if bl else None
                out[(t, k, d)] = dict(cases=len(sel), fails=sum(1 for r in sel if r["status"] == "FAIL"),
                                      worst=w[0] if w else None, worst_n=w[1] if w else None,
                                      l2=max(l2) if l2 else None,
                                      blue=wb[0] if wb else None, blue_n=wb[1] if wb else None,
                                      dump=max(dmp) if dmp else None)
    return out


def md_per_class(rows, label=None):
    pc = per_class(rows)
    has_dump = any(v["dump"] is not None for v in pc.values())
    hdr = ["precision", "kind", "dir", "cases", "fail", "max of max|err|/max|ref|", "at n", "Bluestein primes", "at n", "max L2 rel"]
    if has_dump:
        hdr.append("max vs scalar")
    lines = []
    if label:
        lines.append(f"**{label}**\n")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|:---|:---|:---|" + "---:|" * (len(hdr) - 3))
    for (t, k, d), v in pc.items():
        row = [t, k, d, v["cases"], v["fails"], fmt(v["worst"]), v["worst_n"] or "-", fmt(v["blue"]), v["blue_n"] or "-", fmt(v["l2"])]
        if has_dump:
            row.append(fmt(v["dump"]))
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    # overall per precision
    for t in TYPES:
        vals = [v for (tt, _, _), v in pc.items() if tt == t and v["worst"] is not None]
        if vals:
            w = max(vals, key=lambda v: v["worst"])
            lines.append("| **%s, all** | | | %d | %d | **%s** | %s | | | %s |%s" % (
                t, sum(v["cases"] for v in vals), sum(v["fails"] for v in vals), fmt(w["worst"]), w["worst_n"],
                fmt(max(v["l2"] for v in vals if v["l2"] is not None)),
                (" %s |" % fmt(max((v["dump"] for v in vals if v["dump"] is not None), default=None))) if has_dump else ""))
    return "\n".join(lines) + "\n"


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(2)
    for path in files:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"{path}: empty")
            continue
        print(md_per_class(rows, label=f"{rows[0]['label']} ({path}, {len(rows)} cases)"))


if __name__ == "__main__":
    main()
