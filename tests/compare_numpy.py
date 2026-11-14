#!/usr/bin/env python3
# compare_numpy.py <c2c|r2c|c2r> <float|double> <n> <fwd> <prefix> [prefix2]
# Checks dumped pocketfft output against numpy.fft (reference) and optionally against a second dump.
import sys, numpy as np
kind, typ, n, fwd, pre = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ft = np.float32 if typ == 'float' else np.float64
ct = np.complex64 if typ == 'float' else np.complex128
if kind == 'c2c':
    x = np.fromfile(pre + '.in', dtype=ct); y = np.fromfile(pre + '.out', dtype=ct)
    ref = (np.fft.fft if fwd else np.fft.ifft)(x.astype(np.complex128)) * (1 if fwd else n)
elif kind == 'r2c':
    x = np.fromfile(pre + '.in', dtype=ft); y = np.fromfile(pre + '.out', dtype=ct)
    ref = np.fft.rfft(x.astype(np.float64))
else:
    x = np.fromfile(pre + '.in', dtype=ct); y = np.fromfile(pre + '.out', dtype=ft)
    ref = np.fft.irfft(x.astype(np.complex128), n) * n
err = np.max(np.abs(y.astype(np.complex128) - ref)); scale = np.max(np.abs(ref))
l2 = np.linalg.norm(y.astype(np.complex128) - ref) / np.linalg.norm(ref)
line = f"{kind} {typ} n={n} fwd={fwd}: max_abs_err={err:.3e} rel_to_scale={err/scale:.3e} l2rel={l2:.3e}"
if len(sys.argv) > 6:
    y2 = np.fromfile(sys.argv[6] + '.out', dtype=(ft if kind == 'c2r' else ct))
    d = np.max(np.abs(y.astype(np.complex128) - y2.astype(np.complex128)))
    line += f" rvv_vs_scalar={d/scale:.3e}"
tol = 1e-5
print(line, "PASS" if err/scale <= tol else "FAIL")
sys.exit(0 if err/scale <= tol else 1)
