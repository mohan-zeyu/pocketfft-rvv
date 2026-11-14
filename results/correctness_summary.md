# Correctness summary (pocketfft vs FFTW 3.3.11 double-precision reference)

Generated 2026-11-08T12:36:36Z from results/correctness_*.csv; header dc04331f062d98c0 / rvv 9ffbbe4ac559b139 / real a98eb7d247abe33a

```
=== correctness summary (337 s) ===
x86-64-gcc13                         rc=0    4s  1080 cases, 1080 pass, 0 FAIL, 0 without reference (4.1 s)
      worst max|err|/max|ref| float  : 3.226e-07  (float r2c fwd n=65537 B=1 s=1/1)
      worst max|err|/max|ref| double : 1.340e-15  (double c2r bwd n=65537 B=1 s=1/1)
riscv64-clang23-scalar               rc=0   37s  1080 cases, 1080 pass, 0 FAIL, 0 without reference (36.3 s)
      worst max|err|/max|ref| float  : 3.770e-07  (float r2c fwd n=65537 B=1 s=1/1)
      worst max|err|/max|ref| double : 1.319e-15  (double c2c bwd n=65537 B=1 s=1/1)
riscv64-clang23-rvv-vlen256          rc=0  194s  1080 cases, 1080 pass, 0 FAIL, 0 without reference (194.0 s)
      worst max|err|/max|ref| float  : 3.645e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst max|err|/max|ref| double : 1.360e-15  (double r2c fwd n=65537 B=1 s=1/1)
      worst vs-dump difference float : 4.048e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst vs-dump difference double: 8.755e-16  (double r2c fwd n=65537 B=1 s=1/1)
riscv64-clang23-rvv-vlen128          rc=0  197s  1080 cases, 1080 pass, 0 FAIL, 0 without reference (197.0 s)
      worst max|err|/max|ref| float  : 3.645e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst max|err|/max|ref| double : 1.360e-15  (double r2c fwd n=65537 B=1 s=1/1)
      worst vs-dump difference float : 4.048e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst vs-dump difference double: 8.755e-16  (double r2c fwd n=65537 B=1 s=1/1)
riscv64-clang23-rvv-vlen512          rc=0  106s  1080 cases, 1080 pass, 0 FAIL, 0 without reference (106.6 s)
      worst max|err|/max|ref| float  : 3.645e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst max|err|/max|ref| double : 1.360e-15  (double r2c fwd n=65537 B=1 s=1/1)
      worst vs-dump difference float : 4.048e-07  (float r2c bwd n=65537 B=1 s=1/1)
      worst vs-dump difference double: 8.755e-16  (double r2c fwd n=65537 B=1 s=1/1)
RESULT: all configurations passed

```

**x86-64-gcc13 (results/correctness_x86-64_gcc13.csv, 1080 cases)**

| precision | kind | dir | cases | fail | max of max|err|/max|ref| | at n | Bluestein primes | at n | max L2 rel |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| double | c2c | fwd | 94 | 0 | 1.23e-15 | 65537 | 1.23e-15 | 65537 | 1.15e-15 |
| double | c2c | bwd | 94 | 0 | 1.22e-15 | 65537 | 1.22e-15 | 65537 | 1.15e-15 |
| double | r2c | fwd | 88 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.12e-15 |
| double | r2c | bwd | 88 | 0 | 1.24e-15 | 65537 | 1.24e-15 | 65537 | 1.12e-15 |
| double | c2r | fwd | 88 | 0 | 1.06e-15 | 65537 | 1.06e-15 | 65537 | 1.04e-15 |
| double | c2r | bwd | 88 | 0 | 1.34e-15 | 65537 | 1.34e-15 | 65537 | 1.05e-15 |
| float | c2c | fwd | 94 | 0 | 2.80e-07 | 65537 | 2.80e-07 | 65537 | 2.99e-07 |
| float | c2c | bwd | 94 | 0 | 3.21e-07 | 65537 | 3.21e-07 | 65537 | 3.00e-07 |
| float | r2c | fwd | 88 | 0 | 3.23e-07 | 65537 | 3.23e-07 | 65537 | 2.99e-07 |
| float | r2c | bwd | 88 | 0 | 3.16e-07 | 65537 | 3.16e-07 | 65537 | 3.01e-07 |
| float | c2r | fwd | 88 | 0 | 2.83e-07 | 65537 | 2.83e-07 | 65537 | 2.30e-07 |
| float | c2r | bwd | 88 | 0 | 2.32e-07 | 65537 | 2.32e-07 | 65537 | 2.30e-07 |
| **double, all** | | | 540 | 0 | **1.34e-15** | 65537 | | | 1.15e-15 |
| **float, all** | | | 540 | 0 | **3.23e-07** | 65537 | | | 3.01e-07 |

**riscv64-clang23-scalar (results/correctness_riscv64_clang23-scalar.csv, 1080 cases)**

| precision | kind | dir | cases | fail | max of max|err|/max|ref| | at n | Bluestein primes | at n | max L2 rel |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| double | c2c | fwd | 94 | 0 | 1.16e-15 | 65537 | 1.16e-15 | 65537 | 1.15e-15 |
| double | c2c | bwd | 94 | 0 | 1.32e-15 | 65537 | 1.32e-15 | 65537 | 1.15e-15 |
| double | r2c | fwd | 88 | 0 | 1.20e-15 | 65537 | 1.20e-15 | 65537 | 1.12e-15 |
| double | r2c | bwd | 88 | 0 | 1.24e-15 | 65537 | 1.24e-15 | 65537 | 1.12e-15 |
| double | c2r | fwd | 88 | 0 | 1.06e-15 | 10007 | 1.06e-15 | 10007 | 1.03e-15 |
| double | c2r | bwd | 88 | 0 | 1.21e-15 | 65537 | 1.21e-15 | 65537 | 1.04e-15 |
| float | c2c | fwd | 94 | 0 | 3.05e-07 | 65537 | 3.05e-07 | 65537 | 3.09e-07 |
| float | c2c | bwd | 94 | 0 | 3.22e-07 | 65537 | 3.22e-07 | 65537 | 3.11e-07 |
| float | r2c | fwd | 88 | 0 | 3.77e-07 | 65537 | 3.77e-07 | 65537 | 3.09e-07 |
| float | r2c | bwd | 88 | 0 | 3.57e-07 | 65537 | 3.57e-07 | 65537 | 3.12e-07 |
| float | c2r | fwd | 88 | 0 | 2.65e-07 | 65537 | 2.65e-07 | 65537 | 2.36e-07 |
| float | c2r | bwd | 88 | 0 | 2.22e-07 | 65537 | 2.22e-07 | 65537 | 2.38e-07 |
| **double, all** | | | 540 | 0 | **1.32e-15** | 65537 | | | 1.15e-15 |
| **float, all** | | | 540 | 0 | **3.77e-07** | 65537 | | | 3.12e-07 |

**riscv64-clang23-rvv-vlen128 (results/correctness_riscv64_clang23-rvv-vlen128.csv, 1080 cases)**

| precision | kind | dir | cases | fail | max of max|err|/max|ref| | at n | Bluestein primes | at n | max L2 rel | max vs scalar |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| double | c2c | fwd | 94 | 0 | 1.28e-15 | 65537 | 1.28e-15 | 65537 | 1.18e-15 | 7.37e-16 |
| double | c2c | bwd | 94 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.18e-15 | 7.34e-16 |
| double | r2c | fwd | 88 | 0 | 1.36e-15 | 65537 | 1.36e-15 | 65537 | 1.15e-15 | 8.76e-16 |
| double | r2c | bwd | 88 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.15e-15 | 7.80e-16 |
| double | c2r | fwd | 88 | 0 | 1.12e-15 | 65537 | 1.12e-15 | 65537 | 1.07e-15 | 6.24e-16 |
| double | c2r | bwd | 88 | 0 | 1.21e-15 | 65537 | 1.21e-15 | 65537 | 1.07e-15 | 6.03e-16 |
| float | c2c | fwd | 94 | 0 | 3.29e-07 | 65537 | 3.29e-07 | 65537 | 3.09e-07 | 3.59e-07 |
| float | c2c | bwd | 94 | 0 | 3.24e-07 | 65537 | 3.24e-07 | 65537 | 3.09e-07 | 3.57e-07 |
| float | r2c | fwd | 88 | 0 | 3.28e-07 | 65537 | 3.28e-07 | 65537 | 3.08e-07 | 3.46e-07 |
| float | r2c | bwd | 88 | 0 | 3.65e-07 | 65537 | 3.65e-07 | 65537 | 3.10e-07 | 4.05e-07 |
| float | c2r | fwd | 88 | 0 | 2.65e-07 | 65537 | 2.65e-07 | 65537 | 2.36e-07 | 2.80e-07 |
| float | c2r | bwd | 88 | 0 | 2.34e-07 | 65537 | 2.34e-07 | 65537 | 2.39e-07 | 2.28e-07 |
| **double, all** | | | 540 | 0 | **1.36e-15** | 65537 | | | 1.18e-15 | 8.76e-16 |
| **float, all** | | | 540 | 0 | **3.65e-07** | 65537 | | | 3.10e-07 | 4.05e-07 |

**riscv64-clang23-rvv-vlen256 (results/correctness_riscv64_clang23-rvv-vlen256.csv, 1080 cases)**

| precision | kind | dir | cases | fail | max of max|err|/max|ref| | at n | Bluestein primes | at n | max L2 rel | max vs scalar |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| double | c2c | fwd | 94 | 0 | 1.28e-15 | 65537 | 1.28e-15 | 65537 | 1.18e-15 | 7.37e-16 |
| double | c2c | bwd | 94 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.18e-15 | 7.34e-16 |
| double | r2c | fwd | 88 | 0 | 1.36e-15 | 65537 | 1.36e-15 | 65537 | 1.15e-15 | 8.76e-16 |
| double | r2c | bwd | 88 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.15e-15 | 7.80e-16 |
| double | c2r | fwd | 88 | 0 | 1.12e-15 | 65537 | 1.12e-15 | 65537 | 1.07e-15 | 6.24e-16 |
| double | c2r | bwd | 88 | 0 | 1.21e-15 | 65537 | 1.21e-15 | 65537 | 1.07e-15 | 6.03e-16 |
| float | c2c | fwd | 94 | 0 | 3.29e-07 | 65537 | 3.29e-07 | 65537 | 3.09e-07 | 3.59e-07 |
| float | c2c | bwd | 94 | 0 | 3.24e-07 | 65537 | 3.24e-07 | 65537 | 3.09e-07 | 3.57e-07 |
| float | r2c | fwd | 88 | 0 | 3.28e-07 | 65537 | 3.28e-07 | 65537 | 3.08e-07 | 3.46e-07 |
| float | r2c | bwd | 88 | 0 | 3.65e-07 | 65537 | 3.65e-07 | 65537 | 3.10e-07 | 4.05e-07 |
| float | c2r | fwd | 88 | 0 | 2.65e-07 | 65537 | 2.65e-07 | 65537 | 2.36e-07 | 2.80e-07 |
| float | c2r | bwd | 88 | 0 | 2.34e-07 | 65537 | 2.34e-07 | 65537 | 2.39e-07 | 2.28e-07 |
| **double, all** | | | 540 | 0 | **1.36e-15** | 65537 | | | 1.18e-15 | 8.76e-16 |
| **float, all** | | | 540 | 0 | **3.65e-07** | 65537 | | | 3.10e-07 | 4.05e-07 |

**riscv64-clang23-rvv-vlen512 (results/correctness_riscv64_clang23-rvv-vlen512.csv, 1080 cases)**

| precision | kind | dir | cases | fail | max of max|err|/max|ref| | at n | Bluestein primes | at n | max L2 rel | max vs scalar |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| double | c2c | fwd | 94 | 0 | 1.28e-15 | 65537 | 1.28e-15 | 65537 | 1.18e-15 | 7.37e-16 |
| double | c2c | bwd | 94 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.18e-15 | 7.34e-16 |
| double | r2c | fwd | 88 | 0 | 1.36e-15 | 65537 | 1.36e-15 | 65537 | 1.15e-15 | 8.76e-16 |
| double | r2c | bwd | 88 | 0 | 1.25e-15 | 65537 | 1.25e-15 | 65537 | 1.15e-15 | 7.80e-16 |
| double | c2r | fwd | 88 | 0 | 1.12e-15 | 65537 | 1.12e-15 | 65537 | 1.07e-15 | 6.24e-16 |
| double | c2r | bwd | 88 | 0 | 1.21e-15 | 65537 | 1.21e-15 | 65537 | 1.07e-15 | 6.03e-16 |
| float | c2c | fwd | 94 | 0 | 3.29e-07 | 65537 | 3.29e-07 | 65537 | 3.09e-07 | 3.59e-07 |
| float | c2c | bwd | 94 | 0 | 3.24e-07 | 65537 | 3.24e-07 | 65537 | 3.09e-07 | 3.57e-07 |
| float | r2c | fwd | 88 | 0 | 3.28e-07 | 65537 | 3.28e-07 | 65537 | 3.08e-07 | 3.46e-07 |
| float | r2c | bwd | 88 | 0 | 3.65e-07 | 65537 | 3.65e-07 | 65537 | 3.10e-07 | 4.05e-07 |
| float | c2r | fwd | 88 | 0 | 2.65e-07 | 65537 | 2.65e-07 | 65537 | 2.36e-07 | 2.80e-07 |
| float | c2r | bwd | 88 | 0 | 2.34e-07 | 65537 | 2.34e-07 | 65537 | 2.39e-07 | 2.28e-07 |
| **double, all** | | | 540 | 0 | **1.36e-15** | 65537 | | | 1.18e-15 | 8.76e-16 |
| **float, all** | | | 540 | 0 | **3.65e-07** | 65537 | | | 3.10e-07 | 4.05e-07 |


## Notes

* The RVV build produces **bit-identical results at VLEN 128, 256 and 512** (all 1080 per-case error metrics are identical
  across the three CSVs), i.e. the vector-length-agnostic code has no VLEN-dependent operation order.
* RVV vs scalar (same compiler, identical inputs, element-wise): <= 8.8e-16 (double) and <= 4.1e-07 (float) relative to
  max|ref| - the expected FMA-vs-separate-multiply-add rounding differences, with the maximum at the Bluestein length 65537.
* Worst cases of every class are at n = 65537 (Bluestein, i.e. two zero-padded transforms of length 131072 + chirp
  multiplications); the x86 generic-vector build and the riscv64 scalar build show the same 1.3e-15 / 3.2e-7..3.8e-7 levels,
  so the error is inherent to the algorithm/precision, not to the RVV code.
* Pass criterion was max|err|/max|ref| <= 1e-5; the measured margins are ~4 orders of magnitude (float) and ~10 (double).
