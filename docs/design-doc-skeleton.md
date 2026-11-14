# pocketfft-RVV: Solution design document (skeleton)

> Skeleton prepared 2026-11-08. Section headers follow what the competition's "solution design document" is expected
> to contain; bullets are prompts for the orchestrator. Supporting material: `docs/integration.md` (call chains,
> adoption path), `docs/hardware-run-guide.md` (measurement protocol), `docs/related-work.md` (prior art with links),
> `/home/user/research/survey/fft.md` (survey evidence). Mark every number with its provenance (QEMU trend vs. hardware).

## 0. Front matter
- Project name, team, date, version of this document; one-paragraph abstract (what, for whom, how much faster, how verified).
- Repository layout (`pocketfft_hdronly.h`, `pocketfft_rvv.h`, `bench/`, `tests/`, `scripts/`, `results/`), licence (BSD-3-Clause, same as pocketfft).

## 1. Background and problem statement
- Who computes FFTs on RISC-V servers through pocketfft: NumPy (`numpy.fft`), PyTorch CPU (`torch.fft`, all riscv64 builds because MKL is absent), ITK; SciPy via ducc0 (same algorithm family). Cite the call chains in `integration.md` §2-5.
- Status quo on RISC-V (link `related-work.md`): scalar 1-D transforms everywhere; upstream PR #31 vectorises only batched lines, is VLEN-specific and ignored by GCC 13; FFTW RVV ports unmerged; distro packages scalar.
- Problem statement in one sentence: "a VLEN-agnostic, runtime-dispatched RVV 1.0 backend for pocketfft that accelerates single 1-D and batched c2c/r2c/c2r transforms without changing results beyond rounding, consumable by NumPy/PyTorch/ITK through a header update".
- Goals / non-goals (e.g. no API change; no multithreading changes; DCT/DST and Hartley reuse the real/complex kernels or stay scalar; RVV 0.7.1 out of scope or best-effort via `_xtheadvector`).
- Success criteria: the competition's >= 10% bar, our targets per transform class and VLEN (fill from measurements), bit-level agreement policy, portability (one binary for VLEN 128/256/512/1024, graceful fallback without V).

## 2. Architecture
- Diagram: public API (`c2c`/`r2c`/`c2r`) -> `general_nd`/`general_r2c`/`general_c2r` -> plan (`pocketfft_c`/`pocketfft_r`: `cfftp`/`rfftp`/`fftblue`) -> passes; where `pocketfft_rvv.h` hooks in (scalar-T instantiations of `pass_all`/`rfftp::exec`, `copy_input`/`copy_output`, Bluestein multiplies, batched lines).
- `pocketfft_rvv.h` layering: traits per (type, LMUL) wrapping intrinsics; VL-agnostic loops with `vsetvl`; `enabled()` runtime check (`AT_HWCAP` bit 21 / hwprobe); compile-time gates (`__riscv_v_intrinsic >= 12000`, `POCKETFFT_NO_RVV`, interaction with `POCKETFFT_NO_VECTORS` and `VLEN<T>`).
- Build modes: (a) `-march=rv64gcv` whole-program; (b) rv64gc baseline + `target("arch=+v")` kernels + runtime dispatch (GCC 14+/clang 19+); (c) upstream generic-vector path kept/disabled - state the decision and why (`integration.md` §6).
- Data layout decisions: interleaved complex in memory, split re/im in registers via `vlseg2e`/`vsseg2e`; twiddle storage (precomputed per pass, layout friendly to unit-stride loads); temporary buffers (`arr<T> ch`) and in-place vs out-of-place passes; alignment assumptions (misaligned vector access policy from hwprobe).
- Threading: unchanged (`thread_count`); note NumPy builds with `POCKETFFT_NO_MULTITHREADING` and PyTorch passes `nthreads=1`.

## 3. Optimisation strategy
- 1-D complex passes (radix 2/3/4/5/7/8/11, generic `passg`): vectorise across butterflies (the `ido` or `l1` loop), VL-agnostic strip-mining; per-radix register budget vs 32 vregs; LMUL choice per VLEN (LMUL=1 on VLEN 256, LMUL=2 on VLEN 128?); handling of `ido==1` (no twiddles) vs twiddled stages; tail handling without scalar fallback.
- Twiddle multiplication as 4 FMAs (`vfmul`/`vfnmsac`/`vfmacc`), conjugation by sign-injection (`vfsgnjn`), no `vrgather`; special constants (radix-3/5 rotations) as `.vf` scalar operands.
- Real transforms (`radf*`/`radb*`): where the halfcomplex (FFTPACK) packing forces strided or reversed access; use of `vlsseg2`/negative strides vs copy; the r2c/c2r repacking loops in `general_r2c`/`general_c2r` and NumPy's own `copy_input`.
- Memory-access strategy: avoid `vlse` (4 cycles on X60) except for true transposes; prefetching (`Zicbop`) for large N; cache blocking for N beyond L2 (SG2044 2 MB/cluster, K1 L2 size); when to fall back to the scalar code (tiny N, odd factors).
- Batched/multi-D path: keep `cmplx<vtype_t<T>>` lines (compiler-dependent) or route through the intrinsics kernels; decision matrix by compiler/VLEN; expected gains vs. PR #31 numbers.
- Bluestein (`fftblue`): vectorised chirp multiply and the embedded power-of-two FFT; `good_size` interplay.
- Plan construction cost (`POCKETFFT_CACHE_SIZE` is 0 in NumPy/PyTorch): twiddle computation vectorisation or layout changes, and its effect on small-N benchmarks.
- Compiler-specific notes: GCC vs clang codegen differences observed, `-ffp-contract`, scheduling flags, `-mcpu` tuning; what was tried and rejected (keep a table).

## 4. Verification process
- Reference oracles: scalar pocketfft (same build flags), naive DFT in long double for small N, FFTW double (x86 host) for large N; error metric = max relative L2 error, thresholds (double ~1e-15, float ~1e-6) and justification vs. the competition's 1e-5 wording.
- Test matrix: types {float, double}, kinds {c2c fwd/bwd, r2c, c2r, (r2r/dct/dst if touched)}, N = powers of two, 2^a3^b5^c7^d11^e mixes, primes (Bluestein), N=1..64 exhaustively, large N (2^20+), batches {1, VLEN-1, VLEN, VLEN+1, large}, strides (contiguous, strided, negative/transposed axes), in-place vs out-of-place, odd alignments.
- VLEN sweep under QEMU: 128/256/512/1024 with the same binary; `-cpu rv64,v=false` to prove the dispatch fallback; sanitizers (ASan/UBSan) on QEMU; valgrind not available for RVV - say so.
- Downstream suites: NumPy `numpy/fft/tests/test_pocketfft.py`, PyTorch `test/test_spectral_ops.py`, ITK FFT module tests, SciPy (if ducc0 port exists) - how they were run (native or QEMU system-mode) and results.
- Reproducibility: fixed seeds, exact compiler versions and flags, CI script paths in `scripts/`/`tests/`.

## 5. Performance evaluation method
- Metrics: ns per transform (median of >= 5 launches, >= 0.5 s each), GFLOP/s by the `5 N log2 N` (c2c) / `2.5 N log2 N` (r2c/c2r) convention, speedup vs scalar baseline, cycles and instructions per point (perf), flop/cycle vs core peak.
- Three-tier protocol: (1) **QEMU trend** (dynamic instruction counts, vector share, per-VLEN behaviour; explicitly not timing); (2) **real RISC-V hardware** (K1/X60 VLEN 256, SG2044 VLEN 128 and/or K3 X100; procedure, pinning, governor, counters as in `hardware-run-guide.md`); (3) **x86 and ARM counterparts** (same harness, `-O3 -march=native`, per-core efficiency comparison, upstream pocketfft as shipped on each ISA, optional FFTW/MKL context).
- Baselines to include: scalar pocketfft (`POCKETFFT_NO_RVV`), upstream PR #31 generic-vector build (`-mrvv-vector-bits=zvl`), optionally PFFFT RVV and FFTW `--enable-generic-simd256` for context.
- Workloads: micro (single 1-D sizes), batched (NumPy-style `[B, N]` with B around VLEN), 2-D (`fft2` 512x512, 1024x1024), end-to-end NumPy/PyTorch calls (`np.fft.rfft` on audio-sized frames, `torch.stft`-like batches) measured through Python to show user-visible gains.
- Tables/plots to produce: speedup vs N per kind/type/VLEN; cycles per point; roofline-style flop/cycle; before/after per optimisation step (the "optimisation trend"); cross-ISA bar chart.

## 6. Results (to be filled)
- QEMU trend table; hardware tables per machine; cross-ISA table; downstream (NumPy/PyTorch) end-to-end table; correctness summary; analysis of where gains come from and where they stop (memory-bound regime, small N, Bluestein).

## 7. Risks and mitigations
- Hardware access/time (boards, SG2044 availability), firmware without PMU counters, thermal throttling -> protocol in `hardware-run-guide.md`.
- Compiler issues: GCC 14 target-attribute ICEs (PR115554/115562), clang shuffle lowering, `vector_size` behaviour under scalable mode, `-ffp-contract` differences -> pin versions, keep intrinsics path independent of generic vectors.
- Numerical differences vs scalar (FMA contraction order) -> documented tolerance, tests.
- Upstream acceptance (maintainer prefers portable code; header-only constraints; `POCKETFFT_NAMESPACE`/`POCKETFFT_NO_EXCEPTIONS` compatibility) -> opt-in macro, zero impact on other ISAs, QEMU CI.
- Register pressure / spills on high radices, VLEN 128 vs 256 tuning conflicts, RVV 0.7.1 parts (SG2042/TH1520) excluded.
- Benchmark validity: QEMU not timing, VLEN-specific binaries, background load -> three-tier protocol.

## 8. Upstreaming and adoption plan
- Timeline: PR to `mreineck/pocketfft` (cpp branch) with tests + QEMU CI; NumPy submodule bump (`numpy/fft/pocketfft`, 33ae5dc -> new) and PyTorch `third_party/pocketfft` bump (0fa0ef5 -> new) once released; ITK `UpdateFromUpstream.sh`; ducc0/SciPy follow-up; distro notes (Ubuntu RVA23 default vs Debian rv64gc + runtime dispatch). Details in `integration.md` §6.
- Maintenance: who owns the RVV header, CI matrix, how new VLENs/cores are added, policy for `-mcpu` tuning tables.
- Community communication: FFTW issue #371 / PR #377 cross-reference, RISC-V software ecosystem channels.

## 9. Appendices
- A. Exact build commands and environment dumps (`env.txt` from the run script).
- B. Raw CSVs and perf outputs (`results/`), plotting scripts.
- C. Instruction histograms per kernel (static `llvm-objdump`, dynamic QEMU plugin).
- D. Glossary (VLEN, LMUL, SEW, segment loads, halfcomplex order).
- E. References = `docs/related-work.md`.
