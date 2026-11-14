# Interface adaptation: how NumPy, PyTorch, ITK (and SciPy/ducc0) reach pocketfft, and how an RVV backend gets adopted

Status: 2026-11-08. All quotes were taken from the `main` branches on that date (NumPy `80049a0` 2026-11-07,
PyTorch `cba4a42` 2026-11-08, ITK `468408e` 2026-11-07, pocketfft `cpp` HEAD `4d865cb` 2026-10-18). Line numbers
for pocketfft refer to `/home/user/fft-rvv/pocketfft_hdronly.h`, which is byte-identical to upstream `4d865cb`
(checked with `diff` against `/home/user/research/src/pocketfft-cpp/pocketfft_hdronly.h`). **[V]** = verified
against the fetched source / our experiment, **[L]** = likely, **[U]** = unverified.

## 1. Snapshot

| Consumer | Glue file | Vendored pocketfft | Entry points used | Element types | Threads | Reaches our 1-D RVV passes? |
|---|---|---|---|---|---|---|
| **NumPy** `numpy.fft` | `numpy/fft/_pocketfft_umath.cpp` (gufuncs `fft`, `ifft`, `rfft_n_even`, `rfft_n_odd`, `irfft`) | git submodule `numpy/fft/pocketfft` @ **33ae5dc** (2024-03-22, "unconditionaly disable use of aligned_alloc") - **pre-RVV** | `pocketfft::c2c/r2c/c2r` for batches with `n_outer >= VLEN`; otherwise `detail::get_plan<detail::pocketfft_c<T>>` / `pocketfft_r<T>` + `plan->exec(...)` per row | `double`, `float`, `long double` | none (`#define POCKETFFT_NO_MULTITHREADING`) | **Yes**: every single 1-D call and every batch smaller than `VLEN` (and all padded calls) go straight to `pocketfft_c<T>::exec` -> `cfftp::pass_all` / `rfftp::exec` |
| **PyTorch** CPU `torch.fft` | `aten/src/ATen/native/mkl/SpectralOps.cpp` (`_fft_c2c_mkl`, `_fft_r2c_mkl`, `_fft_c2r_mkl`), active when `AT_POCKETFFT_ENABLED()` | git submodule `third_party/pocketfft` @ **0fa0ef5** (2024-11-30, "include cstdint header") - **pre-RVV** | `pocketfft::c2c/r2c/c2r` with the full tensor shape, byte strides and `axes = dim` | `float`, `double` (complex64/128); Half/BFloat16 rejected | `nthreads` argument **omitted -> 1** | **Yes**: 1-D tensors are one line -> scalar passes; `[B, N]` batches use the vector-line path for `floor(B/VLEN)*VLEN` lines and scalar passes for the rest |
| **ITK** FFT module | `Modules/Filtering/FFT/include/itkPocketFFT*.hxx` | copy at `Modules/ThirdParty/PocketFFT/src/itkpocketfft/pocketfft_hdronly.h`, upstream **c90e55b** (2026-06-30) - **already contains the RISC-V `VLEN` entry** (line 161 of the vendored file); namespace rewritten to `itk::detail::pocketfft` | `c2c`/`r2c`/`c2r` over all image dimensions, in place, `nthreads = GetMultiThreader()->GetMaximumNumberOfThreads()`; 1-D filters call per line | `float`, `double` | yes | **Yes** for the 1-D filters and the residual lines; N-D images mostly hit the vector-line path |
| **SciPy** `scipy.fft` | `scipy/fft/_duccfft/pyduccfft.cxx` -> **ducc0** (`ducc0::c2c/r2c/c2r`), not pocketfft | ducc0 (`duccfft_dep`), `src/ducc0/infra/simd.h` has no RVV path; `std::experimental::simd` is width 1 on RISC-V (our QEMU test) | - | `float`, `double`, `long double` | `workers` | No (different code base; §5 describes what a port takes) |

## 2. Inside pocketfft: the chain from the public API to the radix passes

Public entry points (`namespace pocketfft`, header lines 3649-3760, all with `size_t nthreads=1` as last parameter):

```cpp
3649 template<typename T> void c2c(const shape_t &shape, const stride_t &stride_in,
3650   const stride_t &stride_out, const shape_t &axes, bool forward,
3651   const std::complex<T> *data_in, std::complex<T> *data_out, T fct,
3652   size_t nthreads=1)
...
3658   general_nd<pocketfft_c<T>>(ain, aout, axes, fct, nthreads, ExecC2C{forward});
3697 template<typename T> void r2c(const shape_t &shape_in, ..., size_t axis, bool forward, const T *data_in,
     std::complex<T> *data_out, T fct, size_t nthreads=1)      // -> general_r2c (3708)
3711 template<typename T> void r2c(..., const shape_t &axes, ...) // r2c on axes.back(), then c2c on the rest (3718-3726)
3729 template<typename T> void c2r(..., size_t axis, ...)         // -> general_c2r (3740)
3743 template<typename T> void c2r(..., const shape_t &axes, ...) // c2c on leading axes into a temp, then c2r on the last
```

`shape_t = std::vector<size_t>`, `stride_t = std::vector<ptrdiff_t>` (lines 132-133); **strides are in bytes**.

**`general_nd` (3347-3385)** - the per-axis driver used by `c2c`, `dct`, `dst`, Hartley:

```cpp
3356     if ((!plan) || (len!=plan->length())) plan = get_plan<Tplan>(len);
3359     threading::thread_map(
3360       util::thread_count(nthreads, in.shape(), axes[iax], VLEN<T>::val),
3361       [&] {
3362         constexpr auto vlen = VLEN<T0>::val;
3363         auto storage = alloc_tmp<T0>(in.shape(), len, sizeof(T));
3365         multi_iter<vlen> it(tin, out, axes[iax]);
3366 #ifndef POCKETFFT_NO_VECTORS
3367         if (vlen>1)
3368           while (it.remaining()>=vlen)
3369             { it.advance(vlen);
3371             auto tdatav = reinterpret_cast<add_vec_t<T> *>(storage.data());
3372             exec(it, tin, out, tdatav, *plan, fct); }
3374 #endif
3375         while (it.remaining()>0)
3376           { it.advance(1);
3378           auto buf = allow_inplace && it.stride_out() == sizeof(T) ? &out[it.oofs(0)] : reinterpret_cast<T *>(storage.data());
3380           exec(it, tin, out, buf, *plan, fct); }
```

So for each transformed axis pocketfft walks all 1-D "lines" along that axis: groups of `VLEN<T0>::val` lines are
transposed into a `cmplx<vtype_t<T0>>` buffer (GCC `vector_size` type, lines 3250-3266) and transformed in lock-step
by the *same* pass code instantiated for the vector type; the remaining `remaining() mod VLEN` lines, and **all
lines when there is only one** (a 1-D transform), take the scalar branch. The functor **`ExecC2C` (3387-3399)** is
`copy_input(it, in, buf); plan.exec(buf, fct, forward); copy_output(it, buf, out);` - `copy_input/copy_output`
(3290-3345) are scalar gather/scatter loops over `it.iofs(j, i)` offsets.

**Plans**: `pocketfft_c<T0>` (2682-2713) holds either a `cfftp<T0>` (Cooley-Tukey, factors 8/4/2/3/5/7/11/generic)
or an `fftblue<T0>` (Bluestein, chosen when the largest prime factor squared exceeds the length and the cost model
says so, lines 2694-2706); `pocketfft_r<T0>` (2719-2750) likewise with `rfftp<T0>`. Their `exec` methods (2709,
2746) are the **only** place consumers' data enters the kernels:

```cpp
2709     template<typename T> POCKETFFT_NOINLINE void exec(cmplx<T> c[], T0 fct, bool fwd) const
2710       { packplan ? packplan->exec(c,fct,fwd) : blueplan->exec(c,fct,fwd); }
2746     template<typename T> POCKETFFT_NOINLINE void exec(T c[], T0 fct, bool fwd) const
2747       { packplan ? packplan->exec(c,fct,fwd) : blueplan->exec_r(c,fct,fwd); }
```

**`cfftp<T0>::exec` (1690) -> `pass_all<fwd>` (1641-1687)**: ping-pongs between `c` and a temporary `ch` over the
factor list, calling `pass4<fwd>(ido, l1, p1, p2, fact[k1].tw)`, `pass8`, `pass2`, `pass3`, `pass5`, `pass7`,
`pass11` or `passg` (1653-1671) with `ido = length/l2`; the final loop applies `fct`. Each pass is
`template<bool fwd, typename T> void passN(size_t ido, size_t l1, const T *cc, T *ch, const cmplx<T0> *wa)` with
`T = cmplx<T0>` for scalar lines or `T = cmplx<vtype_t<T0>>` for vector lines (e.g. `pass2` at 1061, `pass4` at
1147, `pass8` at 1359, `pass11` at 1472, `passg` at 1532). **`rfftp<T0>::exec(c, fct, r2hc)` (2446-2528)** does the
same with `radf2/3/4/5/g` (1792-2112) for r2hc and `radb*` (2113-2445) for hc2r, on FFTPACK halfcomplex order.
`general_r2c` (3502-3556) and `general_c2r` (3557-3628) contain their own vector/scalar line loops plus the
halfcomplex <-> complex repacking (`out[it.oofs(j,ii)].Set(tdatav[i][j], tdatav[i+1][j])`, 3522-3531).

Template parameters that matter for an RVV backend: the plan's scalar type `T0 in {float, double, long double}`
(twiddles `cmplx<T0>`), and the data type `T in {T0, vtype_t<T0>}`. `VLEN<long double>::val == 1`, so no vector
instantiation exists for long double; an RVV specialisation must likewise be restricted to float/double.

Other knobs: `VLEN` (159-182; the RISC-V entry `#elif (defined(__riscv_vector) || defined(__riscv_v_intrinsic))`
with 8/4 is from PR #31, commit 862d906), `POCKETFFT_NO_VECTORS` (139-157/180-182), `POCKETFFT_NO_MULTITHREADING`
(96, 697-714: `thread_count` returns 1), `POCKETFFT_PTHREADS` (105), `POCKETFFT_CACHE_SIZE` (60-61, default **0**:
`get_plan` at 2990-2999 returns `std::make_shared<T>(length)` on every call, i.e. **twiddles are recomputed per
call** in NumPy and PyTorch, which neither define the macro), `POCKETFFT_NAMESPACE` (PR #30, 2026-04-10) and
`POCKETFFT_NO_EXCEPTIONS` (4d865cb, 2026-10-18). `thread_count` divides the work by `shape[axis]*vlen` lines and by 4
for `shape[axis] < 1000` (702-713).

**Consequence for this project.** The intrinsics kernels must live in the scalar-`T` instantiations of
`pass_all`/`passN`, `rfftp::exec`/`radfN`/`radbN` and `fftblue` (that is where every single 1-D transform and every
residual line executes), optionally in `copy_input/copy_output` (scalar gathers today) and in the vector-line path
(either keep the generic-vector type or replace it by an intrinsics batched kernel). Nothing in NumPy, PyTorch or
ITK needs to change for the 1-D kernels to be exercised.

## 3. NumPy (`numpy/fft/_pocketfft_umath.cpp`, main @ 80049a0)

Build and configuration (`numpy/fft/meson.build` lines 6-17): the extension is compiled from the single file with
`dependencies: np_core_dep` and no SIMD flags; the submodule must exist (`pocketfft/README.md`). The source:

```cpp
23 #define POCKETFFT_NO_MULTITHREADING
24 #include "pocketfft/pocketfft_hdronly.h"
```

Complex transform loop (`fft_loop<T>`, 87-138), gufunc signature `(n),()->(m)` with `n_outer = dimensions[0]`,
`nin = dimensions[1]`, `nout = dimensions[2]`, strides `si, sf, so` (outer) and `step_in, step_out` (core):

```cpp
101 #ifndef POCKETFFT_NO_VECTORS
109     constexpr auto vlen = pocketfft::detail::VLEN<T>::val;
110     if (vlen > 1 && n_outer >= vlen && nin >= nout && sf == 0) {
111         std::vector<size_t> shape = { n_outer, nout };
112         std::vector<ptrdiff_t> strides_in = { si, step_in };
113         std::vector<ptrdiff_t> strides_out = { so, step_out};
114         std::vector<size_t> axes = { 1 };
115         pocketfft::c2c(shape, strides_in, strides_out, axes, direction,
116                        (std::complex<T> *)ip, (std::complex<T> *)op, *(T *)fp);
117         return;
118     }
119 #endif
124     auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(nout);
125     auto buffered = (step_out != sizeof(std::complex<T>));
126     pocketfft::detail::arr<std::complex<T>> buff(buffered ? nout : 0);
127     for (size_t i = 0; i < n_outer; i++, ip += si, fp += sf, op += so) {
128         std::complex<T> *op_or_buff = buffered ? buff.data() : (std::complex<T> *)op;
129         if (ip != (char*)op_or_buff) {
130             copy_input(ip, step_in, nin, op_or_buff, nout);
131         }
132         plan->exec((pocketfft::detail::cmplx<T> *)op_or_buff, *(T *)fp, direction);
133         if (buffered) {
134             copy_output(op_or_buff, op, step_out, nout);
135         }
136     }
```

Real transforms do the same: `rfft_impl<T>` (140-198) calls `pocketfft::r2c(shape_in={n_outer,npts}, ..., axes={1},
FORWARD, ...)` when `vlen > 1 && n_outer >= vlen && nin >= npts && sf == 0` (157-166) and otherwise
`get_plan<pocketfft_r<T>>(npts)` + `plan->exec(&((T *)op_or_buff)[1], *(T *)fp, pocketfft::FORWARD)` on a buffer
laid out in FFTPACK order (172-196); `irfft_loop<T>` (226-293) calls `pocketfft::c2r(shape_out={n_outer,nout}, ...,
BACKWARD, ...)` when `nin >= npts_in` (242-252), else `plan->exec(op_or_buff, *(T *)fp, pocketfft::BACKWARD)` (287).
Registered loops cover `npy_double`, `npy_float`, `npy_longdouble` (295-341). The Python side maps any `axis` to
the core dimension: `_pocketfft.py:101  return ufunc(a, fct, axes=[(axis,), (), (axis,)], out=out)`, and
`fftn`/`fft2` loop over axes one gufunc call at a time (`_raw_fftnd`, 718 ff.).

What this means on RISC-V **[V]**:

1. With the pinned header (33ae5dc) there is no RISC-V `VLEN` entry, so the header itself defines
   `POCKETFFT_NO_VECTORS` (lines 180-181 of today's header; same logic in 33ae5dc) and the whole `#ifndef
   POCKETFFT_NO_VECTORS` branch above is compiled out: **every NumPy FFT on RISC-V today is the per-row
   `plan->exec` path**, i.e. `pocketfft_c<T>::exec` -> `cfftp::pass_all` or `pocketfft_r<T>::exec` -> `rfftp::exec`.
2. After a header update, batched calls with `n_outer >= VLEN<T>::val` (8 floats / 4 doubles with PR #31's values),
   no padding (`nin >= nout`) and a broadcast factor (`sf == 0`, always true from `_raw_fft`) go through
   `pocketfft::c2c/r2c/c2r` -> `general_nd`/`general_r2c`/`general_c2r` and use the vector-line path for whole
   groups of `VLEN` rows and the scalar passes for the remainder. **Zero-padded calls (`n > a.shape[axis]`) and
   batches smaller than `VLEN` always use the per-row scalar path**, as do all 1-D calls (`n_outer == 1`).
   `n_outer` is whatever the ufunc machinery presents per inner-loop call (the full batch for contiguous inputs;
   smaller chunks for awkward layouts).
3. NumPy depends on these **internal** names: `pocketfft::detail::VLEN<T>::val` (must stay `constexpr`),
   `pocketfft::detail::get_plan<...>` returning something with `->exec(ptr, fct, bool)`,
   `pocketfft::detail::pocketfft_c<T>` / `pocketfft_r<T>`, `pocketfft::detail::arr<T>`, `pocketfft::detail::cmplx<T>`,
   `pocketfft::FORWARD/BACKWARD`, and the FFTPACK halfcomplex layout produced by `pocketfft_r<T>::exec`. The RVV
   header must preserve all of them, including the real-transform packing.
4. NumPy's own `copy_input`/`copy_output` (58-82) are scalar strided copies; for non-contiguous rows they sit in
   front of our kernels and cap the achievable speedup (a NumPy-side follow-up, not a pocketfft change).

## 4. PyTorch (`aten/src/ATen/native/mkl/SpectralOps.cpp`, main @ cba4a42)

When is pocketfft used? `cmake/Dependencies.cmake` 318-330:

```cmake
318 # --- [ PocketFFT
319 set(AT_POCKETFFT_ENABLED 0)
320 if(NOT AT_MKL_ENABLED)
321   set(POCKETFFT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/pocketfft/")
...
328   set(AT_POCKETFFT_ENABLED 1)
```

and `aten/src/ATen/Config.h.in:13  #define AT_POCKETFFT_ENABLED() @AT_POCKETFFT_ENABLED@`. MKL exists only for
x86, so **every riscv64 (and aarch64/ppc64le) CPU build has `AT_POCKETFFT_ENABLED() == 1`** and `#elif
AT_MKL_ENABLED()` (line 329) is never taken there **[V]**. PyTorch defines no `POCKETFFT_*` macro anywhere outside
`third_party` (grep of the tree): threading code is compiled in, the plan cache is off.

The glue (float and double only; `_mkl` is a historical name, these are the CPU kernels behind `aten::_fft_c2c`,
`_fft_r2c`, `_fft_c2r` **[L]** for the dispatch-key mapping, **[V]** for the code):

```cpp
201 #if AT_POCKETFFT_ENABLED()
202 #include <pocketfft_hdronly.h>
209 stride_t stride_from_tensor(const Tensor& t) {
210   stride_t stride(t.strides().begin(), t.strides().end());
211   for(auto& s: stride) {
212    s *= t.element_size();          // pocketfft wants byte strides
217 inline shape_t shape_from_tensor(const Tensor& t) {
218   return shape_t(t.sizes().begin(), t.sizes().end());
257 Tensor _fft_c2r_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
262   pocketfft::shape_t axes(dim.begin(), dim.end());
263   if (self.scalar_type() == kComplexFloat) {
264     pocketfft::c2r(shape_from_tensor(out), stride_from_tensor(self), stride_from_tensor(out), axes, false,
265                    tensor_cdata<float>(self),
266                    out.data_ptr<float>(), compute_fct<float>(out, dim, normalization));
276 Tensor _fft_r2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
288   if (self.scalar_type() == kFloat) {
289     pocketfft::r2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, true,
290                    self.const_data_ptr<float>(),
291                    tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
...
297     TORCH_CHECK_NOT_IMPLEMENTED(false, "real-to-complex FFT is not implemented for ", self.scalar_type(), " on CPU");
300   if (!onesided) { at::native::_fft_fill_with_conjugate_symmetry_(out, dim); }
306 Tensor _fft_c2c_mkl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
313   pocketfft::shape_t axes(dim.begin(), dim.end());
314   if (self.scalar_type() == kComplexFloat) {
315     pocketfft::c2c(shape_from_tensor(self), stride_from_tensor(self), stride_from_tensor(out), axes, forward,
316                    tensor_cdata<float>(self),
317                    tensor_cdata<float>(out), compute_fct<float>(self, dim, normalization));
```

Observations **[V]**:

* The **whole tensor** is handed to pocketfft (shape = all sizes, `axes = dim`), always out-of-place into a fresh
  `at::empty` tensor, with the normalisation folded into `fct` (`compute_fct`, 231-253). `torch.fft.fft(x)` on a 1-D
  tensor is `shape={N}, axes={0}`: one line -> `pass_all` on scalar `cmplx<T>`. `torch.fft.fft(X)` on `[B, N]`
  is `axes={1}` with `B` lines -> vector-line path for `floor(B/VLEN)*VLEN` lines, scalar passes for the rest (and
  for everything while the header is pre-RVV). `torch.fft.fft2`/`fftn` pass several axes; `general_nd` transforms
  them one after another over all lines, so a `[B,H,W]` `fft2` runs `B*W` lines along H and then `B*H` lines along W.
* `nthreads` is **not passed, so pocketfft runs single-threaded** regardless of `torch.get_num_threads()`; the
  per-line kernels are therefore the entire cost (and the natural place for a later PyTorch-side improvement).
* Only `pocketfft::c2c/r2c/c2r` are used - a header bump is sufficient; no internal names are referenced.
* Strided (non-contiguous) tensors are passed by stride, no copy, so `copy_input/copy_output` inside pocketfft see
  arbitrary strides.

## 5. ITK and SciPy/ducc0

**ITK** vendors pocketfft under `Modules/ThirdParty/PocketFFT/src/itkpocketfft/pocketfft_hdronly.h` (refreshed by
`UpdateFromUpstream.sh`, tag `for/itk-pocketfft-cpp-c90e55b` = upstream c90e55b, 2026-06-30, so it already carries
PR #31's RISC-V `VLEN` entry; namespace mangled through `itk_pocketfft.h.in` to `itk::detail::pocketfft`). The
filters call the public API over all image dimensions, in place and multi-threaded, e.g.
`itkPocketFFTForwardFFTImageFilter.hxx` 59-67:

```cpp
  itk::detail::pocketfft::c2c(shape, stride, stride, axes, itk::detail::pocketfft::FORWARD,
                              out, out, InputPixelType{ 1 },
                              this->GetMultiThreader()->GetMaximumNumberOfThreads());
```

(`GetSizeGreatestPrimeFactor()` returns 11: "sizes factoring into 2,3,5,7,11 use fast kernels"). The 1-D filters
(`itkPocketFFTForward1DFFTImageFilter.hxx`) use `shape{ lineLength }` per line - scalar passes today. ITK is the
only consumer that would already compile the generic-vector path on an RVA23 toolchain, with the VLEN-specific
behaviour described in `/home/user/research/survey/fft.md` **[V]**.

**SciPy** no longer uses pocketfft: `scipy/fft/_duccfft/pyduccfft.cxx` includes `ducc0/fft/fft.h` and
`ducc0/fft/fftnd_impl.h` (lines 17-18) and calls `ducc0::c2c(in, res, axes, forward, fct, nthreads)` (154),
`ducc0::r2c(...)` (173) etc.; the ducc0 sources arrive through the Meson dependency `duccfft_dep` (not a git
submodule; location inside the SciPy tree to be confirmed when porting) **[V]**. ducc0's SIMD layer is
`src/ducc0/infra/simd.h` (GitHub mirror `mreineck/ducc`, branch `ducc0`, fetched 2026-11-08, identical to the
survey copy): line 67 chooses `std::experimental::simd` whenever `__has_include(<experimental/simd>) &&
defined(__GLIBCXX__) && __GNUC__>=12`; otherwise the home-grown `template<typename T, size_t len> class vtp`
(line 239) with AVX-512/AVX/SSE2/SVE/NEON backends (446-853) and `vtp<T,1>` as fallback (855/860). On RISC-V the
`experimental::simd` path is taken and **`native_simd<float>::size() == native_simd<double>::size() == 1`** (our
QEMU run of a GCC 13.3 `-march=rv64gcv` binary; also 1 with clang 18 `-mrvv-vector-bits=zvl`; 8/4 on an AVX2 x86
host for comparison) - SciPy's FFT is scalar on RISC-V **[V]**. Adopting the pocketfft-RVV kernels there would take:
(1) a `#elif defined(__riscv_vector)` branch that bypasses the `experimental::simd` selection and defines
`DUCC0_USE_RVV`; (2) a `vtp<T,len>` specialisation over RVV (fixed `len` from `riscv_rvv_vector_bits`/VLEN, or
LMUL-sized intrinsics wrappers) implementing the arithmetic, FMA, abs/sqrt, blend/mask and reduction operators ducc0
uses - this alone vectorises ducc0's batched/multi-D lines, because its passes are templated on the vector type
exactly like pocketfft's; (3) porting the 1-D intrinsics passes into ducc0's `detail_fft` `cfftp`/`rfftp`
(same FFTPACK ancestry, different helper names); (4) upstreaming to ducc (GitLab MPCDF primary, GitHub mirror) and
asking SciPy to refresh its vendored copy. Estimated 2-3 weeks on top of the pocketfft work (survey C3).

## 6. Adoption path

### (a) Upstream PR to `mreineck/pocketfft`, branch `cpp`

* Activity in 2026 shows PRs are reviewed and merged within days (PR #27 meson 2026-01-08, #28 2026-02-26, #30
  namespace 2026-04-10, **#31 RVV 2026-05-01**, #32 2026-06-30, `POCKETFFT_NO_EXCEPTIONS` 2026-10-18) **[V]**. PR #31
  was accepted as a three-line `VLEN` entry; the maintainer "didn't expect any noticeable speedup ... on 1D", so the PR
  must come with hardware numbers for 1-D and batched transforms (K1/K3 VLEN 256, SG2044 VLEN 128).
* Shape of the contribution: a second header (`pocketfft_rvv.h`, or an `#include`-guarded section) enabled when
  `__riscv_vector && __riscv_v_intrinsic >= 12000` and not `POCKETFFT_NO_RVV`; zero change for other ISAs; preserves
  `VLEN`, `get_plan`, `pocketfft_c/r::exec`, halfcomplex layout, `POCKETFFT_NAMESPACE`, `POCKETFFT_NO_EXCEPTIONS`,
  `POCKETFFT_NO_MULTITHREADING`; results equal to the scalar path up to FMA-contraction rounding (document it; the
  scalar path compiled with `-O3` on RISC-V already contracts into FMAs under GCC's default `-ffp-contract=fast`).
* Tests: extend `pocketfft_demo.cc`/the meson test with the N and stride matrix from the design document and a QEMU
  job for VLEN 128/256/512/1024 and `v=false` (PFFFT's merged PR #110 is the precedent for QEMU CI).
* Separate, trivially acceptable fix: make the upstream RISC-V `VLEN` entry conditional on
  `__riscv_v_fixed_vlen` (defined only under `-mrvv-vector-bits=zvl`: clang 18 emits `#define __riscv_v_fixed_vlen 256`
  for `-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl` and nothing without the option; GCC 14 `gcc/config/riscv/riscv-c.cc:216`
  `builtin_define_with_int_value ("__riscv_v_fixed_vlen", TARGET_MIN_VLEN)` - both checked 2026-11-08 **[V]**), so a
  generic `-march=rv64gcv` build no longer produces a VLEN-specific binary by accident (see §6(d) and the crash
  evidence in the survey).

### (b) NumPy: submodule bump (and nothing else)

* `git submodule update --remote numpy/fft/pocketfft` to the merged commit, PR against `numpy/numpy` (precedent: the
  bump to 33ae5dc in 2024); NumPy's gufunc code needs no change - after the bump its `vlen > 1` branches become live on
  RISC-V and its per-row path inherits the 1-D kernels.
* Build flags: `_pocketfft_umath.cpp` is compiled with NumPy's baseline flags only (no `cpu-dispatch` features exist
  for RISC-V in `meson_cpu/`), so `__riscv_vector` is defined iff the **toolchain default `-march` contains V** -
  Ubuntu 25.10/26.04 (RVA23U64 baseline) yes, Debian/Fedora (rv64gc) no. This is why the header should carry its own
  `target("arch=+v")` + runtime dispatch (§6(d)): it makes the RVV path reach Debian-built NumPy without NumPy or
  packaging changes. Official riscv64 wheels: **[U]**; distro packages are the realistic channel.

### (c) PyTorch: `third_party/pocketfft` bump

* Submodule bump PR to `pytorch/pytorch`; `SpectralOps.cpp` is unchanged because only the public API is used.
  PyTorch has no riscv64 CI, so attach QEMU test logs and hardware numbers; x86/aarch64 CI proves the non-RISC-V path
  is untouched (the RVV header is a no-op there).
* Optional follow-ups in the same area: pass `at::get_num_threads()` as `nthreads` (today 1) and enable
  `POCKETFFT_CACHE_SIZE` (plans are rebuilt per call) - both independent of RVV but visible in `torch.fft` timings.
* riscv64 PyTorch is built from source by users/distros, so the same `-march` logic as for NumPy applies.

### (d) Enabling V in downstream builds

| Build environment | Default ISA | What happens with the RVV header | What is needed |
|---|---|---|---|
| Ubuntu 25.10 / **26.04 LTS** riscv64 | **RVA23U64** ("starting with Ubuntu 25.10, RVA23 became the minimum supported baseline"; 26.04 LTS "with RVA23 as our unified, long-term supported baseline" - [Canonical blog](https://ubuntu.com/blog/canonical-and-ubuntu-risc-v-a-2025-retro-and-looking-forward-to-2026), [Phoronix](https://www.phoronix.com/news/Ubuntu-25.10-To-Require-RVA23)) **[V]** | `__riscv_vector` defined by default -> RVV kernels compiled into `python3-numpy`, PyTorch, ITK with no packaging change | nothing (keep the runtime check anyway for safety) |
| Debian riscv64 (trixie/forky) | **RV64GC / lp64d** ("The Debian port uses RV64GC as the hardware baseline and the lp64d ABI", [Debian wiki](https://wiki.debian.org/RISC-V)) **[V]**; Fedora also rv64gc **[L]** | V not enabled -> header compiles scalar unless the kernels are built with target attributes or per-TU flags | runtime dispatch (below) |
| Custom builds (HPC sites, containers) | whatever `CXXFLAGS` say | `-march=rv64gcv` (generic, VLEN-agnostic), `-march=rv64gcv_zvl256b` (VLEN >= 256 only), `-march=rva23u64` (GCC 14+/clang 19+ **[L]**; GCC 13 and clang 18 reject it, tested) | document the flag used |

Mechanisms for the rv64gc case:

1. **Function target attributes + runtime check (recommended, header-only).** `__attribute__((target("arch=+v")))`
   on every function that touches RVV types/intrinsics, guarded by `pocketfft::detail::rvv::enabled()`
   (`getauxval(AT_HWCAP) & (1<<21)`, or `__riscv_hwprobe`/`RISCV_HWPROBE_IMA_V`, glibc >= 2.40 / Linux >= 6.4).
   Support: **GCC 14+** (release notes: "Supports the `target` attribute, which allows users to compile a function
   with specific extensions"; the `riscv_vector.h` `#error "Vector intrinsics require the vector extension."` guard
   present in GCC 13 is gone on the gcc-14 branch and the test `gcc.target/riscv/rvv/base/target_attribute_v_with_intrinsic-1.c`
   exists there - patch series "RISC-V: Allow RVV intrinsic when function target("arch=+v")", March/April 2024) and
   **clang 19+** ([llvm/llvm-project#83674](https://github.com/llvm/llvm-project/pull/83674), merged 2024-03-27).
   Verified negative on this machine: GCC 13.3 cross and clang 18 both stop at that `#error` for an `-march=rv64gc`
   TU with an attributed function **[V]**. Caveats: the attribute must be on each template/member function
   instantiated with vector types (no inheritance by callees), vector types must not leak into un-attributed inline
   code, GCC 14.1 had ICEs with target attributes in some patterns (PR115554/PR115562, reported from glibc work in
   June 2024 **[L]**) - test with the exact distro compiler; `#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to=function)`
   is a convenient clang spelling, GCC's `#pragma GCC target` equivalent on RISC-V **[U]**.
2. **Per-TU flags.** Compile only the TU that includes pocketfft with `-march=rv64gcv` and keep every call behind the
   runtime check. Works with GCC 13/clang 17, but the compiler may auto-vectorise *any* code in that TU, so the TU must
   contain nothing that executes before the check; for header-only pocketfft in NumPy/PyTorch this means the whole
   extension/ATen TU, which is why (1) is preferable.
3. **Function multi-versioning** (`target_clones("default","arch=+v")` / `target_version`): specified in
   [riscv-c-api-doc PR #48](https://github.com/riscv-non-isa/riscv-c-api-doc/pull/48), implemented in GCC 15 per the
   glibc discussion ([libc-alpha, 2025-01](https://gcc.gnu.org/pipermail/libc-alpha/2025-January/163630.html), which
   also reports an ifunc-resolver `gp` issue to watch) **[L]**; attractive later, not required now.

## 7. Where our kernels are reached - summary

| User call | pocketfft path | Instantiation | RVV 1-D kernels hit? |
|---|---|---|---|
| `np.fft.fft/ifft/rfft/irfft` on 1-D input, or with `n` > input length, or batch < `VLEN` | NumPy per-row `plan->exec` | `pocketfft_c<T>::exec` / `pocketfft_r<T>::exec` with scalar `cmplx<T>` / `T` | **yes, 100 %** |
| `np.fft.*` on `[B >= VLEN, N]` contiguous, no padding | `pocketfft::c2c/r2c/c2r` -> `general_*` | vector lines for `floor(B/VLEN)*VLEN` rows (`cmplx<vtype_t<T>>`), scalar for the rest | residual rows; batched path per design decision |
| `torch.fft.fft(x)` 1-D | `pocketfft::c2c(shape={N}, axes={0})` -> `general_nd` single line | scalar | **yes** |
| `torch.fft.fft(X)` `[B, N]`, `torch.stft`-style batches | `general_nd`, `B` lines | mixed as above | residual rows + batched decision |
| `torch.fft.fft2/fftn`, ITK N-D filters | `general_nd` per axis, many lines | mostly vector lines | residual rows; `copy_input/copy_output` gathers |
| `np.fft.rfft`/`torch.fft.rfft` | `rfftp::exec` (`radf*`) + halfcomplex repacking | scalar or vector | **yes** for 1-D; repacking loops are extra work |
| sizes with a large prime factor | `fftblue::exec/exec_r` | scalar or vector | only if Bluestein is vectorised too |

Surprising findings while checking the sources: NumPy `main` still pins 33ae5dc and its glue has not changed shape
(the `vlen`-gated batched branch and the per-row fallback are as in 2024); PyTorch calls pocketfft **single-threaded**
(no `nthreads`) and with **no plan cache**; ITK already ships a post-PR-#31 header; GCC 14/15 define
`__riscv_v_intrinsic` = 12000 although the GCC 14 notes speak of the "version 1.0" intrinsics spec (GCC 16+/clang 20+
define 1000000, so the `>= 12000` check in `pocketfft_rvv.h` is right for all of them); libstdc++'s
`std::experimental::simd` is width 1 on RISC-V, which silently leaves SciPy/ducc0 scalar.
