// fft_check.cpp - correctness oracle for pocketfft_hdronly.h (scalar and RVV code paths).
//
// Compares pocketfft::c2c / r2c / c2r (float and double, forward and backward, 1-D, batched 2-D,
// strided, in-place) against a reference transform computed in DOUBLE precision by FFTW3
// (compile with -DHAVE_FFTW and link -lfftw3).  Without FFTW a long-double naive O(n^2) DFT is
// used for n <= --naive-max; larger cases are reported as "noref" (not a failure).
//
// Inputs are deterministic: a 64-bit LCG seeded from (seed, type, kind, direction, n, batch,
// strides) in pure integer arithmetic, so the identical data is regenerated on any machine /
// compiler / ISA.  This makes --dump / --compare-scalar meaningful across builds:
//   scalar build:  fft_check --dump work/scalar.bin
//   RVV build:     fft_check --compare-scalar work/scalar.bin
// quantifies RVV-vs-scalar differences (expected ~1e-16 relative for double, ~1e-7 for float).
//
// Per case we report: max |err|, max |err| / max |ref| ("norm", the rubric metric, pass if <= tol,
// default 1e-5) and the L2 relative error ||err||/||ref||.  Exit status: 0 iff no case failed.
//
// Build (x86 native, FFTW at $F):
//   g++ -O3 -std=c++17 -I.. -DHAVE_FFTW -I$F/include fft_check.cpp -L$F/lib -lfftw3 -lm -o fft_check
// See scripts/build.sh for the cross builds and tests/README.md for usage.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cerrno>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>
#include <map>
#include <tuple>
#include <chrono>
#include <stdexcept>

#include "pocketfft_hdronly.h"

#ifdef HAVE_FFTW
#include <fftw3.h>
#endif

using pocketfft::shape_t;
using pocketfft::stride_t;

// ------------------------------------------------------------------------------------------------
// deterministic pseudo-random inputs
// ------------------------------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t &x)
  {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
  }

struct LCG
  {
  uint64_t s;
  explicit LCG(uint64_t seed) : s(seed) {}
  uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
  // uniform in [-1,1): 53 random bits scaled by a power of two -> bit-exact on every IEEE machine
  double uniform() { return double(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }
  };

// ------------------------------------------------------------------------------------------------
// case description
// ------------------------------------------------------------------------------------------------
enum Kind { C2C = 0, R2C = 1, C2R = 2 };
static const char *kind_name[3] = {"c2c", "r2c", "c2r"};

struct CaseSpec
  {
  bool is_double;
  Kind kind;
  bool fwd;
  size_t n;        // transform length
  size_t batch;    // number of rows (1 => plain 1-D transform; >1 => shape {batch,n}, axis 1)
  ptrdiff_t sin;   // input element stride along the transform axis (1 = contiguous)
  ptrdiff_t sout;  // output element stride
  bool inplace;    // c2c only: data_out == data_in
  };

static uint64_t case_seed(uint64_t base, const CaseSpec &c)
  {
  uint64_t x = base;
  splitmix64(x);
  x ^= uint64_t(c.is_double) << 1 | uint64_t(c.kind) << 2 | uint64_t(c.fwd) << 5;
  splitmix64(x);
  x ^= uint64_t(c.n);
  splitmix64(x);
  x ^= uint64_t(c.batch) << 8;
  splitmix64(x);
  x ^= uint64_t(c.sin) << 16 | uint64_t(c.sout) << 24 | uint64_t(c.inplace) << 32;
  return splitmix64(x);
  }

// ------------------------------------------------------------------------------------------------
// reference transforms (double precision)
// ------------------------------------------------------------------------------------------------
enum RefMode { REF_AUTO, REF_FFTW, REF_NAIVE, REF_NONE };

typedef std::complex<double> cd;

// naive O(n^2) DFT in long double (80-bit on x86, binary128 (software) on aarch64/riscv64)
struct NaiveTwiddle
  {
  size_t n = 0;
  std::vector<std::complex<long double>> w; // w[m] = exp(-2 pi i m / n)
  void init(size_t n_)
    {
    if (n == n_) return;
    n = n_;
    w.resize(n);
    const long double twopi = 6.283185307179586476925286766559005768L;
    for (size_t m = 0; m < n; ++m)
      {
      long double a = twopi * (long double)(m) / (long double)(n);
      w[m] = std::complex<long double>(std::cos(a), -std::sin(a));
      }
    }
  };

static void naive_c2c(NaiveTwiddle &tw, size_t n, const cd *in, cd *out, bool fwd, size_t kmax)
  {
  tw.init(n);
  for (size_t k = 0; k < kmax; ++k)
    {
    long double sr = 0, si = 0;
    size_t idx = 0;
    for (size_t j = 0; j < n; ++j)
      {
      // idx = (j*k) mod n, maintained incrementally (no overflow)
      std::complex<long double> ww = tw.w[idx];
      long double wr = ww.real(), wi = fwd ? ww.imag() : -ww.imag();
      long double xr = in[j].real(), xi = in[j].imag();
      sr += xr * wr - xi * wi;
      si += xr * wi + xi * wr;
      idx += k; if (idx >= n) idx -= n;
      }
    out[k] = cd(double(sr), double(si));
    }
  }

struct Reference
  {
  RefMode mode;
  size_t naive_max;
  NaiveTwiddle tw;
#ifdef HAVE_FFTW
  size_t fftw_n = 0; Kind fftw_kind = C2C; int fftw_sign = 0;
  fftw_plan plan = nullptr;
  double *rbuf = nullptr; fftw_complex *cbuf = nullptr, *cbuf2 = nullptr;
  void fftw_release()
    {
    if (plan) fftw_destroy_plan(plan);
    plan = nullptr;
    if (rbuf) fftw_free(rbuf); if (cbuf) fftw_free(cbuf); if (cbuf2) fftw_free(cbuf2);
    rbuf = nullptr; cbuf = cbuf2 = nullptr; fftw_n = 0;
    }
  void fftw_prepare(size_t n, Kind kind, int sign)
    {
    if (plan && fftw_n == n && fftw_kind == kind && fftw_sign == sign) return;
    fftw_release();
    fftw_n = n; fftw_kind = kind; fftw_sign = sign;
    size_t nc = n / 2 + 1;
    if (kind == C2C)
      {
      cbuf = fftw_alloc_complex(n); cbuf2 = fftw_alloc_complex(n);
      plan = fftw_plan_dft_1d(int(n), cbuf, cbuf2, sign, FFTW_ESTIMATE);
      }
    else if (kind == R2C)
      {
      rbuf = fftw_alloc_real(n); cbuf = fftw_alloc_complex(nc);
      plan = fftw_plan_dft_r2c_1d(int(n), rbuf, cbuf, FFTW_ESTIMATE);
      }
    else
      {
      cbuf = fftw_alloc_complex(nc); rbuf = fftw_alloc_real(n);
      plan = fftw_plan_dft_c2r_1d(int(n), cbuf, rbuf, FFTW_ESTIMATE);
      }
    if (!plan) { std::fprintf(stderr, "FFTW planning failed for n=%zu\n", n); std::exit(3); }
    }
  ~Reference() { fftw_release(); }
#endif
  bool available(size_t n) const
    {
    if (mode == REF_NONE) return false;
#ifdef HAVE_FFTW
    if (mode == REF_FFTW || mode == REF_AUTO) return true;
#else
    if (mode == REF_FFTW) return false;
#endif
    return n <= naive_max;
    }
  const char *name(size_t n) const
    {
    if (!available(n)) return "none";
#ifdef HAVE_FFTW
    if (mode == REF_FFTW || mode == REF_AUTO) return "fftw3";
#endif
    return "naive-ld";
    }
  bool use_fftw() const
    {
#ifdef HAVE_FFTW
    return mode == REF_FFTW || mode == REF_AUTO;
#else
    return false;
#endif
    }

  // c2c: out[k] = sum_j in[j] exp(-/+ 2 pi i jk/n), unscaled
  void c2c(size_t n, const cd *in, cd *out, bool fwd)
    {
#ifdef HAVE_FFTW
    if (use_fftw())
      {
      fftw_prepare(n, C2C, fwd ? FFTW_FORWARD : FFTW_BACKWARD);
      std::memcpy(cbuf, in, n * sizeof(cd));
      fftw_execute(plan);
      std::memcpy(out, cbuf2, n * sizeof(cd));
      return;
      }
#endif
    naive_c2c(tw, n, in, out, fwd, n);
    }
  // r2c: n reals -> n/2+1 complex, exponent sign -(fwd) or +(bwd) (pocketfft convention: bwd = conj(fwd))
  void r2c(size_t n, const double *in, cd *out, bool fwd)
    {
    size_t nc = n / 2 + 1;
#ifdef HAVE_FFTW
    if (use_fftw())
      {
      fftw_prepare(n, R2C, 0);
      std::memcpy(rbuf, in, n * sizeof(double));
      fftw_execute(plan);
      std::memcpy(out, cbuf, nc * sizeof(cd));
      }
    else
#endif
      {
      std::vector<cd> tmp(n);
      for (size_t j = 0; j < n; ++j) tmp[j] = cd(in[j], 0.);
      naive_c2c(tw, n, tmp.data(), out, true, nc);
      }
    if (!fwd) for (size_t k = 0; k < nc; ++k) out[k] = std::conj(out[k]);
    }
  // c2r: n/2+1 complex (Hermitian half) -> n reals.  pocketfft's "forward" c2r conjugates the
  // input first (exp(-) kernel); FFTW's c2r is the exp(+) kernel.
  void c2r(size_t n, const cd *in, double *out, bool fwd)
    {
    size_t nc = n / 2 + 1;
    std::vector<cd> tmp(in, in + nc);
    if (fwd) for (size_t k = 0; k < nc; ++k) tmp[k] = std::conj(tmp[k]);
#ifdef HAVE_FFTW
    if (use_fftw())
      {
      fftw_prepare(n, C2R, 0);
      std::memcpy(cbuf, tmp.data(), nc * sizeof(cd));
      fftw_execute(plan);
      std::memcpy(out, rbuf, n * sizeof(double));
      return;
      }
#endif
    // Hermitian extension + backward DFT, real part
    std::vector<cd> full(n), res(n);
    for (size_t k = 0; k < nc; ++k) full[k] = tmp[k];
    for (size_t k = nc; k < n; ++k) full[k] = std::conj(tmp[n - k]);
    full[0] = cd(full[0].real(), 0.);
    if ((n & 1) == 0) full[n / 2] = cd(full[n / 2].real(), 0.);
    naive_c2c(tw, n, full.data(), res.data(), false, n);
    for (size_t j = 0; j < n; ++j) out[j] = res[j].real();
    }
  };

// ------------------------------------------------------------------------------------------------
// dump file (outputs of one build, for --compare-scalar)
// ------------------------------------------------------------------------------------------------
struct DumpHeader
  {
  uint32_t is_double, kind, fwd, inplace;
  uint64_t n, batch;
  int64_t sin, sout;
  uint64_t nvals;
  };
static const char DUMP_MAGIC[8] = {'P', 'F', 'F', 'T', 'D', 'M', 'P', '1'};

typedef std::tuple<int, int, int, int, uint64_t, uint64_t, int64_t, int64_t> DumpKey;
static DumpKey dump_key(const CaseSpec &c)
  { return DumpKey(c.is_double, c.kind, c.fwd, c.inplace, c.n, c.batch, c.sin, c.sout); }

struct DumpReader
  {
  FILE *f = nullptr;
  std::map<DumpKey, std::pair<long, uint64_t>> index; // key -> (payload offset, nvals)
  bool open(const char *path)
    {
    f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open dump %s: %s\n", path, std::strerror(errno)); return false; }
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, DUMP_MAGIC, 8) != 0)
      { std::fprintf(stderr, "%s: bad dump magic\n", path); return false; }
    for (;;)
      {
      DumpHeader h;
      if (std::fread(&h, sizeof h, 1, f) != 1) break;
      long off = std::ftell(f);
      size_t esz = h.is_double ? 8 : 4;
      CaseSpec c{h.is_double != 0, Kind(h.kind), h.fwd != 0, h.n, h.batch, ptrdiff_t(h.sin), ptrdiff_t(h.sout), h.inplace != 0};
      index[dump_key(c)] = std::make_pair(off, h.nvals);
      if (std::fseek(f, long(h.nvals * esz), SEEK_CUR) != 0) break;
      }
    return true;
    }
  // reads the stored values converted to double; returns false if the case is not in the dump
  bool load(const CaseSpec &c, std::vector<double> &vals)
    {
    auto it = index.find(dump_key(c));
    if (it == index.end()) return false;
    std::fseek(f, it->second.first, SEEK_SET);
    uint64_t nv = it->second.second;
    vals.resize(nv);
    if (c.is_double)
      { if (std::fread(vals.data(), sizeof(double), nv, f) != nv) return false; }
    else
      {
      std::vector<float> tmp(nv);
      if (std::fread(tmp.data(), sizeof(float), nv, f) != nv) return false;
      for (uint64_t i = 0; i < nv; ++i) vals[i] = tmp[i];
      }
    return true;
    }
  ~DumpReader() { if (f) std::fclose(f); }
  };

// ------------------------------------------------------------------------------------------------
// metrics
// ------------------------------------------------------------------------------------------------
struct Metrics
  {
  double maxabs = 0, refmax = 0, sumerr2 = 0, sumref2 = 0;
  void add(double err, double refmag)
    {
    if (err > maxabs) maxabs = err;
    if (refmag > refmax) refmax = refmag;
    sumerr2 += err * err; sumref2 += refmag * refmag;
    }
  double norm() const { return refmax > 0 ? maxabs / refmax : (maxabs == 0 ? 0. : INFINITY); }
  double l2rel() const { return sumref2 > 0 ? std::sqrt(sumerr2 / sumref2) : (sumerr2 == 0 ? 0. : INFINITY); }
  };

struct CaseResult
  {
  bool have_ref = false, have_dump = false;
  Metrics ref;            // pocketfft vs reference
  double vs_dump = -1;    // max |out - dump| / scale
  bool ok = true;
  std::string note;
  double ms = 0;          // pocketfft call time
  };

struct Options
  {
  bool do_float = true, do_double = true;
  bool do_c2c = true, do_r2c = true, do_c2r = true;
  bool do_fwd = true, do_bwd = true;
  bool extras = true;
  std::vector<size_t> lengths;
  size_t max_n = ~size_t(0);
  size_t dump_max_n = ~size_t(0);
  double tol = 1e-5;
  double dump_tol_float = 1e-4, dump_tol_double = 1e-12;
  double fct_override = 0; // 0 = default (1 fwd, 1/n bwd)
  std::string csv, label = "unlabelled", dump, compare;
  RefMode ref = REF_AUTO;
  size_t naive_max;
  uint64_t seed = 12345;
  int verbose = 1;
  size_t nthreads = 1;
  };

// ------------------------------------------------------------------------------------------------
// running one case
// ------------------------------------------------------------------------------------------------
template<typename T> struct TypeName;
template<> struct TypeName<float> { static constexpr const char *v = "float"; };
template<> struct TypeName<double> { static constexpr const char *v = "double"; };

template<typename T>
static void run_case(const CaseSpec &c, const Options &opt, Reference &ref, DumpReader *dump, FILE *dumpout,
  CaseResult &res)
  {
  typedef std::complex<T> ct;
  const size_t n = c.n, B = c.batch;
  const size_t nin = (c.kind == C2R) ? n / 2 + 1 : n;   // logical elements per input row
  const size_t nout = (c.kind == R2C) ? n / 2 + 1 : n;  // logical elements per output row
  const bool cin = (c.kind != R2C), cout_ = (c.kind != C2R); // complex input / output?
  const T fct = T(opt.fct_override != 0 ? opt.fct_override : (c.fwd ? 1.0 : 1.0 / double(n)));
  const double fct_d = double(fct);

  // ---- generate logical (row-major, contiguous) input, rounded to T, kept also as double
  LCG rng(case_seed(opt.seed, c));
  std::vector<double> in_d(B * nin * (cin ? 2 : 1));
  for (size_t i = 0; i < in_d.size(); ++i) in_d[i] = double(T(rng.uniform()));
  if (c.kind == C2R) // make DC / Nyquist imaginary parts zero so every library interprets the data identically
    for (size_t b = 0; b < B; ++b)
      {
      in_d[(b * nin + 0) * 2 + 1] = 0.;
      if ((n & 1) == 0) in_d[(b * nin + n / 2) * 2 + 1] = 0.;
      }

  // ---- strided buffers for pocketfft (element strides sin/sout along the axis, rows packed after)
  const size_t in_elems = B * nin * size_t(c.sin), out_elems = B * nout * size_t(c.sout);
  std::vector<T> in_raw(in_elems * (cin ? 2 : 1)), out_raw((c.inplace ? 0 : out_elems) * (cout_ ? 2 : 1));
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < nin; ++i)
      {
      size_t src = b * nin + i, dst = (b * nin + i) * size_t(c.sin);
      if (cin) { in_raw[2 * dst] = T(in_d[2 * src]); in_raw[2 * dst + 1] = T(in_d[2 * src + 1]); }
      else in_raw[dst] = T(in_d[src]);
      }
  // poison the output so that untouched elements show up
  for (auto &v : out_raw) v = T(12345.678);

  // ---- pocketfft call
  const size_t esz_in = cin ? sizeof(ct) : sizeof(T), esz_out = cout_ ? sizeof(ct) : sizeof(T);
  shape_t shape, axes;
  stride_t stride_in, stride_out;
  if (B == 1)
    {
    shape = {n}; axes = {0};
    stride_in = {ptrdiff_t(c.sin * ptrdiff_t(esz_in))};
    stride_out = {ptrdiff_t(c.sout * ptrdiff_t(esz_out))};
    }
  else
    {
    shape = {B, n}; axes = {1};
    stride_in = {ptrdiff_t(nin * size_t(c.sin) * esz_in), ptrdiff_t(c.sin * ptrdiff_t(esz_in))};
    stride_out = {ptrdiff_t(nout * size_t(c.sout) * esz_out), ptrdiff_t(c.sout * ptrdiff_t(esz_out))};
    }
  T *outp = c.inplace ? in_raw.data() : out_raw.data();
  auto t0 = std::chrono::steady_clock::now();
  try
    {
    if (c.kind == C2C)
      pocketfft::c2c<T>(shape, stride_in, stride_out, axes, c.fwd,
        reinterpret_cast<const ct *>(in_raw.data()), reinterpret_cast<ct *>(outp), fct, opt.nthreads);
    else if (c.kind == R2C)
      pocketfft::r2c<T>(shape, stride_in, stride_out, axes, c.fwd,
        in_raw.data(), reinterpret_cast<ct *>(outp), fct, opt.nthreads);
    else
      pocketfft::c2r<T>(shape, stride_in, stride_out, axes, c.fwd,
        reinterpret_cast<const ct *>(in_raw.data()), outp, fct, opt.nthreads);
    }
  catch (const std::exception &e)
    {
    res.ok = false; res.note = std::string("exception: ") + e.what();
    return;
    }
  auto t1 = std::chrono::steady_clock::now();
  res.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // ---- gather logical output (contiguous, as double)
  std::vector<double> out_d(B * nout * (cout_ ? 2 : 1));
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < nout; ++i)
      {
      size_t dst = b * nout + i, src = (b * nout + i) * size_t(c.sout);
      if (cout_) { out_d[2 * dst] = outp[2 * src]; out_d[2 * dst + 1] = outp[2 * src + 1]; }
      else out_d[dst] = outp[src];
      }
  for (double v : out_d) if (!std::isfinite(v)) { res.ok = false; res.note = "non-finite output"; }

  // ---- reference
  std::vector<double> ref_d;
  if (ref.available(n))
    {
    res.have_ref = true;
    ref_d.resize(out_d.size());
    std::vector<cd> rin(nin), rout(nout);
    std::vector<double> rr(n);
    for (size_t b = 0; b < B; ++b)
      {
      if (c.kind == C2C)
        {
        for (size_t i = 0; i < n; ++i) rin[i] = cd(in_d[2 * (b * n + i)], in_d[2 * (b * n + i) + 1]);
        ref.c2c(n, rin.data(), rout.data(), c.fwd);
        for (size_t i = 0; i < n; ++i)
          { ref_d[2 * (b * n + i)] = rout[i].real() * fct_d; ref_d[2 * (b * n + i) + 1] = rout[i].imag() * fct_d; }
        }
      else if (c.kind == R2C)
        {
        ref.r2c(n, &in_d[b * n], rout.data(), c.fwd);
        for (size_t i = 0; i < nout; ++i)
          { ref_d[2 * (b * nout + i)] = rout[i].real() * fct_d; ref_d[2 * (b * nout + i) + 1] = rout[i].imag() * fct_d; }
        }
      else
        {
        for (size_t i = 0; i < nin; ++i) rin[i] = cd(in_d[2 * (b * nin + i)], in_d[2 * (b * nin + i) + 1]);
        ref.c2r(n, rin.data(), rr.data(), c.fwd);
        for (size_t i = 0; i < n; ++i) ref_d[b * n + i] = rr[i] * fct_d;
        }
      }
    if (cout_)
      for (size_t i = 0; i < B * nout; ++i)
        {
        double er = out_d[2 * i] - ref_d[2 * i], ei = out_d[2 * i + 1] - ref_d[2 * i + 1];
        res.ref.add(std::hypot(er, ei), std::hypot(ref_d[2 * i], ref_d[2 * i + 1]));
        }
    else
      for (size_t i = 0; i < B * nout; ++i)
        res.ref.add(std::fabs(out_d[i] - ref_d[i]), std::fabs(ref_d[i]));
    if (!(res.ref.norm() <= opt.tol)) { res.ok = false; res.note += " ref-mismatch"; }
    }

  // ---- dump / compare
  if (dumpout && n <= opt.dump_max_n)
    {
    DumpHeader h{uint32_t(c.is_double), uint32_t(c.kind), uint32_t(c.fwd), uint32_t(c.inplace),
      uint64_t(n), uint64_t(B), int64_t(c.sin), int64_t(c.sout), uint64_t(out_d.size())};
    std::fwrite(&h, sizeof h, 1, dumpout);
    if (c.is_double) std::fwrite(out_d.data(), sizeof(double), out_d.size(), dumpout);
    else
      {
      std::vector<float> tmp(out_d.size());
      for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = float(out_d[i]);
      std::fwrite(tmp.data(), sizeof(float), tmp.size(), dumpout);
      }
    }
  if (dump)
    {
    std::vector<double> dv;
    if (dump->load(c, dv) && dv.size() == out_d.size())
      {
      res.have_dump = true;
      double mx = 0, scale = res.have_ref ? res.ref.refmax : 0;
      if (cout_)
        for (size_t i = 0; i < B * nout; ++i)
          {
          mx = std::max(mx, std::hypot(out_d[2 * i] - dv[2 * i], out_d[2 * i + 1] - dv[2 * i + 1]));
          if (!res.have_ref) scale = std::max(scale, std::hypot(dv[2 * i], dv[2 * i + 1]));
          }
      else
        for (size_t i = 0; i < B * nout; ++i)
          {
          mx = std::max(mx, std::fabs(out_d[i] - dv[i]));
          if (!res.have_ref) scale = std::max(scale, std::fabs(dv[i]));
          }
      res.vs_dump = scale > 0 ? mx / scale : (mx == 0 ? 0. : INFINITY);
      double dtol = c.is_double ? opt.dump_tol_double : opt.dump_tol_float;
      if (!(res.vs_dump <= dtol)) { res.ok = false; res.note += " dump-mismatch"; }
      }
    else res.note += " (case not in dump)";
    }
  }

// ------------------------------------------------------------------------------------------------
// case list
// ------------------------------------------------------------------------------------------------
static std::vector<size_t> default_lengths()
  {
  return {
    // powers of two
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576,
    // mixed radix 2^k 3^m 5^n and radix-7 / radix-11 / generic-radix (13) factors
    6, 12, 24, 30, 48, 60, 96, 100, 120, 180, 240, 360, 720, 1000, 1200, 1536, 2160, 3000, 5400,
    12288, 14400, 32400,
    26, 39, 49, 77, 121, 143, 169, 231, 352, 448, 1001, 1331, 2401,
    // primes (small ones use the generic radix pass directly, large ones Bluestein)
    13, 17, 19, 23, 31, 47, 53, 97, 101, 257, 1009, 10007, 65537,
    // odd / non-aligned
    3, 5, 7, 9, 15, 33, 65, 129, 1023, 1025};
  }

static void add_case_set(std::vector<CaseSpec> &out, const Options &o, size_t n, size_t B, ptrdiff_t si, ptrdiff_t so,
  bool inplace)
  {
  if (n > o.max_n) return;
  for (int t = 0; t < 2; ++t)
    {
    bool dbl = (t == 1);
    if (dbl && !o.do_double) continue;
    if (!dbl && !o.do_float) continue;
    for (int k = 0; k < 3; ++k)
      {
      if (k == C2C && !o.do_c2c) continue;
      if (k == R2C && !o.do_r2c) continue;
      if (k == C2R && !o.do_c2r) continue;
      if (inplace && k != C2C) continue;
      for (int d = 0; d < 2; ++d)
        {
        bool fwd = (d == 0);
        if (fwd && !o.do_fwd) continue;
        if (!fwd && !o.do_bwd) continue;
        out.push_back(CaseSpec{dbl, Kind(k), fwd, n, B, si, so, inplace});
        }
      }
    }
  }

static std::vector<CaseSpec> build_cases(const Options &o)
  {
  std::vector<CaseSpec> cases;
  std::vector<size_t> lens = o.lengths.empty() ? default_lengths() : o.lengths;
  std::sort(lens.begin(), lens.end());
  lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
  for (size_t n : lens) add_case_set(cases, o, n, 1, 1, 1, false);
  if (o.extras && o.lengths.empty())
    {
    // batched (2-D shape {B,n}, transform along axis 1): exercises the multi-row / vector paths,
    // B=5 and B=9 leave remainders for every vector width
    for (size_t B : {size_t(5), size_t(9)})
      for (size_t n : {size_t(16), size_t(100), size_t(1009), size_t(4096)})
        add_case_set(cases, o, n, B, 1, 1, false);
    // strided input (3) and output (2)
    for (size_t n : {size_t(17), size_t(1000), size_t(4096)}) add_case_set(cases, o, n, 1, 3, 2, false);
    add_case_set(cases, o, 100, 5, 3, 2, false);
    // in-place c2c
    for (size_t n : {size_t(8), size_t(1000), size_t(4096)})
      { add_case_set(cases, o, n, 1, 1, 1, true); add_case_set(cases, o, n, 5, 1, 1, true); }
    }
  return cases;
  }

// ------------------------------------------------------------------------------------------------
static void usage(const char *argv0)
  {
  std::printf(
    "usage: %s [options]\n"
    "  --type float|double|both     (default both)\n"
    "  --kind c2c|r2c|c2r|all       (default all)\n"
    "  --dir fwd|bwd|both           (default both)\n"
    "  --lengths n1,n2,...          transform lengths (default: built-in list of %zu lengths + extras)\n"
    "  --max-n N                    skip lengths > N (quick runs under QEMU)\n"
    "  --no-extras                  skip batched / strided / in-place extra cases\n"
    "  --batch B                    run ONLY the given lengths as batched {B,n} transforms (with --lengths)\n"
    "  --fct X                      scaling factor for all cases (default: 1 fwd, 1/n bwd)\n"
    "  --tol X                      pass threshold on max|err|/max|ref| (default 1e-5)\n"
    "  --ref auto|fftw|naive|none   reference (default auto: FFTW if built in, else naive-ld up to --naive-max)\n"
    "  --naive-max N                (default 4096 on x86-64, 512 elsewhere: binary128 soft-float is slow)\n"
    "  --csv PATH                   append per-case rows to a CSV file (header written if new/empty)\n"
    "  --label STR                  label column for the CSV (e.g. riscv64-gcc15-rvv-vlen256)\n"
    "  --dump PATH                  write all outputs to a binary dump (for --compare-scalar on another build)\n"
    "  --dump-max-n N               only dump cases with n <= N\n"
    "  --compare-scalar PATH        compare outputs with a dump written by another build (scalar)\n"
    "  --dump-tol-double X / --dump-tol-float X   pass thresholds for the dump comparison (1e-12 / 1e-4)\n"
    "  --seed S                     base seed for the input generator (default 12345)\n"
    "  --threads T                  nthreads argument for pocketfft (default 1)\n"
    "  --quiet / --verbose          per-case output off / on (default on)\n",
    argv0, default_lengths().size());
  }

static std::vector<size_t> parse_list(const char *s)
  {
  std::vector<size_t> v;
  std::string str(s);
  size_t pos = 0;
  while (pos < str.size())
    {
    size_t e = str.find(',', pos);
    if (e == std::string::npos) e = str.size();
    std::string tok = str.substr(pos, e - pos);
    if (!tok.empty())
      {
      // allow products like 7*64
      size_t prod = 1, p = 0;
      while (p < tok.size())
        {
        size_t m = tok.find('*', p);
        if (m == std::string::npos) m = tok.size();
        prod *= size_t(std::strtoull(tok.substr(p, m - p).c_str(), nullptr, 10));
        p = m + 1;
        }
      v.push_back(prod);
      }
    pos = e + 1;
    }
  return v;
  }

int main(int argc, char **argv)
  {
  Options o;
#if defined(__x86_64__) || defined(__i386__)
  o.naive_max = 4096;
#else
  o.naive_max = 512;
#endif
  size_t only_batch = 1;
  for (int i = 1; i < argc; ++i)
    {
    std::string a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs an argument (%s)\n", a.c_str(), what); std::exit(2); }
      return argv[++i]; };
    if (a == "--type") { std::string v = need("float|double|both"); o.do_float = (v != "double"); o.do_double = (v != "float"); }
    else if (a == "--kind") { std::string v = need("c2c|r2c|c2r|all"); o.do_c2c = (v == "all" || v == "c2c"); o.do_r2c = (v == "all" || v == "r2c"); o.do_c2r = (v == "all" || v == "c2r"); }
    else if (a == "--dir") { std::string v = need("fwd|bwd|both"); o.do_fwd = (v != "bwd"); o.do_bwd = (v != "fwd"); }
    else if (a == "--lengths" || a == "--sizes") o.lengths = parse_list(need("list"));
    else if (a == "--max-n") o.max_n = std::strtoull(need("N"), nullptr, 10);
    else if (a == "--no-extras") o.extras = false;
    else if (a == "--batch") only_batch = std::strtoull(need("B"), nullptr, 10);
    else if (a == "--fct") o.fct_override = std::strtod(need("X"), nullptr);
    else if (a == "--tol") o.tol = std::strtod(need("X"), nullptr);
    else if (a == "--ref") { std::string v = need("mode"); o.ref = v == "fftw" ? REF_FFTW : v == "naive" ? REF_NAIVE : v == "none" ? REF_NONE : REF_AUTO; }
    else if (a == "--naive-max") o.naive_max = std::strtoull(need("N"), nullptr, 10);
    else if (a == "--csv") o.csv = need("path");
    else if (a == "--label") o.label = need("str");
    else if (a == "--dump") o.dump = need("path");
    else if (a == "--dump-max-n") o.dump_max_n = std::strtoull(need("N"), nullptr, 10);
    else if (a == "--compare-scalar" || a == "--compare") o.compare = need("path");
    else if (a == "--dump-tol-double") o.dump_tol_double = std::strtod(need("X"), nullptr);
    else if (a == "--dump-tol-float") o.dump_tol_float = std::strtod(need("X"), nullptr);
    else if (a == "--seed") o.seed = std::strtoull(need("S"), nullptr, 10);
    else if (a == "--threads") o.nthreads = std::strtoull(need("T"), nullptr, 10);
    else if (a == "--quiet") o.verbose = 0;
    else if (a == "--verbose") o.verbose = 2;
    else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
#ifdef HAVE_FFTW
  if (o.ref == REF_FFTW || o.ref == REF_AUTO) { /* fine */ }
#else
  if (o.ref == REF_FFTW) { std::fprintf(stderr, "this binary was built without FFTW (-DHAVE_FFTW)\n"); return 2; }
#endif

  std::vector<CaseSpec> cases = build_cases(o);
  if (only_batch > 1) for (auto &c : cases) c.batch = only_batch;

  Reference ref; ref.mode = o.ref; ref.naive_max = o.naive_max;
  DumpReader dumpin;
  if (!o.compare.empty() && !dumpin.open(o.compare.c_str())) return 2;
  FILE *dumpout = nullptr;
  if (!o.dump.empty())
    {
    dumpout = std::fopen(o.dump.c_str(), "wb");
    if (!dumpout) { std::fprintf(stderr, "cannot create %s: %s\n", o.dump.c_str(), std::strerror(errno)); return 2; }
    std::fwrite(DUMP_MAGIC, 1, 8, dumpout);
    }
  FILE *csv = nullptr;
  if (!o.csv.empty())
    {
    bool fresh = true;
    if (FILE *t = std::fopen(o.csv.c_str(), "rb")) { std::fseek(t, 0, SEEK_END); fresh = std::ftell(t) == 0; std::fclose(t); }
    csv = std::fopen(o.csv.c_str(), "ab");
    if (!csv) { std::fprintf(stderr, "cannot open %s: %s\n", o.csv.c_str(), std::strerror(errno)); return 2; }
    if (fresh)
      std::fprintf(csv, "label,type,kind,dir,n,batch,stride_in,stride_out,inplace,fct,ref,max_abs_err,ref_max_abs,"
        "max_err_norm,l2_rel,vs_dump_norm,ms,status\n");
    }

  const char *build_info =
#if defined(POCKETFFT_RVV)
    "RVV";
#elif defined(__riscv)
    "riscv-scalar";
#elif defined(POCKETFFT_NO_VECTORS)
    "no-vectors";
#else
    "generic";
#endif
  std::printf("fft_check: label=%s build=%s cases=%zu ref=%s tol=%g seed=%llu%s%s\n", o.label.c_str(), build_info,
    cases.size(),
#ifdef HAVE_FFTW
    o.ref == REF_NONE ? "none" : o.ref == REF_NAIVE ? "naive-ld" : "fftw3(double)",
#else
    o.ref == REF_NONE ? "none" : "naive-ld",
#endif
    o.tol, (unsigned long long)o.seed, o.compare.empty() ? "" : " compare=", o.compare.c_str());
  std::fflush(stdout);

  size_t npass = 0, nfail = 0, nnoref = 0;
  struct Worst { double v = -1; std::string what; };
  Worst worst_norm[2], worst_dump[2];
  size_t ndump_cmp = 0;
  auto tstart = std::chrono::steady_clock::now();
  for (const CaseSpec &c : cases)
    {
    CaseResult r;
    if (c.is_double) run_case<double>(c, o, ref, o.compare.empty() ? nullptr : &dumpin, dumpout, r);
    else run_case<float>(c, o, ref, o.compare.empty() ? nullptr : &dumpin, dumpout, r);
    const char *status = !r.ok ? "FAIL" : (r.have_ref || r.have_dump) ? "PASS" : "NOREF";
    if (!r.ok) ++nfail; else if (r.have_ref || r.have_dump) ++npass; else ++nnoref;
    char what[160];
    std::snprintf(what, sizeof what, "%s %s %s n=%zu B=%zu s=%td/%td%s", c.is_double ? "double" : "float",
      kind_name[c.kind], c.fwd ? "fwd" : "bwd", c.n, c.batch, c.sin, c.sout, c.inplace ? " inplace" : "");
    if (r.have_ref && r.ref.norm() > worst_norm[c.is_double].v) { worst_norm[c.is_double].v = r.ref.norm(); worst_norm[c.is_double].what = what; }
    if (r.have_dump) { ++ndump_cmp; if (r.vs_dump > worst_dump[c.is_double].v) { worst_dump[c.is_double].v = r.vs_dump; worst_dump[c.is_double].what = what; } }
    if (o.verbose >= 1 || !r.ok)
      {
      std::printf("[%s] %-44s", status, what);
      if (r.have_ref) std::printf(" maxabs=%.3e norm=%.3e l2=%.3e", r.ref.maxabs, r.ref.norm(), r.ref.l2rel());
      if (r.have_dump) std::printf(" vs_dump=%.3e", r.vs_dump);
      std::printf(" %.2fms%s\n", r.ms, r.note.c_str());
      std::fflush(stdout);
      }
    if (csv)
      std::fprintf(csv, "%s,%s,%s,%s,%zu,%zu,%td,%td,%d,%.17g,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%s\n", o.label.c_str(),
        c.is_double ? "double" : "float", kind_name[c.kind], c.fwd ? "fwd" : "bwd", c.n, c.batch, c.sin, c.sout,
        int(c.inplace), o.fct_override != 0 ? o.fct_override : (c.fwd ? 1.0 : 1.0 / double(c.n)),
        r.have_ref ? ref.name(c.n) : "none", r.have_ref ? r.ref.maxabs : -1., r.have_ref ? r.ref.refmax : -1.,
        r.have_ref ? r.ref.norm() : -1., r.have_ref ? r.ref.l2rel() : -1., r.have_dump ? r.vs_dump : -1., r.ms, status);
    }
  double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count();
  if (csv) std::fclose(csv);
  if (dumpout) std::fclose(dumpout);

  std::printf("----\nSUMMARY label=%s: %zu cases, %zu pass, %zu FAIL, %zu without reference (%.1f s)\n", o.label.c_str(),
    cases.size(), npass, nfail, nnoref, total_s);
  for (int t = 0; t < 2; ++t)
    if (worst_norm[t].v >= 0)
      std::printf("  worst max|err|/max|ref| %-6s : %.3e  (%s)\n", t ? "double" : "float", worst_norm[t].v, worst_norm[t].what.c_str());
  for (int t = 0; t < 2; ++t)
    if (worst_dump[t].v >= 0)
      std::printf("  worst vs-dump difference %-6s: %.3e  (%s)\n", t ? "double" : "float", worst_dump[t].v, worst_dump[t].what.c_str());
  if (!o.compare.empty()) std::printf("  dump comparisons: %zu cases\n", ndump_cmp);
  if (nnoref == cases.size()) std::printf("  WARNING: no reference was available for any case\n");
  std::printf("RESULT: %s\n", nfail ? "FAIL" : "PASS");
  return nfail ? 1 : 0;
  }
