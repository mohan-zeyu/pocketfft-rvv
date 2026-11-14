// fft_bench.cpp - timing / instruction-count driver for pocketfft_hdronly.h (scalar and RVV paths).
//
//   fft_bench --type double|float --kind c2c-fwd|c2c-bwd|r2c|c2r --n N [--batch B] [--reps R] [--trials T]
//             [--mode api|plan] [--fct X] [--inplace] [--threads T] [--quiet]
//
// Native: prints ns per transform (minimum over --trials of the mean over --reps) and an FFTW-style
// "mflops" figure (5 n log2 n per complex transform, half that for real ones).
//
// QEMU instruction counting: everything except the timed transforms (process start-up, input
// generation, plan creation, one warm-up call) is identical between two runs that differ only in
// --reps, so the scripts run the binary with --reps N and --reps 2N under the libinsn / librvvclass
// plugin and report (insns(2N) - insns(N)) / N as the per-transform dynamic instruction count.
//
// Modes:
//   api  (default) - public API pocketfft::c2c / r2c / c2r on shape {batch,n} (axis 1) or {n};
//                    POCKETFFT_CACHE_SIZE=16 is defined so that the plan is created once and reused
//                    (what SciPy's pocketfft wrapper does); the per-call cost is plan lookup +
//                    buffer copies + the transform itself.
//   plan           - lower-level pocketfft::detail::pocketfft_c<T> / pocketfft_r<T>::exec() on a
//                    contiguous buffer (input copied in first, the same as the API's in-place path);
//                    batch>1 loops over the rows (no vectorised multi-row path).
//
// Default scaling mirrors NumPy: fct = 1 for forward transforms (c2c-fwd, r2c) and 1/n for inverse
// ones (c2c-bwd, c2r); override with --fct.

#ifndef POCKETFFT_CACHE_SIZE
#define POCKETFFT_CACHE_SIZE 16
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <algorithm>

#include "pocketfft_hdronly.h"

using pocketfft::shape_t;
using pocketfft::stride_t;

struct LCG
  {
  uint64_t s;
  explicit LCG(uint64_t seed) : s(seed) {}
  uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
  double uniform() { return double(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }
  };

enum Kind { C2C_FWD, C2C_BWD, R2C, C2R };
static const char *kind_names[4] = {"c2c-fwd", "c2c-bwd", "r2c", "c2r"};

struct Opts
  {
  bool is_double = true;
  Kind kind = C2C_FWD;
  size_t n = 1024, batch = 1, reps = 0, trials = 5, threads = 1;
  bool plan_mode = false, inplace = false, quiet = false;
  double fct = 0; // 0 => default (1 or 1/n)
  };

// prevents the compiler from optimising anything away and gives a cheap scalar/RVV sanity value
static double g_checksum = 0;

template<typename T> struct Bench
  {
  typedef std::complex<T> ct;
  const Opts &o;
  size_t n, B, nin, nout;
  bool cin, cout_;
  std::vector<T> in, out, work;
  T fct;
  shape_t shape, axes;
  stride_t sin, sout;
  // plan-mode objects
  std::unique_ptr<pocketfft::detail::pocketfft_c<T>> cplan;
  std::unique_ptr<pocketfft::detail::pocketfft_r<T>> rplan;

  explicit Bench(const Opts &o_) : o(o_)
    {
    n = o.n; B = o.batch;
    cin = (o.kind != R2C); cout_ = (o.kind != C2R);
    nin = (o.kind == C2R) ? n / 2 + 1 : n;
    nout = (o.kind == R2C) ? n / 2 + 1 : n;
    fct = T(o.fct != 0 ? o.fct : ((o.kind == C2C_FWD || o.kind == R2C) ? 1.0 : 1.0 / double(n)));
    in.resize(B * nin * (cin ? 2 : 1));
    out.resize(B * nout * (cout_ ? 2 : 1));
    LCG rng(0x5EEDULL + n * 131 + B);
    for (auto &v : in) v = T(rng.uniform());
    if (o.kind == C2R)
      for (size_t b = 0; b < B; ++b)
        { in[2 * (b * nin) + 1] = 0; if ((n & 1) == 0) in[2 * (b * nin + n / 2) + 1] = 0; }
    size_t esz_in = cin ? sizeof(ct) : sizeof(T), esz_out = cout_ ? sizeof(ct) : sizeof(T);
    if (B == 1) { shape = {n}; axes = {0}; sin = {ptrdiff_t(esz_in)}; sout = {ptrdiff_t(esz_out)}; }
    else
      {
      shape = {B, n}; axes = {1};
      sin = {ptrdiff_t(nin * esz_in), ptrdiff_t(esz_in)};
      sout = {ptrdiff_t(nout * esz_out), ptrdiff_t(esz_out)};
      }
    if (o.plan_mode)
      {
      if (o.kind == C2C_FWD || o.kind == C2C_BWD) cplan.reset(new pocketfft::detail::pocketfft_c<T>(n));
      else rplan.reset(new pocketfft::detail::pocketfft_r<T>(n));
      work.resize(std::max(in.size(), out.size()));
      }
    if (o.inplace) { out = in; }
    }

  void once()
    {
    const bool fwd = (o.kind == C2C_FWD) || (o.kind == R2C) ? true : (o.kind == C2C_BWD ? false : false);
    if (!o.plan_mode)
      {
      switch (o.kind)
        {
        case C2C_FWD: case C2C_BWD:
          if (o.inplace)
            pocketfft::c2c<T>(shape, sin, sout, axes, o.kind == C2C_FWD, reinterpret_cast<const ct *>(out.data()),
              reinterpret_cast<ct *>(out.data()), fct, o.threads);
          else
            pocketfft::c2c<T>(shape, sin, sout, axes, o.kind == C2C_FWD, reinterpret_cast<const ct *>(in.data()),
              reinterpret_cast<ct *>(out.data()), fct, o.threads);
          break;
        case R2C:
          pocketfft::r2c<T>(shape, sin, sout, axes, true, in.data(), reinterpret_cast<ct *>(out.data()), fct, o.threads);
          break;
        case C2R:
          pocketfft::c2r<T>(shape, sin, sout, axes, false, reinterpret_cast<const ct *>(in.data()), out.data(), fct, o.threads);
          break;
        }
      (void)fwd;
      }
    else
      {
      // mirror of the API's contiguous path: copy the row into the destination buffer, transform in place
      for (size_t b = 0; b < B; ++b)
        {
        if (cplan)
          {
          ct *dst = reinterpret_cast<ct *>(out.data()) + b * n;
          if (!o.inplace) std::memcpy(dst, in.data() + 2 * b * n, n * sizeof(ct));
          cplan->exec(reinterpret_cast<pocketfft::detail::cmplx<T> *>(dst), fct, o.kind == C2C_FWD);
          }
        else if (o.kind == R2C)
          {
          // real plan works in FFTPACK halfcomplex order on n reals; the API additionally unpacks it
          T *w = work.data();
          std::memcpy(w, in.data() + b * n, n * sizeof(T));
          rplan->exec(w, fct, true);
          ct *dst = reinterpret_cast<ct *>(out.data()) + b * nout;
          dst[0] = ct(w[0], 0);
          size_t i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) dst[ii] = ct(w[i], w[i + 1]);
          if (i < n) dst[ii] = ct(w[i], 0);
          }
        else
          {
          T *w = out.data() + b * n;
          const ct *src = reinterpret_cast<const ct *>(in.data()) + b * nin;
          w[0] = src[0].real();
          size_t i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) { w[i] = src[ii].real(); w[i + 1] = src[ii].imag(); }
          if (i < n) w[i] = src[ii].real();
          rplan->exec(w, fct, false);
          }
        }
      }
    }

  double checksum() const
    {
    double s = 0;
    for (size_t i = 0; i < out.size(); i += std::max<size_t>(1, out.size() / 4096)) s += double(out[i]);
    return s;
    }
  };

template<typename T> static int run(const Opts &o)
  {
  Bench<T> b(o);
  // warm-up: creates (and caches) the plan, touches all buffers
  b.once();
  double best_ns = 1e300, total_s = 0;
  for (size_t t = 0; t < o.trials; ++t)
    {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < o.reps; ++r) b.once();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    total_s += s;
    best_ns = std::min(best_ns, s * 1e9 / double(o.reps));
    }
  g_checksum = b.checksum();
  double flops = ((o.kind == C2C_FWD || o.kind == C2C_BWD) ? 5.0 : 2.5) * double(o.n) * std::log2(double(o.n)) * double(o.batch);
  unsigned long vlen_bits = 0;
#if defined(__riscv_vector)
  { unsigned long vlenb; asm volatile("csrr %0, vlenb" : "=r"(vlenb)); vlen_bits = vlenb * 8; }
#endif
  const char *path =
#if defined(POCKETFFT_RVV)
    "rvv";
#elif defined(__riscv)
    "riscv-scalar";
#elif defined(POCKETFFT_NO_VECTORS)
    "scalar";
#else
    "generic-vector";
#endif
  if (!o.quiet)
    std::printf("fft_bench type=%s kind=%s n=%zu batch=%zu mode=%s reps=%zu trials=%zu ns_per_transform=%.1f "
      "ns_per_row=%.1f mflops=%.1f total_s=%.3f checksum=%.9g path=%s vlen=%lu\n", o.is_double ? "double" : "float",
      kind_names[o.kind], o.n, o.batch, o.plan_mode ? "plan" : "api", o.reps, o.trials, best_ns, best_ns / double(o.batch),
      flops / best_ns * 1e3, total_s, g_checksum, path, vlen_bits);
  return 0;
  }

static void usage()
  {
  std::printf("usage: fft_bench --type double|float --kind c2c-fwd|c2c-bwd|r2c|c2r --n N [--batch B] [--reps R]\n"
    "                 [--trials T] [--mode api|plan] [--fct X] [--inplace] [--threads T] [--quiet]\n"
    "  --reps defaults to max(3, 2e7 / (n*batch)) and --trials to 5 (use --trials 1 under QEMU)\n");
  }

int main(int argc, char **argv)
  {
  Opts o;
  for (int i = 1; i < argc; ++i)
    {
    std::string a = argv[i];
    auto need = [&]() -> const char * { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
    if (a == "--type") { std::string v = need(); o.is_double = (v != "float"); }
    else if (a == "--kind")
      {
      std::string v = need();
      if (v == "c2c-fwd" || v == "c2c") o.kind = C2C_FWD;
      else if (v == "c2c-bwd") o.kind = C2C_BWD;
      else if (v == "r2c") o.kind = R2C;
      else if (v == "c2r") o.kind = C2R;
      else { std::fprintf(stderr, "unknown kind %s\n", v.c_str()); return 2; }
      }
    else if (a == "--n" || a == "--length") o.n = std::strtoull(need(), nullptr, 10);
    else if (a == "--batch") o.batch = std::strtoull(need(), nullptr, 10);
    else if (a == "--reps") o.reps = std::strtoull(need(), nullptr, 10);
    else if (a == "--trials") o.trials = std::strtoull(need(), nullptr, 10);
    else if (a == "--threads") o.threads = std::strtoull(need(), nullptr, 10);
    else if (a == "--mode") { std::string v = need(); o.plan_mode = (v == "plan"); }
    else if (a == "--fct") o.fct = std::strtod(need(), nullptr);
    else if (a == "--inplace") o.inplace = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
  if (o.n == 0 || o.batch == 0) { usage(); return 2; }
  if (o.inplace && !(o.kind == C2C_FWD || o.kind == C2C_BWD)) { std::fprintf(stderr, "--inplace is c2c only\n"); return 2; }
  if (o.reps == 0) o.reps = std::max<size_t>(3, size_t(2e7 / double(o.n * o.batch)));
  if (o.trials == 0) o.trials = 1;
  return o.is_double ? run<double>(o) : run<float>(o);
  }
