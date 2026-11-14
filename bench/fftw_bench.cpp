// fftw_bench.cpp - FFTW3 counterpart of fft_bench.cpp (same arguments, same inputs, same output line format)
// for the native x86 "pocketfft vs mainstream library" timing.
//
//   fftw_bench --type double|float --kind c2c-fwd|c2c-bwd|r2c|c2r --n N [--batch B] [--reps R] [--trials T]
//              [--planner estimate|measure|patient] [--fct X] [--quiet]
//
// Out-of-place transforms, fftw_malloc'ed (SIMD-aligned) buffers, batch B via the advanced interface
// (fftw_plan_many_*; contiguous rows).  Planning time is excluded from the timing; FFTW_MEASURE is the
// planner mode a careful FFTW user would pick, FFTW_ESTIMATE is what a one-shot caller gets.  Like pocketfft's
// backward transforms (fct = 1/n in fft_bench), the inverse kinds (c2c-bwd, c2r) include the 1/n scaling pass
// in the timed region (override with --fct).
//
// Build:  g++ -O3 -std=c++17 -march=x86-64-v3 fftw_bench.cpp -I$FFTW/include -L$FFTW/lib -lfftw3 -lfftw3f -lm

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <fftw3.h>

struct LCG
  {
  uint64_t s;
  explicit LCG(uint64_t seed) : s(seed) {}
  uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
  double uniform() { return double(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }
  };

enum Kind { C2C_FWD, C2C_BWD, R2C, C2R };
static const char *kind_names[4] = {"c2c-fwd", "c2c-bwd", "r2c", "c2r"};

template<typename T> struct Api;
template<> struct Api<double>
  {
  typedef fftw_plan plan_t; typedef fftw_complex cpx;
  static void *alloc(size_t bytes) { return fftw_malloc(bytes); }
  static void free_(void *p) { fftw_free(p); }
  static plan_t many_dft(int n, int B, cpx *in, cpx *out, int sign, unsigned f)
    { return fftw_plan_many_dft(1, &n, B, in, nullptr, 1, n, out, nullptr, 1, n, sign, f); }
  static plan_t many_r2c(int n, int B, double *in, cpx *out, unsigned f)
    { return fftw_plan_many_dft_r2c(1, &n, B, in, nullptr, 1, n, out, nullptr, 1, n / 2 + 1, f); }
  static plan_t many_c2r(int n, int B, cpx *in, double *out, unsigned f)
    { return fftw_plan_many_dft_c2r(1, &n, B, in, nullptr, 1, n / 2 + 1, out, nullptr, 1, n, f); }
  static void execute(plan_t p) { fftw_execute(p); }
  static void destroy(plan_t p) { fftw_destroy_plan(p); }
  };
template<> struct Api<float>
  {
  typedef fftwf_plan plan_t; typedef fftwf_complex cpx;
  static void *alloc(size_t bytes) { return fftwf_malloc(bytes); }
  static void free_(void *p) { fftwf_free(p); }
  static plan_t many_dft(int n, int B, cpx *in, cpx *out, int sign, unsigned f)
    { return fftwf_plan_many_dft(1, &n, B, in, nullptr, 1, n, out, nullptr, 1, n, sign, f); }
  static plan_t many_r2c(int n, int B, float *in, cpx *out, unsigned f)
    { return fftwf_plan_many_dft_r2c(1, &n, B, in, nullptr, 1, n, out, nullptr, 1, n / 2 + 1, f); }
  static plan_t many_c2r(int n, int B, cpx *in, float *out, unsigned f)
    { return fftwf_plan_many_dft_c2r(1, &n, B, in, nullptr, 1, n / 2 + 1, out, nullptr, 1, n, f); }
  static void execute(plan_t p) { fftwf_execute(p); }
  static void destroy(plan_t p) { fftwf_destroy_plan(p); }
  };

struct Opts
  {
  bool is_double = true;
  Kind kind = C2C_FWD;
  size_t n = 1024, batch = 1, reps = 0, trials = 5;
  unsigned planner = FFTW_MEASURE;
  const char *planner_name = "measure";
  double fct = 0;
  bool quiet = false;
  };

template<typename T> static int run(const Opts &o)
  {
  typedef Api<T> A;
  const size_t n = o.n, B = o.batch;
  const bool cin = (o.kind != R2C), cout_ = (o.kind != C2R);
  const size_t nin = (o.kind == C2R) ? n / 2 + 1 : n, nout = (o.kind == R2C) ? n / 2 + 1 : n;
  const size_t in_vals = B * nin * (cin ? 2 : 1), out_vals = B * nout * (cout_ ? 2 : 1);
  T *in = static_cast<T *>(A::alloc(in_vals * sizeof(T)));
  T *out = static_cast<T *>(A::alloc(out_vals * sizeof(T)));
  const T fct = T(o.fct != 0 ? o.fct : ((o.kind == C2C_FWD || o.kind == R2C) ? 1.0 : 1.0 / double(n)));
  // plan first (FFTW_MEASURE clobbers the arrays), then fill the input exactly like fft_bench does
  typename A::plan_t plan;
  unsigned flags = o.planner | FFTW_DESTROY_INPUT;
  switch (o.kind)
    {
    case C2C_FWD: plan = A::many_dft(int(n), int(B), reinterpret_cast<typename A::cpx *>(in), reinterpret_cast<typename A::cpx *>(out), FFTW_FORWARD, flags); break;
    case C2C_BWD: plan = A::many_dft(int(n), int(B), reinterpret_cast<typename A::cpx *>(in), reinterpret_cast<typename A::cpx *>(out), FFTW_BACKWARD, flags); break;
    case R2C:     plan = A::many_r2c(int(n), int(B), in, reinterpret_cast<typename A::cpx *>(out), flags); break;
    default:      plan = A::many_c2r(int(n), int(B), reinterpret_cast<typename A::cpx *>(in), out, flags); break;
    }
  if (!plan) { std::fprintf(stderr, "FFTW planning failed\n"); return 3; }
  LCG rng(0x5EEDULL + n * 131 + B);
  for (size_t i = 0; i < in_vals; ++i) in[i] = T(rng.uniform());
  if (o.kind == C2R)
    for (size_t b = 0; b < B; ++b)
      { in[2 * (b * nin) + 1] = 0; if ((n & 1) == 0) in[2 * (b * nin + n / 2) + 1] = 0; }
  // c2r with FFTW_DESTROY_INPUT may clobber the input: keep a pristine copy and restore it each rep (cost of a
  // memcpy, which the pocketfft API also pays for its copy_input), so every rep transforms the same data
  T *keep = nullptr;
  if (o.kind == C2R) { keep = static_cast<T *>(A::alloc(in_vals * sizeof(T))); std::memcpy(keep, in, in_vals * sizeof(T)); }

  auto once = [&]() {
    if (keep) std::memcpy(in, keep, in_vals * sizeof(T));
    A::execute(plan);
    if (fct != T(1)) for (size_t i = 0; i < out_vals; ++i) out[i] *= fct;
  };
  once();
  double best_ns = 1e300, total_s = 0;
  for (size_t t = 0; t < o.trials; ++t)
    {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < o.reps; ++r) once();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    total_s += s;
    best_ns = std::min(best_ns, s * 1e9 / double(o.reps));
    }
  double cs = 0;
  for (size_t i = 0; i < out_vals; i += std::max<size_t>(1, out_vals / 4096)) cs += double(out[i]);
  double flops = ((o.kind == C2C_FWD || o.kind == C2C_BWD) ? 5.0 : 2.5) * double(n) * std::log2(double(n)) * double(B);
  if (!o.quiet)
    std::printf("fftw_bench type=%s kind=%s n=%zu batch=%zu planner=%s reps=%zu trials=%zu ns_per_transform=%.1f "
      "ns_per_row=%.1f mflops=%.1f total_s=%.3f checksum=%.9g\n", o.is_double ? "double" : "float", kind_names[o.kind],
      n, B, o.planner_name, o.reps, o.trials, best_ns, best_ns / double(B), flops / best_ns * 1e3, total_s, cs);
  A::destroy(plan);
  A::free_(in); A::free_(out); if (keep) A::free_(keep);
  return 0;
  }

int main(int argc, char **argv)
  {
  Opts o;
  for (int i = 1; i < argc; ++i)
    {
    std::string a = argv[i];
    auto need = [&]() -> const char * { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs an argument\n", a.c_str()); std::exit(2); } return argv[++i]; };
    if (a == "--type") { std::string v = need(); o.is_double = (v != "float"); }
    else if (a == "--kind")
      {
      std::string v = need();
      if (v == "c2c-fwd" || v == "c2c") o.kind = C2C_FWD; else if (v == "c2c-bwd") o.kind = C2C_BWD;
      else if (v == "r2c") o.kind = R2C; else if (v == "c2r") o.kind = C2R;
      else { std::fprintf(stderr, "unknown kind %s\n", v.c_str()); return 2; }
      }
    else if (a == "--n" || a == "--length") o.n = std::strtoull(need(), nullptr, 10);
    else if (a == "--batch") o.batch = std::strtoull(need(), nullptr, 10);
    else if (a == "--reps") o.reps = std::strtoull(need(), nullptr, 10);
    else if (a == "--trials") o.trials = std::strtoull(need(), nullptr, 10);
    else if (a == "--fct") o.fct = std::strtod(need(), nullptr);
    else if (a == "--planner")
      {
      std::string v = need();
      if (v == "estimate") { o.planner = FFTW_ESTIMATE; o.planner_name = "estimate"; }
      else if (v == "patient") { o.planner = FFTW_PATIENT; o.planner_name = "patient"; }
      else { o.planner = FFTW_MEASURE; o.planner_name = "measure"; }
      }
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--mode" || a == "--threads") need(); // accepted for argument compatibility with fft_bench
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
  if (o.n == 0 || o.batch == 0) return 2;
  if (o.reps == 0) o.reps = std::max<size_t>(3, size_t(2e7 / double(o.n * o.batch)));
  return o.is_double ? run<double>(o) : run<float>(o);
  }
