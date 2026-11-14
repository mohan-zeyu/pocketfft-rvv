// test_real.cpp - correctness test for the RVV real-FFT passes (r2c / c2r public API).
//
//   test_real genref REF.bin            native (x86) run: compute exact references
//                                       (naive O(n^2) DFT in long double for n<=NAIVE_MAX, long-double
//                                       pocketfft above) and store them as double-double.
//   test_real check REF.bin [DUMP.bin [CMP.bin]]
//                                       run r2c/c2r for all lengths, float and double, compare with
//                                       REF.bin; optionally write the outputs to DUMP.bin and compare
//                                       them with CMP.bin (output of another build, e.g. scalar vs RVV).
//   test_real bench N REPS              r2c of length N (double), REPS times (instruction counting).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include <algorithm>
#include "pocketfft_hdronly.h"

using namespace pocketfft;
typedef long double ld;

static const size_t NAIVE_MAX = 4096;

static std::vector<size_t> lengths()
  {
  std::vector<size_t> v;
  for (size_t n=1; n<=65536; n*=2) v.push_back(n);                      // powers of two (incl. 1,2)
  const size_t misc[] = {
    3,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,35,36,37,40,42,44,45,
    47,48,49,50,52,53,54,56,60,63,64,66,70,72,75,77,80,81,84,88,90,91,96,97,98,100,101,103,104,105,
    108,110,112,120,121,125,126,127,128,130,135,140,143,144,147,150,160,162,169,180,192,200,210,216,
    225,240,243,250,256,270,288,289,300,320,324,343,360,375,384,400,405,432,450,480,486,500,509,540,
    576,600,625,640,648,675,720,729,750,768,800,810,864,900,960,972,1000,1009,1021,1024,1080,1125,
    1152,1200,1215,1250,1280,1296,1350,1440,1458,1500,1536,1600,1620,1728,1800,1875,1920,1944,2000,
    2003,2025,2048,2187,2250,2304,2400,2500,2560,2592,2700,2880,3000,3072,3125,3200,3375,3456,3600,
    3645,3840,3888,4000,4050,4093,4096,4500,4608,5000,5120,5184,6000,6144,6250,6561,7500,7680,8000,
    8100,8191,8192,9000,10000,10240,12000,12288,12500,15000,15625,16384,18000,19683,20000,24576,
    25000,27000,30000,32768,36000,40000,46875,49152,50000,59049,60000,62500,65536 };
  for (size_t n : misc) v.push_back(n);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
  }

// deterministic input in [-1,1) (integer LCG, identical on every platform)
static void fill_input(std::vector<double> &x, size_t n, uint32_t seed)
  {
  uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t(seed)*0x100000001B3ull) ^ (uint64_t(n)<<20);
  x.resize(n);
  for (size_t i=0; i<n; ++i)
    {
    s = s*6364136223846793005ull + 1442695040888963407ull;
    x[i] = double(int64_t(s>>11)) / 4503599627370496.0;   // 53 random bits -> [-1,1)
    }
  }

struct dd { double hi, lo; };
static dd to_dd(ld x) { dd r; r.hi = double(x); r.lo = double(x - ld(r.hi)); return r; }
static ld from_dd(dd d) { return ld(d.hi) + ld(d.lo); }

// reference spectrum (n/2+1 complex values) of the real input x (given as long double)
static void naive_r2c(const std::vector<ld> &x, std::vector<std::complex<ld>> &out)
  {
  size_t n = x.size(), nh = n/2+1;
  out.assign(nh, 0);
  std::vector<ld> cs(n), sn(n);
  const ld twopi = 6.283185307179586476925286766559005768L;
  for (size_t j=0; j<n; ++j) { cs[j] = std::cos(twopi*ld(j)/ld(n)); sn[j] = std::sin(twopi*ld(j)/ld(n)); }
  for (size_t k=0; k<nh; ++k)
    {
    ld re=0, im=0;
    size_t idx=0;
    for (size_t j=0; j<n; ++j)
      {
      re += x[j]*cs[idx]; im -= x[j]*sn[idx];
      idx += k; if (idx>=n) idx -= n;
      }
    out[k] = std::complex<ld>(re, im);
    }
  }

static void fft_r2c_ld(const std::vector<ld> &x, std::vector<std::complex<ld>> &out)
  {
#ifdef TEST_NO_LD_FFT   // avoid instantiating pocketfft for long double (only needed by native genref)
  naive_r2c(x, out);
#else
  size_t n = x.size();
  out.resize(n/2+1);
  shape_t shape{n}; stride_t sin{ptrdiff_t(sizeof(ld))}, sout{ptrdiff_t(sizeof(std::complex<ld>))};
  r2c<ld>(shape, sin, sout, 0, FORWARD, x.data(), out.data(), ld(1));
#endif
  }

// ---------------------------------------------------------------------------------------------
static int genref(const char *fn)
  {
  FILE *f = fopen(fn, "wb");
  if (!f) { perror(fn); return 1; }
  auto lens = lengths();
  std::vector<double> x;
  for (size_t n : lens)
    for (int typ=0; typ<2; ++typ)   // 0 = double input, 1 = float-rounded input
      {
      fill_input(x, n, 12345u + uint32_t(typ));
      std::vector<ld> xl(n);
      for (size_t i=0; i<n; ++i) xl[i] = typ ? ld(float(x[i])) : ld(x[i]);
      std::vector<std::complex<ld>> sp;
      if (n<=NAIVE_MAX) naive_r2c(xl, sp); else fft_r2c_ld(xl, sp);
      uint64_t hdr[2] = {n, uint64_t(typ)};
      fwrite(hdr, sizeof(hdr), 1, f);
      for (auto &c : sp) { dd a = to_dd(c.real()), b = to_dd(c.imag()); fwrite(&a, sizeof a, 1, f); fwrite(&b, sizeof b, 1, f); }
      fprintf(stderr, "ref n=%zu typ=%d (%s)\n", n, typ, n<=NAIVE_MAX ? "naive" : "ld-fft");
      }
  fclose(f);
  return 0;
  }

// ---------------------------------------------------------------------------------------------
struct errs { double l2, maxabs; };

template<typename T> static errs compare_c(const std::vector<std::complex<T>> &a, const std::vector<std::complex<ld>> &ref)
  {
  ld num=0, den=0, mx=0, mxref=0;
  for (size_t i=0; i<ref.size(); ++i)
    {
    std::complex<ld> d = std::complex<ld>(a[i].real(), a[i].imag()) - ref[i];
    num += std::norm(d); den += std::norm(ref[i]);
    mx = std::max(mx, std::abs(d)); mxref = std::max(mxref, std::abs(ref[i]));
    }
  return { double(std::sqrt(num/den)), double(mx/mxref) };
  }
template<typename T> static errs compare_r(const std::vector<T> &a, const std::vector<ld> &ref)
  {
  ld num=0, den=0, mx=0, mxref=0;
  for (size_t i=0; i<ref.size(); ++i)
    {
    ld d = ld(a[i]) - ref[i];
    num += d*d; den += ref[i]*ref[i];
    mx = std::max(mx, std::fabs(d)); mxref = std::max(mxref, std::fabs(ref[i]));
    }
  return { double(std::sqrt(num/den)), double(mx/mxref) };
  }

struct dumpio
  {
  FILE *out = nullptr, *cmp = nullptr;
  double worst = 0; size_t worst_n = 0; int worst_typ = -1; const char *worst_what = "";
  void put(const std::vector<double> &v, size_t n, int typ, const char *what, double tol, bool &ok)
    {
    if (out) fwrite(v.data(), sizeof(double), v.size(), out);
    if (cmp)
      {
      std::vector<double> w(v.size());
      if (fread(w.data(), sizeof(double), w.size(), cmp) != w.size())
        { fprintf(stderr, "CMP file too short at n=%zu typ=%d\n", n, typ); ok = false; return; }
      ld num=0, den=0;
      for (size_t i=0; i<v.size(); ++i) { ld d = ld(v[i])-ld(w[i]); num += d*d; den += ld(w[i])*ld(w[i]); }
      double rel = double(std::sqrt(num/(den>0?den:1)));
      if (rel > worst) { worst = rel; worst_n = n; worst_typ = typ; worst_what = what; }
      if (rel > tol) { printf("  CMP MISMATCH n=%zu typ=%d %s rel=%g\n", n, typ, what, rel); ok = false; }
      }
    }
  };

template<typename T> static bool run_one(size_t n, int typ, FILE *ref, dumpio &dio, double tol, errs &worst_f, errs &worst_b)
  {
  bool ok = true;
  uint64_t hdr[2];
  if (fread(hdr, sizeof hdr, 1, ref) != 1 || hdr[0] != n || int(hdr[1]) != typ)
    { fprintf(stderr, "reference file out of sync at n=%zu typ=%d\n", n, typ); exit(2); }
  size_t nh = n/2+1;
  std::vector<std::complex<ld>> sp(nh);
  for (size_t i=0; i<nh; ++i)
    {
    dd a, b;
    if (fread(&a, sizeof a, 1, ref)!=1 || fread(&b, sizeof b, 1, ref)!=1) { fprintf(stderr, "short reference file\n"); exit(2); }
    sp[i] = std::complex<ld>(from_dd(a), from_dd(b));
    }
  std::vector<double> xd; fill_input(xd, n, 12345u + uint32_t(typ));
  std::vector<T> x(n); for (size_t i=0; i<n; ++i) x[i] = T(xd[i]);
  std::vector<ld> xl(n); for (size_t i=0; i<n; ++i) xl[i] = ld(x[i]);

  shape_t shape{n}; stride_t sr{ptrdiff_t(sizeof(T))}, sc{ptrdiff_t(sizeof(std::complex<T>))};
  // forward
  std::vector<std::complex<T>> out(nh, std::complex<T>(T(-7), T(-7)));
  r2c<T>(shape, sr, sc, 0, FORWARD, x.data(), out.data(), T(1));
  errs ef = compare_c(out, sp);
  // backward: exact spectrum (rounded to T) -> c2r with fct=1/n should give x back
  std::vector<std::complex<T>> in(nh);
  for (size_t i=0; i<nh; ++i) in[i] = std::complex<T>(T(sp[i].real()), T(sp[i].imag()));
  std::vector<T> back(n, T(-7));
  c2r<T>(shape, sc, sr, 0, BACKWARD, in.data(), back.data(), T(1)/T(n));
  errs eb = compare_r(back, xl);
  worst_f.l2 = std::max(worst_f.l2, ef.l2); worst_f.maxabs = std::max(worst_f.maxabs, ef.maxabs);
  worst_b.l2 = std::max(worst_b.l2, eb.l2); worst_b.maxabs = std::max(worst_b.maxabs, eb.maxabs);
  if (!(ef.l2 <= tol) || !(eb.l2 <= tol) || !(ef.maxabs <= tol) || !(eb.maxabs <= tol))
    { printf("  FAIL n=%zu %s: r2c l2=%.3g max=%.3g  c2r l2=%.3g max=%.3g\n", n, typ?"float":"double", ef.l2, ef.maxabs, eb.l2, eb.maxabs); ok = false; }
  // dump / cross-compare
  std::vector<double> v; v.reserve(2*nh+n);
  for (auto &c : out) { v.push_back(double(c.real())); v.push_back(double(c.imag())); }
  dio.put(v, n, typ, "r2c", typ ? 1e-5 : 1e-12, ok);
  v.clear(); for (auto &c : back) v.push_back(double(c));
  dio.put(v, n, typ, "c2r", typ ? 1e-5 : 1e-12, ok);
  return ok;
  }

// multi-dimensional sanity check: exercises the generic (vtype_t, scalar template) path of rfftp::exec
template<typename T> static bool test_nd(double tol)
  {
  bool ok = true;
  const size_t n0 = 7, n1 = 96;   // 7 rows: vector path (vlen 4 for double / 8 for float) + scalar remainder
  std::vector<double> xd; fill_input(xd, n0*n1, 777u);
  std::vector<T> x(n0*n1); for (size_t i=0; i<x.size(); ++i) x[i] = T(xd[i]);
  // axis 1
    {
    std::vector<std::complex<T>> out(n0*(n1/2+1));
    shape_t shape{n0, n1}; stride_t si{ptrdiff_t(n1*sizeof(T)), ptrdiff_t(sizeof(T))},
      so{ptrdiff_t((n1/2+1)*sizeof(std::complex<T>)), ptrdiff_t(sizeof(std::complex<T>))};
    r2c<T>(shape, si, so, 1, FORWARD, x.data(), out.data(), T(1));
    for (size_t r=0; r<n0; ++r)
      {
      std::vector<std::complex<T>> o1(n1/2+1);
      shape_t s1{n1}; stride_t a{ptrdiff_t(sizeof(T))}, b{ptrdiff_t(sizeof(std::complex<T>))};
      r2c<T>(s1, a, b, 0, FORWARD, x.data()+r*n1, o1.data(), T(1));
      ld num=0, den=0;
      for (size_t i=0; i<o1.size(); ++i) { num += std::norm(std::complex<ld>(out[r*(n1/2+1)+i]) - std::complex<ld>(o1[i])); den += std::norm(std::complex<ld>(o1[i])); }
      double rel = double(std::sqrt(num/den));
      if (!(rel <= tol)) { printf("  FAIL nd axis1 row %zu rel=%g\n", r, rel); ok = false; }
      }
    }
  // axis 0 (length 7 -> radfg path, vector type)
    {
    std::vector<std::complex<T>> out((n0/2+1)*n1);
    shape_t shape{n0, n1}; stride_t si{ptrdiff_t(n1*sizeof(T)), ptrdiff_t(sizeof(T))},
      so{ptrdiff_t(n1*sizeof(std::complex<T>)), ptrdiff_t(sizeof(std::complex<T>))};
    r2c<T>(shape, si, so, 0, FORWARD, x.data(), out.data(), T(1));
    for (size_t c=0; c<n1; ++c)
      {
      std::vector<T> col(n0); for (size_t r=0; r<n0; ++r) col[r] = x[r*n1+c];
      std::vector<std::complex<T>> o1(n0/2+1);
      shape_t s1{n0}; stride_t a{ptrdiff_t(sizeof(T))}, b{ptrdiff_t(sizeof(std::complex<T>))};
      r2c<T>(s1, a, b, 0, FORWARD, col.data(), o1.data(), T(1));
      ld num=0, den=0;
      for (size_t i=0; i<o1.size(); ++i) { num += std::norm(std::complex<ld>(out[i*n1+c]) - std::complex<ld>(o1[i])); den += std::norm(std::complex<ld>(o1[i])); }
      double rel = double(std::sqrt(num/den));
      if (!(rel <= tol)) { printf("  FAIL nd axis0 col %zu rel=%g\n", c, rel); ok = false; }
      }
    }
  // 2-D c2r round trip along axis 1
    {
    std::vector<std::complex<T>> sp(n0*(n1/2+1));
    shape_t shape{n0, n1}; stride_t si{ptrdiff_t(n1*sizeof(T)), ptrdiff_t(sizeof(T))},
      so{ptrdiff_t((n1/2+1)*sizeof(std::complex<T>)), ptrdiff_t(sizeof(std::complex<T>))};
    r2c<T>(shape, si, so, 1, FORWARD, x.data(), sp.data(), T(1));
    std::vector<T> back(n0*n1);
    c2r<T>(shape, so, si, 1, BACKWARD, sp.data(), back.data(), T(1)/T(n1));
    ld num=0, den=0;
    for (size_t i=0; i<back.size(); ++i) { ld d = ld(back[i])-ld(x[i]); num += d*d; den += ld(x[i])*ld(x[i]); }
    double rel = double(std::sqrt(num/den));
    if (!(rel <= tol)) { printf("  FAIL nd c2r round trip rel=%g\n", rel); ok = false; }
    }
  return ok;
  }

static int check(int argc, char **argv)
  {
  FILE *ref = fopen(argv[2], "rb");
  if (!ref) { perror(argv[2]); return 1; }
  dumpio dio;
  if (argc>3) { dio.out = fopen(argv[3], "wb"); if (!dio.out) { perror(argv[3]); return 1; } }
  if (argc>4) { dio.cmp = fopen(argv[4], "rb"); if (!dio.cmp) { perror(argv[4]); return 1; } }
#ifdef POCKETFFT_RVV
  printf("build: RVV enabled (runtime enabled()=%d, vlmax e64m1=%zu)\n", int(detail::rvv::enabled()), detail::rvv::traits<double,1>::vlmax());
#else
  printf("build: scalar (no RVV)\n");
#endif
  auto lens = lengths();
  bool ok = true;
  errs wfd{0,0}, wbd{0,0}, wff{0,0}, wbf{0,0};
  const double told = 1e-13, tolf = 2e-5;
  for (size_t n : lens)
    {
    ok &= run_one<double>(n, 0, ref, dio, told, wfd, wbd);
    ok &= run_one<float >(n, 1, ref, dio, tolf, wff, wbf);
    }
  printf("%zu lengths (1..65536), float and double, r2c + c2r\n", lens.size());
  printf("worst double: r2c l2=%.3g max=%.3g | c2r l2=%.3g max=%.3g\n", wfd.l2, wfd.maxabs, wbd.l2, wbd.maxabs);
  printf("worst float : r2c l2=%.3g max=%.3g | c2r l2=%.3g max=%.3g\n", wff.l2, wff.maxabs, wbf.l2, wbf.maxabs);
  if (dio.cmp) printf("cross-build comparison: worst rel diff %.3g (n=%zu %s %s)\n", dio.worst, dio.worst_n, dio.worst_typ?"float":"double", dio.worst_what);
  bool nd = test_nd<double>(1e-13) && test_nd<float>(2e-5);
  printf("multi-D (generic vector path) check: %s\n", nd ? "ok" : "FAILED");
  ok &= nd;
  printf("%s\n", ok ? "ALL PASSED" : "FAILURES");
  if (dio.out) fclose(dio.out);
  if (dio.cmp) fclose(dio.cmp);
  fclose(ref);
  return ok ? 0 : 1;
  }

static int bench(int argc, char **argv)
  {
  size_t n = argc>2 ? std::strtoul(argv[2], nullptr, 10) : 65536;
  size_t reps = argc>3 ? std::strtoul(argv[3], nullptr, 10) : 1;
  bool c2rmode = argc>4 && std::strcmp(argv[4], "c2r")==0;
  std::vector<double> xd; fill_input(xd, n, 1u);
  std::vector<std::complex<double>> out(n/2+1);
  shape_t shape{n}; stride_t sr{ptrdiff_t(sizeof(double))}, sc{ptrdiff_t(sizeof(std::complex<double>))};
  double acc = 0;
  if (!c2rmode)
    for (size_t r=0; r<reps; ++r)
      { r2c<double>(shape, sr, sc, 0, FORWARD, xd.data(), out.data(), 1.0); acc += out[r % out.size()].real(); }
  else
    {
    r2c<double>(shape, sr, sc, 0, FORWARD, xd.data(), out.data(), 1.0);
    std::vector<double> back(n);
    for (size_t r=0; r<reps; ++r)
      { c2r<double>(shape, sc, sr, 0, BACKWARD, out.data(), back.data(), 1.0/double(n)); acc += back[r % n]; }
    }
  printf("bench n=%zu reps=%zu %s acc=%g\n", n, reps, c2rmode?"c2r":"r2c", acc);
  return 0;
  }

int main(int argc, char **argv)
  {
  if (argc>=3 && !std::strcmp(argv[1], "genref")) return genref(argv[2]);
  if (argc>=3 && !std::strcmp(argv[1], "check")) return check(argc, argv);
  if (argc>=2 && !std::strcmp(argv[1], "bench")) return bench(argc, argv);
  fprintf(stderr, "usage: %s genref REF | check REF [DUMP [CMP]] | bench N REPS [c2r]\n", argv[0]);
  return 1;
  }
