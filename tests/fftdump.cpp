// fftdump: run pocketfft c2c (or r2c/c2r) on deterministic pseudo-random data and dump input/output
// usage: fftdump <c2c|r2c|c2r> <float|double> <n> <fwd:1|0> <outprefix> [reps]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <complex>
#include <string>
#include "../pocketfft_hdronly.h"
static uint64_t lcg = 0x9E3779B97F4A7C15ULL;
static double rnd() { lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL; return double(int64_t(lcg >> 11)) / double(1ULL<<52) ; }
template<typename T> int run(const char* kind, size_t n, bool fwd, const std::string& pre, int reps) {
  using namespace pocketfft;
  shape_t shape{n}; shape_t axes{0};
  if (!strcmp(kind,"c2c")) {
    std::vector<std::complex<T>> in(n), out(n);
    for (auto& v: in) v = std::complex<T>(T(rnd()), T(rnd()));
    stride_t st{ptrdiff_t(sizeof(std::complex<T>))};
    for (int r=0;r<reps;r++) c2c<T>(shape, st, st, axes, fwd, in.data(), out.data(), T(1));
    FILE* f=fopen((pre+".in").c_str(),"wb"); fwrite(in.data(), sizeof(std::complex<T>), n, f); fclose(f);
    f=fopen((pre+".out").c_str(),"wb"); fwrite(out.data(), sizeof(std::complex<T>), n, f); fclose(f);
  } else if (!strcmp(kind,"x2c")) {   // NumPy-style: plan->exec() in place on the output buffer
    std::vector<std::complex<T>> in(n), out(n);
    for (auto& v: in) v = std::complex<T>(T(rnd()), T(rnd()));
    auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(n);
    std::copy(in.begin(), in.end(), out.begin());
    // repeated in-place execution on the same buffer (values grow; fine for instruction counting). The dumped
    // output is correct only for reps==1.
    for (int r=0;r<reps;r++) plan->exec(reinterpret_cast<pocketfft::detail::cmplx<T>*>(out.data()), T(1), fwd);
    FILE* f=fopen((pre+".in").c_str(),"wb"); fwrite(in.data(), sizeof(std::complex<T>), n, f); fclose(f);
    f=fopen((pre+".out").c_str(),"wb"); fwrite(out.data(), sizeof(std::complex<T>), n, f); fclose(f);
  } else if (!strcmp(kind,"r2c")) {
    std::vector<T> in(n); std::vector<std::complex<T>> out(n/2+1);
    for (auto& v: in) v = T(rnd());
    stride_t si{ptrdiff_t(sizeof(T))}, so{ptrdiff_t(sizeof(std::complex<T>))};
    for (int r=0;r<reps;r++) r2c<T>(shape, si, so, axes, true, in.data(), out.data(), T(1));
    FILE* f=fopen((pre+".in").c_str(),"wb"); fwrite(in.data(), sizeof(T), n, f); fclose(f);
    f=fopen((pre+".out").c_str(),"wb"); fwrite(out.data(), sizeof(std::complex<T>), n/2+1, f); fclose(f);
  } else if (!strcmp(kind,"c2r")) {
    std::vector<std::complex<T>> in(n/2+1); std::vector<T> out(n);
    for (auto& v: in) v = std::complex<T>(T(rnd()), T(rnd()));
    in[0] = std::complex<T>(in[0].real(), 0); if (n%2==0) in[n/2] = std::complex<T>(in[n/2].real(), 0);
    stride_t si{ptrdiff_t(sizeof(std::complex<T>))}, so{ptrdiff_t(sizeof(T))};
    for (int r=0;r<reps;r++) c2r<T>(shape, si, so, axes, false, in.data(), out.data(), T(1));
    FILE* f=fopen((pre+".in").c_str(),"wb"); fwrite(in.data(), sizeof(std::complex<T>), n/2+1, f); fclose(f);
    f=fopen((pre+".out").c_str(),"wb"); fwrite(out.data(), sizeof(T), n, f); fclose(f);
  } else return 2;
  return 0;
}
int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: fftdump <c2c|r2c|c2r> <float|double> <n> <fwd> <outprefix> [reps]\n"); return 2; }
  size_t n = strtoull(argv[3], 0, 10); bool fwd = atoi(argv[4]); int reps = argc > 6 ? atoi(argv[6]) : 1;
  return !strcmp(argv[2],"float") ? run<float>(argv[1], n, fwd, argv[5], reps) : run<double>(argv[1], n, fwd, argv[5], reps);
}
