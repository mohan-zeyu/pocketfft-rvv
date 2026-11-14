// Native x86 reference: pocketfft (this header, scalar 1-D path on x86) vs FFTW 3.3.11 (MEASURE plans), ns per transform.
#include <chrono>
#include <complex>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <fftw3.h>
#define POCKETFFT_CACHE_SIZE 16
#include "../pocketfft_hdronly.h"
template<typename F> static double time_ns(F f, int reps){ double best=1e300; for(int t=0;t<5;t++){ auto s=std::chrono::steady_clock::now(); for(int r=0;r<reps;r++) f(); auto e=std::chrono::steady_clock::now(); best=std::min(best, std::chrono::duration<double,std::nano>(e-s).count()/reps);} return best; }
int main(){
  printf("%-5s %-9s %14s %14s %8s\n","kind","n","pocketfft_ns","fftw_ns","ratio");
  for (size_t n : {1024ul, 4096ul, 65536ul, 1048576ul}) {
    int reps = n<=4096 ? 2000 : (n<=65536 ? 200 : 10);
    { std::vector<std::complex<double>> in(n), out(n); for (size_t i=0;i<n;i++) in[i]={double(i%7),double(i%3)};
      pocketfft::shape_t sh{n}; pocketfft::stride_t st{ptrdiff_t(16)};
      double tp = time_ns([&]{ pocketfft::c2c<double>(sh, st, st, {0}, true, in.data(), out.data(), 1.0); }, reps);
      fftw_complex *fi=fftw_alloc_complex(n), *fo=fftw_alloc_complex(n); for (size_t i=0;i<n;i++){fi[i][0]=in[i].real(); fi[i][1]=in[i].imag();}
      fftw_plan p=fftw_plan_dft_1d(int(n), fi, fo, FFTW_FORWARD, FFTW_MEASURE); for (size_t i=0;i<n;i++){fi[i][0]=in[i].real(); fi[i][1]=in[i].imag();}
      double tf = time_ns([&]{ fftw_execute(p); }, reps);
      double md=0; for (size_t i=0;i<n;i++) md=std::max(md, std::abs(out[i]-std::complex<double>(fo[i][0],fo[i][1])));
      printf("%-5s %-9zu %14.0f %14.0f %8.2f   (max|diff|=%.1e)\n","c2c",n,tp,tf,tp/tf,md); fftw_destroy_plan(p); fftw_free(fi); fftw_free(fo); }
    { std::vector<double> in(n); std::vector<std::complex<double>> out(n/2+1); for (size_t i=0;i<n;i++) in[i]=double(i%7);
      pocketfft::shape_t sh{n}; pocketfft::stride_t si{ptrdiff_t(8)}, so{ptrdiff_t(16)};
      double tp = time_ns([&]{ pocketfft::r2c<double>(sh, si, so, {0}, true, in.data(), out.data(), 1.0); }, reps);
      double *fi=fftw_alloc_real(n); fftw_complex *fo=fftw_alloc_complex(n/2+1); fftw_plan p=fftw_plan_dft_r2c_1d(int(n), fi, fo, FFTW_MEASURE); for (size_t i=0;i<n;i++) fi[i]=in[i];
      double tf = time_ns([&]{ fftw_execute(p); }, reps);
      double md=0; for (size_t i=0;i<=n/2;i++) md=std::max(md, std::abs(out[i]-std::complex<double>(fo[i][0],fo[i][1])));
      printf("%-5s %-9zu %14.0f %14.0f %8.2f   (max|diff|=%.1e)\n","r2c",n,tp,tf,tp/tf,md); fftw_destroy_plan(p); fftw_free(fi); fftw_free(fo); }
  }
  return 0; }
