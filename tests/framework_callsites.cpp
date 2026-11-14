// framework_callsites.cpp — exercises pocketfft exactly the way NumPy and PyTorch call it, to demonstrate
// interface adaptation: the RVV paths are reached through the unchanged public entry points.
//  * NumPy (numpy/fft/_pocketfft_umath.cpp): per-row plans via detail::get_plan<pocketfft_c<T>>() + exec(),
//    and the batched branch via pocketfft::c2c/r2c/c2r with shape {n_outer, n}, axes {1}.
//  * PyTorch (aten/src/ATen/native/mkl/SpectralOps.cpp, AT_POCKETFFT_ENABLED): pocketfft::c2c/r2c/c2r on the
//    whole tensor with byte strides and the transformed dims as axes (single-threaded).
// Results are cross-checked between the two routes and against a direct 1-D call; outputs are dumped for
// an external NumPy comparison.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include "../pocketfft_hdronly.h"
using namespace pocketfft;
static uint64_t lcg = 0x9E3779B97F4A7C15ULL;
static double rnd() { lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL; return double(int64_t(lcg >> 11)) / double(1ULL<<52); }
template<typename T> static double maxdiff(const std::vector<std::complex<T>>& a, const std::vector<std::complex<T>>& b)
  { double m=0, s=0; for (size_t i=0;i<a.size();i++){ m=std::max(m,(double)std::abs(a[i]-b[i])); s=std::max(s,(double)std::abs(b[i])); } return m/s; }
template<typename T> int run(size_t rows, size_t n, const std::string& pre)
  {
  using C = std::complex<T>;
  std::vector<C> in(rows*n); for (auto& v: in) v = C(T(rnd()), T(rnd()));
  // --- route 1: NumPy per-row path (plan + exec in place on a copy of each row)
  std::vector<C> out_np(rows*n);
  {
  auto plan = detail::get_plan<detail::pocketfft_c<T>>(n);
  for (size_t r=0; r<rows; ++r)
    {
    std::copy(in.begin()+r*n, in.begin()+(r+1)*n, out_np.begin()+r*n);
    plan->exec(reinterpret_cast<detail::cmplx<T>*>(out_np.data()+r*n), T(1), true);   // FORWARD
    }
  }
  // --- route 2: NumPy batched branch / PyTorch style: whole 2-D array, byte strides, axis 1
  std::vector<C> out_pt(rows*n);
  {
  shape_t shape{rows, n}; stride_t st{ptrdiff_t(n*sizeof(C)), ptrdiff_t(sizeof(C))}; shape_t axes{1};
  c2c<T>(shape, st, st, axes, FORWARD, in.data(), out_pt.data(), T(1));
  }
  // --- route 3: PyTorch style on a 3-D tensor [2, rows, n] transforming dim 2 with a non-contiguous outer layout
  std::vector<C> in3(2*rows*n), out3(2*rows*n);
  for (size_t i=0;i<rows*n;i++){ in3[i]=in[i]; in3[rows*n+i]=in[i]; }
  {
  shape_t shape{2, rows, n}; stride_t st{ptrdiff_t(rows*n*sizeof(C)), ptrdiff_t(n*sizeof(C)), ptrdiff_t(sizeof(C))}; shape_t axes{2};
  c2c<T>(shape, st, st, axes, FORWARD, in3.data(), out3.data(), T(1));
  }
  std::vector<C> out3a(out3.begin()+rows*n, out3.end());
  // --- real transforms through both routes
  std::vector<T> rin(rows*n); for (auto& v: rin) v = T(rnd());
  std::vector<C> rout_np(rows*(n/2+1)), rout_pt(rows*(n/2+1));
  {
  auto plan = detail::get_plan<detail::pocketfft_r<T>>(n);
  std::vector<T> tmp(n);
  for (size_t r=0; r<rows; ++r)
    {
    std::copy(rin.begin()+r*n, rin.begin()+(r+1)*n, tmp.begin());
    plan->exec(tmp.data(), T(1), true);
    // NumPy's own halfcomplex -> complex unpacking
    C* o = rout_np.data()+r*(n/2+1); o[0] = C(tmp[0], 0);
    size_t i=1, ii=1; for (; i<n-1; i+=2, ++ii) o[ii] = C(tmp[i], tmp[i+1]); if (i<n) o[ii] = C(tmp[i], 0);
    }
  shape_t shape{rows, n}; stride_t si{ptrdiff_t(n*sizeof(T)), ptrdiff_t(sizeof(T))}, so{ptrdiff_t((n/2+1)*sizeof(C)), ptrdiff_t(sizeof(C))}; shape_t axes{1};
  r2c<T>(shape, si, so, axes, FORWARD, rin.data(), rout_pt.data(), T(1));
  }
  double d12 = maxdiff(out_np, out_pt), d13 = maxdiff(out_np, out3a), dr = maxdiff(rout_np, rout_pt);
  printf("%s rows=%zu n=%zu : numpy-row vs batched/pytorch c2c %.2e ; vs 3-D pytorch %.2e ; r2c numpy-row vs pytorch %.2e\n",
    sizeof(T)==4?"float":"double", rows, n, d12, d13, dr);
  FILE* f=fopen((pre+".in").c_str(),"wb"); fwrite(in.data(), sizeof(C), n, f); fclose(f);   // first row for numpy check
  f=fopen((pre+".out").c_str(),"wb"); fwrite(out_pt.data(), sizeof(C), n, f); fclose(f);
  return (d12 < 1e-5 && d13 < 1e-5 && dr < 1e-5) ? 0 : 1;
  }
int main(int argc, char** argv)
  {
  size_t rows = argc>1 ? atoi(argv[1]) : 3, n = argc>2 ? atoi(argv[2]) : 1000; std::string pre = argc>3 ? argv[3] : "/tmp/fw";
  int rc = run<double>(rows, n, pre+"_d") | run<float>(rows, n, pre+"_f");
  printf(rc==0 ? "FRAMEWORK CALL-SITE CHECK PASS\n" : "FRAMEWORK CALL-SITE CHECK FAIL\n"); return rc;
  }
