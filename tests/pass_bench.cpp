// pass_bench.cpp - per-pass instruction counting: calls one rfftp pass directly (scalar or RVV).
//   pass_bench N K DIR IMPL REPS   (K = factor index, DIR = f|b, IMPL = s|v)
// pre-include everything pocketfft needs, then make rfftp's private members reachable
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <memory>
#include <vector>
#include <complex>
#include <algorithm>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <functional>
#include <new>
#include <cstdio>
#include <cstring>
#define private public
#include "pocketfft_hdronly.h"
#undef private
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace pocketfft::detail;
int main(int argc, char **argv)
  {
  size_t n = std::strtoul(argv[1],0,10), K = std::strtoul(argv[2],0,10);
  bool fwd = argv[3][0]=='f', vec = argv[4][0]=='v';
  size_t reps = std::strtoul(argv[5],0,10);
  rfftp<double> plan(n);
  size_t nf = plan.fact.size();
  // (ido,l1) of factor K as used by exec
  size_t l1=1; for (size_t k=0;k<K;++k) l1*=plan.fact[k].fct;
  size_t ip = plan.fact[K].fct, ido = n/(ip*l1);
  std::vector<double> a(n), b(n);
  for (size_t i=0;i<n;++i) a[i] = double(i%7)-3;
  const double *tw = plan.fact[K].tw;
  for (size_t r=0; r<reps; ++r)
    {
    if (fwd)
      {
      if (ip==4) { if (vec) rvv::radf4(ido,l1,a.data(),b.data(),tw); else plan.radf4(ido,l1,a.data(),b.data(),tw); }
      else if (ip==2) { if (vec) rvv::radf2(ido,l1,a.data(),b.data(),tw); else plan.radf2(ido,l1,a.data(),b.data(),tw); }
      else if (ip==3) { if (vec) rvv::radf3(ido,l1,a.data(),b.data(),tw); else plan.radf3(ido,l1,a.data(),b.data(),tw); }
      else if (ip==5) { if (vec) rvv::radf5(ido,l1,a.data(),b.data(),tw); else plan.radf5(ido,l1,a.data(),b.data(),tw); }
      }
    else
      {
      if (ip==4) { if (vec) rvv::radb4(ido,l1,a.data(),b.data(),tw); else plan.radb4(ido,l1,a.data(),b.data(),tw); }
      else if (ip==2) { if (vec) rvv::radb2(ido,l1,a.data(),b.data(),tw); else plan.radb2(ido,l1,a.data(),b.data(),tw); }
      else if (ip==3) { if (vec) rvv::radb3(ido,l1,a.data(),b.data(),tw); else plan.radb3(ido,l1,a.data(),b.data(),tw); }
      else if (ip==5) { if (vec) rvv::radb5(ido,l1,a.data(),b.data(),tw); else plan.radb5(ido,l1,a.data(),b.data(),tw); }
      }
    }
  printf("n=%zu K=%zu/%zu ip=%zu ido=%zu l1=%zu %s %s reps=%zu chk=%g\n", n, K, nf, ip, ido, l1, fwd?"radf":"radb", vec?"rvv":"scalar", reps, b[n/3]);
  return 0;
  }
