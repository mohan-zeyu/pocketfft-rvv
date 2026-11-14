/*
 * pocketfft_rvv_real.h - RISC-V Vector (RVV 1.0) implementations of the real-input FFTPACK passes
 * (radf2/3/4/5, radb2/3/4/5) of pocketfft (C++ header-only version).
 *
 * Builds on the helper layer in pocketfft_rvv.h (traits<T,LMUL> / ops<T,LMUL>).
 * Vector-length agnostic: every loop uses vsetvl, so one binary runs on VLEN=128/256/512/1024.
 *
 * Copyright (C) 2026 RISC-V FFT optimization project. BSD-3-Clause (same as pocketfft).
 */
#ifndef POCKETFFT_RVV_REAL_H
#define POCKETFFT_RVV_REAL_H

#include "pocketfft_rvv.h"

#ifdef POCKETFFT_RVV

// GCC's inline-size estimate for functions taking/returning RVV tuple types is far too large, so at
// -O2 the small helpers hit --param max-inline-insns-single and are emitted out of line (one call
// per vector add). Force inlining. (Also defined by newer versions of pocketfft_rvv.h.)
#ifndef POCKETFFT_RVV_INLINE
#if defined(__GNUC__) || defined(__clang__)
#define POCKETFFT_RVV_INLINE inline __attribute__((always_inline))
#else
#define POCKETFFT_RVV_INLINE inline
#endif
#endif

#ifdef POCKETFFT_RVV_INSIDE_DETAIL
namespace rvv {
#else
namespace pocketfft {
namespace detail {
namespace rvv {
#endif

// =====================================================================================
// Real (half-complex, FFTPACK layout) passes for single, non-batched transforms.
//
//   forward  (radf*): CC(a,b,c) = cc[a+ido*(b+l1*c)]  (input),  CH(a,b,c) = ch[a+ido*(b+ip*c)] (output)
//   backward (radb*): CC(a,b,c) = cc[a+ido*(b+ip*c)]  (input),  CH(a,b,c) = ch[a+ido*(b+l1*c)] (output)
//   WA(x,i) = wa[i+x*(ido-1)]
//
// The scalar passes have three parts:
//   1. the "edge" loop over k for element 0 (and element ido-1 of the outputs),
//   2. for even ido, a second loop over k for element ido-1 (the Nyquist element of each column),
//   3. the main loop  for k<l1: for (i=2; i<ido; i+=2) { ic=ido-i; ... }
//      which handles the element pairs (i-1,i) and (ic-1,ic). Each pair is a complex number (re,im)
//      stored in natural address order. With the pair index m = i/2-1 (m = 0..nvec-1, nvec=(ido-1)/2)
//      the (i-1,i) pairs of consecutive m are consecutive in memory (stride +2 elements), while the
//      (ic-1,ic) pairs run backwards (stride -2 elements). The twiddles WA(x,i-2),WA(x,i-1) are the
//      complex numbers wa[x*(ido-1) + 2m], wa[x*(ido-1) + 2m + 1].
//
// Parts 1 and 2 are vectorized across k (strided scalar-element loads/stores, no twiddles).
// Part 3 is vectorized either across m (mode "across i": unit-stride segment loads/stores, negative
// unit-stride segment accesses for the (ic-1,ic) pairs, twiddle vectors loaded from memory) or, when
// ido is small, across k (strided segment loads/stores, broadcast twiddles).
// =====================================================================================

// Element types the RVV passes exist for. real_t<T> is T for float/double and an unusable tag type
// otherwise, so that an overload taking real_t<T0>* is never viable for other T0 (e.g. long double).

// A few extra complex-vector helpers on top of ops<T0,LM>.
template<typename T0, size_t LM> struct rops : ops<T0, LM>
  {
  using O    = ops<T0, LM>;
  using R    = traits<T0, LM>;
  using vec  = typename R::vec;
  using cvec = typename R::vec2;
  // a + conj(b)
  static POCKETFFT_RVV_INLINE cvec addc(cvec a, cvec b, size_t vl)
    { return O::mk(R::add(O::re(a), O::re(b), vl), R::sub(O::im(a), O::im(b), vl)); }
  // a - conj(b)
  static POCKETFFT_RVV_INLINE cvec subc(cvec a, cvec b, size_t vl)
    { return O::mk(R::sub(O::re(a), O::re(b), vl), R::add(O::im(a), O::im(b), vl)); }
  // conj(a - b)
  static POCKETFFT_RVV_INLINE cvec csub(cvec a, cvec b, size_t vl)
    { return O::mk(R::sub(O::re(a), O::re(b), vl), R::sub(O::im(b), O::im(a), vl)); }
  // a + f*b + g*c   (f, g scalars)
  static POCKETFFT_RVV_INLINE cvec fma2(cvec a, T0 f, cvec b, T0 g, cvec c, size_t vl)
    { return O::fmaf(O::fmaf(a, f, b, vl), g, c, vl); }
  // f*b + g*c
  static POCKETFFT_RVV_INLINE cvec lin2(T0 f, cvec b, T0 g, cvec c, size_t vl)
    { return O::fmaf(O::mulf(b, f, vl), g, c, vl); }
  // f*b - g*c
  static POCKETFFT_RVV_INLINE cvec lin2m(T0 f, cvec b, T0 g, cvec c, size_t vl)
    { return O::fnmsf(O::mulf(b, f, vl), g, c, vl); }
  };

// Access policy for the main (i) loop. AK=false: lanes run across the pair index m (fixed k);
// AK=true: lanes run across k (fixed m). Pointer arguments always address the pair of lane 0.
template<typename T0, size_t LM, bool AK> struct racc
  {
  using R    = traits<T0, LM>;
  using O    = ops<T0, LM>;
  using vec  = typename R::vec;
  using cvec = typename R::vec2;
  static constexpr ptrdiff_t neg2 = -ptrdiff_t(2*sizeof(T0));
  size_t vl = 0;
  ptrdiff_t bs_in = 0, bs_out = 0;        // AK: byte stride between consecutive k (input / output array)
  const T0 *w = nullptr;                  // points at WA(0,i-2) of lane 0 (AK=false) / of the current i (AK=true)
  ptrdiff_t wstride = 0;                  // ido-1
  // pairs (i-1,i): consecutive m are consecutive pairs
  POCKETFFT_RVV_INLINE cvec ld_i(const T0 *p) const
    { return AK ? R::lsseg2(p, bs_in, vl) : R::lseg2(p, vl); }
  // pairs (ic-1,ic): consecutive m move backwards by one pair
  POCKETFFT_RVV_INLINE cvec ld_ic(const T0 *p) const
    { return AK ? R::lsseg2(p, bs_in, vl) : R::lsseg2(p, neg2, vl); }
  POCKETFFT_RVV_INLINE void st_i(T0 *p, cvec v) const
    { if (AK) R::ssseg2(p, bs_out, v, vl); else R::sseg2(p, v, vl); }
  POCKETFFT_RVV_INLINE void st_ic(T0 *p, cvec v) const
    { if (AK) R::ssseg2(p, bs_out, v, vl); else R::ssseg2(p, neg2, v, vl); }
  POCKETFFT_RVV_INLINE cvec tw(size_t x) const
    {
    const T0 *q = w + ptrdiff_t(x)*wstride;
    return AK ? O::mk(R::splat(q[0], vl), R::splat(q[1], vl)) : R::lseg2(q, vl);
    }
  };

// ---- main-loop driver -----------------------------------------------------------------
// Calls BF::template run<T0,LM>(c, cc, ch, ido, l1, i, k) for every (i,k) of the main loop, with
// i and k those of lane 0. FWD selects the array layout (strides of the across-k mode).
template<typename T0, size_t LM, bool FWD, typename BF>
static POCKETFFT_RVV_INLINE void rdrive(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  {
  using O = ops<T0, LM>;
  constexpr size_t ip = BF::ip;
  const size_t nvec = (ido-1)/2;
  if (nvec==0) return;
  const size_t vlmax = O::vlmax();
  if (nvec >= (vlmax+1)/2 || nvec >= l1)
    {
    // lanes across i
    racc<T0,LM,false> c;
    c.wstride = ptrdiff_t(ido-1);
    for (size_t k=0; k<l1; ++k)
      for (size_t m=0; m<nvec; m+=c.vl)
        {
        c.vl = O::vsetvl(nvec-m);
        c.w = wa + 2*m;
        BF::template run<T0,LM>(c, cc, ch, ido, l1, 2*m+2, k);
        }
    }
  else
    {
    // lanes across k
    racc<T0,LM,true> c;
    c.wstride = ptrdiff_t(ido-1);
    c.bs_in  = ptrdiff_t((FWD ? ido : ido*ip)*sizeof(T0));
    c.bs_out = ptrdiff_t((FWD ? ido*ip : ido)*sizeof(T0));
    for (size_t m=0; m<nvec; ++m)
      {
      c.w = wa + 2*m;
      for (size_t k=0; k<l1; k+=c.vl)
        {
        c.vl = O::vsetvl(l1-k);
        BF::template run<T0,LM>(c, cc, ch, ido, l1, 2*m+2, k);
        }
      }
    }
  }

// =====================================================================================
// radix 2
// =====================================================================================
struct rf2
  {
  static constexpr size_t ip = 2;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + (i-1) + ido*k;      // CC(i-1,k,j) = p + ido*l1*j
    T0 *q = ch + ido*(ip*k);               // CH(a,j,k)   = q + a + ido*j
    auto c0 = c.ld_i(p);
    auto t  = O::twmul(true, c.ld_i(p + ido*l1), c.tw(0), vl);
    c.st_i (q + (i-1),        O::add (c0, t, vl));
    c.st_ic(q + (ic-1) + ido, O::csub(c0, t, vl));
    }
  // k loops: element 0 of each column, and element ido-1 for even ido
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    const ptrdiff_t bi = ptrdiff_t(ido*sizeof(T0)), bo = ptrdiff_t(ip*ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ido*k;
      T0 *q = ch + ip*ido*k;
      auto c0 = R::loads(p, bi, vl), c1 = R::loads(p+ido*l1, bi, vl);
      R::stores(q,             bo, R::add(c0, c1, vl), vl);   // CH(0,0,k)
      R::stores(q+ido-1+ido,   bo, R::sub(c0, c1, vl), vl);   // CH(ido-1,1,k)
      if ((ido&1)==0)
        {
        auto e0 = R::loads(p+ido-1, bi, vl), e1 = R::loads(p+ido-1+ido*l1, bi, vl);
        R::stores(q+ido,     bo, R::neg(e1, vl), vl);          // CH(0,1,k)     = -CC(ido-1,k,1)
        R::stores(q+ido-1,   bo, e0, vl);                      // CH(ido-1,0,k) =  CC(ido-1,k,0)
        }
      k += vl;
      }
    }
  };

struct rb2
  {
  static constexpr size_t ip = 2;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + ido*(ip*k);         // CC(a,j,k) = p + a + ido*j
    T0 *q = ch + (i-1) + ido*k;            // CH(i-1,k,j) = q + ido*l1*j
    auto a = c.ld_i (p + (i-1));
    auto b = c.ld_ic(p + (ic-1) + ido);
    c.st_i(q,         O::addc(a, b, vl));
    c.st_i(q+ido*l1,  O::twmul(false, O::subc(a, b, vl), c.tw(0), vl));
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    const ptrdiff_t bi = ptrdiff_t(ip*ido*sizeof(T0)), bo = ptrdiff_t(ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ip*ido*k;
      T0 *q = ch + ido*k;
      auto c0 = R::loads(p, bi, vl), c1 = R::loads(p+ido-1+ido, bi, vl);
      R::stores(q,        bo, R::add(c0, c1, vl), vl);          // CH(0,k,0)
      R::stores(q+ido*l1, bo, R::sub(c0, c1, vl), vl);          // CH(0,k,1)
      if ((ido&1)==0)
        {
        auto e0 = R::loads(p+ido-1, bi, vl), e1 = R::loads(p+ido, bi, vl);
        R::stores(q+ido-1,        bo, R::mulf(e0, T0(2), vl), vl);   // CH(ido-1,k,0) =  2*CC(ido-1,0,k)
        R::stores(q+ido-1+ido*l1, bo, R::mulf(e1, T0(-2), vl), vl);  // CH(ido-1,k,1) = -2*CC(0,1,k)
        }
      k += vl;
      }
    }
  };

// =====================================================================================
// radix 3   (ido is always odd for ip=3 in pocketfft's factor order)
// =====================================================================================
struct rf3
  {
  static constexpr size_t ip = 3;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    constexpr T0 taur = T0(-0.5), taui = T0(0.8660254037844386467637231707529362L);
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + (i-1) + ido*k;
    T0 *q = ch + ido*(ip*k);
    auto c0 = c.ld_i(p);
    auto d2 = O::twmul(true, c.ld_i(p + ido*l1),   c.tw(0), vl);
    auto d3 = O::twmul(true, c.ld_i(p + 2*ido*l1), c.tw(1), vl);
    auto s  = O::add(d2, d3, vl);                       // d2+d3
    auto u  = O::rot90(false, O::sub(d3, d2, vl), vl);  // i*(d3-d2)
    c.st_i(q + (i-1), O::add(c0, s, vl));               // CH(i,0)
    auto t2 = O::fmaf(c0, taur, s, vl);                 // c0 + taur*s
    auto t3 = O::mulf(u, taui, vl);
    c.st_i (q + (i-1) + 2*ido, O::add (t2, t3, vl));    // CH(i,2)
    c.st_ic(q + (ic-1) + ido,  O::csub(t2, t3, vl));    // CH(ic,1)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 taur = T0(-0.5), taui = T0(0.8660254037844386467637231707529362L);
    const ptrdiff_t bi = ptrdiff_t(ido*sizeof(T0)), bo = ptrdiff_t(ip*ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ido*k;
      T0 *q = ch + ip*ido*k;
      auto c0 = R::loads(p, bi, vl), c1 = R::loads(p+ido*l1, bi, vl), c2 = R::loads(p+2*ido*l1, bi, vl);
      auto cr2 = R::add(c1, c2, vl);
      R::stores(q,           bo, R::add(c0, cr2, vl), vl);                 // CH(0,0,k)
      R::stores(q+2*ido,     bo, R::mulf(R::sub(c2, c1, vl), taui, vl), vl); // CH(0,2,k)
      R::stores(q+ido-1+ido, bo, R::fmaccf(c0, taur, cr2, vl), vl);        // CH(ido-1,1,k)
      k += vl;
      }
    }
  };

struct rb3
  {
  static constexpr size_t ip = 3;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    constexpr T0 taur = T0(-0.5), taui = T0(0.8660254037844386467637231707529362L);
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + ido*(ip*k);
    T0 *q = ch + (i-1) + ido*k;
    auto a0 = c.ld_i (p + (i-1));
    auto a2 = c.ld_i (p + (i-1) + 2*ido);
    auto b1 = c.ld_ic(p + (ic-1) + ido);
    auto t2 = O::addc(a2, b1, vl);                       // a2 + conj(b1)
    auto c3 = O::mulf(O::subc(a2, b1, vl), taui, vl);    // taui*(a2 - conj(b1))
    c.st_i(q, O::add(a0, t2, vl));                       // CH(i,k,0)
    auto c2 = O::fmaf(a0, taur, t2, vl);                 // a0 + taur*t2
    auto r3 = O::rot90(false, c3, vl);                   // i*c3
    c.st_i(q+ido*l1,   O::twmul(false, O::add(c2, r3, vl), c.tw(0), vl));   // CH(i,k,1) = w0*(c2+i*c3)
    c.st_i(q+2*ido*l1, O::twmul(false, O::sub(c2, r3, vl), c.tw(1), vl));   // CH(i,k,2) = w1*(c2-i*c3)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 taur = T0(-0.5), taui = T0(0.8660254037844386467637231707529362L);
    const ptrdiff_t bi = ptrdiff_t(ip*ido*sizeof(T0)), bo = ptrdiff_t(ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ip*ido*k;
      T0 *q = ch + ido*k;
      auto c0  = R::loads(p, bi, vl);
      auto tr2 = R::mulf(R::loads(p+ido-1+ido, bi, vl), T0(2), vl);     // 2*CC(ido-1,1,k)
      auto cr2 = R::fmaccf(c0, taur, tr2, vl);
      R::stores(q, bo, R::add(c0, tr2, vl), vl);                        // CH(0,k,0)
      auto ci3 = R::mulf(R::loads(p+2*ido, bi, vl), T0(2)*taui, vl);    // 2*taui*CC(0,2,k)
      R::stores(q+2*ido*l1, bo, R::add(cr2, ci3, vl), vl);              // CH(0,k,2)
      R::stores(q+ido*l1,   bo, R::sub(cr2, ci3, vl), vl);              // CH(0,k,1)
      k += vl;
      }
    }
  };

// =====================================================================================
// radix 4
// =====================================================================================
struct rf4
  {
  static constexpr size_t ip = 4;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + (i-1) + ido*k;      // CC(i-1,k,j) = p + ido*l1*j
    T0 *q = ch + ido*(ip*k);               // CH(a,j,k)   = q + a + ido*j
    auto c0 = c.ld_i(p);
    auto d1 = O::twmul(true, c.ld_i(p +   ido*l1), c.tw(0), vl);
    auto d2 = O::twmul(true, c.ld_i(p + 2*ido*l1), c.tw(1), vl);
    auto d3 = O::twmul(true, c.ld_i(p + 3*ido*l1), c.tw(2), vl);
    auto t1 = O::add(d1, d3, vl), t4 = O::sub(d1, d3, vl);
    auto t2 = O::add(c0, d2, vl), t3 = O::sub(c0, d2, vl);
    c.st_i (q + (i-1),          O::add (t2, t1, vl));   // CH(i,0)
    c.st_ic(q + (ic-1) + 3*ido, O::csub(t2, t1, vl));   // CH(ic,3)
    auto u = O::rot90(true, t4, vl);                    // -i*t4
    c.st_i (q + (i-1) + 2*ido,  O::add (t3, u, vl));    // CH(i,2)
    c.st_ic(q + (ic-1) + ido,   O::csub(t3, u, vl));    // CH(ic,1)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 hsqt2 = T0(0.707106781186547524400844362104849L);
    const ptrdiff_t bi = ptrdiff_t(ido*sizeof(T0)), bo = ptrdiff_t(ip*ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ido*k;
      T0 *q = ch + ip*ido*k;
      auto c0 = R::loads(p, bi, vl), c1 = R::loads(p+ido*l1, bi, vl),
           c2 = R::loads(p+2*ido*l1, bi, vl), c3 = R::loads(p+3*ido*l1, bi, vl);
      auto tr1 = R::add(c3, c1, vl);
      R::stores(q+2*ido,       bo, R::sub(c3, c1, vl), vl);      // CH(0,2,k)
      auto tr2 = R::add(c0, c2, vl);
      R::stores(q+ido-1+ido,   bo, R::sub(c0, c2, vl), vl);      // CH(ido-1,1,k)
      R::stores(q,             bo, R::add(tr2, tr1, vl), vl);    // CH(0,0,k)
      R::stores(q+ido-1+3*ido, bo, R::sub(tr2, tr1, vl), vl);    // CH(ido-1,3,k)
      if ((ido&1)==0)
        {
        auto e0 = R::loads(p+ido-1, bi, vl), e1 = R::loads(p+ido-1+ido*l1, bi, vl),
             e2 = R::loads(p+ido-1+2*ido*l1, bi, vl), e3 = R::loads(p+ido-1+3*ido*l1, bi, vl);
        auto ti1 = R::mulf(R::add(e1, e3, vl), -hsqt2, vl);
        auto tr1b= R::mulf(R::sub(e1, e3, vl),  hsqt2, vl);
        R::stores(q+ido-1,       bo, R::add(e0, tr1b, vl), vl);  // CH(ido-1,0,k)
        R::stores(q+ido-1+2*ido, bo, R::sub(e0, tr1b, vl), vl);  // CH(ido-1,2,k)
        R::stores(q+3*ido,       bo, R::add(ti1, e2, vl), vl);   // CH(0,3,k)
        R::stores(q+ido,         bo, R::sub(ti1, e2, vl), vl);   // CH(0,1,k)
        }
      k += vl;
      }
    }
  };

struct rb4
  {
  static constexpr size_t ip = 4;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + ido*(ip*k);         // CC(a,j,k)   = p + a + ido*j
    T0 *q = ch + (i-1) + ido*k;            // CH(i-1,k,j) = q + ido*l1*j
    auto a0 = c.ld_i (p + (i-1));
    auto b3 = c.ld_ic(p + (ic-1) + 3*ido);
    auto a2 = c.ld_i (p + (i-1) + 2*ido);
    auto b1 = c.ld_ic(p + (ic-1) + ido);
    auto t2 = O::addc(a0, b3, vl), t1 = O::subc(a0, b3, vl);   // a0 +- conj(b3)
    auto t3 = O::addc(a2, b1, vl), t4 = O::subc(a2, b1, vl);   // a2 +- conj(b1)
    c.st_i(q, O::add(t2, t3, vl));                                              // CH(i,k,0)
    c.st_i(q+2*ido*l1, O::twmul(false, O::sub(t2, t3, vl), c.tw(1), vl));       // CH(i,k,2) = w1*(t2-t3)
    auto r4 = O::rot90(false, t4, vl);                                          // i*t4
    c.st_i(q+  ido*l1, O::twmul(false, O::add(t1, r4, vl), c.tw(0), vl));       // CH(i,k,1) = w0*(t1+i*t4)
    c.st_i(q+3*ido*l1, O::twmul(false, O::sub(t1, r4, vl), c.tw(2), vl));       // CH(i,k,3) = w2*(t1-i*t4)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 sqrt2 = T0(1.414213562373095048801688724209698L);
    const ptrdiff_t bi = ptrdiff_t(ip*ido*sizeof(T0)), bo = ptrdiff_t(ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ip*ido*k;
      T0 *q = ch + ido*k;
      auto c0 = R::loads(p, bi, vl), c3 = R::loads(p+ido-1+3*ido, bi, vl);
      auto tr2 = R::add(c0, c3, vl), tr1 = R::sub(c0, c3, vl);
      auto tr3 = R::mulf(R::loads(p+ido-1+ido, bi, vl), T0(2), vl);   // 2*CC(ido-1,1,k)
      auto tr4 = R::mulf(R::loads(p+2*ido, bi, vl), T0(2), vl);       // 2*CC(0,2,k)
      R::stores(q,          bo, R::add(tr2, tr3, vl), vl);   // CH(0,k,0)
      R::stores(q+2*ido*l1, bo, R::sub(tr2, tr3, vl), vl);   // CH(0,k,2)
      R::stores(q+3*ido*l1, bo, R::add(tr1, tr4, vl), vl);   // CH(0,k,3)
      R::stores(q+  ido*l1, bo, R::sub(tr1, tr4, vl), vl);   // CH(0,k,1)
      if ((ido&1)==0)
        {
        auto e3 = R::loads(p+3*ido, bi, vl), e1 = R::loads(p+ido, bi, vl);           // CC(0,3,k), CC(0,1,k)
        auto e0 = R::loads(p+ido-1, bi, vl), e2 = R::loads(p+ido-1+2*ido, bi, vl);   // CC(ido-1,0,k), CC(ido-1,2,k)
        auto ti1 = R::add(e3, e1, vl), ti2 = R::sub(e3, e1, vl);
        auto tr2b= R::add(e0, e2, vl), tr1b= R::sub(e0, e2, vl);
        R::stores(q+ido-1,          bo, R::add(tr2b, tr2b, vl), vl);                       // CH(ido-1,k,0)
        R::stores(q+ido-1+  ido*l1, bo, R::mulf(R::sub(tr1b, ti1, vl),  sqrt2, vl), vl);   // CH(ido-1,k,1)
        R::stores(q+ido-1+2*ido*l1, bo, R::add(ti2, ti2, vl), vl);                         // CH(ido-1,k,2)
        R::stores(q+ido-1+3*ido*l1, bo, R::mulf(R::add(tr1b, ti1, vl), -sqrt2, vl), vl);   // CH(ido-1,k,3)
        }
      k += vl;
      }
    }
  };

// =====================================================================================
// radix 5   (ido is always odd for ip=5 in pocketfft's factor order)
// =====================================================================================
struct rf5
  {
  static constexpr size_t ip = 5;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    constexpr T0 tr11 = T0(0.3090169943749474241022934171828191L),
                 ti11 = T0(0.9510565162951535721164393333793821L),
                 tr12 = T0(-0.8090169943749474241022934171828191L),
                 ti12 = T0(0.5877852522924731291687059546390728L);
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + (i-1) + ido*k;
    T0 *q = ch + ido*(ip*k);
    auto c0 = c.ld_i(p);
    auto d2 = O::twmul(true, c.ld_i(p +   ido*l1), c.tw(0), vl);
    auto d3 = O::twmul(true, c.ld_i(p + 2*ido*l1), c.tw(1), vl);
    auto d4 = O::twmul(true, c.ld_i(p + 3*ido*l1), c.tw(2), vl);
    auto d5 = O::twmul(true, c.ld_i(p + 4*ido*l1), c.tw(3), vl);
    auto s2 = O::add(d2, d5, vl), u5 = O::rot90(false, O::sub(d5, d2, vl), vl);   // d2+d5, i*(d5-d2)
    auto s3 = O::add(d3, d4, vl), u4 = O::rot90(false, O::sub(d4, d3, vl), vl);   // d3+d4, i*(d4-d3)
    c.st_i(q + (i-1), O::add(O::add(c0, s2, vl), s3, vl));          // CH(i,0)
    auto t2 = O::fma2(c0, tr11, s2, tr12, s3, vl);
    auto t3 = O::fma2(c0, tr12, s2, tr11, s3, vl);
    auto t5 = O::lin2 (ti11, u5, ti12, u4, vl);                     // ti11*u5 + ti12*u4
    auto t4 = O::lin2m(ti12, u5, ti11, u4, vl);                     // ti12*u5 - ti11*u4
    c.st_i (q + (i-1)  + 2*ido, O::add (t2, t5, vl));               // CH(i,2)
    c.st_ic(q + (ic-1) +   ido, O::csub(t2, t5, vl));               // CH(ic,1)
    c.st_i (q + (i-1)  + 4*ido, O::add (t3, t4, vl));               // CH(i,4)
    c.st_ic(q + (ic-1) + 3*ido, O::csub(t3, t4, vl));               // CH(ic,3)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 tr11 = T0(0.3090169943749474241022934171828191L),
                 ti11 = T0(0.9510565162951535721164393333793821L),
                 tr12 = T0(-0.8090169943749474241022934171828191L),
                 ti12 = T0(0.5877852522924731291687059546390728L);
    const ptrdiff_t bi = ptrdiff_t(ido*sizeof(T0)), bo = ptrdiff_t(ip*ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ido*k;
      T0 *q = ch + ip*ido*k;
      auto c0 = R::loads(p, bi, vl), c1 = R::loads(p+ido*l1, bi, vl), c2 = R::loads(p+2*ido*l1, bi, vl),
           c3 = R::loads(p+3*ido*l1, bi, vl), c4 = R::loads(p+4*ido*l1, bi, vl);
      auto cr2 = R::add(c4, c1, vl), ci5 = R::sub(c4, c1, vl);
      auto cr3 = R::add(c3, c2, vl), ci4 = R::sub(c3, c2, vl);
      R::stores(q,             bo, R::add(R::add(c0, cr2, vl), cr3, vl), vl);                   // CH(0,0,k)
      R::stores(q+ido-1+ido,   bo, R::fmaccf(R::fmaccf(c0, tr11, cr2, vl), tr12, cr3, vl), vl); // CH(ido-1,1,k)
      R::stores(q+2*ido,       bo, R::fmaccf(R::mulf(ci5, ti11, vl), ti12, ci4, vl), vl);       // CH(0,2,k)
      R::stores(q+ido-1+3*ido, bo, R::fmaccf(R::fmaccf(c0, tr12, cr2, vl), tr11, cr3, vl), vl); // CH(ido-1,3,k)
      R::stores(q+4*ido,       bo, R::fnmsacf(R::mulf(ci5, ti12, vl), ti11, ci4, vl), vl);      // CH(0,4,k)
      k += vl;
      }
    }
  };

struct rb5
  {
  static constexpr size_t ip = 5;
  template<typename T0, size_t LM, bool AK>
  static POCKETFFT_RVV_INLINE void run(const racc<T0,LM,AK> &c, const T0 *cc, T0 *ch, size_t ido, size_t l1, size_t i, size_t k)
    {
    using O = rops<T0, LM>;
    constexpr T0 tr11 = T0(0.3090169943749474241022934171828191L),
                 ti11 = T0(0.9510565162951535721164393333793821L),
                 tr12 = T0(-0.8090169943749474241022934171828191L),
                 ti12 = T0(0.5877852522924731291687059546390728L);
    const size_t vl = c.vl, ic = ido-i;
    const T0 *p = cc + ido*(ip*k);
    T0 *q = ch + (i-1) + ido*k;
    auto a0 = c.ld_i (p + (i-1));
    auto a2 = c.ld_i (p + (i-1) + 2*ido);
    auto b1 = c.ld_ic(p + (ic-1) + ido);
    auto a4 = c.ld_i (p + (i-1) + 4*ido);
    auto b3 = c.ld_ic(p + (ic-1) + 3*ido);
    auto t2 = O::addc(a2, b1, vl), t5 = O::subc(a2, b1, vl);   // a2 +- conj(b1)
    auto t3 = O::addc(a4, b3, vl), t4 = O::subc(a4, b3, vl);   // a4 +- conj(b3)
    c.st_i(q, O::add(O::add(a0, t2, vl), t3, vl));             // CH(i,k,0)
    auto c2 = O::fma2(a0, tr11, t2, tr12, t3, vl);
    auto c3 = O::fma2(a0, tr12, t2, tr11, t3, vl);
    auto c5 = O::lin2 (ti11, t5, ti12, t4, vl);                // ti11*t5 + ti12*t4
    auto c4 = O::lin2m(ti12, t5, ti11, t4, vl);                // ti12*t5 - ti11*t4
    auto r5 = O::rot90(false, c5, vl), r4 = O::rot90(false, c4, vl);   // i*c5, i*c4
    c.st_i(q+  ido*l1, O::twmul(false, O::add(c2, r5, vl), c.tw(0), vl));   // CH(i,k,1) = w0*(c2+i*c5)
    c.st_i(q+2*ido*l1, O::twmul(false, O::add(c3, r4, vl), c.tw(1), vl));   // CH(i,k,2) = w1*(c3+i*c4)
    c.st_i(q+3*ido*l1, O::twmul(false, O::sub(c3, r4, vl), c.tw(2), vl));   // CH(i,k,3) = w2*(c3-i*c4)
    c.st_i(q+4*ido*l1, O::twmul(false, O::sub(c2, r5, vl), c.tw(3), vl));   // CH(i,k,4) = w3*(c2-i*c5)
    }
  template<typename T0, size_t LM>
  static void edge(size_t ido, size_t l1, const T0 *cc, T0 *ch)
    {
    using R = traits<T0, LM>;
    constexpr T0 tr11 = T0(0.3090169943749474241022934171828191L),
                 ti11 = T0(0.9510565162951535721164393333793821L),
                 tr12 = T0(-0.8090169943749474241022934171828191L),
                 ti12 = T0(0.5877852522924731291687059546390728L);
    const ptrdiff_t bi = ptrdiff_t(ip*ido*sizeof(T0)), bo = ptrdiff_t(ido*sizeof(T0));
    for (size_t k=0; k<l1; )
      {
      const size_t vl = R::vsetvl(l1-k);
      const T0 *p = cc + ip*ido*k;
      T0 *q = ch + ido*k;
      auto c0  = R::loads(p, bi, vl);
      auto ti5 = R::mulf(R::loads(p+2*ido, bi, vl), T0(2), vl);         // 2*CC(0,2,k)
      auto ti4 = R::mulf(R::loads(p+4*ido, bi, vl), T0(2), vl);         // 2*CC(0,4,k)
      auto tr2 = R::mulf(R::loads(p+ido-1+ido, bi, vl), T0(2), vl);     // 2*CC(ido-1,1,k)
      auto tr3 = R::mulf(R::loads(p+ido-1+3*ido, bi, vl), T0(2), vl);   // 2*CC(ido-1,3,k)
      R::stores(q, bo, R::add(R::add(c0, tr2, vl), tr3, vl), vl);       // CH(0,k,0)
      auto cr2 = R::fmaccf(R::fmaccf(c0, tr11, tr2, vl), tr12, tr3, vl);
      auto cr3 = R::fmaccf(R::fmaccf(c0, tr12, tr2, vl), tr11, tr3, vl);
      auto ci5 = R::fmaccf (R::mulf(ti5, ti11, vl), ti12, ti4, vl);     // ti5*ti11 + ti4*ti12
      auto ci4 = R::fnmsacf(R::mulf(ti5, ti12, vl), ti11, ti4, vl);     // ti5*ti12 - ti4*ti11
      R::stores(q+4*ido*l1, bo, R::add(cr2, ci5, vl), vl);              // CH(0,k,4)
      R::stores(q+  ido*l1, bo, R::sub(cr2, ci5, vl), vl);              // CH(0,k,1)
      R::stores(q+3*ido*l1, bo, R::add(cr3, ci4, vl), vl);              // CH(0,k,3)
      R::stores(q+2*ido*l1, bo, R::sub(cr3, ci4, vl), vl);              // CH(0,k,2)
      k += vl;
      }
    }
  };

// ---- LMUL per radix (register pressure grows with the radix) ----------------------------
#ifndef POCKETFFT_RVV_RLMUL2
#define POCKETFFT_RVV_RLMUL2 4
#endif
#ifndef POCKETFFT_RVV_RLMUL3
#define POCKETFFT_RVV_RLMUL3 2
#endif
#ifndef POCKETFFT_RVV_RLMUL4
#define POCKETFFT_RVV_RLMUL4 2
#endif
#ifndef POCKETFFT_RVV_RLMUL5
#define POCKETFFT_RVV_RLMUL5 1
#endif

// GCC's pre-register-allocation instruction scheduler (-fschedule-insns, on by default for RISC-V)
// hoists all loads of a butterfly to the top of the loop body and then spills vector registers
// (radf4 at LMUL=2: ~2x the instruction count). Disable it for the pass entry points; the
// always_inline helpers are inlined into them and inherit the setting. Override with
// -DPOCKETFFT_RVV_NOSCHED= (empty) to keep the compiler's defaults.
#ifndef POCKETFFT_RVV_NOSCHED
#if defined(__GNUC__) && !defined(__clang__)
#define POCKETFFT_RVV_NOSCHED __attribute__((optimize("no-schedule-insns")))
#else
#define POCKETFFT_RVV_NOSCHED
#endif
#endif

// ---- entry points (same signatures as the scalar rfftp members, T = T0) ----------------
// The edge loops have little register pressure and no twiddles; they run at LMUL=4 (halves their
// cost compared with LMUL=2 for every radix, also for radix 5 whose main loop needs LMUL=1).
#ifndef POCKETFFT_RVV_RLMUL_EDGE
#define POCKETFFT_RVV_RLMUL_EDGE 4
#endif
template<typename T0, size_t LM, bool FWD, typename BF>
static POCKETFFT_RVV_INLINE void rpass(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  {
  BF::template edge<T0,POCKETFFT_RVV_RLMUL_EDGE>(ido, l1, cc, ch);
  if (ido<=2) return;
  rdrive<T0,LM,FWD,BF>(ido, l1, cc, ch, wa);
  }

template<typename T0> POCKETFFT_RVV_NOSCHED inline void radf2(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL2, true,  rf2>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radb2(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL2, false, rb2>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radf3(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL3, true,  rf3>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radb3(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL3, false, rb3>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radf4(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL4, true,  rf4>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radb4(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL4, false, rb4>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radf5(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL5, true,  rf5>(ido, l1, cc, ch, wa); }
template<typename T0> POCKETFFT_RVV_NOSCHED inline void radb5(size_t ido, size_t l1, const T0 *cc, T0 *ch, const T0 *wa)
  { rpass<T0, POCKETFFT_RVV_RLMUL5, false, rb5>(ido, l1, cc, ch, wa); }

// copy_and_norm replacement for plain real arrays: c[i] = fct*p1[i] (p1 != c) or c[i] *= fct
template<typename T0> inline void real_scale_copy(T0 *c, const T0 *src, size_t n, T0 fct)
  {
  using R = traits<T0, 4>;
  for (size_t i=0; i<n; )
    {
    size_t vl = R::vsetvl(n-i);
    R::store(c+i, R::mulf(R::load(src+i, vl), fct, vl), vl);
    i += vl;
    }
  }

#ifdef POCKETFFT_RVV_INSIDE_DETAIL
} // namespace rvv
#else
} // namespace rvv
} // namespace detail
} // namespace pocketfft
#endif

#endif // POCKETFFT_RVV
#endif // POCKETFFT_RVV_REAL_H
