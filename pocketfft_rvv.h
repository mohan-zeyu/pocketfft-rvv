/*
 * pocketfft_rvv.h - RISC-V Vector (RVV 1.0) helper layer for pocketfft (C++ header-only version).
 *
 * Vector-length agnostic: every loop uses vsetvl, so one binary runs on VLEN=128/256/512/1024.
 * Requires the frozen RVV intrinsics API (v0.12 / v1.0): GCC >= 14 or clang >= 17, and
 * compilation with the V extension enabled (-march=rv64gcv...). Define POCKETFFT_NO_RVV to
 * disable the RVV paths (useful for A/B measurements).
 *
 * Copyright (C) 2026 RISC-V FFT optimization project. BSD-3-Clause (same as pocketfft).
 */
#ifndef POCKETFFT_RVV_H
#define POCKETFFT_RVV_H

#if defined(__riscv_vector) && defined(__riscv_v_intrinsic) && (__riscv_v_intrinsic >= 12000) \
    && !defined(POCKETFFT_NO_RVV)
#define POCKETFFT_RVV 1
// Force inlining of the helper layer: GCC's size estimate for functions taking RVV tuple types is huge and it
// otherwise leaves ops::add etc. out of line ("max-inline-insns-single limit reached").
#ifndef POCKETFFT_RVV_INLINE
#define POCKETFFT_RVV_INLINE inline __attribute__((always_inline))
#endif
// GCC's pre-register-allocation scheduler hoists all vector loads to the loop top and spills; disable it per
// entry point (clang unaffected). Override with -DPOCKETFFT_RVV_NOSCHED= to disable.
#ifndef POCKETFFT_RVV_NOSCHED
#if defined(__GNUC__) && !defined(__clang__)
#define POCKETFFT_RVV_NOSCHED __attribute__((optimize("no-schedule-insns")))
#else
#define POCKETFFT_RVV_NOSCHED
#endif
#endif
#ifndef POCKETFFT_RVV_INSIDE_DETAIL
#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

#ifdef POCKETFFT_RVV_INSIDE_DETAIL
namespace rvv {
#else
namespace pocketfft {
namespace detail {
namespace rvv {
#endif

// Runtime availability of the V extension (relevant when the library is built with target
// attributes / separate TU flags; in a -march=rv64gcv build this is always true).
inline bool enabled()
  {
  static const bool ok = []()
    {
#if defined(__linux__)
    unsigned long hw = getauxval(AT_HWCAP);
    return (hw & (1UL << ('V' - 'A'))) != 0;   // COMPAT_HWCAP_ISA_V
#else
    return true;
#endif
    }();
  return ok;
  }

// Element types the RVV paths exist for. real_t<T> is T for float/double and an unusable tag type
// otherwise, so that an overload taking real_t<T0>* / cmplx<real_t<T0>>* is never viable for other T0
// (e.g. long double, which pocketfft also instantiates).
struct unsupported_type { unsupported_type() = delete; };
template<typename T> struct real_type { using type = unsupported_type; };
template<> struct real_type<float>  { using type = float; };
template<> struct real_type<double> { using type = double; };
template<typename T> using real_t = typename real_type<T>::type;

template<typename T, size_t LMUL> struct traits;

#define POCKETFFT_RVV_TRAITS(T, SEW, LMUL) \
template<> struct traits<T, LMUL> \
  { \
  using vec  = vfloat##SEW##m##LMUL##_t; \
  using vec2 = vfloat##SEW##m##LMUL##x2_t; \
  static constexpr size_t lmul = LMUL; \
  static POCKETFFT_RVV_INLINE size_t vsetvl(size_t n) { return __riscv_vsetvl_e##SEW##m##LMUL(n); } \
  static POCKETFFT_RVV_INLINE size_t vlmax() { return __riscv_vsetvlmax_e##SEW##m##LMUL(); } \
  static POCKETFFT_RVV_INLINE vec  load (const T *p, size_t vl) { return __riscv_vle##SEW##_v_f##SEW##m##LMUL(p, vl); } \
  static POCKETFFT_RVV_INLINE vec  loads(const T *p, ptrdiff_t bs, size_t vl) { return __riscv_vlse##SEW##_v_f##SEW##m##LMUL(p, bs, vl); } \
  static POCKETFFT_RVV_INLINE void store(T *p, vec v, size_t vl) { __riscv_vse##SEW##_v_f##SEW##m##LMUL(p, v, vl); } \
  static POCKETFFT_RVV_INLINE void stores(T *p, ptrdiff_t bs, vec v, size_t vl) { __riscv_vsse##SEW##_v_f##SEW##m##LMUL(p, bs, v, vl); } \
  static POCKETFFT_RVV_INLINE vec2 lseg2 (const T *p, size_t vl) { return __riscv_vlseg2e##SEW##_v_f##SEW##m##LMUL##x2(p, vl); } \
  static POCKETFFT_RVV_INLINE vec2 lsseg2(const T *p, ptrdiff_t bs, size_t vl) { return __riscv_vlsseg2e##SEW##_v_f##SEW##m##LMUL##x2(p, bs, vl); } \
  static POCKETFFT_RVV_INLINE void sseg2 (T *p, vec2 v, size_t vl) { __riscv_vsseg2e##SEW##_v_f##SEW##m##LMUL##x2(p, v, vl); } \
  static POCKETFFT_RVV_INLINE void ssseg2(T *p, ptrdiff_t bs, vec2 v, size_t vl) { __riscv_vssseg2e##SEW##_v_f##SEW##m##LMUL##x2(p, bs, v, vl); } \
  static POCKETFFT_RVV_INLINE vec  get0(vec2 v) { return __riscv_vget_v_f##SEW##m##LMUL##x2_f##SEW##m##LMUL(v, 0); } \
  static POCKETFFT_RVV_INLINE vec  get1(vec2 v) { return __riscv_vget_v_f##SEW##m##LMUL##x2_f##SEW##m##LMUL(v, 1); } \
  static POCKETFFT_RVV_INLINE vec2 create(vec a, vec b) { return __riscv_vcreate_v_f##SEW##m##LMUL##x2(a, b); } \
  static POCKETFFT_RVV_INLINE vec add (vec a, vec b, size_t vl) { return __riscv_vfadd_vv_f##SEW##m##LMUL(a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec sub (vec a, vec b, size_t vl) { return __riscv_vfsub_vv_f##SEW##m##LMUL(a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec mul (vec a, vec b, size_t vl) { return __riscv_vfmul_vv_f##SEW##m##LMUL(a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec mulf(vec a, T b, size_t vl) { return __riscv_vfmul_vf_f##SEW##m##LMUL(a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec addf(vec a, T b, size_t vl) { return __riscv_vfadd_vf_f##SEW##m##LMUL(a, b, vl); } \
  /* fmacc: acc + a*b ; fnmsac: acc - a*b ; fmsac: a*b - acc */ \
  static POCKETFFT_RVV_INLINE vec fmacc (vec acc, vec a, vec b, size_t vl) { return __riscv_vfmacc_vv_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec fnmsac(vec acc, vec a, vec b, size_t vl) { return __riscv_vfnmsac_vv_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec fmsac (vec acc, vec a, vec b, size_t vl) { return __riscv_vfmsac_vv_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec fmaccf (vec acc, T a, vec b, size_t vl) { return __riscv_vfmacc_vf_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec fnmsacf(vec acc, T a, vec b, size_t vl) { return __riscv_vfnmsac_vf_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec fmsacf (vec acc, T a, vec b, size_t vl) { return __riscv_vfmsac_vf_f##SEW##m##LMUL(acc, a, b, vl); } \
  static POCKETFFT_RVV_INLINE vec neg (vec a, size_t vl) { return __riscv_vfneg_v_f##SEW##m##LMUL(a, vl); } \
  static POCKETFFT_RVV_INLINE vec splat(T x, size_t vl) { return __riscv_vfmv_v_f_f##SEW##m##LMUL(x, vl); } \
  };

POCKETFFT_RVV_TRAITS(double, 64, 1)
POCKETFFT_RVV_TRAITS(double, 64, 2)
POCKETFFT_RVV_TRAITS(double, 64, 4)
POCKETFFT_RVV_TRAITS(float, 32, 1)
POCKETFFT_RVV_TRAITS(float, 32, 2)
POCKETFFT_RVV_TRAITS(float, 32, 4)
#undef POCKETFFT_RVV_TRAITS

// Complex-vector operations for element type T at register-group size LMUL=L.
// A vector of complex numbers in split (planar) form is represented by the 2-member tuple type
// (vfloat64m1x2_t etc.): element 0 = real parts, element 1 = imaginary parts. RVV vector types are
// "sizeless" (cannot be struct members and are a non-deduced context for template deduction), so
// all operations are static members of ops<T,L> and take the active vector length vl explicitly.
template<typename T, size_t L> struct ops
  {
  using R    = traits<T, L>;
  using vec  = typename R::vec;
  using cvec = typename R::vec2;

  static POCKETFFT_RVV_INLINE size_t vsetvl(size_t n) { return R::vsetvl(n); }
  static POCKETFFT_RVV_INLINE size_t vlmax() { return R::vlmax(); }

  // interleaved complex arrays <-> split vectors (segment loads/stores with nf=2)
  template<typename C> static POCKETFFT_RVV_INLINE cvec load(const C *p, size_t vl)
    { return R::lseg2(reinterpret_cast<const T *>(p), vl); }
  // stride in complex elements
  template<typename C> static POCKETFFT_RVV_INLINE cvec loads(const C *p, ptrdiff_t stride, size_t vl)
    { return R::lsseg2(reinterpret_cast<const T *>(p), stride * ptrdiff_t(2 * sizeof(T)), vl); }
  // POCKETFFT_RVV_SPLIT_STORE: write real/imaginary parts with two strided stores instead of one
  // segment store. The segment store needs its two source registers to be adjacent, which GCC's
  // register allocator currently satisfies with extra vmv/spill instructions; clang handles it well.
#ifndef POCKETFFT_RVV_SPLIT_STORE
#define POCKETFFT_RVV_SPLIT_STORE 0   // measured: does not help GCC 15 either; keep segment stores
#endif
  template<typename C> static POCKETFFT_RVV_INLINE void store(C *p, cvec v, size_t vl)
    {
#if POCKETFFT_RVV_SPLIT_STORE
    R::stores(reinterpret_cast<T *>(p), ptrdiff_t(2 * sizeof(T)), R::get0(v), vl);
    R::stores(reinterpret_cast<T *>(p) + 1, ptrdiff_t(2 * sizeof(T)), R::get1(v), vl);
#else
    R::sseg2(reinterpret_cast<T *>(p), v, vl);
#endif
    }
  template<typename C> static POCKETFFT_RVV_INLINE void stores(C *p, ptrdiff_t stride, cvec v, size_t vl)
    {
#if POCKETFFT_RVV_SPLIT_STORE
    R::stores(reinterpret_cast<T *>(p), stride * ptrdiff_t(2 * sizeof(T)), R::get0(v), vl);
    R::stores(reinterpret_cast<T *>(p) + 1, stride * ptrdiff_t(2 * sizeof(T)), R::get1(v), vl);
#else
    R::ssseg2(reinterpret_cast<T *>(p), stride * ptrdiff_t(2 * sizeof(T)), v, vl);
#endif
    }

  static POCKETFFT_RVV_INLINE vec  re(cvec v) { return R::get0(v); }
  static POCKETFFT_RVV_INLINE vec  im(cvec v) { return R::get1(v); }
  static POCKETFFT_RVV_INLINE cvec mk(vec r, vec i) { return R::create(r, i); }

  static POCKETFFT_RVV_INLINE cvec add(cvec a, cvec b, size_t vl)
    { return mk(R::add(re(a), re(b), vl), R::add(im(a), im(b), vl)); }
  static POCKETFFT_RVV_INLINE cvec sub(cvec a, cvec b, size_t vl)
    { return mk(R::sub(re(a), re(b), vl), R::sub(im(a), im(b), vl)); }
  static POCKETFFT_RVV_INLINE cvec mulf(cvec a, T f, size_t vl)
    { return mk(R::mulf(re(a), f, vl), R::mulf(im(a), f, vl)); }
  // a + f*b
  static POCKETFFT_RVV_INLINE cvec fmaf(cvec a, T f, cvec b, size_t vl)
    { return mk(R::fmaccf(re(a), f, re(b), vl), R::fmaccf(im(a), f, im(b), vl)); }
  // a - f*b
  static POCKETFFT_RVV_INLINE cvec fnmsf(cvec a, T f, cvec b, size_t vl)
    { return mk(R::fnmsacf(re(a), f, re(b), vl), R::fnmsacf(im(a), f, im(b), vl)); }
  static POCKETFFT_RVV_INLINE cvec neg(cvec a, size_t vl)
    { return mk(R::neg(re(a), vl), R::neg(im(a), vl)); }
  // full complex product a*b
  static POCKETFFT_RVV_INLINE cvec mul(cvec a, cvec b, size_t vl)
    {
    vec ar = re(a), ai = im(a), br = re(b), bi = im(b);
    return mk(R::fnmsac(R::mul(ar, br, vl), ai, bi, vl), R::fmacc(R::mul(ar, bi, vl), ai, br, vl));
    }
  // pocketfft's special_mul<fwd>: fwd -> a*conj(w), bwd -> a*w   (fwd is a compile-time constant after inlining)
  static POCKETFFT_RVV_INLINE cvec twmul(bool fwd, cvec a, cvec w, size_t vl)
    {
    vec ar = re(a), ai = im(a), wr = re(w), wi = im(w);
    return fwd ? mk(R::fmacc (R::mul(ar, wr, vl), ai, wi, vl),    //  a.r*w.r + a.i*w.i
                    R::fnmsac(R::mul(ai, wr, vl), ar, wi, vl))    //  a.i*w.r - a.r*w.i
               : mk(R::fnmsac(R::mul(ar, wr, vl), ai, wi, vl),    //  a.r*w.r - a.i*w.i
                    R::fmacc (R::mul(ar, wi, vl), ai, wr, vl));   //  a.r*w.i + a.i*w.r
    }
  // pocketfft's ROTX90<fwd>: fwd -> (i, -r) (multiply by -i) ; bwd -> (-i, r) (multiply by +i)
  static POCKETFFT_RVV_INLINE cvec rot90(bool fwd, cvec a, size_t vl)
    { return fwd ? mk(im(a), R::neg(re(a), vl)) : mk(R::neg(im(a), vl), re(a)); }
  // cb = {-t.i*f, t.r*f}  (multiplication by i*f) as used by pass3/pass5
  static POCKETFFT_RVV_INLINE cvec mul_if(cvec t, T f, size_t vl)
    { return mk(R::mulf(R::neg(im(t), vl), f, vl), R::mulf(re(t), f, vl)); }
  };


// =====================================================================================
// Vectorized complex FFT passes (FFTPACK layout) for single, non-batched transforms.
//   CC(i,j,k) = cc[i + ido*(j + ip*k)]      (input,  j = 0..ip-1)
//   CH(i,k,j) = ch[i + ido*(k + l1*j)]      (output)
//   WA(x,i)   = wa[(i-1) + x*(ido-1)]       (twiddles, i = 1..ido-1, x = 0..ip-2)
// Three loop shapes are used, all vector-length agnostic:
//   mode A: fixed k, lanes = consecutive i (unit-stride segment loads, twiddle vectors from memory)
//   mode B: fixed i, lanes = consecutive k (strided segment loads, stride ido*ip in / ido out,
//           twiddles are the same for every lane -> broadcast). Used for i = 0 (no twiddles) and,
//           when ido is small, for all i.
// =====================================================================================

// Access context: TWM = 0 (no twiddles), 1 (twiddle vectors at w + x*wstride), 2 (broadcast scalar twiddles)
template<typename T0, size_t LM, int TWM, bool SI, bool SO> struct ctx
  {
  using O = ops<T0, LM>;
  using R = typename O::R;
  using cvec = typename O::cvec;
  size_t vl = 0;
  ptrdiff_t sin = 1, sout = 1, wstride = 1;
  const cmplx<T0> *w = nullptr;
  POCKETFFT_RVV_INLINE cvec ld(const cmplx<T0> *p) const
    { return SI ? O::loads(p, sin, vl) : O::load(p, vl); }
  POCKETFFT_RVV_INLINE void st(cmplx<T0> *p, cvec v) const
    { if (SO) O::stores(p, sout, v, vl); else O::store(p, v, vl); }
  POCKETFFT_RVV_INLINE cvec tw(size_t x) const
    {
    const cmplx<T0> *q = w + ptrdiff_t(x) * wstride;
    return TWM == 1 ? O::load(q, vl) : O::mk(R::splat(q->r, vl), R::splat(q->i, vl));
    }
  POCKETFFT_RVV_INLINE cvec twm(bool fwd, cvec a, size_t x) const
    { return TWM == 0 ? a : O::twmul(fwd, a, tw(x), vl); }
  };

// ---- butterflies ---------------------------------------------------------------------
// Each butterfly works on one chunk of vl lanes. cc0 = &CC(i,0,k), ccj = distance (in complex
// elements) between CC(i,j,k) and CC(i,j+1,k) (= ido); ch0 = &CH(i,k,0), chj = distance between
// CH(i,k,j) and CH(i,k,j+1) (= ido*l1). The lane stride (unit or strided) lives in the context.
#define POCKETFFT_RVV_BF_HEAD \
  template<bool fwd, typename T0, size_t LM, int TWM, bool SI, bool SO> \
  static POCKETFFT_RVV_INLINE void run(const ctx<T0,LM,TWM,SI,SO> &c, \
    const cmplx<T0> *cc0, ptrdiff_t ccj, cmplx<T0> *ch0, ptrdiff_t chj) \
    { \
    using O = ops<T0, LM>; \
    using R = typename O::R; (void)sizeof(R); \
    const size_t vl = c.vl; \
    auto CC = [=](size_t j) { return cc0 + ptrdiff_t(j)*ccj; }; \
    auto CH = [=](size_t j) { return ch0 + ptrdiff_t(j)*chj; };

struct bf2
  {
  static constexpr size_t ip = 2;
  POCKETFFT_RVV_BF_HEAD
    auto a = c.ld(CC(0)), b = c.ld(CC(1));
    c.st(CH(0), O::add(a, b, vl));
    c.st(CH(1), c.twm(fwd, O::sub(a, b, vl), 0));
    }
  };

struct bf3
  {
  static constexpr size_t ip = 3;
  POCKETFFT_RVV_BF_HEAD
    constexpr T0 tw1r = T0(-0.5),
                 tw1i = (fwd ? -1 : 1) * T0(0.8660254037844386467637231707529362L);
    auto t0 = c.ld(CC(0)), c1 = c.ld(CC(1)), c2 = c.ld(CC(2));
    auto t1 = O::add(c1, c2, vl), t2 = O::sub(c1, c2, vl);
    c.st(CH(0), O::add(t0, t1, vl));
    auto ca = O::fmaf(t0, tw1r, t1, vl);       // t0 + tw1r*t1
    auto cb = O::mul_if(t2, tw1i, vl);         // {-t2.i*tw1i, t2.r*tw1i}
    c.st(CH(1), c.twm(fwd, O::add(ca, cb, vl), 0));
    c.st(CH(2), c.twm(fwd, O::sub(ca, cb, vl), 1));
    }
  };

struct bf4
  {
  static constexpr size_t ip = 4;
  POCKETFFT_RVV_BF_HEAD
    auto c0 = c.ld(CC(0)), c1 = c.ld(CC(1)), c2 = c.ld(CC(2)), c3 = c.ld(CC(3));
    auto t2 = O::add(c0, c2, vl), t1 = O::sub(c0, c2, vl);
    auto t3 = O::add(c1, c3, vl), t4 = O::rot90(fwd, O::sub(c1, c3, vl), vl);
    c.st(CH(0), O::add(t2, t3, vl));
    c.st(CH(1), c.twm(fwd, O::add(t1, t4, vl), 0));
    c.st(CH(2), c.twm(fwd, O::sub(t2, t3, vl), 1));
    c.st(CH(3), c.twm(fwd, O::sub(t1, t4, vl), 2));
    }
  };

struct bf5
  {
  static constexpr size_t ip = 5;
  POCKETFFT_RVV_BF_HEAD
    constexpr T0 tw1r = T0(0.3090169943749474241022934171828191L),
                 tw1i = (fwd ? -1 : 1) * T0(0.9510565162951535721164393333793821L),
                 tw2r = T0(-0.8090169943749474241022934171828191L),
                 tw2i = (fwd ? -1 : 1) * T0(0.5877852522924731291687059546390728L);
    auto t0 = c.ld(CC(0)), c1 = c.ld(CC(1)), c2 = c.ld(CC(2)), c3 = c.ld(CC(3)), c4 = c.ld(CC(4));
    auto t1 = O::add(c1, c4, vl), t4 = O::sub(c1, c4, vl);
    auto t2 = O::add(c2, c3, vl), t3 = O::sub(c2, c3, vl);
    c.st(CH(0), O::add(O::add(t0, t1, vl), t2, vl));
    auto step = [&](size_t u1, size_t u2, T0 twar, T0 twbr, T0 twai, T0 twbi)
      {
      auto ca = O::fmaf(O::fmaf(t0, twar, t1, vl), twbr, t2, vl);                           // t0 + twar*t1 + twbr*t2
      auto cbi = R::fmaccf(R::mulf(O::re(t4), twai, vl), twbi, O::re(t3), vl);              //  twai*t4.r + twbi*t3.r
      auto cbr = R::neg(R::fmaccf(R::mulf(O::im(t4), twai, vl), twbi, O::im(t3), vl), vl);  // -(twai*t4.i + twbi*t3.i)
      auto cb = O::mk(cbr, cbi);
      c.st(CH(u1), c.twm(fwd, O::add(ca, cb, vl), u1 - 1));
      c.st(CH(u2), c.twm(fwd, O::sub(ca, cb, vl), u2 - 1));
      };
    step(1, 4, tw1r, tw2r, +tw1i, +tw2i);
    step(2, 3, tw2r, tw1r, +tw2i, -tw1i);
    }
  };

struct bf8
  {
  static constexpr size_t ip = 8;
  POCKETFFT_RVV_BF_HEAD
    constexpr T0 hsqt2 = T0(0.707106781186547524400844362104849L);
    auto rot45 = [&](typename O::cvec a)   // pocketfft ROTX45
      {
      auto r = O::re(a), im = O::im(a);
      return fwd ? O::mk(R::mulf(R::add(r, im, vl), hsqt2, vl), R::mulf(R::sub(im, r, vl), hsqt2, vl))
                 : O::mk(R::mulf(R::sub(r, im, vl), hsqt2, vl), R::mulf(R::add(im, r, vl), hsqt2, vl));
      };
    auto rot135 = [&](typename O::cvec a)  // pocketfft ROTX135
      {
      auto r = O::re(a), im = O::im(a);
      return fwd ? O::mk(R::mulf(R::sub(im, r, vl), hsqt2, vl), R::mulf(R::add(r, im, vl), -hsqt2, vl))
                 : O::mk(R::mulf(R::add(r, im, vl), -hsqt2, vl), R::mulf(R::sub(r, im, vl), hsqt2, vl));
      };
    auto c1 = c.ld(CC(1)), c5 = c.ld(CC(5)), c3 = c.ld(CC(3)), c7 = c.ld(CC(7));
    auto a1 = O::add(c1, c5, vl), a5 = O::sub(c1, c5, vl);
    auto a3 = O::add(c3, c7, vl), a7 = O::rot90(fwd, O::sub(c3, c7, vl), vl);
    { auto t = a1; a1 = O::add(a1, a3, vl); a3 = O::rot90(fwd, O::sub(t, a3, vl), vl); }
    { auto t = a5; a5 = rot45(O::add(a5, a7, vl)); a7 = rot135(O::sub(t, a7, vl)); }
    auto c0 = c.ld(CC(0)), c4 = c.ld(CC(4)), c2 = c.ld(CC(2)), c6 = c.ld(CC(6));
    auto a0 = O::add(c0, c4, vl), a4 = O::sub(c0, c4, vl);
    auto a2 = O::add(c2, c6, vl), a6 = O::rot90(fwd, O::sub(c2, c6, vl), vl);
    { auto t = a0; a0 = O::add(a0, a2, vl); a2 = O::sub(t, a2, vl); }
    c.st(CH(0), O::add(a0, a1, vl));
    c.st(CH(4), c.twm(fwd, O::sub(a0, a1, vl), 3));
    c.st(CH(2), c.twm(fwd, O::add(a2, a3, vl), 1));
    c.st(CH(6), c.twm(fwd, O::sub(a2, a3, vl), 5));
    { auto t = a4; a4 = O::add(a4, a6, vl); a6 = O::sub(t, a6, vl); }
    c.st(CH(1), c.twm(fwd, O::add(a4, a5, vl), 0));
    c.st(CH(5), c.twm(fwd, O::sub(a4, a5, vl), 4));
    c.st(CH(3), c.twm(fwd, O::add(a6, a7, vl), 2));
    c.st(CH(7), c.twm(fwd, O::sub(a6, a7, vl), 6));
    }
  };
#undef POCKETFFT_RVV_BF_HEAD

// ---- generic pass driver ------------------------------------------------------------------
template<bool fwd, typename T0, size_t LM, typename BF>
static void drive(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  {
  using O = ops<T0, LM>;
  constexpr size_t ip = BF::ip;
  const ptrdiff_t ccj = ptrdiff_t(ido), chj = ptrdiff_t(ido*l1);   // j-strides of CC and CH
  const ptrdiff_t cck = ptrdiff_t(ido*ip), chk = ptrdiff_t(ido);   // k-strides of CC and CH
  if (ido==1)
    {
    // whole pass: lanes across k, strided input (stride ip), unit-stride output, no twiddles
    ctx<T0,LM,0,true,false> c;
    c.sin = cck;
    const cmplx<T0> *p = cc; cmplx<T0> *q = ch;
    for (size_t k=0; k<l1; k+=c.vl, p+=ptrdiff_t(c.vl)*cck, q+=ptrdiff_t(c.vl)*chk)
      {
      c.vl = O::vsetvl(l1-k);
      BF::template run<fwd>(c, p, ccj, q, chj);
      }
    return;
    }
  // i = 0 column: lanes across k, strided input/output, no twiddles
    {
    ctx<T0,LM,0,true,true> c;
    c.sin = cck; c.sout = chk;
    const cmplx<T0> *p = cc; cmplx<T0> *q = ch;
    for (size_t k=0; k<l1; k+=c.vl, p+=ptrdiff_t(c.vl)*cck, q+=ptrdiff_t(c.vl)*chk)
      {
      c.vl = O::vsetvl(l1-k);
      BF::template run<fwd>(c, p, ccj, q, chj);
      }
    }
  const size_t vlmax = O::vlmax();
  if (ido-1 >= (vlmax+1)/2)
    {
    // mode A: lanes across i, twiddle vectors
    ctx<T0,LM,1,false,false> c;
    c.wstride = ptrdiff_t(ido-1);
    for (size_t k=0; k<l1; ++k)
      {
      const cmplx<T0> *p = cc + 1 + ptrdiff_t(k)*cck; cmplx<T0> *q = ch + 1 + ptrdiff_t(k)*chk;
      c.w = wa;
      for (size_t i=1; i<ido; i+=c.vl, p+=c.vl, q+=c.vl, c.w+=c.vl)
        {
        c.vl = O::vsetvl(ido-i);
        BF::template run<fwd>(c, p, ccj, q, chj);
        }
      }
    }
  else
    {
    // mode B: lanes across k, broadcast twiddles
    ctx<T0,LM,2,true,true> c;
    c.sin = cck; c.sout = chk; c.wstride = ptrdiff_t(ido-1);
    for (size_t i=1; i<ido; ++i)
      {
      c.w = wa + (i-1);
      const cmplx<T0> *p = cc + i; cmplx<T0> *q = ch + i;
      for (size_t k=0; k<l1; k+=c.vl, p+=ptrdiff_t(c.vl)*cck, q+=ptrdiff_t(c.vl)*chk)
        {
        c.vl = O::vsetvl(l1-k);
        BF::template run<fwd>(c, p, ccj, q, chj);
        }
      }
    }
  }

// ---- twiddle generation (used by sincos_2pibyn::fill) -----------------------------------------
// dst[2t], dst[2t+1] = (re, im) of exp(2*pi*i*(start+t*step)/N) for t < count, computed exactly like
// sincos_2pibyn::operator[] from its two-level tables v1 (mask+1 entries) and v2 (indexed by idx>>shift),
// with the symmetry idx -> N-idx (conjugate) for 2*idx > N. Indexed segment loads gather the table entries.
#define POCKETFFT_RVV_TW_BODY(STORE) \
  const double *b1 = reinterpret_cast<const double *>(v1), *b2 = reinterpret_cast<const double *>(v2); \
  for (size_t t=0; t<count; ) \
    { \
    size_t vl = __riscv_vsetvl_e64m2(count-t); \
    vuint64m2_t idx = __riscv_vadd_vx_u64m2(__riscv_vmul_vx_u64m2(__riscv_vid_v_u64m2(vl), step, vl), start + t*step, vl); \
    vbool32_t lo = __riscv_vmsleu_vx_u64m2_b32(__riscv_vadd_vv_u64m2(idx, idx, vl), N, vl);          /* 2*idx <= N */ \
    vuint64m2_t idx2 = __riscv_vmerge_vvm_u64m2(__riscv_vrsub_vx_u64m2(idx, N, vl), idx, lo, vl);     /* lo ? idx : N-idx */ \
    vuint64m2_t o1 = __riscv_vsll_vx_u64m2(__riscv_vand_vx_u64m2(idx2, mask, vl), 4, vl); \
    vuint64m2_t o2 = __riscv_vsll_vx_u64m2(__riscv_vsrl_vx_u64m2(idx2, shift, vl), 4, vl); \
    vfloat64m2x2_t x1 = __riscv_vluxseg2ei64_v_f64m2x2(b1, o1, vl); \
    vfloat64m2x2_t x2 = __riscv_vluxseg2ei64_v_f64m2x2(b2, o2, vl); \
    vfloat64m2_t x1r = __riscv_vget_v_f64m2x2_f64m2(x1, 0), x1i = __riscv_vget_v_f64m2x2_f64m2(x1, 1); \
    vfloat64m2_t x2r = __riscv_vget_v_f64m2x2_f64m2(x2, 0), x2i = __riscv_vget_v_f64m2x2_f64m2(x2, 1); \
    vfloat64m2_t r  = __riscv_vfnmsac_vv_f64m2(__riscv_vfmul_vv_f64m2(x1r, x2r, vl), x1i, x2i, vl); \
    vfloat64m2_t im = __riscv_vfmacc_vv_f64m2 (__riscv_vfmul_vv_f64m2(x1r, x2i, vl), x1i, x2r, vl); \
    im = __riscv_vmerge_vvm_f64m2(__riscv_vfneg_v_f64m2(im, vl), im, lo, vl);                       /* lo ? im : -im */ \
    STORE; \
    t += vl; \
    }

POCKETFFT_RVV_NOSCHED inline bool fill_twiddles(double *dst, size_t count, size_t start, size_t step,
  const cmplx<double> *v1, const cmplx<double> *v2, size_t mask, size_t shift, size_t N)
  {
  POCKETFFT_RVV_TW_BODY(__riscv_vsseg2e64_v_f64m2x2(dst + 2*t, __riscv_vcreate_v_f64m2x2(r, im), vl))
  return true;
  }
POCKETFFT_RVV_NOSCHED inline bool fill_twiddles(float *dst, size_t count, size_t start, size_t step,
  const cmplx<double> *v1, const cmplx<double> *v2, size_t mask, size_t shift, size_t N)
  {
  POCKETFFT_RVV_TW_BODY(__riscv_vsseg2e32_v_f32m1x2(dst + 2*t, __riscv_vcreate_v_f32m1x2(__riscv_vfncvt_f_f_w_f32m1(r, vl), __riscv_vfncvt_f_f_w_f32m1(im, vl)), vl))
  return true;
  }
#undef POCKETFFT_RVV_TW_BODY
// tables of another precision (e.g. long double): no vector path
template<typename T, typename TH> inline bool fill_twiddles(T *, size_t, size_t, size_t,
  const cmplx<TH> *, const cmplx<TH> *, size_t, size_t, size_t)
  { return false; }

// ---- plain contiguous copy (used for the 1-D copy_input/copy_output fast path) -------------------
POCKETFFT_RVV_NOSCHED inline void copy_bytes(void *dst, const void *src, size_t nbytes)
  {
  unsigned char *d = static_cast<unsigned char *>(dst); const unsigned char *s = static_cast<const unsigned char *>(src);
  for (size_t i=0; i<nbytes; )
    {
    size_t vl = __riscv_vsetvl_e8m8(nbytes-i);
    __riscv_vse8_v_u8m8(d+i, __riscv_vle8_v_u8m8(s+i, vl), vl);
    i += vl;
    }
  }

// ---- half-complex (FFTPACK r2hc layout) <-> interleaved complex pack/unpack for 1-D r2c/c2r ------
// r2c output: out[ii] = (t[2ii-1], +-t[2ii]) for ii = 1..npairs (npairs = (len-1)/2); conj negates the imaginary part.
template<typename T0> POCKETFFT_RVV_NOSCHED inline void unpack_halfcomplex(cmplx<T0> *out, const T0 *t, size_t npairs, bool conj)
  {
  using O = ops<T0, 4>;
  const cmplx<T0> *src = reinterpret_cast<const cmplx<T0> *>(t + 1);   // pairs (t[1],t[2]), (t[3],t[4]), ...
  for (size_t i=0; i<npairs; )
    {
    size_t vl = O::vsetvl(npairs-i);
    auto v = O::load(src+i, vl);
    if (conj) v = O::mk(O::re(v), O::R::neg(O::im(v), vl));
    O::store(out+1+i, v, vl);
    i += vl;
    }
  }
// c2r input: t[2ii-1] = in[ii].r, t[2ii] = +-in[ii].i for ii = 1..npairs
template<typename T0> POCKETFFT_RVV_NOSCHED inline void pack_halfcomplex(T0 *t, const cmplx<T0> *in, size_t npairs, bool conj)
  {
  using O = ops<T0, 4>;
  cmplx<T0> *dst = reinterpret_cast<cmplx<T0> *>(t + 1);
  for (size_t i=0; i<npairs; )
    {
    size_t vl = O::vsetvl(npairs-i);
    auto v = O::load(in+1+i, vl);
    if (conj) v = O::mk(O::re(v), O::R::neg(O::im(v), vl));
    O::store(dst+i, v, vl);
    i += vl;
    }
  }
template<typename T0> inline void unpack_halfcomplex(cmplx<unsupported_type> *, const T0 *, size_t, bool) {}
template<typename T0> inline void pack_halfcomplex(unsupported_type *, const cmplx<T0> *, size_t, bool) {}

// LMUL per radix (register pressure: bf8 needs ~20 vector registers at LMUL=1)
#ifndef POCKETFFT_RVV_LMUL2
#define POCKETFFT_RVV_LMUL2 4
#endif
#ifndef POCKETFFT_RVV_LMUL3
#define POCKETFFT_RVV_LMUL3 2
#endif
#ifndef POCKETFFT_RVV_LMUL4
#define POCKETFFT_RVV_LMUL4 2
#endif
#ifndef POCKETFFT_RVV_LMUL5
#define POCKETFFT_RVV_LMUL5 1
#endif
#ifndef POCKETFFT_RVV_LMUL8
#define POCKETFFT_RVV_LMUL8 1
#endif

template<bool fwd, typename T0> POCKETFFT_RVV_NOSCHED inline void pass2(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  { drive<fwd, T0, POCKETFFT_RVV_LMUL2, bf2>(ido, l1, cc, ch, wa); }
template<bool fwd, typename T0> POCKETFFT_RVV_NOSCHED inline void pass3(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  { drive<fwd, T0, POCKETFFT_RVV_LMUL3, bf3>(ido, l1, cc, ch, wa); }
template<bool fwd, typename T0> POCKETFFT_RVV_NOSCHED inline void pass4(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  { drive<fwd, T0, POCKETFFT_RVV_LMUL4, bf4>(ido, l1, cc, ch, wa); }
template<bool fwd, typename T0> POCKETFFT_RVV_NOSCHED inline void pass5(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  { drive<fwd, T0, POCKETFFT_RVV_LMUL5, bf5>(ido, l1, cc, ch, wa); }
template<bool fwd, typename T0> POCKETFFT_RVV_NOSCHED inline void pass8(size_t ido, size_t l1, const cmplx<T0> *cc, cmplx<T0> *ch, const cmplx<T0> *wa)
  { drive<fwd, T0, POCKETFFT_RVV_LMUL8, bf8>(ido, l1, cc, ch, wa); }

// scale/copy helpers used at the end of pass_all:  c[i] = src[i]*fct
template<typename T0> POCKETFFT_RVV_NOSCHED inline void scale_copy(cmplx<T0> *c, const cmplx<T0> *src, size_t n, T0 fct)
  {
  using O = ops<T0, 4>;
  for (size_t i=0; i<n; )
    {
    size_t vl = O::vsetvl(n-i);
    O::store(c+i, O::mulf(O::load(src+i, vl), fct, vl), vl);
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

#endif // RVV available
#endif // POCKETFFT_RVV_H
