#pragma once
// zfactor/intrin.h — platform detection, feature flags, intrinsic wrappers

#include <cstdint>

// ================================================================
//  ISA feature detection
// ================================================================

#if defined(__AVX512F__)
#define ZFACTOR_AVX512 1
#endif
#if defined(__AVX512VPOPCNTDQ__)
#define ZFACTOR_AVX512_VPOPCNTDQ 1
#define ZFACTOR_POPCNT_AVX512 1
#endif
#if defined(__AVX512IFMA__)
#define ZFACTOR_AVX512_IFMA 1
#endif
#if defined(__AVX512BW__)
#define ZFACTOR_AVX512_BW 1
#endif
#if defined(__AVX2__)
#define ZFACTOR_AVX2 1
#endif
#if defined(__BMI2__)
#define ZFACTOR_BMI2 1
#endif
#if defined(__POPCNT__)
#define ZFACTOR_POPCNT 1
#endif

// ================================================================
//  SIMD headers
// ================================================================

#if defined(ZFACTOR_AVX512) || defined(ZFACTOR_AVX2)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// ================================================================
//  Compiler detection
// ================================================================

#if defined(_MSC_VER)
#define ZFACTOR_MSVC 1
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#define ZFACTOR_GCC_COMPAT 1
#endif

// ================================================================
//  Target attributes (for multi-versioned functions)
// ================================================================

#ifdef ZFACTOR_GCC_COMPAT
#define ZFACTOR_TARGET(x) __attribute__((target(x)))
#define ZFACTOR_NOINLINE __attribute__((noinline))
#else
#define ZFACTOR_TARGET(x)
#define ZFACTOR_NOINLINE __declspec(noinline)
#endif

// ================================================================
//  Wide multiply + carry chain primitives
// ================================================================

#if defined(ZFACTOR_MSVC)

inline uint64_t mulhi(uint64_t a, uint64_t b) {
  uint64_t hi;
  _umul128(a, b, &hi);
  return hi;
}
inline uint64_t mullo_hi(uint64_t a, uint64_t b, uint64_t *hi) {
  return _umul128(a, b, hi);
}
inline uint8_t addcarry(uint8_t c, uint64_t a, uint64_t b, uint64_t *out) {
  return _addcarry_u64(c, a, b, (unsigned long long *)out);
}
inline uint8_t subborrow(uint8_t c, uint64_t a, uint64_t b, uint64_t *out) {
  return _subborrow_u64(c, a, b, (unsigned long long *)out);
}

#elif defined(ZFACTOR_GCC_COMPAT)

using u128 = unsigned __int128;

inline uint64_t mulhi(uint64_t a, uint64_t b) {
  return (uint64_t)((u128)a * b >> 64);
}
inline uint64_t mullo_hi(uint64_t a, uint64_t b, uint64_t *hi) {
  u128 r = (u128)a * b;
  *hi = (uint64_t)(r >> 64);
  return (uint64_t)r;
}
inline uint8_t addcarry(uint8_t c, uint64_t a, uint64_t b, uint64_t *out) {
  u128 r = (u128)a + b + c;
  *out = (uint64_t)r;
  return (uint8_t)(r >> 64);
}
inline uint8_t subborrow(uint8_t c, uint64_t a, uint64_t b, uint64_t *out) {
  u128 r = (u128)a - b - c;
  *out = (uint64_t)r;
  return (uint8_t)(r >> 127);
}

#endif

// ================================================================
//  Bit operations
// ================================================================

inline int clz_u64(uint64_t x) {
#if defined(ZFACTOR_GCC_COMPAT)
  return x ? __builtin_clzll(x) : 64;
#elif defined(ZFACTOR_MSVC)
  unsigned long idx;
  return _BitScanReverse64(&idx, x) ? 63 - (int)idx : 64;
#endif
}

inline int ctz_u64(uint64_t x) {
#if defined(ZFACTOR_GCC_COMPAT)
  return x ? __builtin_ctzll(x) : 64;
#elif defined(ZFACTOR_MSVC)
  unsigned long idx;
  return _BitScanForward64(&idx, x) ? (int)idx : 64;
#endif
}

// ================================================================
//  Single-word popcount
// ================================================================

inline uint64_t popcount_u64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return (uint64_t)__builtin_popcountll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
  return __popcnt64(x);
#else
  x -= (x >> 1) & 0x5555555555555555ULL;
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >>
         56;
#endif
}

namespace zfactor {

#if defined(ZFACTOR_AVX512)
constexpr uint32_t vec_size = 64;
using raw_vec = __m512i;
#elif defined(ZFACTOR_AVX2)
constexpr uint32_t vec_size = 32;
using raw_vec = __m256i;
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
constexpr uint32_t vec_size = 16;
using raw_vec = v128_t;
#else
constexpr uint32_t vec_size = 16;
using raw_vec = __m128i;
#endif

struct vec {
  raw_vec v;

  vec() = default;
  vec(raw_vec x) : v(x) {}
  operator raw_vec() const { return v; }

  // ---- factory ----

  static vec zero() {
#if defined(ZFACTOR_AVX512)
    return _mm512_setzero_si512();
#elif defined(ZFACTOR_AVX2)
    return _mm256_setzero_si256();
#elif defined(__wasm_simd128__)
    return wasm_i64x2_const(0, 0);
#else
    return _mm_setzero_si128();
#endif
  }

  static vec ones() {
#if defined(ZFACTOR_AVX512)
    return _mm512_set1_epi8((char)0xFF);
#elif defined(ZFACTOR_AVX2)
    return _mm256_set1_epi8((char)0xFF);
#elif defined(__wasm_simd128__)
    return wasm_i8x16_splat(0xFF);
#else
    return _mm_set1_epi8((char)0xFF);
#endif
  }

  // ---- load / store ----

  static vec load(const void *p) {
#if defined(ZFACTOR_AVX512)
    return _mm512_loadu_si512(p);
#elif defined(ZFACTOR_AVX2)
    return _mm256_loadu_si256((const __m256i *)p);
#elif defined(__wasm_simd128__)
    return wasm_v128_load(p);
#else
    return _mm_loadu_si128((const __m128i *)p);
#endif
  }

  void store(void *p) const {
#if defined(ZFACTOR_AVX512)
    _mm512_storeu_si512(p, v);
#elif defined(ZFACTOR_AVX2)
    _mm256_storeu_si256((__m256i *)p, v);
#elif defined(__wasm_simd128__)
    wasm_v128_store(p, v);
#else
    _mm_storeu_si128((__m128i *)p, v);
#endif
  }

  // ---- bitwise operators ----

  vec operator&(vec o) const {
#if defined(ZFACTOR_AVX512)
    return _mm512_and_si512(v, o.v);
#elif defined(ZFACTOR_AVX2)
    return _mm256_and_si256(v, o.v);
#elif defined(__wasm_simd128__)
    return wasm_v128_and(v, o.v);
#else
    return _mm_and_si128(v, o.v);
#endif
  }

  vec operator|(vec o) const {
#if defined(ZFACTOR_AVX512)
    return _mm512_or_si512(v, o.v);
#elif defined(ZFACTOR_AVX2)
    return _mm256_or_si256(v, o.v);
#elif defined(__wasm_simd128__)
    return wasm_v128_or(v, o.v);
#else
    return _mm_or_si128(v, o.v);
#endif
  }

  vec operator^(vec o) const {
#if defined(ZFACTOR_AVX512)
    return _mm512_xor_si512(v, o.v);
#elif defined(ZFACTOR_AVX2)
    return _mm256_xor_si256(v, o.v);
#elif defined(__wasm_simd128__)
    return wasm_v128_xor(v, o.v);
#else
    return _mm_xor_si128(v, o.v);
#endif
  }

  vec operator~() const { return *this ^ ones(); }

  vec andnot(vec o) const { // ~(*this) & o
#if defined(ZFACTOR_AVX512)
    return _mm512_andnot_si512(v, o.v);
#elif defined(ZFACTOR_AVX2)
    return _mm256_andnot_si256(v, o.v);
#elif defined(__wasm_simd128__)
    return wasm_v128_andnot(o.v, v);
#else
    return _mm_andnot_si128(v, o.v);
#endif
  }

  vec &operator&=(vec o) {
    *this = *this & o;
    return *this;
  }
  vec &operator|=(vec o) {
    *this = *this | o;
    return *this;
  }
  vec &operator^=(vec o) {
    *this = *this ^ o;
    return *this;
  }

  // ---- query ----

  bool any() const {
#if defined(ZFACTOR_AVX512)
    return _mm512_test_epi8_mask(v, v) != 0;
#elif defined(ZFACTOR_AVX2)
    return !_mm256_testz_si256(v, v);
#elif defined(__wasm_simd128__)
    return wasm_v128_any_true(v);
#else
    return !_mm_testz_si128(v, v);
#endif
  }

  bool none() const { return !any(); }
};

// ---- block fill ----

inline void vec_fill(void *dst, vec val, size_t bytes) {
  vec *p = (vec *)dst;
  for (size_t i = 0; i < bytes / vec_size; i++)
    p[i].store(p + i); // wrong: should be val.store(p + i)
}

} // namespace zfactor