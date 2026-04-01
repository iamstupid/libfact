#pragma once

// popcnt.h — Bulk popcount and GF(2) inner product
//
// Dispatch hierarchy:
//   1. AVX-512 VPOPCNTDQ + maskz tail  (Ice Lake+ / Zen4+)
//   2. AVX2 three-tier:                 (Haswell+)
//        <8 words  → scalar popcntq
//        <64 words → VPSHUFB + add_epi8 accumulate + maskload tail
//        ≥64 words → Harley-Seal CSA + tier-2 tail
//   3. Scalar fallback
//
// API:
//   popcount_array(data, count)          — bulk popcount of count × u64
//   popcount_and_array(a, b, count)      — popcount of (a[i] & b[i])
//   gf2_dot(a, b, count)                 — parity of popcount_and (0 or 1)

#include <cstdint>
#include <cstddef>
#include <cstring>

#include "intrin.h"

namespace zfactor {

// ================================================================
//  AVX-512 VPOPCNTDQ
// ================================================================

#if defined(ZFACTOR_POPCNT_AVX512)

__attribute__((target("avx512f,avx512vpopcntdq")))
inline uint64_t popcount_array(const uint64_t* data, size_t count) {
    __m512i total = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= count; i += 8)
        total = _mm512_add_epi64(total,
                    _mm512_popcnt_epi64(_mm512_loadu_si512(data + i)));
    if (count & 7) {
        __mmask8 mask = (1u << (count & 7)) - 1;
        total = _mm512_add_epi64(total,
                    _mm512_popcnt_epi64(
                        _mm512_maskz_loadu_epi64(mask, data + i)));
    }
    return _mm512_reduce_add_epi64(total);
}

__attribute__((target("avx512f,avx512vpopcntdq")))
inline uint64_t popcount_and_array(
    const uint64_t* a, const uint64_t* b, size_t count)
{
    __m512i total = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= count; i += 8)
        total = _mm512_add_epi64(total,
                    _mm512_popcnt_epi64(
                        _mm512_and_si512(_mm512_loadu_si512(a + i),
                                         _mm512_loadu_si512(b + i))));
    if (count & 7) {
        __mmask8 mask = (1u << (count & 7)) - 1;
        total = _mm512_add_epi64(total,
                    _mm512_popcnt_epi64(
                        _mm512_and_si512(
                            _mm512_maskz_loadu_epi64(mask, a + i),
                            _mm512_maskz_loadu_epi64(mask, b + i))));
    }
    return _mm512_reduce_add_epi64(total);
}

// ================================================================
//  AVX2 Harley-Seal three-tier
// ================================================================

#elif defined(ZFACTOR_POPCNT_AVX2)

namespace detail {

// ---- constants ----

inline const __m256i PC_LUT = _mm256_setr_epi8(
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);

inline const __m256i NIBBLE = _mm256_set1_epi8(0x0F);

// maskload masks: bit 63 of each i64 lane controls the load
inline const __m256i TAIL_MASK[4] = {
    _mm256_set_epi64x(0, 0, 0, 0),
    _mm256_set_epi64x(0, 0, 0, (long long)0x8000000000000000ULL),
    _mm256_set_epi64x(0, 0,
        (long long)0x8000000000000000ULL,
        (long long)0x8000000000000000ULL),
    _mm256_set_epi64x(0,
        (long long)0x8000000000000000ULL,
        (long long)0x8000000000000000ULL,
        (long long)0x8000000000000000ULL),
};

// ---- primitives ----

__attribute__((target("avx2")))
inline __m256i popcnt_bytes(__m256i v) {
    __m256i lo = _mm256_and_si256(v, NIBBLE);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), NIBBLE);
    return _mm256_add_epi8(_mm256_shuffle_epi8(PC_LUT, lo),
                            _mm256_shuffle_epi8(PC_LUT, hi));
}

__attribute__((target("avx2")))
inline __m256i bytes_to_u64(__m256i byte_acc) {
    return _mm256_sad_epu8(byte_acc, _mm256_setzero_si256());
}

__attribute__((target("avx2")))
inline __m256i popcount256(__m256i v) {
    return bytes_to_u64(popcnt_bytes(v));
}

__attribute__((target("avx2")))
inline uint64_t hsum_epi64(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi64(lo, hi);
    return (uint64_t)_mm_extract_epi64(lo, 0)
         + (uint64_t)_mm_extract_epi64(lo, 1);
}

// ---- Carry-Save Adder ----

#define ZFACTOR_CSA(hi, lo, a, b, c) do {                \
    const __m256i _u = _mm256_xor_si256((a), (b));        \
    (hi) = _mm256_or_si256(_mm256_and_si256((a), (b)),    \
                           _mm256_and_si256(_u, (c)));     \
    (lo) = _mm256_xor_si256(_u, (c));                      \
} while (0)

__attribute__((target("avx2")))
inline __m256i harley_seal_block(
    const __m256i* v,
    __m256i& ones, __m256i& twos, __m256i& fours, __m256i& eights)
{
    __m256i twos_a, twos_b, fours_a, fours_b, eights_a, eights_b, sixteens;

    ZFACTOR_CSA(twos_a,   ones, ones, v[0],  v[1]);
    ZFACTOR_CSA(twos_b,   ones, ones, v[2],  v[3]);
    ZFACTOR_CSA(fours_a,  twos, twos, twos_a, twos_b);

    ZFACTOR_CSA(twos_a,   ones, ones, v[4],  v[5]);
    ZFACTOR_CSA(twos_b,   ones, ones, v[6],  v[7]);
    ZFACTOR_CSA(fours_b,  twos, twos, twos_a, twos_b);
    ZFACTOR_CSA(eights_a, fours, fours, fours_a, fours_b);

    ZFACTOR_CSA(twos_a,   ones, ones, v[8],  v[9]);
    ZFACTOR_CSA(twos_b,   ones, ones, v[10], v[11]);
    ZFACTOR_CSA(fours_a,  twos, twos, twos_a, twos_b);

    ZFACTOR_CSA(twos_a,   ones, ones, v[12], v[13]);
    ZFACTOR_CSA(twos_b,   ones, ones, v[14], v[15]);
    ZFACTOR_CSA(fours_b,  twos, twos, twos_a, twos_b);
    ZFACTOR_CSA(eights_b, fours, fours, fours_a, fours_b);

    ZFACTOR_CSA(sixteens, eights, eights, eights_a, eights_b);

    return popcount256(sixteens);
}

__attribute__((target("avx2")))
inline __m256i flush_accumulators(
    __m256i total,
    __m256i ones, __m256i twos, __m256i fours, __m256i eights)
{
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount256(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount256(fours),  2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount256(twos),   1));
    total = _mm256_add_epi64(total, popcount256(ones));
    return total;
}

#undef ZFACTOR_CSA

// ---- Tier 2: byte-accumulate + maskload tail ----
// Handles 0..63 words. Safe for up to 31 vectors (8×31 = 248 < 256).

__attribute__((target("avx2")))
inline uint64_t popcount_small(const uint64_t* data, size_t count) {
    const __m256i* vdata = (const __m256i*)data;
    size_t nvecs = count / 4;
    size_t tail  = count & 3;

    __m256i acc = _mm256_setzero_si256();

    for (size_t i = 0; i < nvecs; i++)
        acc = _mm256_add_epi8(acc,
                  popcnt_bytes(_mm256_loadu_si256(vdata + i)));

    if (tail) {
        __m256i v = _mm256_maskload_epi64(
            (const long long*)(data + nvecs * 4), TAIL_MASK[tail]);
        acc = _mm256_add_epi8(acc, popcnt_bytes(v));
    }

    return hsum_epi64(bytes_to_u64(acc));
}

__attribute__((target("avx2")))
inline uint64_t popcount_and_small(
    const uint64_t* a, const uint64_t* b, size_t count)
{
    const __m256i* va = (const __m256i*)a;
    const __m256i* vb = (const __m256i*)b;
    size_t nvecs = count / 4;
    size_t tail  = count & 3;

    __m256i acc = _mm256_setzero_si256();

    for (size_t i = 0; i < nvecs; i++)
        acc = _mm256_add_epi8(acc,
                  popcnt_bytes(_mm256_and_si256(
                      _mm256_loadu_si256(va + i),
                      _mm256_loadu_si256(vb + i))));

    if (tail) {
        __m256i v = _mm256_and_si256(
            _mm256_maskload_epi64(
                (const long long*)(a + nvecs * 4), TAIL_MASK[tail]),
            _mm256_maskload_epi64(
                (const long long*)(b + nvecs * 4), TAIL_MASK[tail]));
        acc = _mm256_add_epi8(acc, popcnt_bytes(v));
    }

    return hsum_epi64(bytes_to_u64(acc));
}

} // namespace detail

// ---- Public: popcount_array ----

inline uint64_t popcount_array(const uint64_t* data, size_t count) {
    using namespace detail;

    if (count < 8) {
        uint64_t t = 0;
        for (size_t i = 0; i < count; i++)
            t += popcount_u64(data[i]);
        return t;
    }

    if (count < 64)
        return popcount_small(data, count);

    const __m256i* vdata = (const __m256i*)data;
    size_t nvecs = count / 4;
    size_t full  = nvecs / 16 * 16;

    __m256i total  = _mm256_setzero_si256();
    __m256i ones   = _mm256_setzero_si256();
    __m256i twos   = _mm256_setzero_si256();
    __m256i fours  = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();

    for (size_t i = 0; i < full; i += 16)
        total = _mm256_add_epi64(total,
                    harley_seal_block(vdata + i, ones, twos, fours, eights));

    total = flush_accumulators(total, ones, twos, fours, eights);
    uint64_t result = hsum_epi64(total);

    size_t tail_count = count - full * 4;
    if (tail_count)
        result += popcount_small(data + full * 4, tail_count);

    return result;
}

// ---- Public: popcount_and_array ----

inline uint64_t popcount_and_array(
    const uint64_t* a, const uint64_t* b, size_t count)
{
    using namespace detail;

    if (count < 8) {
        uint64_t t = 0;
        for (size_t i = 0; i < count; i++)
            t += popcount_u64(a[i] & b[i]);
        return t;
    }

    if (count < 64)
        return popcount_and_small(a, b, count);

    const __m256i* va = (const __m256i*)a;
    const __m256i* vb = (const __m256i*)b;
    size_t nvecs = count / 4;
    size_t full  = nvecs / 16 * 16;

    __m256i total  = _mm256_setzero_si256();
    __m256i ones   = _mm256_setzero_si256();
    __m256i twos   = _mm256_setzero_si256();
    __m256i fours  = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();

    for (size_t i = 0; i < full; i += 16) {
        __m256i v[16];
        for (int k = 0; k < 16; k++)
            v[k] = _mm256_and_si256(
                        _mm256_loadu_si256(va + i + k),
                        _mm256_loadu_si256(vb + i + k));
        total = _mm256_add_epi64(total,
                    harley_seal_block(v, ones, twos, fours, eights));
    }

    total = flush_accumulators(total, ones, twos, fours, eights);
    uint64_t result = hsum_epi64(total);

    size_t tail_count = count - full * 4;
    if (tail_count)
        result += popcount_and_small(a + full * 4, b + full * 4, tail_count);

    return result;
}

// ================================================================
//  Scalar fallback
// ================================================================

#else

inline uint64_t popcount_array(const uint64_t* data, size_t count) {
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++)
        total += popcount_u64(data[i]);
    return total;
}

inline uint64_t popcount_and_array(
    const uint64_t* a, const uint64_t* b, size_t count)
{
    uint64_t total = 0;
    for (size_t i = 0; i < count; i++)
        total += popcount_u64(a[i] & b[i]);
    return total;
}

#endif

// ================================================================
//  GF(2) dot product
// ================================================================

inline uint64_t gf2_dot(const uint64_t* a, const uint64_t* b, size_t count) {
    return popcount_and_array(a, b, count) & 1;
}

} // namespace zfactor