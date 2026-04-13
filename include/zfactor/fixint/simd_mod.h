#pragma once

// SIMD batch modular reduction: UInt<N> mod p for 8 primes simultaneously.
//
// Hybrid integer/FP: exact u64 Horner accumulation, f64 quotient estimate
// via FMA + magic-trick floor, integer remainder via mul_epu32 + sub.
//
// Valid for primes p in [1024, 2^31). Primes < 1024 should use the existing
// 15-bit integer kernel in trial.h.
//
// AVX2: 2×4 primes in u64 lanes. Two independent halves per step.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "zfactor/fixint/uint.h"

namespace zfactor::fixint {
namespace simd_mod {

#if defined(__AVX2__)

// ================================================================
//  Precomputed table for a group of 8 primes
// ================================================================

// General group (p < 2^31): u64 integer lanes + f64 inverse.
struct alignas(32) ModGroup8 {
    uint64_t p_u64[2][4];
    uint64_t R_u64[2][4];
    double inv_p[2][4];
};

inline ModGroup8 make_group(const uint32_t* primes_8) {
    ModGroup8 g;
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
            uint32_t p = primes_8[h * 4 + i];
            g.p_u64[h][i] = p;
            g.R_u64[h][i] = p ? ((uint64_t)1 << 32) % p : 0;
            g.inv_p[h][i] = p ? std::nextafter(1.0 / (double)p, 0.0) : 0.0;
        }
    }
    return g;
}

// FP group (p < 2^25): all-FP with 32-bit chunk Horner (2N iterations).
// R = 2^32 mod p.
struct alignas(32) ModGroup8FP {
    double R_f64[2][4];
    double p_f64[2][4];
    double inv_p[2][4];
};

inline ModGroup8FP make_group_fp(const uint32_t* primes_8) {
    ModGroup8FP g;
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
            uint32_t p = primes_8[h * 4 + i];
            g.p_f64[h][i] = (double)p;
            g.R_f64[h][i] = p ? (double)(((uint64_t)1 << 32) % p) : 0.0;
            g.inv_p[h][i] = p ? std::nextafter(1.0 / (double)p, 0.0) : 0.0;
        }
    }
    return g;
}

// FP-fast group (p < 2^15): all-FP with 64-bit limb Horner (N iterations).
// Uses R = 2^32 mod p for merging two 32-bit halves, R2 = 2^64 mod p for
// the outer Horner step. Since p < 2^15:
//   hi * R < 2^32 * 2^15 = 2^47, exact in f64
//   hilo = hi*R + lo < 2^47 + 2^32 < 2^48, exact
//   r * R2 < 2^16 * 2^15 = 2^31 (r < 2p < 2^16)
//   v = r*R2 + hilo < 2^31 + 2^48 < 2^49, exact
struct alignas(32) ModGroup8FPFast {
    double R_f64[2][4];     // 2^32 mod p (for chunk merge)
    double R2_f64[2][4];    // 2^64 mod p (for outer Horner)
    double p_f64[2][4];
    double inv_p[2][4];
};

inline ModGroup8FPFast make_group_fp_fast(const uint32_t* primes_8) {
    ModGroup8FPFast g;
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
            uint32_t p = primes_8[h * 4 + i];
            uint64_t R  = p ? ((uint64_t)1 << 32) % p : 0;
            uint64_t R2 = p ? (uint64_t)(((unsigned __int128)1 << 64) % p) : 0;
            g.R_f64[h][i]  = (double)R;
            g.R2_f64[h][i] = (double)R2;
            g.p_f64[h][i]  = (double)p;
            g.inv_p[h][i]  = p ? std::nextafter(1.0 / (double)p, 0.0) : 0.0;
        }
    }
    return g;
}

// ================================================================
//  u64 -> f64 approximate conversion (AVX2, no _mm256_cvtepi64_pd)
// ================================================================

static inline __m256d u64_to_f64_approx(__m256i v) {
    __m256i lo_i = _mm256_and_si256(v, _mm256_set1_epi64x(0xFFFFFFFF));
    __m256i hi_i = _mm256_srli_epi64(v, 32);

    const __m256i magic_bits = _mm256_set1_epi64x(0x4330000000000000ULL);
    const __m256d magic_f64  = _mm256_set1_pd(4503599627370496.0);  // 2^52

    __m256d lo_f = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(lo_i, magic_bits)), magic_f64);
    __m256d hi_f = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(hi_i, magic_bits)), magic_f64);

    return _mm256_fmadd_pd(hi_f, _mm256_set1_pd(4294967296.0), lo_f);
}

// ================================================================
//  Horner steps
// ================================================================

static constexpr double FP_BIAS  = -0.5 + 0x1p-32;
static constexpr double FP_MAGIC = 0x1p52;

// --- General path (p < 2^31): u64 accumulation, FP quotient ---

static inline __m256i horner_step_4(
    __m256i r,          // accumulator, < 2p in low 32 of each u64 lane
    __m256i R_vec,      // 2^32 mod p, u64 lanes
    __m256i p_vec,      // prime, u64 lanes
    __m256d inv_p,      // underestimate 1/p
    uint32_t chunk)     // 32-bit chunk broadcast
{
    __m256i chunk_vec = _mm256_set1_epi64x(chunk);
    __m256d bias_vec  = _mm256_set1_pd(FP_BIAS);
    __m256d magic_vec = _mm256_set1_pd(FP_MAGIC);

    // v = r * R + chunk  (exact u64)
    __m256i v = _mm256_add_epi64(_mm256_mul_epu32(r, R_vec), chunk_vec);

    // q = floor(v * inv_p) via FMA + magic
    __m256d v_f64     = u64_to_f64_approx(v);
    __m256d q_biased  = _mm256_fmadd_pd(v_f64, inv_p, bias_vec);
    __m256d q_snapped = _mm256_add_pd(q_biased, magic_vec);
    __m256i q = _mm256_castpd_si256(q_snapped);

    // vmod = v - q * p  (result < 2p, fits low 32 bits)
    return _mm256_sub_epi64(v, _mm256_mul_epu32(q, p_vec));
}

// --- FP path (p < 2^25): all-FP with 32-bit chunk Horner ---
//
// r < 2p < 2^26, R < p < 2^25. So r*R < 2^51, plus chunk < 2^32,
// v < 2^51 + 2^32 < 2^52. Everything is exact in f64.

static inline __m256d horner_step_4_fp(
    __m256d r,          // accumulator as f64, < 2p
    __m256d R_f64,      // 2^32 mod p as f64
    __m256d p_f64,      // prime as f64
    __m256d inv_p,      // underestimate 1/p
    double chunk_f64)   // 32-bit chunk as f64
{
    __m256d bias_vec  = _mm256_set1_pd(FP_BIAS);
    __m256d magic_vec = _mm256_set1_pd(FP_MAGIC);

    // v = r * R + chunk  (exact in f64 since < 2^52)
    __m256d v = _mm256_fmadd_pd(r, R_f64, _mm256_set1_pd(chunk_f64));

    // q = floor(v * inv_p)
    __m256d q_biased  = _mm256_fmadd_pd(v, inv_p, bias_vec);
    __m256d q_snapped = _mm256_add_pd(q_biased, magic_vec);
    __m256d q         = _mm256_sub_pd(q_snapped, magic_vec);

    // r = v - q * p  (exact, result < 2p < 2^26)
    return _mm256_fnmadd_pd(q, p_f64, v);
}

// --- FP-fast path (p < 2^15): all-FP, one step per 64-bit limb ---
//
// Merges two 32-bit halves via fma(hi, R, lo), then one outer Horner step
// with R2 = 2^64 mod p.  N iterations instead of 2N.
// Bounds: hi*R < 2^47, hilo < 2^48, r*R2 < 2^31, v < 2^49. All exact.

static inline __m256d horner_step_4_fp_fast(
    __m256d r,          // accumulator, < 2p < 2^16
    __m256d R_f64,      // 2^32 mod p
    __m256d R2_f64,     // 2^64 mod p
    __m256d p_f64,
    __m256d inv_p,
    uint64_t limb)
{
    __m256d bias_vec  = _mm256_set1_pd(FP_BIAS);
    __m256d magic_vec = _mm256_set1_pd(FP_MAGIC);

    double hi_f = (double)(uint32_t)(limb >> 32);
    double lo_f = (double)(uint32_t)(limb);

    // hilo = hi * R + lo  (exact: < 2^48)
    __m256d hilo = _mm256_fmadd_pd(_mm256_set1_pd(hi_f), R_f64, _mm256_set1_pd(lo_f));

    // v = r * R2 + hilo  (exact: < 2^49)
    __m256d v = _mm256_fmadd_pd(r, R2_f64, hilo);

    // q = floor(v * inv_p)
    __m256d q_biased  = _mm256_fmadd_pd(v, inv_p, bias_vec);
    __m256d q_snapped = _mm256_add_pd(q_biased, magic_vec);
    __m256d q         = _mm256_sub_pd(q_snapped, magic_vec);

    return _mm256_fnmadd_pd(q, p_f64, v);
}

// ================================================================
//  Full reduction: UInt<N> mod 8 primes
// ================================================================
//
// Returns two __m256i (half0, half1), each with 4 remainders in the
// low 32 bits of u64 lanes.

template<int N>
struct ModResult8 { __m256i h0; __m256i h1; };

template<int N>
ModResult8<N> simd_mod_8(const mpn::limb_t* d, const ModGroup8& g) {
    __m256i R0 = _mm256_loadu_si256((const __m256i*)g.R_u64[0]);
    __m256i R1 = _mm256_loadu_si256((const __m256i*)g.R_u64[1]);
    __m256i p0 = _mm256_loadu_si256((const __m256i*)g.p_u64[0]);
    __m256i p1 = _mm256_loadu_si256((const __m256i*)g.p_u64[1]);
    __m256d inv0 = _mm256_loadu_pd(g.inv_p[0]);
    __m256d inv1 = _mm256_loadu_pd(g.inv_p[1]);

    __m256i r0 = _mm256_setzero_si256();
    __m256i r1 = _mm256_setzero_si256();

    for (int i = N - 1; i >= 0; --i) {
        uint64_t limb = d[i];
        uint32_t hi = (uint32_t)(limb >> 32);
        uint32_t lo = (uint32_t)(limb);

        r0 = horner_step_4(r0, R0, p0, inv0, hi);
        r1 = horner_step_4(r1, R1, p1, inv1, hi);
        r0 = horner_step_4(r0, R0, p0, inv0, lo);
        r1 = horner_step_4(r1, R1, p1, inv1, lo);
    }

    // csub per half: if r >= p, r -= p
    // Unsigned compare via sub + sign check: if (r - p) didn't borrow, r >= p.
    // Or: compare low 32 bits as unsigned.
    auto csub = [](__m256i r, __m256i p) {
        __m256i diff = _mm256_sub_epi64(r, p);
        // If no borrow (r >= p), high bits of diff won't be all-ones.
        // Since r, p < 2^31 and result fits in low 32 bits, check bit 63:
        // if r >= p then diff in [0, p) and bit 63 = 0.
        // if r < p then diff wraps and bit 63 = 1.
        __m256i borrow = _mm256_srai_epi32(_mm256_shuffle_epi32(diff, 0xF5), 31);
        // borrow is all-ones in each u64 lane if r < p, zero if r >= p.
        // Pick: r >= p ? diff : r
        return _mm256_blendv_epi8(diff, r, borrow);
    };

    r0 = csub(r0, p0);
    r1 = csub(r1, p1);

    return {r0, r1};
}

// Fast-path reduction for p < 2^25: all-FP, no integer in Horner loop.
// Returns 4+4 f64 remainders; caller converts to u32.

template<int N>
struct ModResult8FP { __m256d h0; __m256d h1; };

template<int N>
ModResult8FP<N> simd_mod_8_fp(const mpn::limb_t* d, const ModGroup8FP& g) {
    __m256d R0 = _mm256_loadu_pd(g.R_f64[0]);
    __m256d R1 = _mm256_loadu_pd(g.R_f64[1]);
    __m256d p0 = _mm256_loadu_pd(g.p_f64[0]);
    __m256d p1 = _mm256_loadu_pd(g.p_f64[1]);
    __m256d inv0 = _mm256_loadu_pd(g.inv_p[0]);
    __m256d inv1 = _mm256_loadu_pd(g.inv_p[1]);

    __m256d r0 = _mm256_setzero_pd();
    __m256d r1 = _mm256_setzero_pd();

    for (int i = N - 1; i >= 0; --i) {
        uint64_t limb = d[i];
        double hi = (double)(uint32_t)(limb >> 32);
        double lo = (double)(uint32_t)(limb);

        r0 = horner_step_4_fp(r0, R0, p0, inv0, hi);
        r1 = horner_step_4_fp(r1, R1, p1, inv1, hi);
        r0 = horner_step_4_fp(r0, R0, p0, inv0, lo);
        r1 = horner_step_4_fp(r1, R1, p1, inv1, lo);
    }

    // csub: if r >= p, r -= p
    auto csub_fp = [](__m256d r, __m256d p) {
        __m256d diff = _mm256_sub_pd(r, p);
        return _mm256_blendv_pd(diff, r, diff);
    };

    r0 = csub_fp(r0, p0);
    r1 = csub_fp(r1, p1);

    return {r0, r1};
}

// FP-fast reduction for p < 2^15: one step per limb (N iterations).
template<int N>
ModResult8FP<N> simd_mod_8_fp_fast(const mpn::limb_t* d, const ModGroup8FPFast& g) {
    __m256d R0  = _mm256_loadu_pd(g.R_f64[0]);
    __m256d R1  = _mm256_loadu_pd(g.R_f64[1]);
    __m256d R20 = _mm256_loadu_pd(g.R2_f64[0]);
    __m256d R21 = _mm256_loadu_pd(g.R2_f64[1]);
    __m256d p0  = _mm256_loadu_pd(g.p_f64[0]);
    __m256d p1  = _mm256_loadu_pd(g.p_f64[1]);
    __m256d inv0 = _mm256_loadu_pd(g.inv_p[0]);
    __m256d inv1 = _mm256_loadu_pd(g.inv_p[1]);

    __m256d r0 = _mm256_setzero_pd();
    __m256d r1 = _mm256_setzero_pd();

    for (int i = N - 1; i >= 0; --i) {
        r0 = horner_step_4_fp_fast(r0, R0, R20, p0, inv0, d[i]);
        r1 = horner_step_4_fp_fast(r1, R1, R21, p1, inv1, d[i]);
    }

    auto csub_fp = [](__m256d r, __m256d p) {
        __m256d diff = _mm256_sub_pd(r, p);
        return _mm256_blendv_pd(diff, r, diff);
    };
    r0 = csub_fp(r0, p0);
    r1 = csub_fp(r1, p1);
    return {r0, r1};
}

// ================================================================
//  Batch API
// ================================================================

struct SimdModTable {
    std::vector<ModGroup8> groups;
    uint32_t num_primes = 0;

    static SimdModTable build(const uint32_t* primes, uint32_t count) {
        SimdModTable t;
        t.num_primes = count;
        uint32_t ngroups = (count + 7) / 8;
        t.groups.resize(ngroups);
        uint32_t buf[8];
        for (uint32_t g = 0; g < ngroups; ++g) {
            for (int i = 0; i < 8; ++i) {
                uint32_t idx = g * 8 + i;
                buf[i] = (idx < count) ? primes[idx] : 1;
            }
            t.groups[g] = make_group(buf);
        }
        return t;
    }
};

struct SimdModTableFPFast {
    std::vector<ModGroup8FPFast> groups;
    uint32_t num_primes = 0;

    static SimdModTableFPFast build(const uint32_t* primes, uint32_t count) {
        SimdModTableFPFast t;
        t.num_primes = count;
        uint32_t ngroups = (count + 7) / 8;
        t.groups.resize(ngroups);
        uint32_t buf[8];
        for (uint32_t g = 0; g < ngroups; ++g) {
            for (int i = 0; i < 8; ++i) {
                uint32_t idx = g * 8 + i;
                buf[i] = (idx < count) ? primes[idx] : 3;
            }
            t.groups[g] = make_group_fp_fast(buf);
        }
        return t;
    }
};

struct SimdModTableFP {
    std::vector<ModGroup8FP> groups;
    uint32_t num_primes = 0;

    static SimdModTableFP build(const uint32_t* primes, uint32_t count) {
        SimdModTableFP t;
        t.num_primes = count;
        uint32_t ngroups = (count + 7) / 8;
        t.groups.resize(ngroups);
        uint32_t buf[8];
        for (uint32_t g = 0; g < ngroups; ++g) {
            for (int i = 0; i < 8; ++i) {
                uint32_t idx = g * 8 + i;
                buf[i] = (idx < count) ? primes[idx] : 3;
            }
            t.groups[g] = make_group_fp(buf);
        }
        return t;
    }
};

template<int N>
void batch_mod(const UInt<N>& a, const SimdModTable& table, uint32_t* out) {
    uint32_t ngroups = (uint32_t)table.groups.size();
    // Full groups: safe to write 8 u32 at once
    uint32_t full = (ngroups > 0 && (table.num_primes & 7)) ? ngroups - 1 : ngroups;
    __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (uint32_t g = 0; g < full; ++g) {
        auto [r0, r1] = simd_mod_8<N>(a.d, table.groups[g]);
        __m256 packed = _mm256_shuffle_ps(
            _mm256_castsi256_ps(r0), _mm256_castsi256_ps(r1), 0x88);
        _mm256_storeu_si256((__m256i*)(out + g * 8),
            _mm256_permutevar8x32_epi32(_mm256_castps_si256(packed), perm));
    }
    // Last partial group: store via buffer to avoid overrun
    if (full < ngroups) {
        auto [r0, r1] = simd_mod_8<N>(a.d, table.groups[full]);
        __m256 packed = _mm256_shuffle_ps(
            _mm256_castsi256_ps(r0), _mm256_castsi256_ps(r1), 0x88);
        alignas(32) uint32_t buf[8];
        _mm256_store_si256((__m256i*)buf,
            _mm256_permutevar8x32_epi32(_mm256_castps_si256(packed), perm));
        uint32_t rem = table.num_primes - full * 8;
        for (uint32_t i = 0; i < rem; ++i)
            out[full * 8 + i] = buf[i];
    }
}

// Fast-path batch_mod for p < 2^25 (all-FP).
template<int N>
void batch_mod_fp(const UInt<N>& a, const SimdModTableFP& table, uint32_t* out) {
    uint32_t ngroups = (uint32_t)table.groups.size();
    uint32_t full = (ngroups > 0 && (table.num_primes & 7)) ? ngroups - 1 : ngroups;
    __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (uint32_t g = 0; g < full; ++g) {
        auto [r0, r1] = simd_mod_8_fp<N>(a.d, table.groups[g]);
        // Convert f64 remainders to u32: cvttpd_epi32 gives 4 x i32 in __m128i
        __m128i i0 = _mm256_cvttpd_epi32(r0);
        __m128i i1 = _mm256_cvttpd_epi32(r1);
        // Combine into 8 x u32: [r0..r3, r4..r7]
        _mm256_storeu_si256((__m256i*)(out + g * 8),
            _mm256_setr_m128i(i0, i1));
    }
    if (full < ngroups) {
        auto [r0, r1] = simd_mod_8_fp<N>(a.d, table.groups[full]);
        alignas(32) uint32_t buf[8];
        __m128i i0 = _mm256_cvttpd_epi32(r0);
        __m128i i1 = _mm256_cvttpd_epi32(r1);
        _mm256_store_si256((__m256i*)buf, _mm256_setr_m128i(i0, i1));
        uint32_t rem = table.num_primes - full * 8;
        for (uint32_t i = 0; i < rem; ++i)
            out[full * 8 + i] = buf[i];
    }
}

// FP-fast batch_mod for p < 2^15 (full-limb Horner, N iterations).
template<int N>
void batch_mod_fp_fast(const UInt<N>& a, const SimdModTableFPFast& table, uint32_t* out) {
    uint32_t ngroups = (uint32_t)table.groups.size();
    uint32_t full = (ngroups > 0 && (table.num_primes & 7)) ? ngroups - 1 : ngroups;
    for (uint32_t g = 0; g < full; ++g) {
        auto [r0, r1] = simd_mod_8_fp_fast<N>(a.d, table.groups[g]);
        __m128i i0 = _mm256_cvttpd_epi32(r0);
        __m128i i1 = _mm256_cvttpd_epi32(r1);
        _mm256_storeu_si256((__m256i*)(out + g * 8), _mm256_setr_m128i(i0, i1));
    }
    if (full < ngroups) {
        auto [r0, r1] = simd_mod_8_fp_fast<N>(a.d, table.groups[full]);
        alignas(32) uint32_t buf[8];
        __m128i i0 = _mm256_cvttpd_epi32(r0);
        __m128i i1 = _mm256_cvttpd_epi32(r1);
        _mm256_store_si256((__m256i*)buf, _mm256_setr_m128i(i0, i1));
        uint32_t rem = table.num_primes - full * 8;
        for (uint32_t i = 0; i < rem; ++i)
            out[full * 8 + i] = buf[i];
    }
}

#endif // __AVX2__

} // namespace simd_mod
} // namespace zfactor::fixint
