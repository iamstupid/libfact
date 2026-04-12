#pragma once

// Trial division by a cached table of small primes.
//
// Strategy:
//   1. Strip factors of 2 via ctz + multi-limb rshift (one pass, no division).
//   2. Walk odd primes from the cached table.  For each p:
//        - Fast-reject via mod_small (one high-to-low libdivide sweep).
//        - On hit, peel with divexact_and_mod_next: a single sweep produces
//          both the new quotient and (n/p) mod p, so successive peelings
//          don't redo the same limb traversal twice.
//   3. Return the list of (p, e) factors; n is mutated to the remaining
//      cofactor (== 1 if n was fully smooth).
//
// Small-prime bound is capped at 2^32 so that mod/div over UInt<N> is built
// out of plain uint64_t / uint64_t libdivide calls on 32-bit halves of each
// limb, avoiding any 128/64 division.

#include <cstdint>
#include <vector>

#if defined(__AVX2__)
  #include <immintrin.h>
  #ifndef LIBDIVIDE_AVX2
    #define LIBDIVIDE_AVX2
  #endif
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #ifndef LIBDIVIDE_AVX512
    #define LIBDIVIDE_AVX512
  #endif
#endif

#include "libdivide.h"

#include "zfactor/fixint/uint.h"
#include "zfactor/sieve.h"

namespace zfactor {

struct SmallFactor {
    uint32_t p;
    uint32_t e;
};

namespace detail_trial {

using libdiv_u64 = libdivide::divider<uint64_t>;

// r = n mod p, no quotient.  Fast-reject path for the hot loop.
// Each limb is split into two 32-bit halves so the numerator of every
// libdivide call stays < 2^64 (r < p <= 2^32 - 1).
template<int N>
inline uint32_t mod_small(const fixint::mpn::limb_t* d,
                          uint32_t p,
                          const libdiv_u64& div) noexcept {
    uint64_t r = 0;
    const uint64_t pp = p;
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        uint64_t t = (r << 32) | (lim >> 32);
        r = t - (t / div) * pp;
        t = (r << 32) | (lim & 0xFFFFFFFFu);
        r = t - (t / div) * pp;
    }
    return static_cast<uint32_t>(r);
}

// Peel one factor of p: divide n by p exactly (caller guarantees p | n) AND
// compute (n/p) mod p, in a single high-to-low pass over the limbs.  The
// return value tells the caller whether another factor of p remains, so the
// peel loop doesn't need to follow every divexact with a separate mod_small.
template<int N>
inline uint32_t divexact_and_mod_next(fixint::mpn::limb_t* d,
                                      uint32_t p,
                                      const libdiv_u64& div) noexcept {
    uint64_t r_div = 0;   // rolling remainder of n/p (terminates at 0)
    uint64_t r_mod = 0;   // rolling mod of the freshly-produced quotient
    const uint64_t pp = p;
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];

        // --- division, high half ---
        uint64_t t = (r_div << 32) | (lim >> 32);
        uint64_t q_hi = t / div;
        r_div = t - q_hi * pp;

        // --- division, low half ---
        t = (r_div << 32) | (lim & 0xFFFFFFFFu);
        uint64_t q_lo = t / div;
        r_div = t - q_lo * pp;

        uint64_t q_limb = (q_hi << 32) | q_lo;
        d[i] = q_limb;

        // --- rolling mod of the quotient limb we just wrote ---
        t = (r_mod << 32) | (q_limb >> 32);
        r_mod = t - (t / div) * pp;
        t = (r_mod << 32) | (q_limb & 0xFFFFFFFFu);
        r_mod = t - (t / div) * pp;
    }
    return static_cast<uint32_t>(r_mod);
}

template<int N>
inline bool is_one(const fixint::mpn::limb_t* d) noexcept {
    if (d[0] != 1) return false;
    for (int i = 1; i < N; ++i) if (d[i] != 0) return false;
    return true;
}

} // namespace detail_trial

// Cached table of small primes and their libdivide reciprocals.
//
// Parallel arrays (SoA) rather than a single vector<{p, div}>: the hot loop
// in trial_divide() reads divs[] for every libdivide call and primes[] only
// on rejection or hit, so keeping them apart avoids dragging unused bytes
// through cache.
class TrialDivTable {
public:
    std::vector<uint32_t> primes;
    std::vector<detail_trial::libdiv_u64> divs;

    // All primes p with 2 <= p < bound.  bound must be <= 2^32.
    static TrialDivTable build(uint32_t bound) {
        TrialDivTable t;
        if (bound < 3) return t;
        auto ps = primes_below(bound);
        t.primes.reserve(ps.size());
        t.divs.reserve(ps.size());
        for (uint64_t p : ps) {
            t.primes.push_back(static_cast<uint32_t>(p));
            t.divs.emplace_back(p);
        }
        return t;
    }

    std::size_t size() const noexcept { return primes.size(); }
};

// Peel off all small prime factors of n found in `table`.
//
// On return, `n` holds the remaining cofactor (== 1 if n was fully smooth
// with respect to the table).  The returned vector lists each prime that
// divided n along with its exponent, in ascending order.
template<int N>
inline std::vector<SmallFactor> trial_divide(fixint::UInt<N>& n,
                                             const TrialDivTable& table) {
    std::vector<SmallFactor> out;
    if (n.is_zero()) return out;

    // --- 1. Strip factors of 2 via ctz + rshift, no division. ---
    {
        unsigned s = fixint::mpn::ctz<N>(n.d);
        if (s > 0) {
            fixint::mpn::rshift<N>(n.d, n.d, s);
            out.push_back({2, s});
        }
        if (detail_trial::is_one<N>(n.d)) return out;
    }

    // --- 2. Walk odd primes only (skip primes[0] == 2 when present). ---
    const std::size_t m = table.size();
    std::size_t start = (m > 0 && table.primes[0] == 2) ? 1 : 0;
    for (std::size_t i = start; i < m; ++i) {
        const uint32_t p = table.primes[i];
        const auto& dv = table.divs[i];

        uint32_t r = detail_trial::mod_small<N>(n.d, p, dv);
        if (r != 0) continue;

        // We know p | n.  One fused pass gives us the new quotient and
        // (n/p) mod p in the same high-to-low walk, so the peel loop
        // avoids a second mod_small traversal per iteration.
        uint32_t e = 1;
        r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
        while (r == 0) {
            r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
            ++e;
        }
        out.push_back({p, e});

        if (detail_trial::is_one<N>(n.d)) break;
    }
    return out;
}

} // namespace zfactor

// ============================================================================
// SIMD trial division  (AVX2, 8 × u32 lanes)
// ============================================================================
//
// Batches 8 odd primes at a time and tests one candidate n against each group
// in parallel.  Design constraints:
//
//   * 15-bit primes only (p < 2^15 = 32768).  This caps r, R = 2^32 mod p,
//     and chunk_mod all below 2^15, so the Horner fold (r·R + chunk_mod)
//     stays below 2^31 — comfortably inside u32.
//
//   * 32-bit chunks: each 64-bit limb is processed as (hi32, lo32) high-to-low,
//     same Horner shape as the scalar path but using libdivide's packed
//     u32-branchfree formula for the per-lane mod step.
//
//   * libdivide u32 branchfree yields  { magic (u32), more (u8) }.
//     We pack 8 magics into a __m256i and 8 `more` values into a u32 vector,
//     then call libdivide's own lane-wise mullhi primitive — the divisors ARE
//     still libdivide divisors, we're just feeding a vector of them instead
//     of one broadcasted magic.
//
//   * On a hit, fall back to the existing scalar divexact_and_mod_next using
//     an on-the-fly u64 libdivide divider.  Hits are rare, setup cost is
//     ~10 cycles per hit — noise.
//
// Table size is rounded DOWN to a multiple of 8; the tail (<= 7 primes) is
// discarded — negligible coverage loss for an all-SIMD inner loop.
// ============================================================================

#if defined(__AVX2__)

namespace zfactor {

class SimdTrialDivTable {
public:
    static constexpr uint32_t K = 8;
    static constexpr uint32_t MAX_PRIME = 1u << 15;  // exclusive

    // SoA: each group of K consecutive primes occupies K slots in every
    // array.  Aligned to 32 bytes so AVX2 aligned loads are cheap.
    std::vector<uint32_t> primes;   // padded, size == num_groups * K
    std::vector<uint32_t> magics;   // libdivide branchfree magic, per prime
    std::vector<uint32_t> mores;    // libdivide branchfree shift, per prime (as u32 for srlv)
    std::vector<uint32_t> R;        // 2^32 mod p, per prime
    std::size_t num_groups = 0;

    // Also keep scalar libdivide u64 dividers for the peel path on hits.
    std::vector<detail_trial::libdiv_u64> peel_divs;

    // Build with primes in [3, min(bound, MAX_PRIME)), rounded down to K.
    static SimdTrialDivTable build(uint32_t bound) {
        if (bound > MAX_PRIME) bound = MAX_PRIME;
        SimdTrialDivTable t;
        if (bound < 3) return t;

        // primesieve natively fills a u32 vector, and (3, ...) skips p=2.
        auto ps = primes_range_as<uint32_t>(3, bound);
        std::size_t trimmed = (ps.size() / K) * K;
        ps.resize(trimmed);
        t.num_groups = trimmed / K;

        t.primes  = std::move(ps);
        t.magics.resize(trimmed);
        t.mores.resize(trimmed);
        t.R.resize(trimmed);
        t.peel_divs.reserve(trimmed);

        for (std::size_t i = 0; i < trimmed; ++i) {
            uint32_t p = t.primes[i];
            auto bf = libdivide::libdivide_u32_branchfree_gen(p);
            t.magics[i] = bf.magic;
            t.mores[i]  = bf.more;
            t.R[i]      = static_cast<uint32_t>((uint64_t(1) << 32) % p);
            t.peel_divs.emplace_back(uint64_t(p));
        }
        return t;
    }

    std::size_t size() const noexcept { return primes.size(); }
};

namespace detail_trial {

// 8 × (u32 × u32 → high u32), lane-wise, with no broadcasting assumption.
// libdivide's own libdivide_mullhi_u32_vec256 is documented as requiring b
// to be broadcasted — its second mul_epu32 reads b's even lanes twice — so
// we need our own version that shifts b as well to pick up its odd lanes.
static inline __m256i packed_mullhi_u32(__m256i a, __m256i b) noexcept {
    __m256i hi_even = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
    __m256i a_odd   = _mm256_srli_epi64(a, 32);
    __m256i b_odd   = _mm256_srli_epi64(b, 32);
    __m256i mask    = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    __m256i hi_odd  = _mm256_and_si256(_mm256_mul_epu32(a_odd, b_odd), mask);
    return _mm256_or_si256(hi_even, hi_odd);
}

// SIMD mod by a packed vector of divisors.  Identical to libdivide's u32
// branchfree formula but with per-lane magic/shift rather than a broadcast.
//
//     q = mullhi(n, magic)
//     t = ((n - q) >> 1) + q
//     q = t >> more                (variable per-lane shift)
//     r = n - q * p
static inline __m256i simd_mod_u32(__m256i n,
                                   __m256i magic,
                                   __m256i more,
                                   __m256i p) noexcept {
    __m256i q = packed_mullhi_u32(n, magic);
    __m256i t = _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(n, q), 1), q);
    q = _mm256_srlv_epi32(t, more);
    return _mm256_sub_epi32(n, _mm256_mullo_epi32(q, p));
}

// Compute r_vec for a single group of 8 primes against multi-limb n.
//
// Two kernels with different latency/throughput tradeoffs:
//
//   _short — single libdivide reduction per chunk via the carry-correction
//            trick (prod = r*R; sum = prod + chunk32; if wrap: sum += R).
//            Shorter dep chain per iteration.  Wins for small N where
//            per-iteration latency dominates (N <= 4).
//
//   _long  — two libdivide reductions per chunk: chunk_mod(chunk) is
//            independent of r, so the OoO core can pipeline it against the
//            previous iteration's r-dependent reduction.  Wins for large N
//            where steady-state throughput dominates (N >= 5).

template<int N>
static inline __m256i simd_group_mod_short(const fixint::mpn::limb_t* d,
                                           __m256i p_vec,
                                           __m256i magic_vec,
                                           __m256i more_vec,
                                           __m256i R_vec) noexcept {
    __m256i r = _mm256_setzero_si256();
    const __m256i sign_flip = _mm256_set1_epi32(int(0x80000000u));
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        for (int half = 1; half >= 0; --half) {
            uint32_t chunk = (half == 1)
                                 ? static_cast<uint32_t>(lim >> 32)
                                 : static_cast<uint32_t>(lim);
            __m256i chunk_vec = _mm256_set1_epi32(static_cast<int>(chunk));

            __m256i prod = _mm256_mullo_epi32(r, R_vec);
            __m256i sum  = _mm256_add_epi32(prod, chunk_vec);

            // Unsigned wrap detect via signed cmpgt + sign flip.
            __m256i ovf_mask = _mm256_cmpgt_epi32(
                _mm256_xor_si256(prod, sign_flip),
                _mm256_xor_si256(sum,  sign_flip));
            __m256i correction = _mm256_and_si256(R_vec, ovf_mask);
            sum = _mm256_add_epi32(sum, correction);

            r = simd_mod_u32(sum, magic_vec, more_vec, p_vec);
        }
    }
    return r;
}

template<int N>
static inline __m256i simd_group_mod_long(const fixint::mpn::limb_t* d,
                                          __m256i p_vec,
                                          __m256i magic_vec,
                                          __m256i more_vec,
                                          __m256i R_vec) noexcept {
    __m256i r = _mm256_setzero_si256();
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        for (int half = 1; half >= 0; --half) {
            uint32_t chunk = (half == 1)
                                 ? static_cast<uint32_t>(lim >> 32)
                                 : static_cast<uint32_t>(lim);
            __m256i chunk_vec = _mm256_set1_epi32(static_cast<int>(chunk));

            // Independent of r — gives OoO room to overlap with the
            // previous iteration's tail.
            __m256i chunk_mod = simd_mod_u32(chunk_vec, magic_vec, more_vec, p_vec);

            __m256i prod = _mm256_mullo_epi32(r, R_vec);
            __m256i sum  = _mm256_add_epi32(prod, chunk_mod);
            r = simd_mod_u32(sum, magic_vec, more_vec, p_vec);
        }
    }
    return r;
}

template<int N>
static inline __m256i simd_group_mod(const fixint::mpn::limb_t* d,
                                     __m256i p_vec,
                                     __m256i magic_vec,
                                     __m256i more_vec,
                                     __m256i R_vec) noexcept {
    if constexpr (N <= 4)
        return simd_group_mod_short<N>(d, p_vec, magic_vec, more_vec, R_vec);
    else
        return simd_group_mod_long<N>(d, p_vec, magic_vec, more_vec, R_vec);
}

} // namespace detail_trial

// Peel off all small prime factors of n using the SIMD table.
//
// On return, n holds the remaining cofactor (== 1 if fully smooth wrt the
// SIMD table).  Hits are peeled via the existing scalar fused divexact.
template<int N>
inline std::vector<SmallFactor> trial_divide_simd(fixint::UInt<N>& n,
                                                  const SimdTrialDivTable& table) {
    std::vector<SmallFactor> out;
    if (n.is_zero()) return out;

    // 1. Strip factors of 2 via ctz + rshift — no division.
    {
        unsigned s = fixint::mpn::ctz<N>(n.d);
        if (s > 0) {
            fixint::mpn::rshift<N>(n.d, n.d, s);
            out.push_back({2, s});
        }
        if (detail_trial::is_one<N>(n.d)) return out;
    }

    // 2. SIMD sweep: 8 odd primes at a time.
    const std::size_t G = table.num_groups;
    alignas(32) uint32_t lanes[8];

    for (std::size_t g = 0; g < G; ++g) {
        const uint32_t* base_p = &table.primes[g * 8];
        __m256i p_vec     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base_p));
        __m256i magic_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&table.magics[g * 8]));
        __m256i more_vec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&table.mores[g * 8]));
        __m256i R_vec     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&table.R[g * 8]));

        __m256i r_vec = detail_trial::simd_group_mod<N>(n.d, p_vec, magic_vec, more_vec, R_vec);

        // Fast reject: did any lane hit zero?
        __m256i zero = _mm256_cmpeq_epi32(r_vec, _mm256_setzero_si256());
        int mask = _mm256_movemask_epi8(zero);
        if (mask == 0) continue;

        // Rare path: store r, pick out the hit lanes, peel via scalar.
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), r_vec);
        for (int l = 0; l < 8; ++l) {
            if (lanes[l] != 0) continue;
            uint32_t p = base_p[l];
            const auto& dv = table.peel_divs[g * 8 + l];

            // Peel: we know p | n_old, but n may have shifted since the
            // current group was computed (earlier lanes in the same group
            // could have already peeled their prime).  Re-check before
            // committing, since a smaller prime q in this group may have
            // completely stripped n of this factor already.
            if (detail_trial::mod_small<N>(n.d, p, dv) != 0) continue;

            uint32_t e = 1;
            uint32_t r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
            while (r == 0) {
                r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
                ++e;
            }
            out.push_back({p, e});

            if (detail_trial::is_one<N>(n.d)) return out;
        }
    }
    return out;
}

} // namespace zfactor

#endif // __AVX2__

// ============================================================================
// SIMD trial division  (AVX-512, 16 × u32 lanes)
// ============================================================================
//
// Same shape as the AVX2 path but with K=16 lanes via __m512i.  All the
// constraint analysis (15-bit primes, 32-bit chunks, Horner with R=2^32 mod p)
// carries over unchanged — only the intrinsic widths grow.
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512DQ__)

namespace zfactor {

class Simd512TrialDivTable {
public:
    static constexpr uint32_t K = 16;
    static constexpr uint32_t MAX_PRIME = 1u << 15;

    std::vector<uint32_t> primes;   // padded, size == num_groups * K
    std::vector<uint32_t> magics;
    std::vector<uint32_t> mores;
    std::vector<uint32_t> R;
    std::size_t num_groups = 0;

    std::vector<detail_trial::libdiv_u64> peel_divs;

    static Simd512TrialDivTable build(uint32_t bound) {
        if (bound > MAX_PRIME) bound = MAX_PRIME;
        Simd512TrialDivTable t;
        if (bound < 3) return t;

        auto ps = primes_range_as<uint32_t>(3, bound);
        std::size_t trimmed = (ps.size() / K) * K;
        ps.resize(trimmed);
        t.num_groups = trimmed / K;

        t.primes  = std::move(ps);
        t.magics.resize(trimmed);
        t.mores.resize(trimmed);
        t.R.resize(trimmed);
        t.peel_divs.reserve(trimmed);

        for (std::size_t i = 0; i < trimmed; ++i) {
            uint32_t p = t.primes[i];
            auto bf = libdivide::libdivide_u32_branchfree_gen(p);
            t.magics[i] = bf.magic;
            t.mores[i]  = bf.more;
            t.R[i]      = static_cast<uint32_t>((uint64_t(1) << 32) % p);
            t.peel_divs.emplace_back(uint64_t(p));
        }
        return t;
    }

    std::size_t size() const noexcept { return primes.size(); }
};

namespace detail_trial {

// 16 × (u32 × u32 → high u32) lane-wise.  libdivide's mullhi_u32_vec512
// assumes b is broadcasted (its second mul_epu32 only reads b's even lanes),
// so we shift b by 32 ourselves to pick up the odd lanes.
static inline __m512i packed_mullhi_u32_512(__m512i a, __m512i b) noexcept {
    __m512i hi_even = _mm512_srli_epi64(_mm512_mul_epu32(a, b), 32);
    __m512i a_odd   = _mm512_srli_epi64(a, 32);
    __m512i b_odd   = _mm512_srli_epi64(b, 32);
    __m512i mask    = _mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0,
                                       -1, 0, -1, 0, -1, 0, -1, 0);
    __m512i hi_odd  = _mm512_and_si512(_mm512_mul_epu32(a_odd, b_odd), mask);
    return _mm512_or_si512(hi_even, hi_odd);
}

static inline __m512i simd_mod_u32_512(__m512i n,
                                       __m512i magic,
                                       __m512i more,
                                       __m512i p) noexcept {
    __m512i q = packed_mullhi_u32_512(n, magic);
    __m512i t = _mm512_add_epi32(_mm512_srli_epi32(_mm512_sub_epi32(n, q), 1), q);
    q = _mm512_srlv_epi32(t, more);
    return _mm512_sub_epi32(n, _mm512_mullo_epi32(q, p));
}

// Two AVX-512 kernels with the same short/long dispatch as the AVX2 path.
// AVX-512 has native u32 unsigned compare + mask_add, so the _short kernel
// is cleaner — no sign-flip dance.

template<int N>
static inline __m512i simd_group_mod_512_short(const fixint::mpn::limb_t* d,
                                               __m512i p_vec,
                                               __m512i magic_vec,
                                               __m512i more_vec,
                                               __m512i R_vec) noexcept {
    __m512i r = _mm512_setzero_si512();
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        for (int half = 1; half >= 0; --half) {
            uint32_t chunk = (half == 1)
                                 ? static_cast<uint32_t>(lim >> 32)
                                 : static_cast<uint32_t>(lim);
            __m512i chunk_vec = _mm512_set1_epi32(static_cast<int>(chunk));

            __m512i prod = _mm512_mullo_epi32(r, R_vec);
            __m512i sum  = _mm512_add_epi32(prod, chunk_vec);
            __mmask16 ovf = _mm512_cmplt_epu32_mask(sum, prod);
            sum = _mm512_mask_add_epi32(sum, ovf, sum, R_vec);

            r = simd_mod_u32_512(sum, magic_vec, more_vec, p_vec);
        }
    }
    return r;
}

template<int N>
static inline __m512i simd_group_mod_512_long(const fixint::mpn::limb_t* d,
                                              __m512i p_vec,
                                              __m512i magic_vec,
                                              __m512i more_vec,
                                              __m512i R_vec) noexcept {
    __m512i r = _mm512_setzero_si512();
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        for (int half = 1; half >= 0; --half) {
            uint32_t chunk = (half == 1)
                                 ? static_cast<uint32_t>(lim >> 32)
                                 : static_cast<uint32_t>(lim);
            __m512i chunk_vec = _mm512_set1_epi32(static_cast<int>(chunk));

            __m512i chunk_mod = simd_mod_u32_512(chunk_vec, magic_vec, more_vec, p_vec);

            __m512i prod = _mm512_mullo_epi32(r, R_vec);
            __m512i sum  = _mm512_add_epi32(prod, chunk_mod);
            r = simd_mod_u32_512(sum, magic_vec, more_vec, p_vec);
        }
    }
    return r;
}

template<int N>
static inline __m512i simd_group_mod_512(const fixint::mpn::limb_t* d,
                                         __m512i p_vec,
                                         __m512i magic_vec,
                                         __m512i more_vec,
                                         __m512i R_vec) noexcept {
    if constexpr (N <= 4)
        return simd_group_mod_512_short<N>(d, p_vec, magic_vec, more_vec, R_vec);
    else
        return simd_group_mod_512_long<N>(d, p_vec, magic_vec, more_vec, R_vec);
}

} // namespace detail_trial

template<int N>
inline std::vector<SmallFactor> trial_divide_simd512(fixint::UInt<N>& n,
                                                     const Simd512TrialDivTable& table) {
    std::vector<SmallFactor> out;
    if (n.is_zero()) return out;

    {
        unsigned s = fixint::mpn::ctz<N>(n.d);
        if (s > 0) {
            fixint::mpn::rshift<N>(n.d, n.d, s);
            out.push_back({2, s});
        }
        if (detail_trial::is_one<N>(n.d)) return out;
    }

    const std::size_t G = table.num_groups;
    alignas(64) uint32_t lanes[16];

    for (std::size_t g = 0; g < G; ++g) {
        const uint32_t* base_p = &table.primes[g * 16];
        __m512i p_vec     = _mm512_loadu_si512(base_p);
        __m512i magic_vec = _mm512_loadu_si512(&table.magics[g * 16]);
        __m512i more_vec  = _mm512_loadu_si512(&table.mores[g * 16]);
        __m512i R_vec     = _mm512_loadu_si512(&table.R[g * 16]);

        __m512i r_vec = detail_trial::simd_group_mod_512<N>(
            n.d, p_vec, magic_vec, more_vec, R_vec);

        __mmask16 zmask = _mm512_cmpeq_epi32_mask(r_vec, _mm512_setzero_si512());
        if (zmask == 0) continue;

        _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), r_vec);
        while (zmask) {
            int l = __builtin_ctz(zmask);
            zmask &= zmask - 1;

            uint32_t p = base_p[l];
            const auto& dv = table.peel_divs[g * 16 + l];

            if (detail_trial::mod_small<N>(n.d, p, dv) != 0) continue;

            uint32_t e = 1;
            uint32_t r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
            while (r == 0) {
                r = detail_trial::divexact_and_mod_next<N>(n.d, p, dv);
                ++e;
            }
            out.push_back({p, e});

            if (detail_trial::is_one<N>(n.d)) return out;
        }
    }
    return out;
}

} // namespace zfactor

#endif // __AVX512F__ && __AVX512DQ__
