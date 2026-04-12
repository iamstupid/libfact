#pragma once

// Modular pre-filter for perfect-power detection.
//
// Design (per user spec):
//   * Candidate exponents = first 16 prime exponents
//     {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}.
//     Bit i of the filter mask corresponds to PERFECT_POWER_EXPONENTS[i],
//     so the mask fits in uint16.
//
//   * Per-prime LUT: for filter prime p and residue r in [0, p), store a
//     uint16 bitmask of which q's have an x with x^q ≡ r (mod p).  At
//     runtime, OR all hits together via AND across primes.
//
//   * Reuse the AVX-512 trial-division kernel (simd_group_mod_512) to
//     compute n mod p for 16 primes in parallel.  The post-processing is
//     a 16-way scalar LUT lookup + AND-reduce per group.
//
// Correctness: if n = m^q then n mod p = (m mod p)^q mod p, so the LUT
// entry for r = n mod p must have bit q set.  ANDing across many primes
// drives the false-positive rate down exponentially for low q.  For high
// q (q >= 19) only a handful of primes have gcd(q, p-1) > 1, so the
// filter is weaker there — the bit is more likely to survive even when
// n is not a q-th power.  That's fine, it just means the iroot check
// still has work to do for the high exponents (which is rare anyway).
//
// Intended workflow:
//   1. Trial-divide n via trial_divide_simd512 (existing).
//   2. Run PerfectPowerFilter::filter() on the cofactor.
//   3. For each set bit i, verify with iroot<N>(cofactor, q_i).
//
// The filter is stronger after small factors are stripped, since residue
// zero always passes (0^q = 0 for any q).

#include <cstdint>
#include <vector>

#include "libdivide.h"

#include "zfactor/fixint/uint.h"
#include "zfactor/sieve.h"
#include "zfactor/trial.h"

namespace zfactor {

inline constexpr uint32_t PERFECT_POWER_EXPONENTS[16] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
};
inline constexpr unsigned PERFECT_POWER_NQ = 16;

namespace detail_pp {

inline uint64_t powmod_u64(uint64_t base, uint64_t exp, uint64_t mod) noexcept {
    if (mod == 1) return 0;
    uint64_t r = 1 % mod;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) r = (r * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return r;
}

} // namespace detail_pp

#if defined(__AVX512F__) && defined(__AVX512DQ__)

class PerfectPowerFilter {
public:
    static constexpr uint32_t K = 16;          // AVX-512 lane count
    // primes p with 3 <= p < MAX_PRIME.  pi(313) - 1 (skip 2) = 64 = 4·K,
    // so the table fits exactly into 4 SIMD groups with no tail loss.
    static constexpr uint32_t MAX_PRIME = 314;

    // SoA SIMD table — same shape as Simd512TrialDivTable.
    std::vector<uint32_t> primes;
    std::vector<uint32_t> magics;
    std::vector<uint32_t> mores;
    std::vector<uint32_t> R;
    std::size_t num_groups = 0;

    // Per-prime LUT, packed into one flat uint16 array.
    // lut[lut_offset[i] + r] is the bitmask for primes[i] at residue r.
    std::vector<uint32_t> lut_offset;
    std::vector<uint16_t> lut;

    static PerfectPowerFilter build() {
        PerfectPowerFilter t;

        // Skip p=2 — every residue mod 2 is trivially a q-th power, so
        // including it would waste a lane with no filtering benefit.
        auto ps = primes_range_as<uint32_t>(3, MAX_PRIME);
        std::size_t trimmed = (ps.size() / K) * K;
        ps.resize(trimmed);
        t.num_groups = trimmed / K;

        t.primes = std::move(ps);
        t.magics.resize(trimmed);
        t.mores.resize(trimmed);
        t.R.resize(trimmed);
        t.lut_offset.resize(trimmed);

        std::size_t lut_total = 0;
        for (uint32_t p : t.primes) lut_total += p;
        t.lut.assign(lut_total, 0);

        std::size_t off = 0;
        for (std::size_t i = 0; i < trimmed; ++i) {
            uint32_t p = t.primes[i];
            auto bf = libdivide::libdivide_u32_branchfree_gen(p);
            t.magics[i] = bf.magic;
            t.mores[i]  = bf.more;
            t.R[i]      = static_cast<uint32_t>((uint64_t(1) << 32) % p);
            t.lut_offset[i] = static_cast<uint32_t>(off);

            // Build the LUT slice for this prime: for each q, walk x in
            // [0, p) and mark x^q mod p as a "q-th power residue".
            for (unsigned qi = 0; qi < PERFECT_POWER_NQ; ++qi) {
                uint32_t q = PERFECT_POWER_EXPONENTS[qi];
                uint16_t bit = static_cast<uint16_t>(uint16_t(1) << qi);
                for (uint32_t x = 0; x < p; ++x) {
                    uint32_t r = static_cast<uint32_t>(detail_pp::powmod_u64(x, q, p));
                    t.lut[off + r] |= bit;
                }
            }
            off += p;
        }
        return t;
    }

    // Returns a uint16 mask: bit i is set iff the filter could not rule
    // out n being a (PERFECT_POWER_EXPONENTS[i])-th power.  An all-zero
    // result means n is definitely not a perfect q-th power for any q in
    // the candidate set.
    template<int N>
    uint16_t filter(const fixint::UInt<N>& n) const noexcept {
        uint16_t mask = static_cast<uint16_t>(0xFFFFu);
        alignas(64) uint32_t lanes[K];

        for (std::size_t g = 0; g < num_groups; ++g) {
            __m512i p_vec     = _mm512_loadu_si512(&primes[g * K]);
            __m512i magic_vec = _mm512_loadu_si512(&magics[g * K]);
            __m512i more_vec  = _mm512_loadu_si512(&mores[g * K]);
            __m512i R_vec     = _mm512_loadu_si512(&R[g * K]);

            __m512i r_vec = detail_trial::simd_group_mod_512<N>(
                n.d, p_vec, magic_vec, more_vec, R_vec);
            _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), r_vec);

            for (unsigned l = 0; l < K; ++l) {
                uint32_t off = lut_offset[g * K + l];
                mask &= lut[off + lanes[l]];
            }
            if (mask == 0) return 0;
        }
        return mask;
    }
};

#endif // __AVX512F__ && __AVX512DQ__

} // namespace zfactor
