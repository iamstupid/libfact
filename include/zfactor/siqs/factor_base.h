#pragma once

// SIQS Factor Base construction.
//
// The factor base for kn consists of primes p such that the Legendre symbol
// (kn / p) ≥ 0 (i.e., kn is a quadratic residue mod p, or p | kn).
// For each such p we compute sqrt(kn) mod p via Tonelli-Shanks.
//
// The factor base is stored in Structure-of-Arrays layout for SIMD-friendly
// access during sieving: parallel arrays of prime[], modsqrt[], logp[].

#include <cmath>
#include <cstdint>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/jacobi.h"
#include "zfactor/sqrt_mod.h"
#include "zfactor/sieve.h"
#include "zfactor/siqs/params.h"

namespace zfactor::siqs {

static constexpr uint32_t BLOCKSIZE = 32768;

struct FactorBase {
    // Parallel arrays, all of length fb_size (starting from index 0 = prime -1,
    // index 1 = prime 2, index 2+ = odd primes).
    // Index 0 is a sentinel for the sign row (prime = "−1").
    std::vector<uint32_t> prime;
    std::vector<uint32_t> modsqrt;   // sqrt(kn) mod p (only one root stored; second is p - modsqrt)
    std::vector<uint8_t>  logp;      // floor(log2(p) + 0.5)

    uint32_t fb_size;       // total number of factor base entries
    uint32_t multiplier;    // Knuth-Schroeppel multiplier k

    // Threshold indices for sieve stages:
    uint32_t min_sieve_fb;  // first FB index to sieve (skip tiny primes for SPV)
    uint32_t med_B;         // first FB index with prime >= BLOCKSIZE (start of bucket sieve)

    // Large prime bounds:
    uint64_t large_prime_max;   // single LP cutoff
    double   large_prime_max2;  // double LP cutoff (LP_max ^ dlp_exp)
    double   dlp_exp;

    // Build the factor base for k*n.
    template<int N>
    static FactorBase build(const fixint::UInt<N>& n, const SiqsParams& params, uint32_t multiplier) {
        FactorBase fb;
        fb.multiplier = multiplier;
        fb.dlp_exp = params.dlp_exp;

        // Compute kn = k * n
        fixint::UInt<N> kn = n;
        {
            // Multiply by small multiplier
            uint64_t carry = 0;
            for (int i = 0; i < N; i++) {
                unsigned __int128 w = (unsigned __int128)kn.d[i] * multiplier + carry;
                kn.d[i] = (uint64_t)w;
                carry = (uint64_t)(w >> 64);
            }
        }

        // Target FB size from params
        uint32_t target_fb = params.fb_size;

        // Generate enough primes.  We need roughly 2*target_fb primes
        // since about half will have (kn/p) = -1.
        // Upper bound: p_n ~ n * ln(n), so we estimate max prime.
        uint64_t max_prime_est = std::max<uint64_t>(
            100000ULL, (uint64_t)(target_fb * 20));
        auto primes = zfactor::primes_range_as<uint32_t>(2, max_prime_est);

        fb.prime.reserve(target_fb + 16);
        fb.modsqrt.reserve(target_fb + 16);
        fb.logp.reserve(target_fb + 16);

        // Index 0: the "−1" sentinel (for sign of sieve values)
        fb.prime.push_back(0);      // sentinel, not a real prime
        fb.modsqrt.push_back(0);
        fb.logp.push_back(0);

        // Index 1: prime 2 (always in FB)
        fb.prime.push_back(2);
        fb.modsqrt.push_back(1);
        fb.logp.push_back(1);

        uint32_t filled = 2;  // indices 0 and 1
        for (size_t pi = 1; pi < primes.size() && filled < target_fb; pi++) {
            uint32_t p = primes[pi];
            if (p == 2) continue;

            uint32_t kn_modp = uint_mod_u32<N>(kn, p);

            if (kn_modp == 0) {
                // p divides kn. If p divides the multiplier, include with
                // logp/2 (one root).  If p divides n itself, the caller
                // should have removed it already, but include anyway.
                uint8_t lp = std::max<uint8_t>(1, (uint8_t)(std::log2((double)p) / 2.0 + 0.5));
                fb.prime.push_back(p);
                // Find root by brute force for small p dividing multiplier
                uint32_t root = 0;
                for (uint32_t r = 0; r < p; r++) {
                    if (((uint64_t)r * r) % p == kn_modp) { root = r; break; }
                }
                fb.modsqrt.push_back(root);
                fb.logp.push_back(lp);
                filled++;
                continue;
            }

            // Check if kn is a QR mod p
            if (jacobi_u64(kn_modp, p) != 1)
                continue;

            uint32_t root = sqrt_mod_prime(kn_modp, p);

            uint8_t lp = (uint8_t)(std::log2((double)p) + 0.5);
            if (lp < 1) lp = 1;

            fb.prime.push_back(p);
            fb.modsqrt.push_back(root);
            fb.logp.push_back(lp);
            filled++;
        }

        fb.fb_size = filled;

        // If we didn't find enough primes, try a larger range
        if (filled < target_fb) {
            // Extend with more primes
            uint64_t new_max = max_prime_est * 4;
            auto more = zfactor::primes_range_as<uint32_t>(max_prime_est, new_max);
            for (size_t pi = 0; pi < more.size() && filled < target_fb; pi++) {
                uint32_t p = more[pi];
                uint32_t kn_modp = uint_mod_u32<N>(kn, p);
                if (kn_modp == 0) continue;
                if (jacobi_u64(kn_modp, p) != 1) continue;
                uint32_t root = sqrt_mod_prime(kn_modp, p);
                uint8_t lp = (uint8_t)(std::log2((double)p) + 0.5);
                fb.prime.push_back(p);
                fb.modsqrt.push_back(root);
                fb.logp.push_back(lp);
                filled++;
            }
            fb.fb_size = filled;
        }

        // Compute threshold indices
        // min_sieve_fb: skip primes < ~50 (too small to sieve efficiently)
        fb.min_sieve_fb = 2;  // start sieving from index 2 (first odd prime)
        for (uint32_t i = 2; i < fb.fb_size; i++) {
            if (fb.prime[i] >= 50) { fb.min_sieve_fb = i; break; }
        }

        // med_B: first prime >= BLOCKSIZE (these need bucket sieve)
        fb.med_B = fb.fb_size;
        for (uint32_t i = 2; i < fb.fb_size; i++) {
            if (fb.prime[i] >= BLOCKSIZE) { fb.med_B = i; break; }
        }

        // Large prime bounds
        uint32_t pmax = fb.prime[fb.fb_size - 1];
        fb.large_prime_max = (uint64_t)params.lp_mult * pmax;
        fb.large_prime_max2 = std::pow((double)fb.large_prime_max, params.dlp_exp);

        return fb;
    }
};

} // namespace zfactor::siqs
