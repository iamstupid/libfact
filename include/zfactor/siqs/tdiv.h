#pragma once

// SIQS Trial Division.
//
// After sieving, scan for locations where the accumulated logp sum exceeds
// the threshold (indicating the sieve value Q(x) is likely smooth over the
// factor base).  For each candidate, reconstruct Q(x) as a multi-precision
// value, trial divide by factor base primes, and check the cofactor.
//
// Relation types:
//   - Full: cofactor == 1
//   - Single Large Prime (SLP): cofactor is prime and < large_prime_max
//   - Double Large Prime (DLP): cofactor splits into two primes, each < large_prime_max

#include <cstdint>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/bpsw.h"
#include "zfactor/squfof.h"

#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/poly.h"
#include "zfactor/siqs/sieve.h"
#include "zfactor/siqs/relation.h"

namespace zfactor::siqs {

// Scan one sieve block for smooth candidates and trial divide them.
// Appends found relations to rels.
template<int N>
void scan_and_tdiv(uint32_t block_idx,
                   const uint8_t* sieve,  // BLOCKSIZE bytes
                   const SiqsPoly<N>& poly,
                   const FactorBase& fb,
                   const fixint::UInt<N>& kn,
                   uint32_t half_sieve,
                   uint32_t poly_a_idx,
                   RelationSet& rels) {
    uint32_t block_start = block_idx * BLOCKSIZE;

    // Phase 1: Scan for candidates (byte with high bit set = wrapped below 0)
    // The sieve was initialized to threshold; subtracting logp decreases it.
    // If it wraps past 0 (high bit set), the location is a candidate.
    for (uint32_t j = 0; j < BLOCKSIZE; j += 8) {
        // Quick 8-byte check
        uint64_t word = *(const uint64_t*)(sieve + j);
        if (!(word & 0x8080808080808080ULL)) continue;

        for (uint32_t k = 0; k < 8; k++) {
            if (!(sieve[j + k] & 0x80)) continue;

            uint32_t loc = block_start + j + k;
            int32_t x = (int32_t)loc - (int32_t)half_sieve;

            // Compute Q(x) = ((Ax + B)^2 - kn) / A
            // This avoids manual sign tracking of the polynomial terms.
            // We compute in UInt<2N> to avoid overflow, then divide by A.

            uint32_t abs_x = (uint32_t)std::abs(x);
            bool x_neg = (x < 0);

            // Compute val = |A*x + B| in UInt<N>
            fixint::UInt<N> ax = {};
            {
                uint64_t carry = 0;
                for (int i = 0; i < N; i++) {
                    unsigned __int128 w = (unsigned __int128)poly.A.d[i] * abs_x + carry;
                    ax.d[i] = (uint64_t)w;
                    carry = (uint64_t)(w >> 64);
                }
            }

            fixint::UInt<N> val = {};
            bool val_neg = false;
            if (!x_neg) {
                fixint::mpn::add<N>(val.d, ax.d, poly.B.d);
            } else {
                if (fixint::mpn::cmp<N>(poly.B.d, ax.d) >= 0) {
                    fixint::mpn::sub<N>(val.d, poly.B.d, ax.d);
                } else {
                    fixint::mpn::sub<N>(val.d, ax.d, poly.B.d);
                    val_neg = true;
                }
            }

            // val_sq = val^2 in UInt<2N>
            fixint::UInt<2*N> val_sq = {};
            fixint::mpn::mul<N>(val_sq.d, val.d, val.d);

            // Compute |val^2 - kn| and its sign
            fixint::UInt<2*N> kn_wide = {};
            for (int i = 0; i < N; i++) kn_wide.d[i] = kn.d[i];

            fixint::UInt<2*N> diff_wide = {};
            bool Q_neg = false;
            if (fixint::mpn::cmp<2*N>(val_sq.d, kn_wide.d) >= 0) {
                fixint::mpn::sub<2*N>(diff_wide.d, val_sq.d, kn_wide.d);
            } else {
                fixint::mpn::sub<2*N>(diff_wide.d, kn_wide.d, val_sq.d);
                Q_neg = true;
            }

            // Q = diff_wide / A (exact division)
            fixint::UInt<2*N> A_wide = {};
            for (int i = 0; i < N; i++) A_wide.d[i] = poly.A.d[i];
            fixint::UInt<2*N> Q_wide = {}, R_wide = {};
            fixint::mpn::divrem<2*N>(Q_wide.d, R_wide.d, diff_wide.d, A_wide.d);

            // Check remainder is 0 (division should be exact)
            {
                bool r_zero = true;
                for (int i = 0; i < 2*N; i++) if (R_wide.d[i] != 0) r_zero = false;
                if (!r_zero) continue;  // Not exact division — skip this candidate
            }

            fixint::UInt<N> Q = {};
            for (int i = 0; i < N; i++) Q.d[i] = Q_wide.d[i];

            if (Q.is_zero()) continue;

            // Trial divide Q by factor base primes
            Relation rel;
            rel.sieve_offset = loc;
            rel.poly_idx = poly.poly_index;
            rel.a_poly_idx = poly_a_idx;
            rel.sign = Q_neg ? 1 : 0;
            rel.large_prime[0] = 1;
            rel.large_prime[1] = 1;

            // The SIQS identity: (Ax+B)^2 = A*Q(x) (mod kn).
            // We need the full factorization of A*Q(x).
            // Add one copy of each A-factor prime to account for the factor of A.
            for (int af = 0; af < poly.s; af++) {
                rel.fb_offsets.push_back((uint16_t)poly.a_factors[af]);
            }

            // Now trial divide Q(x) by factor base primes.
            // First strip factors of 2
            {
                int tz = 0;
                while ((Q.d[0] & 1) == 0 && !Q.is_zero()) {
                    // Right shift by 1
                    uint64_t carry = 0;
                    for (int i = N - 1; i >= 0; i--) {
                        uint64_t next_carry = Q.d[i] & 1;
                        Q.d[i] = (Q.d[i] >> 1) | (carry << 63);
                        carry = next_carry;
                    }
                    tz++;
                }
                for (int t = 0; t < tz; t++)
                    rel.fb_offsets.push_back(1);  // FB index 1 = prime 2
            }

            // Trial divide by odd primes — use actual division for correctness
            for (uint32_t fi = 2; fi < fb.fb_size; fi++) {
                uint32_t p = fb.prime[fi];
                if (p == 0) continue;

                // Check divisibility by computing Q mod p directly
                uint32_t qmod = uint_mod_u32<N>(Q, p);
                if (qmod != 0) continue;

                // Divide out all powers of p
                while (true) {
                    uint32_t rem = uint_mod_u32<N>(Q, p);
                    if (rem != 0) break;
                    rel.fb_offsets.push_back((uint16_t)fi);
                    // Exact divide Q by p
                    uint64_t r = 0;
                    for (int i = N - 1; i >= 0; i--) {
                        unsigned __int128 w = ((unsigned __int128)r << 64) | Q.d[i];
                        Q.d[i] = (uint64_t)(w / p);
                        r = (uint64_t)(w % p);
                    }
                }
            }

            // Check cofactor
            // Q should now be the unfactored part.
            // If Q == 1: full relation
            // If Q < large_prime_max and probably prime: SLP relation
            // If Q < large_prime_max^dlp_exp: try DLP via SQUFOF

            bool is_one = (Q.d[0] == 1);
            for (int i = 1; i < N && is_one; i++)
                is_one = (Q.d[i] == 0);

            if (is_one) {
                // Full relation
                rels.add_relation(std::move(rel));
                continue;
            }

            // Check if cofactor fits in 64 bits
            bool fits_u64 = true;
            for (int i = 1; i < N; i++)
                if (Q.d[i] != 0) { fits_u64 = false; break; }

            if (!fits_u64) continue;  // cofactor too large

            uint64_t cofactor = Q.d[0];

            if (cofactor <= fb.large_prime_max) {
                // Single large prime relation
                rel.large_prime[0] = cofactor;
                rels.add_relation(std::move(rel));
                continue;
            }

            // Try DLP: cofactor must be < dlp_bound and composite
            if ((double)cofactor > fb.large_prime_max2) continue;

            // Quick primality check
            if (cofactor < (1ULL << 32)) {
                // Small enough for deterministic test
                fixint::UInt<1> cof1;
                cof1.d[0] = cofactor;
                if (zfactor::bpsw<1>(cof1)) continue;  // prime cofactor too big for SLP
            }

            // Try SQUFOF to split
            uint64_t factor = zfactor::squfof(cofactor);
            if (factor > 1 && factor < cofactor) {
                uint64_t other = cofactor / factor;
                if (factor <= fb.large_prime_max && other <= fb.large_prime_max) {
                    // DLP relation!
                    rel.large_prime[0] = std::min(factor, other);
                    rel.large_prime[1] = std::max(factor, other);
                    rels.add_relation(std::move(rel));
                }
            }
        }
    }
}

} // namespace zfactor::siqs
