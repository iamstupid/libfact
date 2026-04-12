#pragma once

// SIQS Sieve Kernel.
//
// Two-tier sieve:
// - Small/medium primes (< BLOCKSIZE): sieved directly into byte array.
//   For each prime p with roots r1, r2, walk the arithmetic progressions
//   r1, r1+p, r1+2p, ... and r2, r2+p, ... subtracting logp from each byte.
// - Large primes (>= BLOCKSIZE): bucket sieve.  During polynomial init,
//   scatter entries (fb_idx, location) into per-block buckets.  Then when
//   processing each block, drain the bucket and apply the logp values.
//
// The sieve array represents sieve locations [0, 2*M) where M = num_blocks * BLOCKSIZE.
// The actual polynomial offset x = location - M (so x ranges [-M, M)).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/poly.h"

namespace zfactor::siqs {

// Bucket entry: packed fb_index + location within block
struct BucketEntry {
    uint16_t fb_idx;
    uint16_t loc;
};

struct SieveState {
    // Sieve array: one byte per location within current block.
    alignas(64) uint8_t sieve[BLOCKSIZE];

    // Bucket storage for large primes.
    // One bucket per sieve block.
    std::vector<std::vector<BucketEntry>> buckets;

    uint32_t num_blocks;    // per side
    uint32_t sieve_len;     // total = 2 * num_blocks * BLOCKSIZE
    uint32_t half_sieve;    // = num_blocks * BLOCKSIZE (offset for x=0)
    uint8_t  threshold;     // log(Q(x)) - log(LP_bound) approximately

    void init(const SiqsParams& params, const FactorBase& fb) {
        num_blocks = params.num_blocks;
        sieve_len = 2 * num_blocks * BLOCKSIZE;
        half_sieve = num_blocks * BLOCKSIZE;

        // Threshold: approximate log2(Q(x)) at the edges minus the large
        // prime tolerance.  We use a conservative value.
        // For SIQS, sieve values ≈ M * sqrt(kn/2), so log2(Q) ≈ log2(M) + bits(kn)/2.
        // We allow primes up to large_prime_max, so tolerance = log2(large_prime_max).
        // threshold = log2(Q) - tolerance
        // This will be set by the caller after computing actual values.
        threshold = 0;  // Will be set per-polynomial

        // Allocate buckets
        uint32_t total_blocks = 2 * num_blocks;
        buckets.resize(total_blocks);
        for (auto& b : buckets) b.reserve(256);
    }

    // Fill bucket entries for large primes.
    // Called once per polynomial (after root computation).
    template<int N>
    void fill_buckets(const SiqsPoly<N>& poly, const FactorBase& fb) {
        uint32_t total_blocks = 2 * num_blocks;
        for (auto& b : buckets) b.clear();

        for (uint32_t i = fb.med_B; i < fb.fb_size; i++) {
            if (poly.root1[i] == -1) continue;  // A-factor

            uint32_t p = fb.prime[i];
            uint8_t lp = fb.logp[i];

            // Root1: walk r1, r1+p, r1+2p, ... through [0, sieve_len)
            uint32_t r1 = (uint32_t)poly.root1[i];
            while (r1 < sieve_len) {
                uint32_t block = r1 / BLOCKSIZE;
                uint16_t loc = (uint16_t)(r1 % BLOCKSIZE);
                if (block < total_blocks) {
                    buckets[block].push_back({(uint16_t)i, loc});
                }
                r1 += p;
            }

            // Root2
            uint32_t r2 = (uint32_t)poly.root2[i];
            while (r2 < sieve_len) {
                uint32_t block = r2 / BLOCKSIZE;
                uint16_t loc = (uint16_t)(r2 % BLOCKSIZE);
                if (block < total_blocks) {
                    buckets[block].push_back({(uint16_t)i, loc});
                }
                r2 += p;
            }
        }
    }

    // Sieve one block with small/medium primes.
    template<int N>
    void sieve_block(uint32_t block_idx, const SiqsPoly<N>& poly,
                     const FactorBase& fb) {
        // Initialize sieve to threshold
        std::memset(sieve, threshold, BLOCKSIZE);

        uint32_t block_start = block_idx * BLOCKSIZE;

        // Phase 1: Small and medium primes (direct sieve)
        for (uint32_t i = fb.min_sieve_fb; i < fb.med_B; i++) {
            if (poly.root1[i] == -1) continue;

            uint32_t p = fb.prime[i];
            uint8_t lp = fb.logp[i];

            // Compute starting positions within this block
            uint32_t r1 = (uint32_t)poly.root1[i];
            uint32_t r2 = (uint32_t)poly.root2[i];

            // Advance roots to be >= block_start
            if (r1 < block_start) {
                uint32_t skip = (block_start - r1 + p - 1) / p;
                r1 += skip * p;
            }
            if (r2 < block_start) {
                uint32_t skip = (block_start - r2 + p - 1) / p;
                r2 += skip * p;
            }

            // Convert to block-local offsets
            uint32_t off1 = r1 - block_start;
            uint32_t off2 = r2 - block_start;

            // Ensure off1 <= off2 for the merged loop
            if (off1 > off2) std::swap(off1, off2);

            // Sieve both roots
            while (off2 < BLOCKSIZE) {
                sieve[off1] -= lp;
                sieve[off2] -= lp;
                off1 += p;
                off2 += p;
            }
            // off2 past end, off1 might have one more
            if (off1 < BLOCKSIZE) {
                sieve[off1] -= lp;
            }
        }

        // Phase 2: Large primes from bucket
        if (block_idx < buckets.size()) {
            for (auto& entry : buckets[block_idx]) {
                sieve[entry.loc] -= fb.logp[entry.fb_idx];
            }
        }
    }
};

} // namespace zfactor::siqs
