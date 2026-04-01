#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace zfactor {

// Segmented sieve of Eratosthenes for factor base generation and ECM stage-2.
//
// Odd-only byte-array approach: each segment is L1-sized (32 KB), one byte per
// odd number. Byte array avoids read-modify-write overhead of bit packing while
// staying within L1 for the hot sieve loop.
//
// API:
//   generate(limit)          — returns sorted vector of all primes <= limit
//   for_each(limit, fn)      — streams primes to callback without storing
//   count(limit)             — counts primes <= limit

class PrimeSieve {
    // 32 KB segment fits L1 data cache. Covers 65536 consecutive integers.
    static constexpr uint32_t L1_BYTES = 1u << 15;

public:
    // Generate all primes up to `limit` (inclusive). Limit must be < 2^32.
    static std::vector<uint32_t> generate(uint32_t limit) {
        std::vector<uint32_t> primes;
        if (limit < 2) return primes;

        if (limit >= 100) {
            double est = 1.15 * limit / std::log(static_cast<double>(limit));
            primes.reserve(static_cast<size_t>(est));
        }
        primes.push_back(2);
        if (limit < 3) return primes;

        // Odd-number indexing: global index i represents number 2*i + 1.
        // Index 0 = 1 (skipped), index 1 = 3, index 2 = 5, ...
        uint32_t max_idx = (limit - 1) / 2;

        // --- Small sieve: primes up to sqrt(limit) ---
        uint32_t sqrt_lim = static_cast<uint32_t>(std::sqrt(static_cast<double>(limit)));
        while (static_cast<uint64_t>(sqrt_lim + 1) * (sqrt_lim + 1) <= limit) ++sqrt_lim;

        uint32_t small_size = sqrt_lim / 2 + 1;
        std::vector<uint8_t> small(small_size, 0);

        struct SP { uint32_t p; uint32_t next; };
        std::vector<SP> sps;

        for (uint32_t i = 1; i < small_size; ++i) {
            if (!small[i]) {
                uint32_t p = 2 * i + 1;
                // Mark composites in small sieve starting at p*p.
                // Global index of p*p is (p*p-1)/2 = 2*i*(i+1). Stride = p.
                uint64_t j0 = 2ULL * i * (i + 1);
                for (uint64_t j = j0; j < small_size; j += p)
                    small[static_cast<uint32_t>(j)] = 1;
                sps.push_back({p, static_cast<uint32_t>(j0)});
            }
        }

        // --- Segmented sieve ---
        alignas(64) uint8_t seg[L1_BYTES];

        for (uint32_t lo = 1; lo <= max_idx; lo += L1_BYTES) {
            uint32_t hi = lo + L1_BYTES;
            if (hi > max_idx + 1) hi = max_idx + 1;
            uint32_t len = hi - lo;

            std::memset(seg, 0, len);

            for (auto& sp : sps) {
                if (sp.next >= hi) continue;
                uint32_t j = sp.next - lo;
                uint32_t p = sp.p;
                for (; j < len; j += p)
                    seg[j] = 1;
                sp.next = lo + j;
            }

            for (uint32_t j = 0; j < len; ++j) {
                if (!seg[j])
                    primes.push_back(2 * (lo + j) + 1);
            }
        }

        return primes;
    }

    // Stream all primes up to `limit` to callback. Supports uint64_t limits.
    // Callback signature: void(uint64_t prime).
    // Practical limit: sqrt(limit) must fit in memory for the small sieve
    // (~5 MB for limit = 10^14, ~50 MB for limit = 10^16).
    template<typename F>
    static void for_each(uint64_t limit, F&& callback) {
        if (limit < 2) return;
        callback(uint64_t(2));
        if (limit < 3) return;

        uint64_t max_idx = (limit - 1) / 2;
        uint64_t sqrt_lim = static_cast<uint64_t>(std::sqrt(static_cast<double>(limit)));
        while ((sqrt_lim + 1) * (sqrt_lim + 1) <= limit) ++sqrt_lim;

        uint32_t small_size = static_cast<uint32_t>(sqrt_lim / 2 + 1);
        std::vector<uint8_t> small(small_size, 0);

        struct SP { uint32_t p; uint64_t next; };
        std::vector<SP> sps;

        for (uint32_t i = 1; i < small_size; ++i) {
            if (!small[i]) {
                uint32_t p = 2 * i + 1;
                uint64_t j0 = 2ULL * i * (i + 1);
                for (uint64_t j = j0; j < small_size; j += p)
                    small[static_cast<uint32_t>(j)] = 1;
                sps.push_back({p, j0});
            }
        }

        alignas(64) uint8_t seg[L1_BYTES];

        for (uint64_t lo = 1; lo <= max_idx; lo += L1_BYTES) {
            uint64_t hi = lo + L1_BYTES;
            if (hi > max_idx + 1) hi = max_idx + 1;
            uint32_t len = static_cast<uint32_t>(hi - lo);

            std::memset(seg, 0, len);

            for (auto& sp : sps) {
                if (sp.next >= hi) continue;
                uint32_t j = static_cast<uint32_t>(sp.next - lo);
                uint32_t p = sp.p;
                for (; j < len; j += p)
                    seg[j] = 1;
                sp.next = lo + j;
            }

            for (uint32_t j = 0; j < len; ++j) {
                if (!seg[j])
                    callback(2 * (lo + j) + 1);
            }
        }
    }

    // Count primes up to `limit` without allocating storage.
    static uint64_t count(uint64_t limit) {
        uint64_t cnt = 0;
        for_each(limit, [&cnt](uint64_t) { ++cnt; });
        return cnt;
    }
};

} // namespace zfactor
