// Side-by-side comparison of Pollard rho (Brent) and SQUFOF on balanced
// semiprimes across a range of bit widths.  The "balanced" case — both
// factors ~sqrt(n) — is SQUFOF's sweet spot; rho's runtime is O(n^(1/4))
// while SQUFOF is closer to O(n^(1/4)) with much lower constants.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/modular.h"
#include "zfactor/bpsw.h"
#include "zfactor/rho.h"
#include "zfactor/squfof.h"

using namespace zfactor;
using namespace zfactor::fixint;

static uint64_t next_prime_u64(uint64_t v) {
    if ((v & 1) == 0) ++v;
    while (!bpsw<1>(UInt<1>(v))) v += 2;
    return v;
}

// Build a balanced semiprime with each factor ~2^bits.
static uint64_t balanced_semiprime(unsigned bits, std::mt19937_64& rng) {
    uint64_t base = uint64_t(1) << bits;
    uint64_t jitter_mask = (uint64_t(1) << std::min<unsigned>(bits, 20)) - 1;
    uint64_t p = next_prime_u64(base + (rng() & jitter_mask));
    uint64_t q = next_prime_u64(base + (rng() & jitter_mask));
    while (q == p) q = next_prime_u64(q + 2);
    if (p > UINT64_MAX / q) return 0;  // overflow guard
    return p * q;
}

struct Result { double rho_ms, squfof_ms; uint64_t p_rho, p_squfof; };

static Result bench_one(uint64_t n, int reps) {
    Result r{};
    {
        MontCtx<1> mctx;
        mctx.init(UInt<1>(n));
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) {
            UInt<1> d;
            for (uint64_t c : {1ULL, 2ULL, 3ULL, 5ULL, 7ULL}) {
                d = pollard_rho_brent<1>(mctx, c, 2);
                if (d.d[0] > 1 && d.d[0] != n) break;
            }
            r.p_rho = d.d[0];
        }
        auto t1 = std::chrono::steady_clock::now();
        r.rho_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    }
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) {
            r.p_squfof = squfof(n);
        }
        auto t1 = std::chrono::steady_clock::now();
        r.squfof_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    }
    return r;
}

int main() {
    std::mt19937_64 rng(0xFACE1234DEADBEEFULL);

    std::printf("Balanced semiprimes (p ~ q ~ 2^bits).  Times in microseconds per call.\n");
    std::printf("%-6s %-20s %12s %12s %12s\n",
                "bits", "n", "rho (us)", "squfof (us)", "speedup");

    unsigned widths[] = {10, 14, 18, 20, 22, 24, 26, 28, 30, 31};

    for (unsigned bw : widths) {
        // Generate a few semiprimes at this width and average
        constexpr int NUM = 5;
        double rho_total = 0, sq_total = 0;
        int ok = 0;
        uint64_t sample_n = 0;
        for (int i = 0; i < NUM; ++i) {
            uint64_t n = balanced_semiprime(bw, rng);
            if (n == 0) continue;
            sample_n = n;
            // Scale reps to keep total wall time reasonable
            int reps = bw <= 16 ? 1000 : (bw <= 22 ? 200 : (bw <= 26 ? 50 : 10));
            Result r = bench_one(n, reps);
            // Verify both found the factor
            if (r.p_rho > 1 && r.p_rho != n && r.p_squfof > 1 && r.p_squfof != n) {
                rho_total += r.rho_ms;
                sq_total  += r.squfof_ms;
                ++ok;
            }
        }
        if (ok == 0) { std::printf("%-6u (all failed)\n", bw); continue; }
        double rho_us = 1000.0 * rho_total / ok;
        double sq_us  = 1000.0 * sq_total  / ok;
        std::printf("%-6u %-20llu %12.3f %12.3f %11.2fx\n",
                    bw, (unsigned long long)sample_n, rho_us, sq_us,
                    rho_us / sq_us);
    }
    return 0;
}
