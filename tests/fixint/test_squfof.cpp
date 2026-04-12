#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/squfof.h"
#include "zfactor/fixint/uint.h"
#include "zfactor/bpsw.h"
#include "zfactor/rho.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

static bool is_prime_u64(uint64_t n) { return bpsw<1>(UInt<1>(n)); }

// ---- hand-picked small semiprimes ----

static void test_known_semiprimes() {
    std::printf("  known semiprimes...\n");

    struct Case { uint64_t n, p, q; };
    Case cases[] = {
        { 7919ULL * 7907ULL,       7907, 7919 },
        { 104729ULL * 104723ULL,   104723, 104729 },
        { 1000003ULL * 1000033ULL, 1000003, 1000033 },
        { 1048573ULL * 1048583ULL, 1048573, 1048583 },
        { 4294967291ULL * 4294967279ULL, 4294967279ULL, 4294967291ULL },  // ~2^64
        { 2305843009213693951ULL * 3ULL, 3, 2305843009213693951ULL },     // M61 * 3
    };

    for (auto& t : cases) {
        uint64_t d = squfof(t.n);
        CHECK(d != 0, "squfof found a factor");
        if (d == 0) continue;

        CHECK(d != 1 && d != t.n, "factor is non-trivial");
        CHECK(t.n % d == 0, "factor divides n");
        uint64_t cofactor = t.n / d;
        bool matches = (d == t.p && cofactor == t.q) || (d == t.q && cofactor == t.p);
        CHECK(matches, "factor matches expected p/q");
    }
}

// ---- random semiprimes in the SQUFOF sweet spot ----

static uint64_t next_prime_u64(uint64_t v) {
    if ((v & 1) == 0) ++v;
    while (!is_prime_u64(v)) v += 2;
    return v;
}

static void test_random_semiprimes() {
    std::printf("  random semiprimes in SQUFOF sweet spot...\n");
    std::mt19937_64 rng(0xBADF00DCAFEBABEULL);

    // Build semiprimes with both factors ~2^20 .. ~2^30 (product 2^40..2^60)
    int tested = 0;
    for (int iter = 0; iter < 30 && tested < 20; ++iter) {
        // Pick random primes around a target bit width
        unsigned bw = 20 + (iter % 11);  // 20..30 bits
        uint64_t lo = uint64_t(1) << bw;
        uint64_t p = next_prime_u64(lo + (rng() & ((uint64_t(1) << bw) - 1)));
        uint64_t q = next_prime_u64(lo + (rng() & ((uint64_t(1) << bw) - 1)));
        if (p == q) continue;

        // Make sure product fits in u64
        if (q > UINT64_MAX / p) continue;
        uint64_t n = p * q;

        uint64_t d = squfof(n);
        CHECK(d != 0, "random semiprime factored");
        if (d == 0) continue;
        CHECK(d != 1 && d != n, "non-trivial");
        CHECK(n % d == 0, "divides");
        bool ok = (d == p || d == q);
        CHECK(ok, "factor matches p or q");
        ++tested;
    }
    std::printf("    tested %d random semiprimes\n", tested);
}

// ---- timing: SQUFOF vs rho on balanced semiprimes ----

static void bench_vs_rho() {
    std::printf("  SQUFOF vs rho timing (balanced semiprimes)...\n");

    struct B { const char* label; uint64_t n; };
    B items[] = {
        // Pairs of primes with each factor around the stated bit count.
        { "30-bit × 30-bit", uint64_t(1073741827ULL) * uint64_t(1073741831ULL) },  // ~60-bit
        { "32-bit × 32-bit", uint64_t(4294967311ULL) * uint64_t(4294967357ULL) },  // ~64-bit (may overflow!)
    };
    // Filter out the ones that overflowed
    std::printf("    %-18s  %12s  %12s\n", "target", "rho (ms)", "squfof (ms)");

    for (auto& it : items) {
        if (it.n == 0) continue;  // skipped

        // rho timing
        double rho_ms = 0;
        {
            MontCtx<1> mctx;
            mctx.init(UInt<1>(it.n));
            UInt<1> d;
            auto t0 = std::chrono::steady_clock::now();
            for (uint64_t c : {1ULL, 2ULL, 3ULL, 5ULL}) {
                d = pollard_rho_brent<1>(mctx, c, 2);
                if (d.d[0] > 1 && d.d[0] != it.n) break;
            }
            auto t1 = std::chrono::steady_clock::now();
            rho_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            CHECK(d.d[0] > 1 && d.d[0] != it.n, "rho found factor");
        }

        // squfof timing
        double squfof_ms = 0;
        {
            auto t0 = std::chrono::steady_clock::now();
            uint64_t d = squfof(it.n);
            auto t1 = std::chrono::steady_clock::now();
            squfof_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            CHECK(d > 1 && d != it.n, "squfof found factor");
            if (d > 1 && d != it.n) {
                CHECK(it.n % d == 0, "squfof factor divides");
            }
        }

        std::printf("    %-18s  %12.4f  %12.4f\n", it.label, rho_ms, squfof_ms);
    }
}

int main() {
    test_known_semiprimes();
    test_random_semiprimes();
    bench_vs_rho();

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
