#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/perfect_power.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// Helper: build n = base^exp as a UInt<N>.  Caller picks N large enough.
template<int N>
static UInt<N> pow_uint(uint64_t base, unsigned exp) {
    UInt<N> r{};
    r.d[0] = 1;
    UInt<N> b{};
    b.d[0] = base;
    for (unsigned i = 0; i < exp; ++i) {
        mpn::limb_t prod[2 * N] = {};
        mpn::mul<N>(prod, r.d, b.d);
        for (int j = 0; j < N; ++j) r.d[j] = prod[j];
    }
    return r;
}

// Index of exponent q in PERFECT_POWER_EXPONENTS, or -1.
static int exp_index(uint32_t q) {
    for (unsigned i = 0; i < PERFECT_POWER_NQ; ++i)
        if (PERFECT_POWER_EXPONENTS[i] == q) return int(i);
    return -1;
}

// ---- positive: known perfect powers must keep the corresponding bit set ----

static void test_positive_known_powers(const PerfectPowerFilter& f) {
    std::printf("  positive: known perfect powers...\n");

    // base must be coprime to all filter primes (< 314) for clean filtering,
    // but the filter is correct regardless — bit q stays set whenever
    // n IS a true q-th power.  We test both coprime and not.

    struct Case {
        uint64_t base;
        uint32_t q;
        const char* msg;
    };
    Case cases[] = {
        {1009, 2,  "1009^2"},
        {1009, 3,  "1009^3"},
        {1009, 5,  "1009^5"},
        {1009, 7,  "1009^7"},
        {1009, 11, "1009^11"},
        {1013, 13, "1013^13"},
        {7,    2,  "7^2"},
        {7,    3,  "7^3"},
        {3,    5,  "3^5"},
        {2,    17, "2^17"},
        {2,    23, "2^23"},
        {2,    53, "2^53"},
    };

    for (const auto& c : cases) {
        UInt<16> n = pow_uint<16>(c.base, c.q);
        uint16_t m = f.filter(n);
        int idx = exp_index(c.q);
        CHECK(idx >= 0, "exponent in candidate set");
        bool bit_set = (m & (uint16_t(1) << idx)) != 0;
        if (!bit_set) {
            std::fprintf(stderr, "  %s: mask = 0x%04x, bit %d (q=%u) NOT set\n",
                         c.msg, m, idx, c.q);
        }
        CHECK(bit_set, c.msg);
    }
}

// ---- negative: a prime > MAX_PRIME is not a perfect power; q=2 bit cleared ----

static void test_negative_known_primes(const PerfectPowerFilter& f) {
    std::printf("  negative: known primes (not powers)...\n");

    // Primes > 314, so they don't share factors with the filter table.
    // For each, q=2 should be aggressively rejected (each odd prime gives
    // ~50% filtering for squares; 64 primes = ~10^-19 false positive).
    uint64_t primes[] = {
        317, 331, 1009, 1013, 1000003, 999999937ULL, 18446744073709551557ULL
    };

    for (uint64_t p : primes) {
        UInt<1> n{}; n.d[0] = p;
        uint16_t m = f.filter(n);
        bool q2_cleared = (m & 1) == 0;
        if (!q2_cleared) {
            std::fprintf(stderr, "  prime %llu: mask = 0x%04x, bit q=2 NOT cleared\n",
                         (unsigned long long)p, m);
        }
        CHECK(q2_cleared, "prime: q=2 cleared");
    }
}

// ---- false positive rate on random non-powers ----

static void test_filter_false_positive_rate(const PerfectPowerFilter& f) {
    std::printf("  false positive rate on random non-powers...\n");

    std::mt19937_64 rng(0xC0FFEE12345ull);
    int q2_set = 0, q3_set = 0, q5_set = 0;
    int total = 5000;
    for (int i = 0; i < total; ++i) {
        // Random N=4 (256-bit).  Random integers are non-perfect-powers
        // with probability ~1, so this measures the filter's false-pos rate.
        UInt<4> n{};
        for (int j = 0; j < 4; ++j) n.d[j] = rng();
        n.d[3] |= (1ULL << 63);  // top bit set
        n.d[0] |= 1;             // odd

        uint16_t m = f.filter(n);
        if (m & (1u << 0)) ++q2_set;
        if (m & (1u << 1)) ++q3_set;
        if (m & (1u << 2)) ++q5_set;
    }
    std::printf("    q=2 false positives: %d / %d  (%.2f%%)\n",
                q2_set, total, 100.0 * q2_set / total);
    std::printf("    q=3 false positives: %d / %d  (%.2f%%)\n",
                q3_set, total, 100.0 * q3_set / total);
    std::printf("    q=5 false positives: %d / %d  (%.2f%%)\n",
                q5_set, total, 100.0 * q5_set / total);
    // Theoretical: q=2 should be essentially 0, q=3 maybe 1-2 stray hits.
    CHECK(q2_set == 0, "q=2 false positive rate is essentially zero");
    CHECK(q3_set <= total / 100, "q=3 false positive rate < 1%");
}

// ---- multi-limb perfect powers ----

static void test_multilimb_powers(const PerfectPowerFilter& f) {
    std::printf("  multi-limb perfect powers...\n");

    // 65537^5 (Fermat prime cubed-and-then-some, fits in N=2)
    {
        UInt<2> n = pow_uint<2>(65537, 5);
        uint16_t m = f.filter(n);
        int idx = exp_index(5);
        CHECK((m & (1u << idx)) != 0, "65537^5: q=5 bit set");
    }
    // 65537^11 (~ 176 bits, N=3)
    {
        UInt<3> n = pow_uint<3>(65537, 11);
        uint16_t m = f.filter(n);
        int idx = exp_index(11);
        CHECK((m & (1u << idx)) != 0, "65537^11: q=11 bit set");
    }
    // 1000003^32 is ~640 bits, needs N=16 to hold without truncation.
    // 32 = 2^5, so n is a perfect square (and 4th, 8th, 16th, 32nd power);
    // q=2 must be set.
    {
        UInt<16> n = pow_uint<16>(1000003, 32);
        uint16_t m = f.filter(n);
        CHECK((m & 1) != 0, "1000003^32: q=2 bit set");
    }
}

#endif // AVX-512

int main() {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    auto f = PerfectPowerFilter::build();
    std::printf("PerfectPowerFilter: %zu primes, %zu groups, lut=%zu bytes\n",
                f.primes.size(), f.num_groups, f.lut.size() * sizeof(uint16_t));

    test_positive_known_powers(f);
    test_negative_known_primes(f);
    test_filter_false_positive_rate(f);
    test_multilimb_powers(f);
#else
    std::printf("AVX-512 not available, skipping perfect power tests\n");
#endif

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
