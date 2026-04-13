#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

#include "zfactor/fixint/uint.h"

// Scalar reference: params.h uint_mod_u32
template<int N>
inline uint32_t scalar_mod(const zfactor::fixint::UInt<N>& a, uint32_t p) {
    uint64_t r = 0;
    for (int i = N - 1; i >= 0; i--) {
        unsigned __int128 w = (unsigned __int128)r << 64 | a.d[i];
        r = (uint64_t)(w % p);
    }
    return (uint32_t)r;
}

#if defined(__AVX2__)
#include "zfactor/fixint/simd_mod.h"

using namespace zfactor::fixint;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* desc) {
    if (ok) { ++g_pass; }
    else { ++g_fail; fprintf(stderr, "  FAIL: %s\n", desc); }
}

// Build a UInt<N> from a u128 value (for constructing test inputs)
template<int N>
UInt<N> from_u128(unsigned __int128 v) {
    UInt<N> r{};
    r.d[0] = (uint64_t)v;
    if constexpr (N >= 2) r.d[1] = (uint64_t)(v >> 64);
    return r;
}

// Build a UInt<N> with value = n * p + rem  (for targeted boundary tests)
template<int N>
UInt<N> make_boundary_value(uint64_t n_mult, uint32_t p, uint32_t rem) {
    // Compute n_mult * p + rem as multi-limb
    UInt<N> result{};
    // Use 128-bit arithmetic for the base product
    unsigned __int128 prod = (unsigned __int128)n_mult * p + rem;
    result.d[0] = (uint64_t)prod;
    if constexpr (N >= 2) result.d[1] = (uint64_t)(prod >> 64);
    return result;
}

// Test one (value, prime) pair against scalar reference via batch_mod API
template<int N>
void test_one(const UInt<N>& val, uint32_t p, const char* tag) {
    uint32_t expected = scalar_mod<N>(val, p);

    uint32_t primes8[8];
    for (int i = 0; i < 8; ++i) primes8[i] = p;

    auto table = simd_mod::SimdModTable::build(primes8, 8);
    uint32_t results[8];
    simd_mod::batch_mod<N>(val, table, results);

    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (results[i] != expected) {
            ok = false;
            fprintf(stderr, "    lane %d: got %u, expected %u (p=%u, tag=%s)\n",
                    i, results[i], expected, p, tag);
        }
    }
    check(ok, tag);
}

// Boundary tests: the critical values around n*p + floor(p/2)
template<int N>
void test_boundary_cases(uint32_t p) {
    char desc[256];

    // Test with various multipliers n
    uint64_t multipliers[] = {0, 1, 2, 100, 65535, 1000000, (uint64_t)p, (uint64_t)p * 2};
    for (uint64_t n : multipliers) {
        // n*p (remainder = 0)
        auto v0 = make_boundary_value<N>(n, p, 0);
        snprintf(desc, sizeof(desc), "n=%llu p=%u rem=0", (unsigned long long)n, p);
        test_one<N>(v0, p, desc);

        // n*p - 1 (remainder = p-1)
        if (n > 0) {
            auto vm1 = make_boundary_value<N>(n - 1, p, p - 1);
            snprintf(desc, sizeof(desc), "n=%llu p=%u rem=p-1", (unsigned long long)n, p);
            test_one<N>(vm1, p, desc);
        }

        // n*p + floor(p/2) (remainder = p/2)
        {
            uint32_t half = p / 2;
            auto vh = make_boundary_value<N>(n, p, half);
            snprintf(desc, sizeof(desc), "n=%llu p=%u rem=p/2=%u", (unsigned long long)n, p, half);
            test_one<N>(vh, p, desc);
        }

        // n*p + floor(p/2) + 1
        {
            uint32_t half1 = p / 2 + 1;
            auto vh1 = make_boundary_value<N>(n, p, half1);
            snprintf(desc, sizeof(desc), "n=%llu p=%u rem=p/2+1=%u", (unsigned long long)n, p, half1);
            test_one<N>(vh1, p, desc);
        }
    }
}

// Random tests with multiple distinct primes per group
template<int N>
void test_random_mixed_group(std::mt19937_64& rng, int count) {
    char desc[256];

    for (int t = 0; t < count; ++t) {
        // Generate 8 distinct primes in [1024, 2^31)
        // Use odd random values; not guaranteed prime but doesn't matter for mod correctness
        uint32_t primes8[8];
        for (int i = 0; i < 8; ++i) {
            primes8[i] = 1024 + (rng() % (0x7FFFFFFFU - 1024));
            if (primes8[i] < 1024) primes8[i] = 1024;
        }

        // Generate random UInt<N>
        UInt<N> val{};
        for (int i = 0; i < N; ++i)
            val.d[i] = rng();

        // Scalar reference for each prime
        uint32_t expected[8];
        for (int i = 0; i < 8; ++i)
            expected[i] = scalar_mod<N>(val, primes8[i]);

        // SIMD via batch_mod (handles deinterleaving)
        auto table = simd_mod::SimdModTable::build(primes8, 8);
        uint32_t results[8];
        simd_mod::batch_mod<N>(val, table, results);

        bool ok = true;
        for (int i = 0; i < 8; ++i) {
            if (results[i] != expected[i]) {
                ok = false;
                fprintf(stderr, "    mixed group trial %d lane %d: got %u, expected %u (p=%u)\n",
                        t, i, results[i], expected[i], primes8[i]);
            }
        }
        snprintf(desc, sizeof(desc), "random_mixed<N=%d> trial %d", N, t);
        check(ok, desc);
    }
}

// Test the batch_mod API
template<int N>
void test_batch_api(std::mt19937_64& rng) {
    // Generate some primes
    std::vector<uint32_t> primes;
    for (int i = 0; i < 50; ++i)
        primes.push_back(1024 + (rng() % (0x7FFFFFFFU - 1024)));

    UInt<N> val{};
    for (int i = 0; i < N; ++i)
        val.d[i] = rng();

    auto table = simd_mod::SimdModTable::build(primes.data(), (uint32_t)primes.size());
    std::vector<uint32_t> results(primes.size());
    simd_mod::batch_mod<N>(val, table, results.data());

    bool ok = true;
    for (size_t i = 0; i < primes.size(); ++i) {
        uint32_t expected = scalar_mod<N>(val, primes[i]);
        if (results[i] != expected) {
            ok = false;
            fprintf(stderr, "    batch<N=%d> prime[%zu]=%u: got %u expected %u\n",
                    N, i, primes[i], results[i], expected);
        }
    }
    char desc[64];
    snprintf(desc, sizeof(desc), "batch_mod<N=%d> 50 primes", N);
    check(ok, desc);
}

template<int N>
void run_tests_for_N() {
    fprintf(stderr, "--- N = %d ---\n", N);
    std::mt19937_64 rng(42 + N);

    // Boundary tests for specific prime ranges
    uint32_t test_primes[] = {
        1021,           // small prime near 1024
        1031,           // small prime just above 1024
        4099,           // medium
        32749,          // near 2^15
        32771,          // just above 2^15
        65537,          // 2^16 + 1
        1000003,        // large
        (1u << 20) + 7, // 2^20 range
        (1u << 30) + 3, // 2^30 range
        0x7FFFFFFFU,    // 2^31 - 1 (max)
    };

    for (uint32_t p : test_primes) {
        if (p < 2) continue;
        test_boundary_cases<N>(p);
    }

    // Random mixed groups
    test_random_mixed_group<N>(rng, 1000);

    // Batch API
    test_batch_api<N>(rng);

    // Edge case: n = 0
    {
        UInt<N> zero{};
        test_one<N>(zero, 1031, "n=0 p=1031");
    }

    // Edge case: n = all ones
    {
        UInt<N> allones{};
        memset(allones.d, 0xFF, sizeof(allones.d));
        test_one<N>(allones, 1031, "n=0xFF..F p=1031");
        test_one<N>(allones, 0x7FFFFFFFU, "n=0xFF..F p=2^31-1");
    }
}

int main() {
    fprintf(stderr, "=== test_simd_mod ===\n\n");

    run_tests_for_N<1>();
    run_tests_for_N<2>();
    run_tests_for_N<4>();
    run_tests_for_N<8>();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#else // no AVX2

int main() {
    fprintf(stderr, "test_simd_mod: skipped (no AVX2)\n");
    return 0;
}

#endif
