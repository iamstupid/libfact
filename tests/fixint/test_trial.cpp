#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/trial.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

// Build a UInt<N> from a uint64_t
template<int N>
static UInt<N> from_u64(uint64_t v) {
    UInt<N> r;
    for (int i = 0; i < N; ++i) r.d[i] = 0;
    r.d[0] = v;
    return r;
}

template<int N>
static uint64_t to_u64(const UInt<N>& a) {
    return a.d[0];
}

// ---- single-limb basics ----

static void test_basic(const TrialDivTable& table) {
    std::printf("  basic factorizations...\n");

    // 360 = 2^3 * 3^2 * 5
    {
        auto n = from_u64<1>(360);
        auto f = trial_divide<1>(n, table);
        CHECK(f.size() == 3, "360: three distinct primes");
        CHECK(f[0].p == 2 && f[0].e == 3, "360: 2^3");
        CHECK(f[1].p == 3 && f[1].e == 2, "360: 3^2");
        CHECK(f[2].p == 5 && f[2].e == 1, "360: 5");
        CHECK(to_u64(n) == 1, "360: fully smooth");
    }

    // 1 → no factors, cofactor 1
    {
        auto n = from_u64<1>(1);
        auto f = trial_divide<1>(n, table);
        CHECK(f.empty(), "1: no factors");
        CHECK(to_u64(n) == 1, "1: cofactor is 1");
    }

    // 2 → one factor, cofactor 1
    {
        auto n = from_u64<1>(2);
        auto f = trial_divide<1>(n, table);
        CHECK(f.size() == 1 && f[0].p == 2 && f[0].e == 1, "2: 2^1");
        CHECK(to_u64(n) == 1, "2: cofactor 1");
    }

    // Large prime beyond table → no factors, cofactor unchanged
    {
        // 1000003 is prime and is past our bound-1000 table
        auto n = from_u64<1>(1000003ULL);
        auto f = trial_divide<1>(n, table);
        CHECK(f.empty(), "1000003: no small factors");
        CHECK(to_u64(n) == 1000003, "1000003: cofactor unchanged");
    }

    // Composite with small factor + large prime cofactor
    // 2 * 1000003 = 2000006
    {
        auto n = from_u64<1>(2000006ULL);
        auto f = trial_divide<1>(n, table);
        CHECK(f.size() == 1 && f[0].p == 2 && f[0].e == 1, "2*1000003: 2^1");
        CHECK(to_u64(n) == 1000003, "2*1000003: cofactor = 1000003");
    }

    // Prime power: 2^20 = 1048576
    {
        auto n = from_u64<1>(1ULL << 20);
        auto f = trial_divide<1>(n, table);
        CHECK(f.size() == 1 && f[0].p == 2 && f[0].e == 20, "2^20");
        CHECK(to_u64(n) == 1, "2^20: cofactor 1");
    }

    // Primorial-like: 2*3*5*7*11*13 = 30030
    {
        auto n = from_u64<1>(30030);
        auto f = trial_divide<1>(n, table);
        CHECK(f.size() == 6, "30030: 6 primes");
        uint64_t prod = 1;
        for (auto& x : f) {
            CHECK(x.e == 1, "30030: all e=1");
            prod *= x.p;
        }
        CHECK(prod == 30030, "30030: prime product matches");
        CHECK(to_u64(n) == 1, "30030: cofactor 1");
    }
}

// ---- random single-limb cross-check against naive factorization ----

// Mirror trial_divide's behaviour exactly: walk every integer 2..bound-1
// and peel off factors.  Composites never hit because their prime factors
// are removed first, so this only records primes — same as iterating the
// sieved table.  Leaves the remaining cofactor in *cofactor_out.
static std::vector<SmallFactor> naive_factor(uint64_t n,
                                             uint32_t bound,
                                             uint64_t* cofactor_out) {
    std::vector<SmallFactor> out;
    for (uint32_t p = 2; p < bound; ++p) {
        if (n % p == 0) {
            uint32_t e = 0;
            while (n % p == 0) { n /= p; ++e; }
            out.push_back({p, e});
            if (n == 1) break;
        }
    }
    *cofactor_out = n;
    return out;
}

static void test_random_u64(const TrialDivTable& table, uint32_t bound) {
    std::printf("  random single-limb cross-check...\n");
    std::mt19937_64 rng(0xABCDEF1234567890ULL);
    for (int iter = 0; iter < 2000; ++iter) {
        uint64_t v = (rng() % ((1ULL << 40) - 1)) + 1;
        auto n = from_u64<1>(v);
        auto f = trial_divide<1>(n, table);

        uint64_t ref_cofactor = 0;
        auto ref = naive_factor(v, bound, &ref_cofactor);

        CHECK(f.size() == ref.size(), "random: count matches");
        bool ok = f.size() == ref.size();
        for (std::size_t i = 0; ok && i < f.size(); ++i) {
            if (f[i].p != ref[i].p || f[i].e != ref[i].e) ok = false;
        }
        CHECK(ok, "random: factor list matches");
        CHECK(to_u64(n) == ref_cofactor, "random: cofactor matches reference");

        // Reconstruct: product of small factors * cofactor == original
        uint64_t prod = to_u64(n);
        for (auto& x : f)
            for (uint32_t k = 0; k < x.e; ++k) prod *= x.p;
        CHECK(prod == v, "random: product reconstructs n");
    }
}

// ---- multi-limb: construct n with known factors then verify ----

static void test_multilimb() {
    std::printf("  multi-limb factorizations...\n");
    auto table = TrialDivTable::build(10000);

    // N=2: 2^64 * small stuff.  Use n = 2^65 * 3^2 * 5.
    // 2^65 overflows 1 limb; 3^2 * 5 * 2^65 needs 2 limbs.
    {
        UInt<2> n{};
        n.d[0] = 0;
        n.d[1] = 45ULL << 1; // 45 * 2 = 90, so value is 90 * 2^64 = 2^65 * 45
        // 45 = 9*5 = 3^2 * 5; value = 2^65 * 3^2 * 5
        auto f = trial_divide<2>(n, table);
        CHECK(f.size() == 3, "2^65*45: three primes");
        CHECK(f[0].p == 2 && f[0].e == 65, "2^65*45: 2^65");
        CHECK(f[1].p == 3 && f[1].e == 2,  "2^65*45: 3^2");
        CHECK(f[2].p == 5 && f[2].e == 1,  "2^65*45: 5");
        CHECK(n.d[0] == 1 && n.d[1] == 0,  "2^65*45: cofactor 1");
    }

    // N=3: multiply a large prime cofactor with small factors
    // cofactor = 0xFFFFFFFFFFFFFFC5 (a 64-bit prime), n = cofactor * 2^2 * 7 * 11
    {
        UInt<3> n{};
        const uint64_t big_prime = 0xFFFFFFFFFFFFFFC5ULL;
        // value = big_prime * 4 * 7 * 11 = big_prime * 308
        // Compute big_prime * 308 as a 128-bit-ish product:
        unsigned __int128 v = (unsigned __int128)big_prime * 308ULL;
        n.d[0] = (uint64_t)v;
        n.d[1] = (uint64_t)(v >> 64);
        n.d[2] = 0;
        auto f = trial_divide<3>(n, table);
        CHECK(f.size() == 3, "big*308: three primes");
        CHECK(f[0].p == 2 && f[0].e == 2,  "big*308: 2^2");
        CHECK(f[1].p == 7 && f[1].e == 1,  "big*308: 7");
        CHECK(f[2].p == 11 && f[2].e == 1, "big*308: 11");
        CHECK(n.d[0] == big_prime && n.d[1] == 0 && n.d[2] == 0,
              "big*308: cofactor matches");
    }

    // N=4: purely smooth product across 4 limbs.
    // Build n = 2^10 * 3^10 * 5^10 * 7^10 and verify roundtrip
    {
        UInt<4> n{};
        // 2^10 * 3^10 * 5^10 * 7^10 = 30030... use mpn::mul1 style via uint128
        // Keep it simple: compute via uint128 then fall out into limbs.
        unsigned __int128 v = 1;
        uint32_t ps[] = {2, 3, 5, 7};
        for (uint32_t p : ps)
            for (int k = 0; k < 10; ++k) v *= p;
        n.d[0] = (uint64_t)v;
        n.d[1] = (uint64_t)(v >> 64);
        n.d[2] = 0;
        n.d[3] = 0;
        auto f = trial_divide<4>(n, table);
        CHECK(f.size() == 4, "smooth^10: four primes");
        for (auto& x : f)
            CHECK(x.e == 10, "smooth^10: all e=10");
        CHECK(n.d[0] == 1 && n.d[1] == 0 && n.d[2] == 0 && n.d[3] == 0,
              "smooth^10: cofactor 1");
    }
}

#if defined(__AVX2__)
// ---- SIMD path cross-check vs scalar ----
static void test_simd_vs_scalar() {
    std::printf("  SIMD vs scalar cross-check (N=1..4)...\n");
    auto st = zfactor::SimdTrialDivTable::build(1u << 15);
    // Scalar table covers the same primes so results are apples-to-apples.
    auto sc = TrialDivTable::build(1u << 15);
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    auto st512 = zfactor::Simd512TrialDivTable::build(1u << 15);
#endif

    std::mt19937_64 rng(0x51BDCFECCAFEBABEull);
    auto cmp_lists = [](const std::vector<SmallFactor>& a,
                        const std::vector<SmallFactor>& b) -> bool {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].p != b[i].p || a[i].e != b[i].e) return false;
        return true;
    };

    auto check_one = [&](auto dummy) {
        constexpr int N = decltype(dummy)::value;
        for (int iter = 0; iter < 500; ++iter) {
            UInt<N> n_sc{}, n_sd{};
            for (int j = 0; j < N; ++j) {
                uint64_t v = rng();
                n_sc.d[j] = v;
                n_sd.d[j] = v;
            }
            // Avoid n == 0
            if (N >= 1) { n_sc.d[0] |= 1; n_sd.d[0] |= 1; }

            UInt<N> n_avx2 = n_sd;
            auto fs = trial_divide<N>(n_sc, sc);
            auto fd = trial_divide_simd<N>(n_avx2, st);

            CHECK(cmp_lists(fs, fd), "simd avx2/scalar: factor lists match");
            bool same_cofactor = true;
            for (int j = 0; j < N; ++j)
                if (n_sc.d[j] != n_avx2.d[j]) same_cofactor = false;
            CHECK(same_cofactor, "simd avx2/scalar: cofactor matches");

#if defined(__AVX512F__) && defined(__AVX512DQ__)
            UInt<N> n_avx512 = n_sd;
            auto f512 = zfactor::trial_divide_simd512<N>(n_avx512, st512);
            CHECK(cmp_lists(fs, f512), "simd avx512/scalar: factor lists match");
            bool same_512 = true;
            for (int j = 0; j < N; ++j)
                if (n_sc.d[j] != n_avx512.d[j]) same_512 = false;
            CHECK(same_512, "simd avx512/scalar: cofactor matches");
#endif
        }
    };

    check_one(std::integral_constant<int, 1>{});
    check_one(std::integral_constant<int, 2>{});
    check_one(std::integral_constant<int, 3>{});
    check_one(std::integral_constant<int, 4>{});
}

static void test_simd_smooth() {
    std::printf("  SIMD smooth composites...\n");
    auto st = zfactor::SimdTrialDivTable::build(1u << 15);

    // 2^4 * 3^2 * 5 * 7 * 1009 = 16 * 9 * 5 * 7 * 1009 = 5085360
    {
        auto n = from_u64<1>(5085360ULL);
        auto f = trial_divide_simd<1>(n, st);
        CHECK(f.size() == 5, "5085360: 5 primes");
        CHECK(f[0].p == 2 && f[0].e == 4, "5085360: 2^4");
        CHECK(f[1].p == 3 && f[1].e == 2, "5085360: 3^2");
        CHECK(f[2].p == 5 && f[2].e == 1, "5085360: 5");
        CHECK(f[3].p == 7 && f[3].e == 1, "5085360: 7");
        CHECK(f[4].p == 1009 && f[4].e == 1, "5085360: 1009");
        CHECK(to_u64(n) == 1, "5085360: cofactor 1");
    }

    // Large smooth N=3: 2^70 * 3 * 5 * 1009 * 10007.
    // (Avoid primes near the 2^15 ceiling — the SIMD table rounds DOWN to a
    // multiple of 8 and discards the trailing primes.  10007 is safely inside.)
    {
        unsigned __int128 v = 1;
        v *= 3ULL;
        v *= 5ULL;
        v *= 1009ULL;
        v *= 10007ULL;
        // v * 2^70 : v < 2^33, so (v << 6) fits in uint128, then low word goes
        // into limb 1 and spillover into limb 2.
        UInt<3> out{};
        __uint128_t vv = v << 6;
        out.d[0] = 0;
        out.d[1] = (uint64_t)vv;
        out.d[2] = (uint64_t)(vv >> 64);

        auto f = trial_divide_simd<3>(out, st);
        CHECK(f.size() == 5, "big smooth: 5 primes");
        CHECK(f[0].p == 2 && f[0].e == 70, "big smooth: 2^70");
        CHECK(f[1].p == 3, "big smooth: has 3");
        CHECK(f[2].p == 5, "big smooth: has 5");
        CHECK(f[3].p == 1009, "big smooth: has 1009");
        CHECK(f[4].p == 10007, "big smooth: has 10007");
        CHECK(out.d[0] == 1 && out.d[1] == 0 && out.d[2] == 0, "big smooth: cofactor 1");
    }
}
#endif // __AVX2__

int main() {
    auto table = TrialDivTable::build(1000);
    std::printf("TrialDivTable: %zu primes below 1000\n", table.size());

    test_basic(table);
    test_random_u64(table, 1000);
    test_multilimb();

#if defined(__AVX2__)
    test_simd_vs_scalar();
    test_simd_smooth();
#endif

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
