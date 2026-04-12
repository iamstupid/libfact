#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/modular.h"
#include "zfactor/bpsw.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

// ---- single-limb known primes / composites ----

static void test_small_primes() {
    std::printf("  small primes...\n");
    // Primes
    int small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                          59, 61, 67, 71, 73, 79, 83, 89, 97};
    for (int p : small_primes) {
        CHECK(bpsw<1>(UInt<1>(uint64_t(p))), "small prime");
    }
    // Some composites
    int composites[] = {0, 1, 4, 6, 8, 9, 10, 15, 21, 25, 27, 33, 35, 49, 51, 55, 65, 77, 91, 99, 100};
    for (int c : composites) {
        CHECK(!bpsw<1>(UInt<1>(uint64_t(c))), "small composite");
    }
}

// ---- Carmichael numbers (the canonical Miller-Rabin pitfall) ----

static void test_carmichaels() {
    std::printf("  Carmichael numbers...\n");
    // First several Carmichael numbers
    uint64_t carms[] = {561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841,
                        29341, 41041, 46657, 52633, 62745, 63973, 75361,
                        // larger
                        162401, 294409, 56052361, 118901521, 172947529};
    for (uint64_t c : carms) {
        CHECK(!bpsw<1>(UInt<1>(c)), "Carmichael composite");
    }
}

// ---- strong base-2 pseudoprimes (BPSW must catch these) ----

static void test_strong_psp_2() {
    std::printf("  strong base-2 pseudoprimes...\n");
    // First strong pseudoprimes to base 2 (these pass MR base 2 but fail Lucas)
    uint64_t psps[] = {2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141,
                       52633, 65281, 74665, 80581, 85489, 88357, 90751};
    for (uint64_t p : psps) {
        CHECK(!bpsw<1>(UInt<1>(p)), "strong base-2 pseudoprime");
    }
}

// ---- strong Lucas pseudoprimes (BPSW must catch these) ----

static void test_strong_lucas_psp() {
    std::printf("  strong Lucas pseudoprimes...\n");
    // First strong Lucas pseudoprimes (Selfridge parameters).  These pass
    // strong Lucas but fail MR base 2 — BPSW catches them via the MR step.
    uint64_t psps[] = {5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199,
                       40309, 58519, 75077, 97439, 100127, 113573, 115639};
    for (uint64_t p : psps) {
        CHECK(!bpsw<1>(UInt<1>(p)), "strong Lucas pseudoprime");
    }
}

// ---- large primes ----

static void test_large_primes_1limb() {
    std::printf("  large 1-limb primes...\n");
    uint64_t primes[] = {
        2305843009213693951ULL,  // M61
        18446744073709551557ULL, // 2^64 - 59
        18446744073709551533ULL, // 2^64 - 83
        9999999999999999961ULL,  // largest prime < 10^19
        4294967291ULL,           // 2^32 - 5
        4294967311ULL,           // 2^32 + 15
    };
    for (uint64_t p : primes) {
        CHECK(bpsw<1>(UInt<1>(p)), "large 1-limb prime");
    }
}

// ---- composites ----

static void test_large_composites_1limb() {
    std::printf("  large 1-limb composites...\n");
    // n^2 of various primes — these are all squares so BPSW should reject
    uint64_t squares[] = {
        uint64_t(101) * 101,
        uint64_t(1009) * 1009,
        uint64_t(65537) * 65537,
        uint64_t(2147483647) * 3,  // M31 * 3
    };
    for (uint64_t c : squares) {
        CHECK(!bpsw<1>(UInt<1>(c)), "large composite");
    }
    // 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417
    CHECK(!bpsw<1>(UInt<1>(UINT64_MAX)), "2^64 - 1 composite");
}

// ---- multi-limb primes ----

static void test_multilimb_primes() {
    std::printf("  multi-limb primes...\n");

    // 2^127 - 1 (M127)
    {
        UInt<2> p;
        p.d[0] = UINT64_MAX;
        p.d[1] = (1ULL << 63) - 1;
        CHECK(bpsw<2>(p), "M127");
    }
    // 2^89 - 1 (M89), in 2 limbs
    {
        UInt<2> p;
        p.d[0] = UINT64_MAX;
        p.d[1] = (1ULL << 25) - 1;
        CHECK(bpsw<2>(p), "M89");
    }
    // 2^128 - 159 prime
    {
        UInt<2> p;
        p.d[0] = UINT64_MAX - 158;
        p.d[1] = UINT64_MAX;
        CHECK(bpsw<2>(p), "2^128-159");
    }
    // NIST P-192 prime: 2^192 - 2^64 - 1
    {
        UInt<3> p;
        p.d[0] = UINT64_MAX;
        p.d[1] = UINT64_MAX - 1;
        p.d[2] = UINT64_MAX;
        CHECK(bpsw<3>(p), "NIST P-192");
    }
    // NIST P-256 prime: 2^256 - 2^224 + 2^192 + 2^96 - 1
    {
        UInt<4> p;
        p.d[0] = UINT64_MAX;
        p.d[1] = 0x00000000FFFFFFFFULL;
        p.d[2] = 0;
        p.d[3] = 0xFFFFFFFF00000001ULL;
        CHECK(bpsw<4>(p), "NIST P-256");
    }
    // secp256k1 prime: 2^256 - 2^32 - 977
    {
        UInt<4> p;
        p.d[0] = 0xFFFFFFFEFFFFFC2FULL;
        p.d[1] = UINT64_MAX;
        p.d[2] = UINT64_MAX;
        p.d[3] = UINT64_MAX;
        CHECK(bpsw<4>(p), "secp256k1");
    }
    // 2^512 - 38117 prime
    {
        UInt<8> p;
        for (int i = 0; i < 8; ++i) p.d[i] = UINT64_MAX;
        p.d[0] = UINT64_MAX - 38116;
        CHECK(bpsw<8>(p), "2^512-38117");
    }
}

// ---- multi-limb composites ----

static void test_multilimb_composites() {
    std::printf("  multi-limb composites...\n");
    // 2^128 - 1 = (2^64-1)(2^64+1)
    {
        UInt<2> c;
        c.d[0] = UINT64_MAX;
        c.d[1] = UINT64_MAX;
        CHECK(!bpsw<2>(c), "2^128-1");
    }
    // 2^256 - 1
    {
        UInt<4> c;
        for (int i = 0; i < 4; ++i) c.d[i] = UINT64_MAX;
        CHECK(!bpsw<4>(c), "2^256-1");
    }
    // 2^512 - 1
    {
        UInt<8> c;
        for (int i = 0; i < 8; ++i) c.d[i] = UINT64_MAX;
        CHECK(!bpsw<8>(c), "2^512-1");
    }
    // 2^253 - 1 (composite Mersenne, factor includes 8191)
    {
        UInt<4> c;
        c.d[0] = UINT64_MAX;
        c.d[1] = UINT64_MAX;
        c.d[2] = UINT64_MAX;
        c.d[3] = (1ULL << 61) - 1;
        CHECK(!bpsw<4>(c), "2^253-1");
    }
}

// ---- exhaustive small range ----

static bool naive_is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0) return false;
    for (uint64_t i = 3; i * i <= n; i += 2)
        if (n % i == 0) return false;
    return true;
}

static void test_exhaustive() {
    std::printf("  exhaustive [0, 5000)...\n");
    int mismatches = 0;
    for (uint64_t n = 0; n < 5000; ++n) {
        bool b = bpsw<1>(UInt<1>(n));
        bool naive = naive_is_prime(n);
        if (b != naive) {
            ++mismatches;
            std::fprintf(stderr, "  mismatch at n=%llu (bpsw=%d, naive=%d)\n",
                         (unsigned long long)n, b, naive);
        }
    }
    CHECK(mismatches == 0, "bpsw matches naive on [0, 5000)");
}

// ---- bpsw_with_ctx: caller-managed Montgomery context ----
//
// Verify the no-init variant agrees with bpsw() when the caller has
// already set up a MontScope for n.  Also exercise context reuse — the
// common pattern in rho / ECM is to test multiple candidates (or stage
// through several cofactors) under the same MontScope.

template<int N>
static bool with_ctx(const UInt<N>& n) {
    MontCtx<N> mctx;
    mctx.init(n);
    MontScope<N> scope(mctx);
    return bpsw_with_ctx<N>(n);
}

static void test_with_ctx_agrees() {
    std::printf("  bpsw_with_ctx matches bpsw across [3, 5000)...\n");
    int mismatches = 0;
    for (uint64_t n = 3; n < 5000; n += 2) {  // odd, >= 3 — prereq for bpsw_with_ctx
        bool b_full = bpsw<1>(UInt<1>(n));
        bool b_ctx  = with_ctx<1>(UInt<1>(n));
        if (b_full != b_ctx) {
            ++mismatches;
            std::fprintf(stderr, "  with_ctx mismatch at n=%llu (full=%d, ctx=%d)\n",
                         (unsigned long long)n, b_full, b_ctx);
        }
    }
    CHECK(mismatches == 0, "bpsw_with_ctx matches bpsw on odd [3, 5000)");
}

static void test_with_ctx_multilimb() {
    std::printf("  bpsw_with_ctx multi-limb...\n");

    // NIST P-256
    {
        UInt<4> p;
        p.d[0] = UINT64_MAX;
        p.d[1] = 0x00000000FFFFFFFFULL;
        p.d[2] = 0;
        p.d[3] = 0xFFFFFFFF00000001ULL;
        CHECK(with_ctx<4>(p), "with_ctx NIST P-256");
    }
    // 2^512 - 38117
    {
        UInt<8> p;
        for (int i = 0; i < 8; ++i) p.d[i] = UINT64_MAX;
        p.d[0] = UINT64_MAX - 38116;
        CHECK(with_ctx<8>(p), "with_ctx 2^512-38117");
    }
    // 2^256 - 1 (composite)
    {
        UInt<4> c;
        for (int i = 0; i < 4; ++i) c.d[i] = UINT64_MAX;
        CHECK(!with_ctx<4>(c), "with_ctx 2^256-1");
    }
}

int main() {
    test_small_primes();
    test_carmichaels();
    test_strong_psp_2();
    test_strong_lucas_psp();
    test_large_primes_1limb();
    test_large_composites_1limb();
    test_multilimb_primes();
    test_multilimb_composites();
    test_exhaustive();
    test_with_ctx_agrees();
    test_with_ctx_multilimb();

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
