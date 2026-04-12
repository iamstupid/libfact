#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/jacobi.h"
#include "zfactor/sqrt_mod.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

// ============================================================================
// Jacobi
// ============================================================================

static void test_jacobi_table() {
    std::printf("  jacobi: known table values...\n");

    // (a/3): 0, 1, -1
    CHECK(jacobi_u64(0, 3) == 0,  "(0/3)");
    CHECK(jacobi_u64(1, 3) == 1,  "(1/3)");
    CHECK(jacobi_u64(2, 3) == -1, "(2/3)");

    // (a/5): 0, 1, -1, -1, 1
    CHECK(jacobi_u64(0, 5) ==  0, "(0/5)");
    CHECK(jacobi_u64(1, 5) ==  1, "(1/5)");
    CHECK(jacobi_u64(2, 5) == -1, "(2/5)");
    CHECK(jacobi_u64(3, 5) == -1, "(3/5)");
    CHECK(jacobi_u64(4, 5) ==  1, "(4/5)");

    // (a/7): 0, 1, 1, -1, 1, -1, -1
    CHECK(jacobi_u64(0, 7) ==  0, "(0/7)");
    CHECK(jacobi_u64(1, 7) ==  1, "(1/7)");
    CHECK(jacobi_u64(2, 7) ==  1, "(2/7)");
    CHECK(jacobi_u64(3, 7) == -1, "(3/7)");
    CHECK(jacobi_u64(4, 7) ==  1, "(4/7)");
    CHECK(jacobi_u64(5, 7) == -1, "(5/7)");
    CHECK(jacobi_u64(6, 7) == -1, "(6/7)");

    // Composite n: (5/9) = (5/3)^2 = (-1)^2 = 1
    CHECK(jacobi_u64(5, 9) == 1, "(5/9)");

    // gcd(a,n) > 1
    CHECK(jacobi_u64(15, 9) == 0, "(15/9)");
    CHECK(jacobi_u64(6, 15) == 0, "(6/15)");

    // n == 1
    CHECK(jacobi_u64(0, 1) == 1, "(0/1)");
    CHECK(jacobi_u64(7, 1) == 1, "(7/1)");
}

// For prime p, jacobi(a, p) == Euler criterion a^((p-1)/2) mod p, mapped to ±1.
static void test_jacobi_via_euler() {
    std::printf("  jacobi: cross-check vs Euler criterion...\n");
    auto check_prime = [](uint64_t p) {
        for (uint64_t a = 0; a < p; ++a) {
            int j = jacobi_u64(a, p);
            uint64_t pm = detail_sqrt_mod::powmod_u64(a, (p - 1) / 2, p);
            int expected;
            if (a == 0)       expected = 0;
            else if (pm == 1) expected = 1;
            else              expected = -1;  // pm should be p-1
            CHECK(j == expected, "jacobi vs Euler");
        }
    };
    check_prime(3);
    check_prime(11);
    check_prime(23);
    check_prime(31);
    check_prime(101);
    check_prime(257);
    check_prime(997);
}

static void test_jacobi_multilimb() {
    std::printf("  jacobi: multi-limb cross-check vs u64...\n");
    auto check_pair = [](uint64_t a64, uint64_t n64) {
        // n64 must be odd
        if ((n64 & 1) == 0) n64 |= 1;
        if (n64 < 3) n64 = 3;
        UInt<2> a{}, n{};
        a.d[0] = a64;
        n.d[0] = n64;
        int j_ml  = jacobi<2>(a, n);
        int j_u64 = jacobi_u64(a64, n64);
        CHECK(j_ml == j_u64, "ml jacobi == u64 jacobi");
    };
    check_pair(5, 7);
    check_pair(123, 997);
    check_pair(1000000007ULL, 1000000009ULL);

    // Truly multi-limb n: 2^64 - 59 = 18446744073709551557 (a known prime)
    {
        const uint64_t big_prime = 0xFFFFFFFFFFFFFFC5ULL;
        UInt<2> a{}, n{};
        n.d[0] = big_prime;
        a.d[0] = 12345;
        int j_ml  = jacobi<2>(a, n);
        int j_u64 = jacobi_u64(12345, big_prime);
        CHECK(j_ml == j_u64, "ml jacobi (2^64-59)");
    }

    // a > n case: a should be reduced mod n first
    {
        UInt<2> a{}, n{};
        n.d[0] = 7;
        a.d[0] = 100;
        int j_ml  = jacobi<2>(a, n);
        int j_u64 = jacobi_u64(100, 7);
        CHECK(j_ml == j_u64, "ml jacobi a > n");
    }
}

// ============================================================================
// sqrt_mod_prime
// ============================================================================

static void test_sqrt_exhaustive() {
    std::printf("  sqrt_mod_prime: exhaustive small primes...\n");
    auto check_prime = [](uint64_t p) {
        for (uint64_t a = 0; a < p; ++a) {
            uint64_t x = sqrt_mod_prime(a, p);
            int j = jacobi_u64(a, p);
            if (a == 0) {
                CHECK(x == 0, "sqrt(0) == 0");
            } else if (j == 1) {
                uint64_t x2 = (x * x) % p;
                CHECK(x2 == a, "sqrt^2 == a (QR)");
                CHECK(x != 0, "sqrt of QR != 0");
            } else {
                CHECK(x == 0, "sqrt of non-QR returns 0");
            }
        }
    };

    // p ≡ 3 (mod 4) — closed form
    check_prime(3);
    check_prime(7);
    check_prime(11);
    check_prime(19);
    check_prime(23);
    check_prime(31);
    check_prime(43);
    check_prime(67);
    check_prime(83);

    // p ≡ 5 (mod 8) — Atkin
    check_prime(5);
    check_prime(13);
    check_prime(29);
    check_prime(37);
    check_prime(53);
    check_prime(61);
    check_prime(101);

    // p ≡ 1 (mod 8) — Tonelli-Shanks
    check_prime(17);
    check_prime(41);
    check_prime(73);
    check_prime(89);
    check_prime(97);
    check_prime(113);
    check_prime(193);
    check_prime(257);
    check_prime(353);
}

static void test_sqrt_large() {
    std::printf("  sqrt_mod_prime: large primes...\n");
    using detail_sqrt_mod::mulmod_u64;

    // A few primes covering the size range we care about for QS factor bases.
    uint64_t primes[] = {
        65521ULL,                  // < 2^16, ≡ 1 mod 8
        4294967291ULL,             // ~2^32, ≡ 3 mod 4
        4294967311ULL,             // ~2^32, ≡ 3 mod 4
        18014398509481951ULL,      // ~2^54
        1125899906842597ULL,       // ~2^50
        18446744073709551557ULL,   // 2^64 - 59
    };

    std::mt19937_64 rng(0xCAFEBABEDEADBEEFull);
    for (uint64_t p : primes) {
        int qr_found = 0, nqr_found = 0;
        for (int iter = 0; iter < 200; ++iter) {
            uint64_t a = rng() % p;
            uint64_t x = sqrt_mod_prime(a, p);
            int j = jacobi_u64(a, p);
            if (a == 0) {
                CHECK(x == 0, "large sqrt(0) == 0");
            } else if (j == 1) {
                uint64_t x2 = mulmod_u64(x, x, p);
                CHECK(x2 == a, "large sqrt^2 == a");
                ++qr_found;
            } else {
                CHECK(x == 0, "large non-QR returns 0");
                ++nqr_found;
            }
        }
        // Sanity: roughly half should be QRs
        CHECK(qr_found > 50 && nqr_found > 50, "approximate QR distribution");
    }
}

int main() {
    test_jacobi_table();
    test_jacobi_via_euler();
    test_jacobi_multilimb();
    test_sqrt_exhaustive();
    test_sqrt_large();

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
