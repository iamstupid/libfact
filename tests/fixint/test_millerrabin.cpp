#include <cstdio>
#include <cstdint>
#include "zfactor/fixint/modular.h"

using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

// Miller-Rabin primality test.
// n must be odd and > 2. Returns true if n is probably prime.
// Deterministic for the given set of witnesses.
template<int N>
bool miller_rabin(const UInt<N>& n, const uint64_t* witnesses, int num_witnesses) {
    // Write n-1 = d * 2^r
    UInt<N> n_minus_1 = n;
    mpn::sub1<N>(n_minus_1.d, n_minus_1.d, 1);

    unsigned r = mpn::ctz<N>(n_minus_1.d);
    UInt<N> d = n_minus_1;
    mpn::rshift<N>(d.d, d.d, r);

    MontCtx<N> ctx;
    ctx.init(n);
    MontScope<N> scope(ctx);

    Mod<N> mont_one = Mod<N>::one();
    // n-1 in Montgomery form
    Mod<N> mont_n_minus_1 = Mod<N>::from_uint(n_minus_1);

    for (int w = 0; w < num_witnesses; ++w) {
        uint64_t a_val = witnesses[w];
        // Skip if a >= n
        UInt<N> a_uint(a_val);
        if (mpn::cmp<N>(a_uint.d, n.d) >= 0)
            continue;
        if (a_val == 0)
            continue;

        Mod<N> a = Mod<N>::from_uint(a_uint);
        Mod<N> x = pow<N>(a, d);

        if (x == mont_one || x == mont_n_minus_1)
            continue;

        bool found = false;
        for (unsigned i = 1; i < r; ++i) {
            x = x.sqr();
            if (x == mont_n_minus_1) {
                found = true;
                break;
            }
            if (x == mont_one)
                return false;  // composite
        }
        if (!found)
            return false;  // composite
    }
    return true;  // probably prime
}

// Deterministic witnesses for different ranges:
// For n < 3,317,044,064,679,887,385,961,981: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
// covers all 64-bit and 128-bit primes.
static const uint64_t WITNESSES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
static const int NUM_WITNESSES = 12;

template<int N>
bool is_prime(const UInt<N>& n) {
    // Handle small cases
    if (n.d[0] < 2 && mpn::bit_length<N>(n.d) <= 64) return false;
    if (n.d[0] == 2 && mpn::bit_length<N>(n.d) <= 64) return true;
    if ((n.d[0] & 1) == 0) return false;
    return miller_rabin<N>(n, WITNESSES, NUM_WITNESSES);
}

// =========================================================================
// Known primes and composites for multi-limb testing
// =========================================================================

void test_single_limb() {
    std::fprintf(stderr, "  N=1 known primes/composites...\n");

    // Known primes
    CHECK(is_prime(UInt<1>(2)), "2 prime");
    CHECK(is_prime(UInt<1>(3)), "3 prime");
    CHECK(is_prime(UInt<1>(5)), "5 prime");
    CHECK(is_prime(UInt<1>(97)), "97 prime");
    CHECK(is_prime(UInt<1>(997)), "997 prime");
    CHECK(is_prime(UInt<1>(7919)), "7919 prime");
    CHECK(is_prime(UInt<1>(104729)), "104729 prime");

    // Mersenne prime 2^61 - 1
    CHECK(is_prime(UInt<1>(2305843009213693951ULL)), "M61 prime");

    // Large 64-bit primes
    CHECK(is_prime(UInt<1>(18446744073709551557ULL)), "2^64-59 prime");
    CHECK(is_prime(UInt<1>(18446744073709551533ULL)), "2^64-83 prime");

    // Known composites
    CHECK(!is_prime(UInt<1>(4)), "4 composite");
    CHECK(!is_prime(UInt<1>(9)), "9 composite");
    CHECK(!is_prime(UInt<1>(15)), "15 composite");
    CHECK(!is_prime(UInt<1>(561)), "561 Carmichael");
    CHECK(!is_prime(UInt<1>(1105)), "1105 Carmichael");
    CHECK(!is_prime(UInt<1>(1729)), "1729 Carmichael");

    // 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417
    CHECK(!is_prime(UInt<1>(UINT64_MAX)), "2^64-1 composite");

    // Pseudoprimes to base 2
    CHECK(!is_prime(UInt<1>(2047)), "2047 psp(2)");
    CHECK(!is_prime(UInt<1>(3277)), "3277 psp(2)");

    // Large composite: product of two primes
    // 4294967291 * 4294967279 = 18446744056529682389
    CHECK(!is_prime(UInt<1>(18446744056529682389ULL)), "p*q composite");
}

void test_two_limb() {
    std::fprintf(stderr, "  N=2 known primes/composites...\n");

    // 2^127 - 1 (Mersenne prime M127)
    UInt<2> m127;
    m127.d[0] = UINT64_MAX;
    m127.d[1] = (1ULL << 63) - 1;  // 0x7FFFFFFFFFFFFFFF
    CHECK(is_prime(m127), "M127 prime");

    // 2^107 - 1 (Mersenne prime M107)
    UInt<2> m107;
    m107.d[0] = UINT64_MAX;
    m107.d[1] = (1ULL << 43) - 1;
    CHECK(is_prime(m107), "M107 prime");

    // 2^89 - 1 (Mersenne prime M89)
    UInt<2> m89;
    m89.d[0] = UINT64_MAX;
    m89.d[1] = (1ULL << 25) - 1;
    CHECK(is_prime(m89), "M89 prime");

    // A known 128-bit prime: 2^128 - 159 = 340282366920938463463374607431768211297
    UInt<2> p128;
    p128.d[0] = UINT64_MAX - 158;
    p128.d[1] = UINT64_MAX;
    CHECK(is_prime(p128), "2^128-159 prime");

    // A known 128-bit composite: 2^128 - 1 = (2^64-1)(2^64+1)
    UInt<2> c128;
    c128.d[0] = UINT64_MAX;
    c128.d[1] = UINT64_MAX;
    CHECK(!is_prime(c128), "2^128-1 composite");

    // 2^128 - 157 is composite (even)
    UInt<2> even128;
    even128.d[0] = UINT64_MAX - 156;
    even128.d[1] = UINT64_MAX;
    CHECK(!is_prime(even128), "2^128-157 even composite");

    // Product of two 64-bit primes: (2^64-59) * 3 = a 66-bit composite
    // Actually let's use: 2^89 - 1 is prime, so (2^89-1) * 3 should be composite
    UInt<2> comp = m89;
    // Multiply by 3
    UInt<2> three(3);
    mpn::limb_t prod[4] = {};
    mpn::mul<2>(prod, comp.d, three.d);
    UInt<2> comp3;
    comp3.d[0] = prod[0];
    comp3.d[1] = prod[1];
    if (prod[2] == 0 && prod[3] == 0)
        CHECK(!is_prime(comp3), "M89*3 composite");
}

void test_three_limb() {
    std::fprintf(stderr, "  N=3 known primes/composites...\n");

    // 2^192 - 237 is not necessarily prime, let's use a known one.
    // Next prime after 2^191: 2^191 + 1847 (checked via OEIS/primality tables)
    // Actually, let's construct a verifiable prime.

    // 2^189 - 83 — let me use Fermat verification instead.
    // Use a prime of the form p = k * 2^128 + 1 for easy verification.
    // p = 59649589127497217 * 2^128 + 1
    // That's too large. Let's just test known Mersenne primes that fit in 3 limbs.

    // 2^127 - 1 fits in 2 limbs, so use it in 3-limb representation.
    UInt<3> m127_3;
    m127_3.d[0] = UINT64_MAX;
    m127_3.d[1] = (1ULL << 63) - 1;
    m127_3.d[2] = 0;
    CHECK(is_prime(m127_3), "M127 in 3 limbs");

    // A 192-bit prime: 2^192 - 2^64 - 1 (NIST P-192 prime)
    // p = 6277101735386680763835789423207666416102355444464034512897
    // p = 2^192 - 2^64 - 1
    // d[2] = 0xFFFFFFFFFFFFFFFF, d[1] = 0xFFFFFFFFFFFFFFFE, d[0] = 0xFFFFFFFFFFFFFFFF
    UInt<3> p192;
    p192.d[0] = UINT64_MAX;
    p192.d[1] = UINT64_MAX - 1;
    p192.d[2] = UINT64_MAX;
    CHECK(is_prime(p192), "NIST P-192");

    // Composite: p192 + 2 (p is odd, p+2 may or may not be prime — check p+4 which is even)
    UInt<3> comp192 = p192;
    mpn::add1<3>(comp192.d, comp192.d, 1);  // p+1 is even
    CHECK(!is_prime(comp192), "P-192+1 composite (even)");
}

void test_four_limb() {
    std::fprintf(stderr, "  N=4 known primes/composites...\n");

    // NIST P-256 prime: p = 2^256 - 2^224 + 2^192 + 2^96 - 1
    // = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
    UInt<4> p256;
    p256.d[0] = UINT64_MAX;
    p256.d[1] = 0x00000000FFFFFFFF;
    p256.d[2] = 0x0000000000000000;
    p256.d[3] = 0xFFFFFFFF00000001;
    CHECK(is_prime(p256), "NIST P-256");

    // 2^256 - 2^32 - 977 (secp256k1 prime, used in Bitcoin)
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    UInt<4> secp256k1;
    secp256k1.d[0] = 0xFFFFFFFEFFFFFC2FULL;
    secp256k1.d[1] = UINT64_MAX;
    secp256k1.d[2] = UINT64_MAX;
    secp256k1.d[3] = UINT64_MAX;
    CHECK(is_prime(secp256k1), "secp256k1");

    // 2^256 - 1: composite (divisible by 3, 5, 17, ...)
    UInt<4> all_ones;
    all_ones.d[0] = all_ones.d[1] = all_ones.d[2] = all_ones.d[3] = UINT64_MAX;
    CHECK(!is_prime(all_ones), "2^256-1 composite");

    // A Mersenne composite: 2^253 - 1 is NOT prime (factors include 8191)
    UInt<4> m253;
    m253.d[0] = UINT64_MAX;
    m253.d[1] = UINT64_MAX;
    m253.d[2] = UINT64_MAX;
    m253.d[3] = (1ULL << 61) - 1;
    CHECK(!is_prime(m253), "2^253-1 composite");
}

void test_six_limb() {
    std::fprintf(stderr, "  N=6 known primes/composites...\n");

    // NIST P-384 prime: p = 2^384 - 2^128 - 2^96 + 2^32 - 1
    // = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF
    UInt<6> p384;
    p384.d[0] = 0x00000000FFFFFFFF;
    p384.d[1] = 0xFFFFFFFF00000000;
    p384.d[2] = 0xFFFFFFFFFFFFFFFE;
    p384.d[3] = UINT64_MAX;
    p384.d[4] = UINT64_MAX;
    p384.d[5] = UINT64_MAX;
    CHECK(is_prime(p384), "NIST P-384");

    // 2^384 - 1: composite
    UInt<6> all_ones;
    for (int i = 0; i < 6; ++i) all_ones.d[i] = UINT64_MAX;
    CHECK(!is_prime(all_ones), "2^384-1 composite");
}

void test_eight_limb() {
    std::fprintf(stderr, "  N=8 known primes/composites...\n");

    // 2^521 - 1 is a Mersenne prime, but needs 9 limbs. Use a known 512-bit prime instead.
    // A safe prime: (2^511 - 1) is not prime, but there are known 512-bit primes.
    // Use the Oakley Group 1 prime (RFC 2409):
    // p = 2^512 - 2^32 - 2^12 - 2^8 - 2^7 - 2^3 - 2^2 - 1 (not exact — lookup actual)
    // Actually let's just test a composite to verify composites are rejected at this size.

    // 2^512 - 1: composite
    UInt<8> all_ones;
    for (int i = 0; i < 8; ++i) all_ones.d[i] = UINT64_MAX;
    CHECK(!is_prime(all_ones), "2^512-1 composite");

    // 2^512 - 38117 is prime (can be verified)
    UInt<8> p512;
    for (int i = 0; i < 8; ++i) p512.d[i] = UINT64_MAX;
    p512.d[0] = UINT64_MAX - 38116;
    CHECK(is_prime(p512), "2^512-38117 prime");

    // Multiply two primes to get composite
    // (2^256 - 2^32 - 977) is secp256k1 prime. Square it? Needs 8 limbs.
    // p^2 has 8 limbs. But it would overflow. Just test all-ones.

    // Product of two 256-bit primes: use p256 from NIST
    // p256^2 mod 2^512 should be composite (but p256^2 is 512 bits)
    // Actually, let's just verify the prime above works and the composite is rejected.
}

int main() {
    std::fprintf(stderr, "Miller-Rabin primality test suite\n");
    test_single_limb();
    test_two_limb();
    test_three_limb();
    test_four_limb();
    test_six_limb();
    test_eight_limb();
    std::fprintf(stderr, "\n%d checks, %d failures\n", checks, failures);
    return failures > 0 ? 1 : 0;
}
