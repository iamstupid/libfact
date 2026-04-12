#include <cstdio>
#include <cstdlib>
#include <random>
#include "zfactor/fixint/gcd.h"
#include "zfactor/fixint/barrett.h"

using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

static uint64_t ref_gcd(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
}

// ============================================================================
// Single-word tests
// ============================================================================

static void test_gcd_u64() {
    std::printf("  gcd_u64...\n");
    CHECK(gcd_u64(0, 0) == 0, "gcd(0,0)");
    CHECK(gcd_u64(0, 7) == 7, "gcd(0,7)");
    CHECK(gcd_u64(12, 8) == 4, "gcd(12,8)");
    CHECK(gcd_u64(17, 13) == 1, "gcd(17,13)");
    CHECK(gcd_u64(UINT64_MAX, 1) == 1, "gcd(max,1)");

    std::mt19937_64 rng(42);
    for (int i = 0; i < 2000; ++i) {
        uint64_t a = rng(), b = rng();
        CHECK(gcd_u64(a, b) == ref_gcd(a, b), "gcd_u64 random");
    }
}

static void test_xgcd_u64() {
    std::printf("  xgcd_u64...\n");
    std::mt19937_64 rng(123);
    for (int i = 0; i < 2000; ++i) {
        uint64_t a = rng() & 0x7FFFFFFF, b = rng() & 0x7FFFFFFF;
        if (a == 0 && b == 0) continue;
        int64_t x, y;
        uint64_t g = xgcd_u64(&x, &y, a, b);
        CHECK(g == ref_gcd(a, b), "xgcd gcd");
        CHECK(static_cast<int64_t>(a) * x + static_cast<int64_t>(b) * y
              == static_cast<int64_t>(g), "xgcd Bezout");
    }
}

static void test_modinv_u64() {
    std::printf("  modinv_u64...\n");
    CHECK(modinv_u64(3, 7) == 5, "3^-1 mod 7");
    CHECK(modinv_u64(2, 11) == 6, "2^-1 mod 11");
    std::mt19937_64 rng(456);
    for (int i = 0; i < 2000; ++i) {
        uint64_t m = rng() | 3;
        uint64_t a = rng() % m;
        if (a == 0) a = 1;
        if (ref_gcd(a, m) != 1) continue;
        uint64_t inv = modinv_u64(a, m);
        unsigned __int128 prod = (unsigned __int128)a * inv;
        CHECK(static_cast<uint64_t>(prod % m) == 1, "modinv random");
    }
    { // Large modulus near 2^64
        uint64_t m = UINT64_MAX - 58;
        CHECK(static_cast<uint64_t>((unsigned __int128)7 * modinv_u64(7, m) % m) == 1,
              "modinv near 2^64");
    }
}

// ============================================================================
// Multi-limb divrem tests
// ============================================================================

template<int N>
void test_divrem() {
    std::printf("  divrem<UInt<%d>>...\n", N);
    std::mt19937_64 rng(100 + N);

    // a / 1 = a
    for (int i = 0; i < 10; ++i) {
        UInt<N> a; for (int j = 0; j < N; ++j) a.d[j] = rng();
        CHECK(a / UInt<N>(1) == a, "a/1=a");
        CHECK((a % UInt<N>(1)).is_zero(), "a%1=0");
    }

    // q * b + r == a
    for (int i = 0; i < 100; ++i) {
        UInt<N> a, b;
        for (int j = 0; j < N; ++j) { a.d[j] = rng(); b.d[j] = rng(); }
        if (b.is_zero()) continue;
        UInt<N> q = a / b, r = a % b;
        CHECK(mpn::cmp<N>(r.d, b.d) < 0, "r < b");
        auto wide = q * b;
        UInt<N> recon;
        mpn::copy<N>(recon.d, wide.data());
        recon += r;
        CHECK(recon == a, "q*b+r == a");
    }
}

// ============================================================================
// Multi-limb GCD tests
// ============================================================================

template<int N>
void test_gcd() {
    std::printf("  gcd<UInt<%d>>...\n", N);
    std::mt19937_64 rng(789 + N);

    CHECK(gcd<N>(UInt<N>(0), UInt<N>(0)).is_zero(), "gcd(0,0)");
    CHECK(gcd<N>(UInt<N>(12), UInt<N>(8)) == UInt<N>(4), "gcd(12,8)");

    // Commutativity + cross-check with single-word
    for (int i = 0; i < 200; ++i) {
        uint64_t a = rng(), b = rng();
        CHECK(gcd<N>(UInt<N>(a), UInt<N>(b)) == UInt<N>(gcd_u64(a, b)),
              "multi vs single");
    }

    // Scaling
    for (int i = 0; i < 100; ++i) {
        uint64_t k = (rng() % 1000) + 1;
        uint64_t av = (rng() % 10000) + 1, bv = (rng() % 10000) + 1;
        CHECK(gcd<N>(UInt<N>(k * av), UInt<N>(k * bv)) == UInt<N>(k * ref_gcd(av, bv)),
              "scaling");
    }
}

// ============================================================================
// Lehmer GCD tests
// ============================================================================

template<int N>
void test_lehmer() {
    std::printf("  lehmer<UInt<%d>>...\n", N);
    std::mt19937_64 rng(321 + N);

    // Must match binary GCD
    for (int i = 0; i < 100; ++i) {
        UInt<N> a, b;
        for (int j = 0; j < N; ++j) { a.d[j] = rng(); b.d[j] = rng(); }
        CHECK(lehmer_gcd<N>(a, b) == gcd<N>(a, b), "lehmer matches binary");
    }

    // Asymmetric sizes
    for (int i = 0; i < 30; ++i) {
        UInt<N> a, b;
        for (int j = 0; j < N; ++j) a.d[j] = rng();
        b.d[0] = rng() | 1;
        for (int j = 1; j < N; ++j) b.d[j] = 0;
        CHECK(lehmer_gcd<N>(a, b) == gcd<N>(a, b), "lehmer asymmetric");
    }
}

// ============================================================================
// Barrett tests
// ============================================================================

template<int N>
void test_barrett() {
    std::printf("  barrett<UInt<%d>>...\n", N);
    std::mt19937_64 rng(555 + N);

    for (int trial = 0; trial < 3; ++trial) {
        UInt<N> d;
        for (int j = 0; j < N; ++j) d.d[j] = rng();
        d.d[0] |= 1;
        d.d[N - 1] |= (1ULL << 63);

        BarrettCtx<N> ctx;
        ctx.init(d);

        for (int i = 0; i < 20; ++i) {
            UInt<N> a;
            for (int j = 0; j < N; ++j) a.d[j] = rng();
            CHECK(ctx.mod(a) == a % d, "barrett mod");
        }

        for (int i = 0; i < 10; ++i) {
            UInt<N> a;
            for (int j = 0; j < N; ++j) a.d[j] = rng();
            UInt<N> q, r;
            ctx.divrem(q, r, a);
            CHECK(r == a % d, "barrett divrem r");
            CHECK(q == a / d, "barrett divrem q");
        }
    }

    // Note: Barrett requires d to be "normalized" (top bit of d[N-1] set).
    // Non-normalized d values would need a larger inverse buffer.
}

// ============================================================================

int main() {
    std::printf("=== Single-word ===\n");
    test_gcd_u64();
    test_xgcd_u64();
    test_modinv_u64();

    std::printf("\n=== Divrem ===\n");
    test_divrem<1>();
    test_divrem<2>();
    test_divrem<4>();
    test_divrem<8>();

    std::printf("\n=== GCD ===\n");
    test_gcd<1>();
    test_gcd<2>();
    test_gcd<4>();
    test_gcd<8>();

    std::printf("\n=== Lehmer GCD ===\n");
    test_lehmer<1>();
    test_lehmer<2>();
    test_lehmer<4>();
    test_lehmer<8>();

    std::printf("\n=== Barrett ===\n");
    test_barrett<1>();
    test_barrett<2>();
    test_barrett<4>();
    test_barrett<8>();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
