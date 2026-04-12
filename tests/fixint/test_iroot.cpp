#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/iroot.h"

using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

template<int N>
static UInt<N> from_u64(uint64_t v) {
    UInt<N> r{};
    r.d[0] = v;
    return r;
}

template<int N>
static uint64_t to_u64(const UInt<N>& a) { return a.d[0]; }

// ---- isqrt single-limb spot checks ----

static void test_isqrt_basic() {
    std::printf("  isqrt basic...\n");
    auto check = [](uint64_t n, uint64_t expected) {
        auto r = isqrt<1>(from_u64<1>(n));
        CHECK(to_u64(r) == expected, "isqrt single-limb");
    };
    check(0, 0);
    check(1, 1);
    check(2, 1);
    check(3, 1);
    check(4, 2);
    check(15, 3);
    check(16, 4);
    check(99, 9);
    check(100, 10);
    check(101, 10);
    check(123456789ULL, 11111);  // 11111^2 = 123454321
    check((1ULL << 62) - 1, 2147483647ULL);    // sqrt(2^62 - 1) = 2^31 - 1 + change
    check(1ULL << 62, 2147483648ULL);          // sqrt(2^62) = 2^31
    check(uint64_t(-1), 4294967295ULL);        // sqrt(2^64 - 1) = 2^32 - 1
}

// ---- icbrt single-limb spot checks ----

static void test_icbrt_basic() {
    std::printf("  icbrt basic...\n");
    auto check = [](uint64_t n, uint64_t expected) {
        auto r = icbrt<1>(from_u64<1>(n));
        CHECK(to_u64(r) == expected, "icbrt single-limb");
    };
    check(0, 0);
    check(1, 1);
    check(7, 1);
    check(8, 2);
    check(26, 2);
    check(27, 3);
    check(63, 3);
    check(64, 4);
    check(125, 5);
    check(999, 9);
    check(1000, 10);
    check(1001, 10);
    check(1ULL << 30, 1024);   // cbrt(2^30) = 2^10
    check(uint64_t(-1), 2642245ULL);  // cbrt(2^64 - 1) ≈ 2642245.95
}

// ---- random cross-check vs naive C double sqrt ----

static void test_isqrt_random_u64() {
    std::printf("  isqrt random u64 cross-check...\n");
    std::mt19937_64 rng(0xDEADBEEFCAFE);
    for (int i = 0; i < 5000; ++i) {
        uint64_t n = rng();
        uint64_t r = to_u64(isqrt<1>(from_u64<1>(n)));
        // r^2 <= n < (r+1)^2
        unsigned __int128 r2 = (unsigned __int128)r * r;
        unsigned __int128 r1_2 = (unsigned __int128)(r + 1) * (r + 1);
        bool ok = (r2 <= n) && (r1_2 > n);
        CHECK(ok, "isqrt: r^2 <= n < (r+1)^2");
    }
}

static void test_icbrt_random_u64() {
    std::printf("  icbrt random u64 cross-check...\n");
    std::mt19937_64 rng(0xBADC0DECAFE);
    for (int i = 0; i < 5000; ++i) {
        uint64_t n = rng();
        uint64_t r = to_u64(icbrt<1>(from_u64<1>(n)));
        // r^3 <= n < (r+1)^3
        unsigned __int128 r2 = (unsigned __int128)r * r;
        unsigned __int128 r3 = r2 * r;
        unsigned __int128 r1 = r + 1;
        unsigned __int128 r1_3 = r1 * r1 * r1;
        bool ok = (r3 <= n) && (r1_3 > n);
        CHECK(ok, "icbrt: r^3 <= n < (r+1)^3");
    }
}

// ---- multi-limb: build n = r^2 (or r^2 + small) and verify isqrt(n) == r ----

template<int N>
static void test_isqrt_multilimb() {
    std::printf("  isqrt multi-limb N=%d...\n", N);
    std::mt19937_64 rng(0x1234ull + N);

    for (int iter = 0; iter < 200; ++iter) {
        // Build a random N-limb r, compute r^2 (which fits in 2N limbs),
        // then take a random offset in [0, 2*r] so floor(sqrt(r^2 + off)) == r.
        UInt<N> r{};
        for (int j = 0; j < N; ++j) r.d[j] = rng();
        r.d[N - 1] &= (uint64_t(1) << 31) - 1;  // keep r small enough that r^2 fits in N limbs

        // r^2 in N limbs (high N must be zero by construction)
        mpn::limb_t r2[2 * N] = {};
        mpn::sqr<N>(r2, r.d);
        bool fits = true;
        for (int j = N; j < 2 * N; ++j) if (r2[j] != 0) fits = false;
        if (!fits) continue;

        UInt<N> n;
        for (int j = 0; j < N; ++j) n.d[j] = r2[j];

        // Pick offset in [0, 2*r] -- guaranteed n + offset has same floor sqrt
        // (since (r+1)^2 = r^2 + 2r + 1).
        uint64_t off = rng() % ((to_u64(r) > 0) ? (2 * to_u64(r)) : 1);
        // Add off into n's low limb (won't propagate past limb 0 for off < 2^64)
        // Use a proper add to be safe
        UInt<N> off_n{};
        off_n.d[0] = off;
        UInt<N> n_off;
        mpn::add<N>(n_off.d, n.d, off_n.d);

        UInt<N> got = isqrt<N>(n_off);
        bool eq = true;
        for (int j = 0; j < N; ++j)
            if (got.d[j] != r.d[j]) eq = false;
        CHECK(eq, "isqrt multi-limb: got == r");

        // Also check is_square: only true when off == 0
        bool sq = is_square<N>(n);
        CHECK(sq, "is_square: r^2 detected");

        if (off > 0) {
            bool sq_off = is_square<N>(n_off);
            // r^2 + off (with off in (0, 2r+1)) is NOT a perfect square
            CHECK(!sq_off, "is_square: r^2 + small not a square");
        }
    }
}

template<int N>
static void test_icbrt_multilimb() {
    std::printf("  icbrt multi-limb N=%d...\n", N);
    std::mt19937_64 rng(0x5678ull + N);

    for (int iter = 0; iter < 200; ++iter) {
        // Build a random small r, compute r^3 in N limbs, verify icbrt
        UInt<N> r{};
        for (int j = 0; j < N; ++j) r.d[j] = rng();
        // Mask r so that r^3 fits in N limbs (r < 2^(64N/3))
        unsigned bits_r = (64 * N) / 3 - 1;
        unsigned top_limb = bits_r / 64;
        unsigned top_bit  = bits_r % 64;
        for (unsigned j = top_limb + 1; j < unsigned(N); ++j) r.d[j] = 0;
        if (top_limb < unsigned(N))
            r.d[top_limb] &= (uint64_t(1) << top_bit) - 1;

        // r^3 = r * r^2
        mpn::limb_t r2[2 * N] = {};
        mpn::sqr<N>(r2, r.d);
        UInt<N> r2_lo;
        for (int j = 0; j < N; ++j) r2_lo.d[j] = r2[j];

        mpn::limb_t r3[2 * N] = {};
        mpn::mul<N>(r3, r.d, r2_lo.d);
        bool fits = true;
        for (int j = N; j < 2 * N; ++j) if (r3[j] != 0) fits = false;
        if (!fits) continue;

        UInt<N> n;
        for (int j = 0; j < N; ++j) n.d[j] = r3[j];

        UInt<N> got = icbrt<N>(n);
        bool eq = true;
        for (int j = 0; j < N; ++j)
            if (got.d[j] != r.d[j]) eq = false;
        CHECK(eq, "icbrt multi-limb: got == r");

        bool cb = is_cube<N>(n);
        CHECK(cb, "is_cube: r^3 detected");
    }
}

int main() {
    test_isqrt_basic();
    test_icbrt_basic();
    test_isqrt_random_u64();
    test_icbrt_random_u64();
    test_isqrt_multilimb<1>();
    test_isqrt_multilimb<2>();
    test_isqrt_multilimb<3>();
    test_isqrt_multilimb<4>();
    test_isqrt_multilimb<8>();
    test_icbrt_multilimb<1>();
    test_icbrt_multilimb<2>();
    test_icbrt_multilimb<3>();
    test_icbrt_multilimb<4>();
    test_icbrt_multilimb<8>();

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
