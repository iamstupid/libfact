// test_zint_ntt_dispatch.cpp
//
// Correctness test for the separated-radix NTT dispatcher in
// third_party/zint/ntt/p30x3/engines.hpp.
//
// Strategy: for a sweep of sizes that stresses each engine's smooth set and
// the boundaries between engines, multiply random inputs and verify the
// product via a small-prime modular consistency check (same technique as
// third_party/zint/tests/test_zint_correctness.cpp).
//
// Build:
//   cmake --build <build_dir> --target test_zint_ntt_dispatch
// Run:
//   ./test_zint_ntt_dispatch

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>

#include "zint/zint.hpp"
#include "zint/rng.hpp"
#include "zint/ntt/api.hpp"

using u32 = uint32_t;
using u64 = uint64_t;

static void fail(const char* where, u64 size, u64 prime, u64 lhs, u64 rhs) {
    std::printf("FAIL at %s size=%llu mod=%llu: lhs=%llu rhs=%llu\n",
        where, (unsigned long long)size,
        (unsigned long long)prime, (unsigned long long)lhs, (unsigned long long)rhs);
    std::fflush(stdout);
    std::exit(1);
}

// compute x mod prime where x is a little-endian u32 limb array of length n.
static u64 limbs_mod(const u32* x, size_t n, u64 prime) {
    u64 r = 0;
    for (size_t i = n; i-- > 0;) {
        // r = (r * 2^32 + x[i]) mod prime
        __uint128_t v = ((__uint128_t)r << 32) | x[i];
        r = (u64)(v % prime);
    }
    return r;
}

static void test_size(size_t na, size_t nb, std::mt19937_64& rng) {
    std::vector<u32> a(na), b(nb);
    for (auto& v : a) v = (u32)rng();
    for (auto& v : b) v = (u32)rng();
    // Make sure top limb nonzero so sizes are accurate
    if (na) a[na - 1] |= 1u << 31;
    if (nb) b[nb - 1] |= 1u << 31;

    size_t out_n = na + nb;
    std::vector<u32> out(out_n, 0);

    // Call the zint NTT dispatcher directly.
    bool ok = zint::ntt::big_multiply(out.data(), out_n, a.data(), na, b.data(), nb);
    if (!ok) {
        std::printf("FAIL: big_multiply returned false at size na=%zu nb=%zu\n", na, nb);
        std::fflush(stdout);
        std::exit(1);
    }

    // Verify with small-prime consistency: several random-ish 31-bit primes.
    static const u64 small_primes[] = {
        2147483647ULL,  // 2^31 - 1 (Mersenne prime)
        2147483629ULL,
        2147483587ULL,
        1000000007ULL,
        999999937ULL,
    };
    for (u64 p : small_primes) {
        u64 am = limbs_mod(a.data(), na, p);
        u64 bm = limbs_mod(b.data(), nb, p);
        u64 cm = limbs_mod(out.data(), out_n, p);
        u64 expect = (u64)((__uint128_t)am * bm % p);
        if (cm != expect) {
            fail("test_size", na + nb, p, cm, expect);
        }
    }
}

int main() {
    std::mt19937_64 rng(0xC0FFEE);

    // Sweep sizes stressing all three engines and their boundaries.
    // Each entry is a single multiplication size (na == nb == size).
    const size_t sizes[] = {
        // Small (Engine A only)
        4, 8, 12, 16, 24, 32,
        // Mid-small (A, B, C all compete)
        48, 64, 80, 96, 128, 160, 192, 256, 320, 512,
        // Mid
        1024, 1280, 1536, 2048, 2560, 3072,
        // Approaching baseline max (old P30X3_MAX_NTT = 12.6M)
        100000, 500000, 1000000, 2000000,
        // Beyond old baseline (sizes that previously went to p50x4)
        5000000,   // na+nb = 10M (old baseline = 12.6M; still fits A)
        8000000,   // na+nb = 16M (fits A's 16M container)
        // These cross engine boundaries:
        10000000,  // na+nb = 20M (A needs 32M; C needs 20.9M; C wins)
        12000000,  // na+nb = 24M (A: 32M; B: 25M; C: 20.9M->40M... actually 24M fits in B at 25.2M)
        15000000,  // na+nb = 30M (A: 32M; B: doesn't fit; C: 40M; A wins)
        18000000,  // na+nb = 36M (A: doesn't fit; B: doesn't fit; C: 40M)
        // Near C's cap (don't go above to keep scratch memory reasonable)
        20000000,  // na+nb = 40M, C at its cap
    };

    for (size_t s : sizes) {
        // Do a few random trials at each size
        for (int trial = 0; trial < 3; ++trial) {
            test_size(s, s, rng);
        }
        // Also test asymmetric: na != nb
        test_size(s, s / 2 + 1, rng);
        test_size(s / 2 + 1, s, rng);
    }

    std::printf("All %zu sizes passed.\n", sizeof(sizes) / sizeof(sizes[0]));
    return 0;
}
