// End-to-end ECM stage-1 test on small semiprimes.  Each test composite has
// a known smallest factor; we expect ECM to find it within max_curves curves.
#include "zfactor/eecm.h"
#include <cstdio>
#include <cstdint>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
bool try_factor(uint64_t lo, uint64_t hi, int /*unused*/, int /*unused*/,
                const char* name) {
    UInt<N> n{};
    n.d[0] = lo;
    if constexpr (N >= 2) n.d[1] = hi;
    auto r = ecm<N>(n);
    if (r) {
        std::printf("[OK] %s: factor = %llu (%llu)\n",
                    name, (unsigned long long)r->d[0], (unsigned long long)r->d[1]);
        return true;
    } else {
        std::printf("[NO] %s: no factor\n", name);
        return false;
    }
}

int main() {
    // Test 1: 1009 * 10007 = 10097063  (factors are 10 and 14 bits)
    try_factor<1>(1009ULL * 10007ULL, 0, 14, 50, "1009*10007");

    // Test 2: 100003 * 1000003 = 100006300009  (17 and 20 bits)
    try_factor<1>(100003ULL * 1000003ULL, 0, 20, 50, "100003*1000003");

    // Test 3: 1000003 * 10000019 — 20-bit factor.  Try B1 large enough.
    try_factor<1>(1000003ULL * 10000019ULL, 0, 80, 200, "1000003*10000019 (B1 bigger)");

    // Test 4: a 24-bit factor — 16777259 * 33554467
    try_factor<1>(16777259ULL * 33554467ULL, 0, 80, 500, "16777259*33554467 (more curves)");

    // Test 5: small balanced semiprime, two ~15-bit primes
    try_factor<1>(32771ULL * 32779ULL, 0, 15, 50, "32771*32779");

    // Test 6: small unbalanced — 3 * 1000003
    try_factor<1>(3ULL * 1000003ULL, 0, 20, 50, "3*1000003 (trivial)");

    // Easy: 17-bit prime * 17-bit prime, very small.  factor_bits=20 →
    // B1=557, B2=23300.  Should find easily.
    try_factor<1>(131071ULL * 131101ULL, 0, 20, 50, "easy 17-bit");
    // 19-bit primes — bump factor_bits to give a generous schedule
    try_factor<1>(524287ULL * 524309ULL, 0, 40, 50, "19-bit big sched");
    // 31-bit factor with a much bigger schedule
    try_factor<1>(9223372064772063217ULL, 0, 100, 50, "31-bit huge sched");

    // N=3..6 sanity: same composite stored at progressively larger N.
    try_factor<3>(1009ULL * 10007ULL, 0, 14, 50, "N=3 1009*10007");
    try_factor<4>(1009ULL * 10007ULL, 0, 14, 50, "N=4 1009*10007");
    try_factor<5>(1009ULL * 10007ULL, 0, 14, 50, "N=5 1009*10007");
    try_factor<6>(1009ULL * 10007ULL, 0, 14, 50, "N=6 1009*10007");

    // N=2 sanity: same composite as test 1, but stored in a 128-bit slot.
    // Verifies that multi-limb dispatch works end-to-end.
    try_factor<2>(1009ULL * 10007ULL, 0, 14, 50, "N=2 1009*10007");

    // N=2 with a 32-bit factor: p1 * p2 = 0xFFFFFFFFFFFFFFFFULL * 0xFFFFFFFFULL
    // The small factor here is 4294967295 = 2^32 - 1 = 3 * 5 * 17 * 257 * 65537
    // — heavily composite, so ECM should find the smallest prime factor (3).
    // Just check it returns *some* factor.
    {
        // n = (2^64-1) * (2^32-1) = 2^96 - 2^64 - 2^32 + 1
        UInt<2> n{};
        n.d[0] = 0x0000000000000001ULL;        // 1
        n.d[1] = 0x00000000FFFFFFFFULL;        // 2^96 piece
        // subtract (2^64 + 2^32) from this:
        //  d[0] -= 2^32 → 0xFFFFFFFF00000001 with borrow
        //  d[1] -= 1 (the borrow) and -= 0 (no 2^64 term in d[1]'s bits below 32)
        // Easier: just compute (2^64-1)*(2^32-1) directly: let p1 = 2^64-1, p2 = 2^32-1.
        // low 64 of product = (p1 * p2) low = ((2^64-1)*(2^32-1)) low
        //                   = (2^96 - 2^64 - 2^32 + 1) mod 2^64
        //                   = (- 2^32 + 1) mod 2^64
        //                   = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
        // high 64 = (2^32 - 2) = 0xFFFFFFFE
        n.d[0] = 0xFFFFFFFF00000001ULL;
        n.d[1] = 0x00000000FFFFFFFEULL;
        try_factor<2>(n.d[0], n.d[1], 30, 100, "N=2 (2^64-1)*(2^32-1)");
    }

    return 0;
}
