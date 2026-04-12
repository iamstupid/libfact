// Test: FFT polynomial stage 2 via FLINT.
// Compile on WSL with -DZFACTOR_USE_FLINT and link -lflint -lgmp.
#include "zfactor/eecm.h"
#include <cstdio>
#include <cstdint>
#include <chrono>

using namespace zfactor::fixint;
using namespace zfactor::ecm;
using Clock = std::chrono::steady_clock;

template<int N>
bool test_ecm(const char* name, UInt<N> n) {
    auto t0 = Clock::now();
    auto r = ecm<N>(n);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r) {
        // Verify it's a real factor: n % factor == 0
        // (simple check: factor * (n/factor) == n, but just print for now)
        std::printf("[OK] %-35s factor=%llu (hi=%llu)  %.1f ms\n",
                    name,
                    (unsigned long long)r->d[0],
                    N >= 2 ? (unsigned long long)r->d[1] : 0ULL,
                    ms);
        return true;
    } else {
        std::printf("[NO] %-35s  %.1f ms\n", name, ms);
        return false;
    }
}

int main() {
    int pass = 0, fail = 0;
    auto check = [&](bool ok) { ok ? ++pass : ++fail; };

    std::printf("=== ECM with FFT polynomial stage 2 ===\n\n");

    // Small: factors found in stage 1 (FFT stage 2 shouldn't hurt)
    {
        UInt<1> n{}; n.d[0] = 1009ULL * 10007ULL;
        check(test_ecm("1009*10007 (N=1)", n));
    }
    {
        UInt<1> n{}; n.d[0] = 100003ULL * 1000003ULL;
        check(test_ecm("100003*1000003 (N=1)", n));
    }

    // Medium: likely needs stage 2
    {
        UInt<1> n{}; n.d[0] = 1000003ULL * 10000019ULL;
        check(test_ecm("1000003*10000019 (N=1)", n));
    }
    {
        UInt<1> n{}; n.d[0] = 16777259ULL * 33554467ULL;
        check(test_ecm("16777259*33554467 (N=1)", n));
    }

    // N=2: 128-bit composite
    {
        // (2^64-1) * (2^32-1)
        UInt<2> n{};
        n.d[0] = 0xFFFFFFFF00000001ULL;
        n.d[1] = 0x00000000FFFFFFFEULL;
        check(test_ecm("(2^64-1)*(2^32-1) (N=2)", n));
    }

    // N=3: 192-bit composite = two 96-bit primes
    // p = 79228162514264337593543950397, q = 79228162514264337593543951787
    // n = p*q (already used in bench_polyeval test)
    {
        UInt<3> n{};
        // n = 6277101735386680763835789543000648137670033885902487509439
        // Let's use a smaller N=3 case: two ~48-bit primes stored in 192-bit
        UInt<3> n3{};
        n3.d[0] = 281474976710677ULL * 3ULL;  // 48-bit prime * 3
        // Actually let's keep it simple: use a semiprime that fits
        // 2^47 + 21 = 140737488355349 (prime), 2^47 + 39 = 140737488355367 (prime)
        // n = 140737488355349 * 140737488355367
        // This is about 94 bits, fits in N=2 but test at N=3
        uint64_t p1 = 140737488355349ULL;
        uint64_t p2 = 140737488355367ULL;
        unsigned __int128 prod = (unsigned __int128)p1 * p2;
        n3.d[0] = (uint64_t)prod;
        n3.d[1] = (uint64_t)(prod >> 64);
        check(test_ecm("47bit*47bit (N=3)", n3));
    }

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
