// Factor a 160-bit semiprime (two 80-bit prime factors) via ECM.
#include "zfactor/eecm.h"
#include <cstdio>
#include <chrono>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

int main() {
    // n = 634699177039978900322741 * 900519778841796214669567
    //   = 571559162539111861129462442476192304254970723147  (159 bits)
    UInt<3> n{};
    n.d[0] = 0x76726AC9CA837B4BULL;
    n.d[1] = 0xD048AE31446E57DFULL;
    n.d[2] = 0x00000000641D9968ULL;

    std::printf("Factoring 159-bit semiprime (two 80-bit primes)...\n");
    std::printf("n = ");
    for (int i = 2; i >= 0; --i) std::printf("%016llx", (unsigned long long)n.d[i]);
    std::printf("\n\n");

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto r = ecm<3>(n);
    auto t1 = clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (r) {
        std::printf("Factor found: ");
        for (int i = 2; i >= 0; --i) std::printf("%016llx", (unsigned long long)r->d[i]);
        std::printf("\n  = %llu * 2^64 + %llu\n",
                    (unsigned long long)r->d[1], (unsigned long long)r->d[0]);
        std::printf("Time: %.1f ms\n", ms);
    } else {
        std::printf("No factor found after %.1f ms\n", ms);
    }
    return 0;
}
