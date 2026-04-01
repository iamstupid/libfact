#include "util/primesieve.h"
#include "util/timer.h"
#include <cstdio>

int main() {
    using namespace zfactor;

    printf("=== zfactor: PrimeSieve Benchmark ===\n\n");

    constexpr uint32_t limits[] = {
        1'000'000, 10'000'000, 100'000'000, 1'000'000'000
    };

    printf("--- generate (returns vector<uint32_t>) ---\n");
    for (uint32_t lim : limits) {
        Timer t;
        auto primes = PrimeSieve::generate(lim);
        double ms = t.elapsed_ms();
        printf("  generate(%-13u): %8zu primes, %8.2f ms\n",
               lim, primes.size(), ms);
    }

    printf("\n--- for_each (streaming, no allocation) ---\n");
    for (uint32_t lim : limits) {
        uint64_t count = 0;
        Timer t;
        PrimeSieve::for_each(lim, [&count](uint64_t) { ++count; });
        double ms = t.elapsed_ms();
        printf("  for_each(%-13u): %8llu primes, %8.2f ms\n",
               lim, static_cast<unsigned long long>(count), ms);
    }

    return 0;
}
