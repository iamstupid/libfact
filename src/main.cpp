#include "zfactor/sieve.h"
#include "util/timer.h"
#include <cstdio>

int main() {
    using namespace zfactor;

    printf("=== zfactor: primesieve smoke test ===\n\n");

    constexpr uint64_t limits[] = {
        1'000'000ULL, 10'000'000ULL, 100'000'000ULL, 1'000'000'000ULL
    };

    printf("--- prime_count ---\n");
    for (uint64_t lim : limits) {
        Timer t;
        uint64_t cnt = prime_count(lim);
        double ms = t.elapsed_ms();
        printf("  pi(%-13llu) = %10llu  (%7.2f ms)\n",
               (unsigned long long)lim,
               (unsigned long long)cnt,
               ms);
    }

    printf("\n--- iterator stream ---\n");
    for (uint64_t lim : limits) {
        Timer t;
        PrimeIter it(0, lim);
        uint64_t cnt = 0;
        while (it.next() != 0) ++cnt;
        double ms = t.elapsed_ms();
        printf("  iter (%-13llu) = %10llu  (%7.2f ms)\n",
               (unsigned long long)lim,
               (unsigned long long)cnt,
               ms);
    }

    return 0;
}
