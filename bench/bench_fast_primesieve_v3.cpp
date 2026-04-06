#include <cstdio>
#include <vector>
#include "util/fast_primesieve_v3.h"
#include "util/timer.h"

int main() {
    using namespace zfactor;
    using namespace zfactor::sieve_v3;

    config c1{};
    c1.num_threads = 1;
    config c4{};
    c4.num_threads = 4;

    struct CountCase { const char* name; uint64_t lo; uint64_t hi; } counts[] = {
        {"pi(1e8)", 2ULL, 100000000ULL},
        {"[1e12,1e12+1e6]", 1000000000000ULL, 1000001000001ULL},
        {"[1e12,1e12+1e8]", 1000000000000ULL, 1000100000001ULL},
    };

    struct GenCase { const char* name; uint64_t lo; uint64_t hi; } gens[] = {
        {"[0,1e8)", 0ULL, 100000000ULL},
        {"[1e12,1e12+1e6]", 1000000000000ULL, 1000001000001ULL},
    };

    std::puts("== count ==");
    for (auto cc : counts) {
        Timer t1;
        auto a = count_primes(cc.lo, cc.hi, c1);
        double ms1 = t1.elapsed_ms();
        Timer t4;
        auto b = count_primes(cc.lo, cc.hi, c4);
        double ms4 = t4.elapsed_ms();
        std::printf("%s: count=%llu t1=%.2fms t4=%.2fms\n",
            cc.name,
            (unsigned long long)a,
            ms1,
            ms4);
        if (a != b) return 1;
    }

    std::puts("== generate ==");
    for (auto gc : gens) {
        Timer t1;
        auto a = prime_sieve::range(gc.lo, gc.hi, c1);
        double ms1 = t1.elapsed_ms();
        Timer t4;
        auto b = prime_sieve::range(gc.lo, gc.hi, c4);
        double ms4 = t4.elapsed_ms();
        std::printf("%s: size=%zu t1=%.2fms t4=%.2fms\n",
            gc.name,
            a.size(),
            ms1,
            ms4);
        if (a != b) return 2;
    }

    return 0;
}
