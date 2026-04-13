// Benchmark: divexact1 (modular inverse) vs divexact_and_mod_next (libdivide)
//
// Measures the raw peel cost: given n divisible by p, how fast can we
// compute n/p and check if p still divides the result?

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/detail/divexact1.h"

#if defined(__AVX2__)
#include "zfactor/trial.h"
#endif

using namespace zfactor::fixint;

template<typename T>
inline void escape(const T& v) { asm volatile("" : : "g"(&v) : "memory"); }

// Build a value = p^e * cofactor, where cofactor is random odd and coprime to p.
template<int N>
UInt<N> make_divisible(uint32_t p, int e, std::mt19937_64& rng) {
    UInt<N> result(1);
    for (int i = 0; i < e; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < N; j++) {
            unsigned __int128 w = (unsigned __int128)result.d[j] * p + carry;
            result.d[j] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }
    // Multiply by a random odd cofactor that fits
    uint64_t cofactor = rng() | 1;
    while (cofactor % p == 0) cofactor += 2;
    {
        uint64_t carry = 0;
        for (int j = 0; j < N; j++) {
            unsigned __int128 w = (unsigned __int128)result.d[j] * cofactor + carry;
            result.d[j] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }
    return result;
}

static constexpr int COUNT = 4096;
static constexpr int TRIALS = 500;

template<int N>
void bench_peel(uint32_t p) {
    std::mt19937_64 rng(42 + N + p);

    // Build test values: each has exactly 1 factor of p
    std::vector<UInt<N>> vals(COUNT);
    for (int i = 0; i < COUNT; i++)
        vals[i] = make_divisible<N>(p, 1, rng);

    uint64_t inv_p = inverse_mod_2_64(p);
    libdivide::divider<uint64_t> div_p(p);
    uint64_t scratch[N];

    // --- Bench divexact1 (peel once, check borrow) ---
    double best_de = 1e18;
    for (int t = 0; t < TRIALS; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t sink = 0;
        for (int i = 0; i < COUNT; i++) {
            UInt<N> c = vals[i];
            uint64_t b = divexact1<N>(scratch, c.d, inv_p, p);
            sink ^= scratch[0] ^ b;
            escape(scratch);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        escape(sink);
        best_de = std::min(best_de, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }

    // --- Bench divexact_and_mod_next (peel once, returns mod) ---
    double best_dam = 1e18;
#if defined(__AVX2__)
    for (int t = 0; t < TRIALS; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t sink = 0;
        for (int i = 0; i < COUNT; i++) {
            UInt<N> c = vals[i];
            uint32_t r = zfactor::detail_trial::divexact_and_mod_next<N>(c.d, p, div_p);
            sink ^= c.d[0] ^ r;
            escape(c);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        escape(sink);
        best_dam = std::min(best_dam, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
#endif

    double de_ns = best_de / COUNT;
    double dam_ns = best_dam / COUNT;
    printf("  N=%-2d  p=%-7u  divexact1=%5.1f ns  divexact_mod=%5.1f ns  speedup=%.2fx\n",
           N, p, de_ns, dam_ns, dam_ns / de_ns);
}

template<int N>
void bench_peel_multi(uint32_t p, int npeels) {
    std::mt19937_64 rng(42 + N + p + npeels);

    std::vector<UInt<N>> vals(COUNT);
    for (int i = 0; i < COUNT; i++)
        vals[i] = make_divisible<N>(p, npeels, rng);

    uint64_t inv_p = inverse_mod_2_64(p);
    libdivide::divider<uint64_t> div_p(p);

    // --- LLVM asm peel loop ---
    double best_peel = 1e18;
    for (int t = 0; t < TRIALS; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t sink = 0;
        for (int i = 0; i < COUNT; i++) {
            UInt<N> c = vals[i];
            uint32_t e = divexact1_ip_peel<N>(c.d, inv_p, p);
            sink ^= c.d[0] ^ e;
            escape(c);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        escape(sink);
        best_peel = std::min(best_peel, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }

    // --- divexact_and_mod_next: peel loop ---
    double best_dam = 1e18;
#if defined(__AVX2__)
    for (int t = 0; t < TRIALS; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t sink = 0;
        for (int i = 0; i < COUNT; i++) {
            UInt<N> c = vals[i];
            uint32_t e = 1;
            uint32_t r = zfactor::detail_trial::divexact_and_mod_next<N>(c.d, p, div_p);
            while (r == 0) {
                r = zfactor::detail_trial::divexact_and_mod_next<N>(c.d, p, div_p);
                ++e;
            }
            sink ^= c.d[0] ^ e;
            escape(c);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        escape(sink);
        best_dam = std::min(best_dam, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
#endif

    double peel_ns = best_peel / COUNT;
    double dam_ns = best_dam / COUNT;
    printf("  N=%-2d  p=%-7u  %d peels:  llvm_peel=%5.1f ns (%4.1f/peel)  libdiv=%5.1f ns (%4.1f/peel)  speedup=%.1fx\n",
           N, p, npeels, peel_ns, peel_ns/npeels, dam_ns, dam_ns/npeels, dam_ns/peel_ns);
}

int main() {
    printf("=== bench_divexact: single peel ===\n");
    for (uint32_t p : {3u, 7u, 127u, 1021u, 32749u}) {
        bench_peel<1>(p);
        bench_peel<2>(p);
        bench_peel<4>(p);
        bench_peel<8>(p);
        printf("\n");
    }

    printf("=== bench_divexact: peel loops ===\n");
    for (uint32_t p : {3u, 127u, 32749u}) {
        for (int npeels : {1, 3, 5, 10}) {
            bench_peel_multi<1>(p, npeels);
            bench_peel_multi<2>(p, npeels);
            bench_peel_multi<4>(p, npeels);
            bench_peel_multi<8>(p, npeels);
            printf("\n");
        }
    }
    return 0;
}
