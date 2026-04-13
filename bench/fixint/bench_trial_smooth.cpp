// Benchmark trial division on smooth numbers.
//
// Generates random B-smooth UInt<N> values (products of random small primes),
// then measures trial division throughput for three variants:
//   1. avx2 int15:        existing integer SIMD kernel + libdivide peel
//   2. fp:                FP-fast SIMD kernel + libdivide peel
//   3. fp+divexact:       FP-fast SIMD kernel + modular-inverse divexact peel
//
// This benchmarks the PEEL path (which dominates for smooth numbers) rather
// than the screening path (which dominates for random composites).

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/trial.h"
#include "zfactor/sieve.h"

using zfactor::fixint::UInt;

template<typename T>
inline void escape(const T& v) { __asm__ __volatile__("" : : "g"(&v) : "memory"); }

// Build a B-smooth UInt<N>: product of random primes < B, with ~target_bits total.
template<int N>
UInt<N> make_smooth(const std::vector<uint32_t>& primes, std::mt19937_64& rng,
                    unsigned target_bits) {
    UInt<N> result(1);
    while (result.bit_length() < target_bits) {
        uint32_t p = primes[rng() % primes.size()];
        // Multiply: result *= p
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)result.d[i] * p + carry;
            result.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
        if (carry != 0) {
            // Overflow: start over
            result = UInt<N>(1);
        }
    }
    return result;
}

#if defined(__AVX2__)

static constexpr int COUNT = 256;

template<int N>
struct Batch { UInt<N> n[COUNT]; };

template<int N, typename Table, typename Fn>
double run_bench(const Table& table, const Batch<N>& batch, Fn&& divide) {
    // Warmup
    for (int w = 0; w < 3; ++w) {
        uint64_t sink = 0;
        for (int i = 0; i < COUNT; ++i) {
            UInt<N> c = batch.n[i];
            auto f = divide(c, table);
            sink ^= c.d[0] ^ (uint64_t)f.size();
            escape(c);
        }
        escape(sink);
    }

    int rounds = std::max(1, 2000 / N);
    auto t0 = std::chrono::steady_clock::now();
    uint64_t sink = 0;
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < COUNT; ++i) {
            UInt<N> c = batch.n[i];
            auto f = divide(c, table);
            sink ^= c.d[0] ^ (uint64_t)f.size();
            escape(c);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    escape(sink);
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / (double(COUNT) * rounds);
}

template<int N>
void bench_n(uint32_t B) {
    // Get primes < B for smooth number generation
    auto primes = zfactor::primes_range_as<uint32_t>(2, B);
    if (primes.empty()) return;

    std::mt19937_64 rng(42424242ULL + N + B);
    Batch<N> batch;
    unsigned target_bits = std::min(N * 64 - 8, N * 48);  // ~75% full
    for (int i = 0; i < COUNT; ++i)
        batch.n[i] = make_smooth<N>(primes, rng, target_bits);

    // Count average factors for context
    double avg_factors = 0;
    {
        auto t = zfactor::SimdTrialDivTableFP::build(B);
        for (int i = 0; i < COUNT; ++i) {
            UInt<N> c = batch.n[i];
            auto f = zfactor::trial_divide_simd_fp<N>(c, t);
            avg_factors += f.size();
        }
        avg_factors /= COUNT;
    }

    // Build tables
    auto table_int15 = zfactor::SimdTrialDivTable::build(B);
    auto table_fp    = zfactor::SimdTrialDivTableFP::build(B);
    auto table_fpde  = zfactor::SimdTrialDivTableFPDE::build(B);

    double ns_int15 = 0;
    // Skip int15 — no early exit, too slow on smooth numbers
    // double ns_int15 = run_bench<N>(table_int15, batch,
    //     [](UInt<N>& c, const zfactor::SimdTrialDivTable& t) {
    //         return zfactor::trial_divide_simd<N>(c, t);
    //     });
    double ns_fp = run_bench<N>(table_fp, batch,
        [](UInt<N>& c, const zfactor::SimdTrialDivTableFP& t) {
            return zfactor::trial_divide_simd_fp<N>(c, t);
        });
    double ns_fpde = run_bench<N>(table_fpde, batch,
        [](UInt<N>& c, const zfactor::SimdTrialDivTableFPDE& t) {
            return zfactor::trial_divide_simd_fp_de<N>(c, t);
        });

    printf("  N=%-2d  avg_factors=%.0f  int15=%7.0f  fp=%7.0f (%.1fx)  fp+de=%7.0f (%.1fx) ns/cand\n",
           N, avg_factors, ns_int15, ns_fp, ns_int15/ns_fp, ns_fpde, ns_int15/ns_fpde);
}

int main() {
    printf("=== bench_trial_smooth: trial division on B-smooth numbers ===\n\n");

    for (uint32_t B : {1000u, 10000u, 32768u}) {
        printf("--- B = %u ---\n", B);
        bench_n<1>(B);
        bench_n<2>(B);
        bench_n<4>(B);
        printf("\n");
    }
    return 0;
}

#else
int main() {
    printf("bench_trial_smooth: skipped (no AVX2)\n");
    return 0;
}
#endif
