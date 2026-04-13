// Trial division benchmark — measures scalar mod_small/divexact_small cost
// per candidate against a cached prime table.  Baseline before SIMD work.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/trial.h"

namespace {

using zfactor::fixint::UInt;
using zfactor::TrialDivTable;
using zfactor::trial_divide;
#if defined(__AVX2__)
using zfactor::SimdTrialDivTable;
using zfactor::trial_divide_simd;
using zfactor::SimdTrialDivTableFP;
using zfactor::trial_divide_simd_fp;
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
using zfactor::Simd512TrialDivTable;
using zfactor::trial_divide_simd512;
#endif

template<typename T>
inline void escape(const T& v) { __asm__ __volatile__("" : : "g"(&v) : "memory"); }

constexpr std::size_t COUNT = 256;

template<int N>
struct Batch { std::array<UInt<N>, COUNT> n; };

// Build COUNT random UInt<N> values with top bit set, so every candidate
// actually exercises all N limbs.
template<int N>
Batch<N> make_batch(std::mt19937_64& rng) {
    Batch<N> b;
    for (std::size_t i = 0; i < COUNT; ++i) {
        for (int j = 0; j < N; ++j) b.n[i].d[j] = rng();
        b.n[i].d[N - 1] |= (1ULL << 63);
        b.n[i].d[0] |= 1; // odd, so at least the p=2 test fails fast
    }
    return b;
}

template<int N, typename Table, typename Fn>
double run_trial_bench(const Table& table, const Batch<N>& batch, Fn&& divide) {
    // Warmup
    for (int w = 0; w < 3; ++w) {
        UInt<N> sink{};
        for (std::size_t i = 0; i < COUNT; ++i) {
            UInt<N> c = batch.n[i];
            auto f = divide(c, table);
            sink.d[0] ^= c.d[0] ^ static_cast<uint64_t>(f.size());
            escape(c);
        }
        escape(sink);
    }
    std::size_t rounds = std::max<std::size_t>(1, 2000 / N);
    auto t0 = std::chrono::steady_clock::now();
    UInt<N> sink{};
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t i = 0; i < COUNT; ++i) {
            UInt<N> c = batch.n[i];
            auto f = divide(c, table);
            sink.d[0] ^= c.d[0] ^ static_cast<uint64_t>(f.size());
            escape(c);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    escape(sink);
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (double(COUNT) * double(rounds));
}

template<int N>
void bench_n(const TrialDivTable& scalar_table
#if defined(__AVX2__)
             , const SimdTrialDivTable& simd_table
             , const SimdTrialDivTableFP& simd_fp_table
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
             , const Simd512TrialDivTable& simd512_table
#endif
            ) {
    std::mt19937_64 rng(0xA11CE + N);
    auto batch = make_batch<N>(rng);

    double scalar_ns = run_trial_bench<N>(scalar_table, batch,
        [](UInt<N>& c, const TrialDivTable& t) { return trial_divide<N>(c, t); });
    double scalar_per_prime = scalar_ns / double(scalar_table.size());
    std::printf("  scalar  N=%-2d  %8.0f ns/cand  %6.2f ns/prime   %5.2f Mcand/s\n",
                N, scalar_ns, scalar_per_prime, 1000.0 / scalar_ns);

#if defined(__AVX2__)
    double simd_ns = run_trial_bench<N>(simd_table, batch,
        [](UInt<N>& c, const SimdTrialDivTable& t) { return trial_divide_simd<N>(c, t); });
    double simd_per_prime = simd_ns / double(simd_table.size());
    double simd_speedup = scalar_ns / simd_ns;
    std::printf("  avx2    N=%-2d  %8.0f ns/cand  %6.2f ns/prime   %5.2f Mcand/s   %.2fx\n",
                N, simd_ns, simd_per_prime, 1000.0 / simd_ns, simd_speedup);

    double fp_ns = run_trial_bench<N>(simd_fp_table, batch,
        [](UInt<N>& c, const SimdTrialDivTableFP& t) { return trial_divide_simd_fp<N>(c, t); });
    double fp_per_prime = fp_ns / double(simd_fp_table.size());
    double fp_speedup = scalar_ns / fp_ns;
    std::printf("  fp      N=%-2d  %8.0f ns/cand  %6.2f ns/prime   %5.2f Mcand/s   %.2fx\n",
                N, fp_ns, fp_per_prime, 1000.0 / fp_ns, fp_speedup);
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    double simd512_ns = run_trial_bench<N>(simd512_table, batch,
        [](UInt<N>& c, const Simd512TrialDivTable& t) { return trial_divide_simd512<N>(c, t); });
    double simd512_per_prime = simd512_ns / double(simd512_table.size());
    double simd512_speedup = scalar_ns / simd512_ns;
    std::printf("  avx512  N=%-2d  %8.0f ns/cand  %6.2f ns/prime   %5.2f Mcand/s   %.2fx\n",
                N, simd512_ns, simd512_per_prime, 1000.0 / simd512_ns, simd512_speedup);
#endif
}

} // namespace

int main() {
    const uint32_t bounds[] = {1000, 10000, 32768};

    for (uint32_t B : bounds) {
        auto scalar_table = TrialDivTable::build(B);
#if defined(__AVX2__)
        auto simd_table = SimdTrialDivTable::build(B);
        auto simd_fp_table = SimdTrialDivTableFP::build(B);
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        auto simd512_table = Simd512TrialDivTable::build(B);
        std::printf("=== bound=%u   scalar=%zu  avx2=%zu  avx512=%zu ===\n",
                    B, scalar_table.size(), simd_table.size(), simd512_table.size());
#elif defined(__AVX2__)
        std::printf("=== bound=%u   scalar=%zu  avx2=%zu ===\n",
                    B, scalar_table.size(), simd_table.size());
#else
        std::printf("=== bound=%u   scalar=%zu ===\n", B, scalar_table.size());
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
        bench_n<1>(scalar_table, simd_table, simd512_table);
        bench_n<2>(scalar_table, simd_table, simd512_table);
        bench_n<3>(scalar_table, simd_table, simd512_table);
        bench_n<4>(scalar_table, simd_table, simd512_table);
        bench_n<6>(scalar_table, simd_table, simd512_table);
        bench_n<8>(scalar_table, simd_table, simd512_table);
#elif defined(__AVX2__)
        bench_n<1>(scalar_table, simd_table, simd_fp_table);
        bench_n<2>(scalar_table, simd_table, simd_fp_table);
        bench_n<3>(scalar_table, simd_table, simd_fp_table);
        bench_n<4>(scalar_table, simd_table, simd_fp_table);
        bench_n<6>(scalar_table, simd_table, simd_fp_table);
        bench_n<8>(scalar_table, simd_table, simd_fp_table);
#else
        bench_n<1>(scalar_table);
        bench_n<2>(scalar_table);
        bench_n<3>(scalar_table);
        bench_n<4>(scalar_table);
        bench_n<6>(scalar_table);
        bench_n<8>(scalar_table);
#endif
        std::puts("");
    }
    return 0;
}
