#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>

#include "zfactor/fixint/uint.h"

// Scalar reference
template<int N>
inline uint32_t scalar_mod(const zfactor::fixint::UInt<N>& a, uint32_t p) {
    uint64_t r = 0;
    for (int i = N - 1; i >= 0; i--) {
        unsigned __int128 w = (unsigned __int128)r << 64 | a.d[i];
        r = (uint64_t)(w % p);
    }
    return (uint32_t)r;
}

#if defined(__AVX2__)
#include "zfactor/fixint/simd_mod.h"
#include "zfactor/trial.h"
using namespace zfactor::fixint;
using namespace zfactor;

template<int N>
void bench_one(const char* label, uint32_t prime_lo, uint32_t prime_hi, int num_primes,
               bool bench_fp = false) {
    std::mt19937_64 rng(12345 + N + prime_lo);

    std::vector<uint32_t> primes(num_primes);
    for (int i = 0; i < num_primes; ++i) {
        primes[i] = prime_lo + (rng() % (prime_hi - prime_lo));
        primes[i] |= 1;
        if (primes[i] < prime_lo) primes[i] = prime_lo | 1;
    }

    UInt<N> val{};
    for (int i = 0; i < N; ++i)
        val.d[i] = rng();

    std::vector<uint32_t> results(num_primes);
    constexpr int TRIALS = 200;

    // --- Scalar ---
    double best_scalar = 1e18;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_primes; ++i)
            results[i] = scalar_mod<N>(val, primes[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        best_scalar = std::min(best_scalar, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }

    // --- SIMD hybrid (general) ---
    auto table = simd_mod::SimdModTable::build(primes.data(), num_primes);
    simd_mod::batch_mod<N>(val, table, results.data());  // warm
    double best_simd = 1e18;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        simd_mod::batch_mod<N>(val, table, results.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        best_simd = std::min(best_simd, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }

    // --- SIMD all-FP (p < 2^25 only) ---
    double best_fp = 1e18;
    if (bench_fp) {
        auto table_fp = simd_mod::SimdModTableFP::build(primes.data(), num_primes);
        simd_mod::batch_mod_fp<N>(val, table_fp, results.data());  // warm
        for (int t = 0; t < TRIALS; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            simd_mod::batch_mod_fp<N>(val, table_fp, results.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            best_fp = std::min(best_fp, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
    }

    // --- Existing 15-bit integer kernel (only if all primes < 2^15) ---
    // Build a SimdTrialDivTable from the SAME primes for apples-to-apples.
    double best_int15 = 1e18;
    bool can_int15 = (prime_hi <= (1u << 15));
    if (can_int15) {
        // Build table from our primes (pad to multiple of 8)
        uint32_t ngroups_15 = (num_primes + 7) / 8;
        std::vector<uint32_t> primes_padded(ngroups_15 * 8, 3);
        std::vector<uint32_t> magics(ngroups_15 * 8);
        std::vector<uint32_t> mores(ngroups_15 * 8);
        std::vector<uint32_t> R_arr(ngroups_15 * 8);
        for (int i = 0; i < num_primes; ++i) primes_padded[i] = primes[i];
        for (size_t i = 0; i < primes_padded.size(); ++i) {
            uint32_t p = primes_padded[i];
            auto bf = libdivide::libdivide_u32_branchfree_gen(p);
            magics[i] = bf.magic;
            mores[i] = bf.more;
            R_arr[i] = (uint32_t)(((uint64_t)1 << 32) % p);
        }
        volatile uint32_t sink = 0;
        alignas(32) uint32_t lanes[8];
        for (int t = 0; t < TRIALS; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t gi = 0; gi < ngroups_15; ++gi) {
                __m256i p_vec = _mm256_loadu_si256((const __m256i*)&primes_padded[gi * 8]);
                __m256i m_vec = _mm256_loadu_si256((const __m256i*)&magics[gi * 8]);
                __m256i s_vec = _mm256_loadu_si256((const __m256i*)&mores[gi * 8]);
                __m256i R_vec = _mm256_loadu_si256((const __m256i*)&R_arr[gi * 8]);
                __m256i r = detail_trial::simd_group_mod<N>(val.d, p_vec, m_vec, s_vec, R_vec);
                _mm256_store_si256((__m256i*)lanes, r);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            sink = lanes[0];
            best_int15 = std::min(best_int15, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
    }

    double sc = best_scalar / num_primes;
    double si = best_simd / num_primes;

    // --- FP-fast (p < 2^15): full-limb Horner, N iterations ---
    double best_fp_fast = 1e18;
    bool can_fp_fast = (prime_hi <= (1u << 15));
    if (can_fp_fast) {
        auto table_fpf = simd_mod::SimdModTableFPFast::build(primes.data(), num_primes);
        simd_mod::batch_mod_fp_fast<N>(val, table_fpf, results.data());
        for (int t = 0; t < TRIALS; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            simd_mod::batch_mod_fp_fast<N>(val, table_fpf, results.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            best_fp_fast = std::min(best_fp_fast, std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
    }

    printf("  N=%d  %-22s  scalar=%5.1f  hybrid=%5.1f (%.1fx)", N, label, sc, si, best_scalar/best_simd);
    if (bench_fp) printf("  fp=%5.1f (%.1fx)", best_fp/num_primes, best_scalar/best_fp);
    if (can_fp_fast) printf("  fp_fast=%5.1f (%.1fx)", best_fp_fast/num_primes, best_scalar/best_fp_fast);
    if (can_int15) printf("  int15=%5.1f (%.1fx)", best_int15/num_primes, best_scalar/best_int15);
    printf(" ns/p\n");
}

template<int N>
void bench_N() {
    printf("--- N = %d (%d bits) ---\n", N, N * 64);
    bench_one<N>("[2^10, 2^15)",  1024,      32768,     4096, true);
    bench_one<N>("[2^15, 2^20)",  32768,     1048576,   4096, true);
    bench_one<N>("[2^20, 2^25)",  1048576,   (1u<<25),  4096, true);
    bench_one<N>("[2^25, 2^31)",  (1u<<25),  0x7FFFFFFFU, 4096);
    printf("\n");
}

int main() {
    printf("=== bench_simd_mod: hybrid FP kernel vs scalar ===\n\n");
    bench_N<1>();
    bench_N<2>();
    bench_N<4>();
    bench_N<8>();
    return 0;
}

#else
int main() {
    printf("bench_simd_mod: skipped (no AVX2)\n");
    return 0;
}
#endif
