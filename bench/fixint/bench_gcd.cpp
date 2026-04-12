// GCD benchmark: binary GCD, Lehmer GCD, and GMP's mpn_gcd.
//
// Build:
//   clang++ -std=c++20 -O3 -march=native -DZFACTOR_HAS_BMI2=1 -DZFACTOR_HAS_ADX=1 \
//     -I include -I third_party -I /usr/include/x86_64-linux-gnu \
//     bench/fixint/bench_gcd.cpp -L build_wsl -lzfactor_fixint_backend -lgmp \
//     -o bench_gcd

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <gmp.h>

#include "zfactor/fixint/gcd.h"

namespace {

using zfactor::fixint::mpn::limb_t;
using zfactor::fixint::UInt;
using zfactor::fixint::gcd;
using zfactor::fixint::lehmer_gcd;

template<typename T>
inline void escape(const T& v) { __asm__ __volatile__("" : : "g"(&v) : "memory"); }

constexpr std::size_t COUNT = 256;

template<typename Fn>
double run_bench(std::size_t items, std::size_t rounds, Fn&& fn) {
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < rounds; ++r) fn();
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (double(items) * double(rounds));
}

template<int N>
struct Batch {
    std::array<UInt<N>, COUNT> a;
    std::array<UInt<N>, COUNT> b;
};

template<int N>
Batch<N> make_random(std::mt19937_64& rng) {
    Batch<N> batch;
    for (std::size_t i = 0; i < COUNT; ++i) {
        for (int j = 0; j < N; ++j) {
            batch.a[i].d[j] = rng();
            batch.b[i].d[j] = rng();
        }
        // Make both odd to avoid trivial cases dominating
        batch.a[i].d[0] |= 1;
        batch.b[i].d[0] |= 1;
        // Force top bit for full-width values
        batch.a[i].d[N - 1] |= (1ULL << 63);
        batch.b[i].d[N - 1] |= (1ULL << 63);
    }
    return batch;
}

template<int N>
void bench_n() {
    std::mt19937_64 rng(0xDEADBEEF + N);
    auto batch = make_random<N>(rng);
    // GCD is slower per call than basic ops; fewer rounds
    std::size_t rounds = std::max<std::size_t>(1, 20000 / N);

    // ----- Binary GCD (Stein) -----
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < COUNT; ++i) {
                UInt<N> g = gcd<N>(batch.a[i], batch.b[i]);
                sink.d[0] ^= g.d[0];
                escape(g);
            }
        };
        for (int w = 0; w < 3; ++w) fn();
        double ns = run_bench(COUNT, rounds, fn);
        std::printf("binary gcd      N=%-2d  %8.2f ns/op  %7.2f Mop/s\n",
                    N, ns, 1000.0 / ns);
        escape(sink);
    }

    // ----- Lehmer GCD -----
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < COUNT; ++i) {
                UInt<N> g = lehmer_gcd<N>(batch.a[i], batch.b[i]);
                sink.d[0] ^= g.d[0];
                escape(g);
            }
        };
        for (int w = 0; w < 3; ++w) fn();
        double ns = run_bench(COUNT, rounds, fn);
        std::printf("lehmer gcd      N=%-2d  %8.2f ns/op  %7.2f Mop/s\n",
                    N, ns, 1000.0 / ns);
        escape(sink);
    }

    // ----- GMP mpn_gcd -----
    // mpn_gcd requires odd v, and au, av must satisfy specific constraints.
    // The simplest interface: mpn_gcd(rp, ap, an, bp, bn) where
    // bp[bn-1] != 0, bp is odd, ap >= bp, an >= bn.
    {
        mp_limb_t sink = 0;
        // Pre-arrange so a >= b for each pair
        std::array<std::array<mp_limb_t, N>, COUNT> ga, gb;
        for (std::size_t i = 0; i < COUNT; ++i) {
            for (int j = 0; j < N; ++j) {
                ga[i][j] = batch.a[i].d[j];
                gb[i][j] = batch.b[i].d[j];
            }
            // Swap if needed so a >= b
            if (mpn_cmp(ga[i].data(), gb[i].data(), N) < 0) {
                std::swap(ga[i], gb[i]);
            }
            // Both already have low bit set (odd) from make_random
        }
        // mpn_gcd is destructive on inputs! Make per-call copies.
        auto fn = [&]() {
            for (std::size_t i = 0; i < COUNT; ++i) {
                mp_limb_t a_copy[N], b_copy[N], result[N];
                for (int j = 0; j < N; ++j) {
                    a_copy[j] = ga[i][j];
                    b_copy[j] = gb[i][j];
                }
                mp_size_t rn = mpn_gcd(result, a_copy, N, b_copy, N);
                sink ^= result[0] ^ static_cast<mp_limb_t>(rn);
                escape(result);
            }
        };
        for (int w = 0; w < 3; ++w) fn();
        double ns = run_bench(COUNT, rounds, fn);
        std::printf("GMP mpn_gcd     N=%-2d  %8.2f ns/op  %7.2f Mop/s\n",
                    N, ns, 1000.0 / ns);
        escape(sink);
    }

    std::puts("");
}

} // namespace

int main() {
    static_assert(sizeof(mp_limb_t) == sizeof(limb_t));
    std::puts("=== GCD: binary vs Lehmer vs GMP ===\n");
    std::puts("(both inputs are full-width odd N-limb)\n");
    bench_n<1>();
    bench_n<2>();
    bench_n<3>();
    bench_n<4>();
    bench_n<5>();
    bench_n<6>();
    bench_n<7>();
    bench_n<8>();
    return 0;
}
