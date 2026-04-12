// Compare zfactor divrem vs GMP mpn_tdiv_qr.
//
// Build directly:
//   clang++ -std=c++20 -O3 -march=native -DZFACTOR_HAS_BMI2=1 -DZFACTOR_HAS_ADX=1 \
//     -I include -I third_party -I /usr/include/x86_64-linux-gnu \
//     bench/fixint/bench_divrem_gmp.cpp -L build_wsl -lzfactor_fixint_backend -lgmp \
//     -o bench_divrem_gmp

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <gmp.h>

#include "zfactor/fixint/uint.h"

namespace {

using zfactor::fixint::mpn::limb_t;
using zfactor::fixint::UInt;

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
    UInt<N> d;
    std::array<std::array<limb_t, 2 * N>, COUNT> a;
};

template<int N>
Batch<N> make_batch(std::mt19937_64& rng) {
    Batch<N> b;
    for (int j = 0; j < N; ++j) b.d.d[j] = rng();
    b.d.d[0] |= 1;
    b.d.d[N - 1] |= (1ULL << 63);
    for (auto& x : b.a) {
        for (int j = 0; j < 2 * N; ++j) x[j] = rng();
        x[2 * N - 1] |= (1ULL << 63);
    }
    return b;
}

template<int N>
void bench_n() {
    std::mt19937_64 rng(0xC0FFEE + N);
    auto batch = make_batch<N>(rng);
    std::size_t rounds = std::max<std::size_t>(1, 200000 / N);

    // ----- zfactor divrem -----
    {
        limb_t sink = 0;
        auto fn = [&]() {
            for (std::size_t i = 0; i < COUNT; ++i) {
                limb_t q[N + 1] = {};
                limb_t r[N] = {};
                zfactor::fixint::mpn::divrem_wide<2 * N, N>(q, r, batch.a[i].data(), batch.d.d);
                sink ^= q[0] ^ r[0];
                escape(q);
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        double ns = run_bench(COUNT, rounds, fn);
        std::printf("zfactor 2N/N      N=%-2d  %7.2f ns/op  %7.2f Mop/s\n",
                    N, ns, 1000.0 / ns);
        escape(sink);
    }

    // ----- GMP mpn_tdiv_qr -----
    {
        mp_limb_t sink = 0;
        auto fn = [&]() {
            for (std::size_t i = 0; i < COUNT; ++i) {
                mp_limb_t q[N + 1] = {};
                mp_limb_t r[N] = {};
                // mpn_tdiv_qr(qp, rp, qxn, np, nn, dp, dn)
                // qxn must be 0; qp gets nn-dn+1 limbs
                mpn_tdiv_qr(q, r, 0,
                            reinterpret_cast<const mp_limb_t*>(batch.a[i].data()), 2 * N,
                            reinterpret_cast<const mp_limb_t*>(batch.d.d), N);
                sink ^= q[0] ^ r[0];
                escape(q);
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        double ns = run_bench(COUNT, rounds, fn);
        std::printf("GMP mpn_tdiv_qr   N=%-2d  %7.2f ns/op  %7.2f Mop/s\n",
                    N, ns, 1000.0 / ns);
        escape(sink);
    }
    std::puts("");
}

} // namespace

int main() {
    static_assert(sizeof(mp_limb_t) == sizeof(limb_t));
    std::puts("=== divrem 2N/N: zfactor vs GMP ===\n");
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
