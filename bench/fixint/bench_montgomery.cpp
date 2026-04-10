#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>

#include "zfactor/fixint/modular.h"

namespace {

using zfactor::fixint::mpn::limb_t;
using zfactor::fixint::UInt;
using zfactor::fixint::MontCtx;
using zfactor::fixint::MontScope;
using zfactor::fixint::Mod;

template<typename T>
inline void escape(const T& value) {
    __asm__ __volatile__("" : : "g"(&value) : "memory");
}

inline void compiler_barrier() {
    __asm__ __volatile__("" ::: "memory");
}

struct Stats {
    double ns_per_item = 0.0;
    double mops = 0.0;
};

template<typename Fn>
Stats run_bench(std::size_t items, std::size_t rounds, Fn&& fn) {
    auto start = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < rounds; ++r)
        fn();
    auto stop = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    double total_items = double(items) * double(rounds);
    return {ns / total_items, total_items / ns * 1e3};
}

inline void print_stats(const char* op, int n, const Stats& s) {
    std::printf("%-12s N=%-2d  %8.2f ns/op  %8.2f Mop/s\n", op, n, s.ns_per_item, s.mops);
}

template<int N>
UInt<N> make_odd_mod(std::mt19937_64& rng) {
    UInt<N> n;
    for (int i = 0; i < N; ++i) n.d[i] = rng();
    n.d[0] |= 1;
    n.d[N - 1] |= (1ULL << 63);  // ensure full-width
    return n;
}

template<int N>
UInt<N> random_mod(std::mt19937_64& rng, const UInt<N>& mod) {
    UInt<N> r;
    for (int i = 0; i < N; ++i) r.d[i] = rng();
    // Mask to mod's bit length
    unsigned mod_bits = zfactor::fixint::mpn::bit_length<N>(mod.d);
    if (mod_bits < unsigned(N) * 64) {
        unsigned top_limb = mod_bits / 64;
        unsigned top_bit = mod_bits % 64;
        for (unsigned i = top_limb + 1; i < unsigned(N); ++i) r.d[i] = 0;
        if (top_limb < unsigned(N) && top_bit < 64)
            r.d[top_limb] &= (1ULL << (top_bit + 1)) - 1;
    }
    while (zfactor::fixint::mpn::cmp<N>(r.d, mod.d) >= 0)
        zfactor::fixint::mpn::sub<N>(r.d, r.d, mod.d);
    return r;
}

template<int N>
struct MontBatch {
    static constexpr std::size_t count = 256;
    MontCtx<N> ctx;
    std::array<Mod<N>, count> a{};
    std::array<Mod<N>, count> b{};

    MontBatch() {
        std::mt19937_64 rng(0xBADC0FFEu + N * 31u);
        UInt<N> mod = make_odd_mod<N>(rng);
        ctx.init(mod);
        MontScope<N> scope(ctx);
        for (std::size_t i = 0; i < count; ++i) {
            a[i] = Mod<N>::from_uint(random_mod<N>(rng, mod));
            b[i] = Mod<N>::from_uint(random_mod<N>(rng, mod));
        }
    }
};

template<int N>
Stats bench_montmul(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<Mod<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = batch.a[i] * batch.b[i];
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_montsqr(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<Mod<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = batch.a[i].sqr();
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_modadd(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<Mod<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = batch.a[i] + batch.b[i];
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_modsub(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<Mod<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = batch.a[i] - batch.b[i];
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_modpow(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    // Use a fixed 64-bit exponent for consistent work per call
    UInt<N> exp(0xDEADBEEFCAFEBABEULL);
    std::array<Mod<N>, 4> out{};
    return run_bench(4, rounds, [&]() {
        for (std::size_t i = 0; i < 4; ++i) {
            out[i] = zfactor::fixint::pow<N>(batch.a[i], exp);
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_to_mont(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<UInt<N>, MontBatch<N>::count> inputs{};
    std::mt19937_64 rng(999 + N);
    for (std::size_t i = 0; i < MontBatch<N>::count; ++i)
        inputs[i] = random_mod<N>(rng, batch.ctx.mod);

    std::array<Mod<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = Mod<N>::from_uint(inputs[i]);
            escape(out[i]);
        }
    });
}

template<int N>
Stats bench_from_mont(MontBatch<N>& batch, std::size_t rounds) {
    MontScope<N> scope(batch.ctx);
    std::array<UInt<N>, MontBatch<N>::count> out{};
    return run_bench(MontBatch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < MontBatch<N>::count; ++i) {
            out[i] = batch.a[i].to_uint();
            escape(out[i]);
        }
    });
}

template<int N>
void bench_family() {
    constexpr std::size_t light = 40'000;
    constexpr std::size_t heavy = 12'000;
    constexpr std::size_t pow_rounds = 200;

    MontBatch<N> batch;

    auto montmul  = bench_montmul<N>(batch, heavy);
    auto montsqr  = bench_montsqr<N>(batch, heavy);
    auto modadd   = bench_modadd<N>(batch, light);
    auto modsub   = bench_modsub<N>(batch, light);
    auto to_mont  = bench_to_mont<N>(batch, heavy);
    auto from_mont = bench_from_mont<N>(batch, heavy);
    auto modpow   = bench_modpow<N>(batch, pow_rounds);

    print_stats("montmul", N, montmul);
    print_stats("montsqr", N, montsqr);
    print_stats("modadd", N, modadd);
    print_stats("modsub", N, modsub);
    print_stats("to_mont", N, to_mont);
    print_stats("from_mont", N, from_mont);
    print_stats("modpow64", N, modpow);

    // sqr/mul speedup ratio
    if (montmul.ns_per_item > 0 && montsqr.ns_per_item > 0)
        std::printf("  sqr/mul ratio: %.2fx\n", montmul.ns_per_item / montsqr.ns_per_item);
    std::printf("\n");
}

} // namespace

int main() {
    std::puts("== zfactor Montgomery benchmark ==\n");
    bench_family<1>();
    bench_family<2>();
    bench_family<3>();
    bench_family<4>();
    bench_family<6>();
    bench_family<8>();
    return 0;
}
