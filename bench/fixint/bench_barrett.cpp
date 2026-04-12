#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/barrett.h"
#include "zfactor/fixint/detail/mpn_common.h"

namespace {

using zfactor::fixint::mpn::limb_t;
using zfactor::fixint::UInt;
using zfactor::fixint::BarrettCtx;

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
    std::printf("%-22s N=%-2d  %8.2f ns/op  %8.2f Mop/s\n", op, n, s.ns_per_item, s.mops);
}

template<int N>
UInt<N> make_normalized_d(std::mt19937_64& rng) {
    UInt<N> d;
    for (int i = 0; i < N; ++i) d.d[i] = rng();
    d.d[0] |= 1;
    d.d[N - 1] |= (1ULL << 63);  // top bit set => normalized
    return d;
}

template<int N>
struct Batch {
    static constexpr std::size_t count = 256;
    UInt<N> d{};
    std::array<UInt<N>, count> a{};
    BarrettCtx<N> ctx;
};

template<int N>
Batch<N> make_batch(std::mt19937_64& rng) {
    Batch<N> batch;
    batch.d = make_normalized_d<N>(rng);
    batch.ctx.init(batch.d);
    for (auto& x : batch.a) {
        for (int i = 0; i < N; ++i) x.d[i] = rng();
        x.d[N - 1] = ~limb_t(0);
    }
    return batch;
}

// Wide batch: dividend is 2N limbs, divisor is N limbs.
// This is the typical Barrett use case (reducing a multiplication result).
template<int N>
struct WideBatch {
    static constexpr std::size_t count = 256;
    UInt<N> d{};
    std::array<std::array<limb_t, 2 * N>, count> a{};
    BarrettCtx<N> ctx;
};

template<int N>
WideBatch<N> make_wide_batch(std::mt19937_64& rng) {
    WideBatch<N> batch;
    batch.d = make_normalized_d<N>(rng);
    batch.ctx.init(batch.d);
    for (auto& x : batch.a) {
        for (int i = 0; i < 2 * N; ++i) x[i] = rng();
        // Make sure x has full 2N limbs
        x[2 * N - 1] |= (1ULL << 63);
    }
    return batch;
}

template<int N>
void bench_n() {
    std::mt19937_64 rng(0xC0FFEE + N);
    auto batch = make_batch<N>(rng);
    constexpr std::size_t count = Batch<N>::count;

    // Tune rounds so each benchmark runs ~0.5 sec
    std::size_t rounds = std::max<std::size_t>(1, 200000 / N);

    // ----------------- schoolbook a % d -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                UInt<N> r = batch.a[i] % batch.d;
                sink.d[0] ^= r.d[0];
                escape(r);
            }
        };
        // warmup
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("schoolbook a%d", N, s);
        escape(sink);
    }

    // ----------------- Barrett ctx.mod(a) -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                UInt<N> r = batch.ctx.mod(batch.a[i]);
                sink.d[0] ^= r.d[0];
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("barrett ctx.mod", N, s);
        escape(sink);
    }

    // ----------------- schoolbook a / d (full divrem) -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                UInt<N> q = batch.a[i] / batch.d;
                sink.d[0] ^= q.d[0];
                escape(q);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("schoolbook a/d", N, s);
        escape(sink);
    }

    // ----------------- Barrett ctx.divrem -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                UInt<N> q, r;
                batch.ctx.divrem(q, r, batch.a[i]);
                sink.d[0] ^= q.d[0] ^ r.d[0];
                escape(q);
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("barrett ctx.divrem", N, s);
        escape(sink);
    }

    std::puts("");
}

template<int N>
void bench_wide() {
    std::mt19937_64 rng(0xBEEF + N);
    auto batch = make_wide_batch<N>(rng);
    constexpr std::size_t count = WideBatch<N>::count;
    std::size_t rounds = std::max<std::size_t>(1, 100000 / N);

    // ----------------- schoolbook divrem<2N, N> -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                limb_t q[N + 1];
                limb_t r[N];
                zfactor::fixint::mpn::divrem_wide<2 * N, N>(q, r, batch.a[i].data(), batch.d.d);
                sink.d[0] ^= r[0] ^ q[0];
                escape(q);
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("schoolbook 2N/N", N, s);
        escape(sink);
    }

    // ----------------- Barrett mod_wide -----------------
    {
        UInt<N> sink{};
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                UInt<N> r = batch.ctx.mod_wide(batch.a[i].data());
                sink.d[0] ^= r.d[0];
                escape(r);
            }
        };
        for (int w = 0; w < 5; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("barrett mod_wide", N, s);
        escape(sink);
    }

    std::puts("");
}

void bench_init() {
    // Measure one-time Barrett init cost (Newton inverse computation)
    std::mt19937_64 rng(0xBEEF);
    auto run = [&](auto N_const) {
        constexpr int N = decltype(N_const)::value;
        constexpr std::size_t count = 64;
        std::array<UInt<N>, count> ds;
        for (auto& d : ds) d = make_normalized_d<N>(rng);
        std::size_t rounds = std::max<std::size_t>(1, 50000 / N);
        BarrettCtx<N> ctx;
        auto fn = [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                ctx.init(ds[i]);
                escape(ctx);
            }
        };
        for (int w = 0; w < 3; ++w) fn();
        auto s = run_bench(count, rounds, fn);
        print_stats("barrett init", N, s);
    };
    run(std::integral_constant<int, 1>{});
    run(std::integral_constant<int, 2>{});
    run(std::integral_constant<int, 3>{});
    run(std::integral_constant<int, 4>{});
    run(std::integral_constant<int, 5>{});
    run(std::integral_constant<int, 6>{});
    run(std::integral_constant<int, 7>{});
    run(std::integral_constant<int, 8>{});
    std::puts("");
}

} // namespace

int main() {
    std::puts("=== Narrow: dividend = divisor = N limbs ===\n");
    bench_n<1>();
    bench_n<2>();
    bench_n<3>();
    bench_n<4>();
    bench_n<5>();
    bench_n<6>();
    bench_n<7>();
    bench_n<8>();

    std::puts("=== Wide: dividend = 2N limbs, divisor = N limbs ===\n");
    bench_wide<1>();
    bench_wide<2>();
    bench_wide<3>();
    bench_wide<4>();
    bench_wide<5>();
    bench_wide<6>();
    bench_wide<7>();
    bench_wide<8>();

    std::puts("=== Barrett init cost (one-time) ===");
    bench_init();
    return 0;
}
