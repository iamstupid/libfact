// Benchmark raw Edwards curve operation throughput (latency-bound chain).
// Measures ed_dbl and ed_add in a dependent chain so we see true latency.
#include "zfactor/eecm/curve_setup.h"
#include "zfactor/edwards.h"
#include <cstdio>
#include <cstdint>
#include <chrono>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
void bench_one() {
    // Use a large prime for the modulus.  Known primes near 2^(64*N):
    //   N=1: 2^64 - 59
    //   N=2: 2^128 - 159
    //   N=3: 2^192 - 237
    //   N=4: 2^256 - 189
    //   N=5: 2^320 - 197
    //   N=6: 2^384 - 317
    // These are all prime (verified via Sage/PARI).
    constexpr uint64_t offsets[] = {0, 59, 159, 237, 189, 197, 317};
    UInt<N> mod{};
    for (int i = 0; i < N; ++i) mod.d[i] = ~uint64_t(0);
    mod.d[0] -= (offsets[N] - 1);  // 2^(64N) - offset

    MontCtx<N> ctx;
    ctx.init(mod);
    // Try several k values until we get a valid curve (some k hit a factor
    // of the modulus, which means setup_curve returns factor_found=true).
    CurveSetupResult<N> cs;
    for (uint64_t k = 2; k < 100; ++k) {
        cs = setup_curve<N>(k, ctx);
        if (!cs.factor_found) break;
    }
    if (cs.factor_found) { std::printf("N=%d: setup failed\n", N); return; }

    auto P = cs.P0;
    auto& curve = cs.curve;

    // Warm up
    for (int i = 0; i < 100; ++i) P = ed_dbl<N>(P, ctx);

    using clock = std::chrono::steady_clock;
    constexpr int ITERS = 100000;

    // Benchmark ed_dbl (dependent chain: each dbl feeds the next)
    {
        auto t0 = clock::now();
        for (int i = 0; i < ITERS; ++i) {
            P = ed_dbl<N>(P, ctx);
        }
        auto t1 = clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ITERS;
        std::printf("N=%d  ed_dbl: %7.2f ns   (%7.2f Mop/s)\n", N, ns, 1e3 / ns);
    }

    // Benchmark ed_add (dependent chain: Q = add(Q, P_fixed))
    auto P_fixed = cs.P0;  // a different point
    {
        auto t0 = clock::now();
        for (int i = 0; i < ITERS; ++i) {
            P = ed_add<N>(P, P_fixed, curve, ctx);
        }
        auto t1 = clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ITERS;
        std::printf("N=%d  ed_add: %7.2f ns   (%7.2f Mop/s)\n", N, ns, 1e3 / ns);
    }

    // Prevent dead-code elim
    volatile uint64_t sink = P.X.d[0] ^ P.Y.d[0];
    (void)sink;
}

int main() {
    std::printf("Edwards curve op latency (dependent chain, Zen4)\n\n");
    bench_one<1>();
    bench_one<2>();
    bench_one<3>();
    bench_one<4>();
    bench_one<5>();
    bench_one<6>();
    return 0;
}
