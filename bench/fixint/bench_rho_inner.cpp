// Microbenchmark for the Pollard rho Brent inner loop — measures raw
// ns/iteration of `y = f(y); q *= (x - y)` across N = 1..8.
//
// This is the metric we want to optimize.  Isolating it from the Brent
// cycle detector / GCD overhead gives a clean steady-state number that
// directly reflects Mont kernel quality.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/modular.h"

using namespace zfactor;
using namespace zfactor::fixint;

template<int N>
static UInt<N> make_modulus() {
    // An odd composite with random-looking limbs — we don't care about
    // factorability, only about the Mont kernel behaviour.
    std::mt19937_64 rng(0xABCDEF01u + N);
    UInt<N> n{};
    for (int i = 0; i < N; ++i) n.d[i] = rng();
    n.d[0] |= 1ull;                            // odd
    n.d[N - 1] |= (1ull << 63);                // full width
    return n;
}

// Version A: uses Mod<N> wrapper which does TLS ctx lookups per op.
template<int N>
[[gnu::noinline]]
static uint64_t run_inner_mod(const MontCtx<N>& mctx, uint64_t iters,
                              uint64_t x0_seed, uint64_t c_seed) {
    MontScope<N> scope(mctx);

    Mod<N> c{};  c.v.d[0] = c_seed;
    Mod<N> y{};  y.v.d[0] = x0_seed;
    Mod<N> x = y;
    Mod<N> q = Mod<N>::one();

    for (uint64_t i = 0; i < iters; ++i) {
        y = y.sqr() + c;
        q *= (x - y);
    }

    uint64_t sink = 0;
    for (int i = 0; i < N; ++i) sink ^= y.v.d[i] ^ q.v.d[i];
    return sink;
}

// Version B: raw Mont primitives, ctx passed as a local reference so the
// compiler can hoist the mod/pos_inv loads out of the loop.
template<int N>
[[gnu::noinline]]
static uint64_t run_inner_raw(const MontCtx<N>& mctx, uint64_t iters,
                              uint64_t x0_seed, uint64_t c_seed) {
    mpn::limb_t c[N]  = {}; c[0]  = c_seed;
    mpn::limb_t y[N]  = {}; y[0]  = x0_seed;
    mpn::limb_t x[N]  = {}; x[0]  = x0_seed;
    mpn::limb_t q[N]  = {}; q[0]  = mctx.r_mod.d[0];
    for (int i = 1; i < N; ++i) q[i] = mctx.r_mod.d[i];

    mpn::limb_t tmp[N];
    mpn::limb_t diff[N];

    for (uint64_t i = 0; i < iters; ++i) {
        // y = y^2 + c
        montsqr<N>(tmp, y, mctx);
        {
            uint8_t cy = mpn::add<N>(y, tmp, c);
            if (cy) mpn::sub<N>(y, y, mctx.mod.d);
            else    mpn::csub<N>(y, y, mctx.mod.d);
        }
        // diff = x - y
        {
            mpn::limb_t bw = mpn::sub<N>(diff, x, y);
            mpn::cadd<N>(diff, diff, mctx.mod.d, bw);
        }
        // q = q * diff
        montmul<N>(tmp, q, diff, mctx);
        for (int k = 0; k < N; ++k) q[k] = tmp[k];
    }

    uint64_t sink = 0;
    for (int i = 0; i < N; ++i) sink ^= y[i] ^ q[i];
    return sink;
}

// "Bound" Mod<N>: a self-contained value type that carries a const
// MontCtx<N>* alongside its limbs.  No TLS.  Demonstrates the cost of
// the abstraction when the ctx is bound at the value level instead of
// stashed in a thread-local stack.
template<int N>
struct ModBound {
    UInt<N> v{};
    const MontCtx<N>* mctx = nullptr;

    static ModBound from_raw(const MontCtx<N>& m, uint64_t low) {
        ModBound r;
        r.mctx = &m;
        r.v.d[0] = low;
        for (int i = 1; i < N; ++i) r.v.d[i] = 0;
        return r;
    }
    static ModBound mont_one(const MontCtx<N>& m) {
        ModBound r;
        r.mctx = &m;
        r.v = m.r_mod;
        return r;
    }

    [[gnu::always_inline]] ModBound sqr() const {
        ModBound r; r.mctx = mctx;
        montsqr<N>(r.v.d, v.d, *mctx);
        return r;
    }

    [[gnu::always_inline]] friend ModBound operator+(const ModBound& a, const ModBound& b) {
        ModBound r; r.mctx = a.mctx;
        uint8_t cy = mpn::add<N>(r.v.d, a.v.d, b.v.d);
        if (cy) mpn::sub<N>(r.v.d, r.v.d, a.mctx->mod.d);
        else    mpn::csub<N>(r.v.d, r.v.d, a.mctx->mod.d);
        return r;
    }

    [[gnu::always_inline]] friend ModBound operator-(const ModBound& a, const ModBound& b) {
        ModBound r; r.mctx = a.mctx;
        mpn::limb_t bw = mpn::sub<N>(r.v.d, a.v.d, b.v.d);
        mpn::cadd<N>(r.v.d, r.v.d, a.mctx->mod.d, bw);
        return r;
    }

    [[gnu::always_inline]] friend ModBound operator*(const ModBound& a, const ModBound& b) {
        ModBound r; r.mctx = a.mctx;
        montmul<N>(r.v.d, a.v.d, b.v.d, *a.mctx);
        return r;
    }

    [[gnu::always_inline]] ModBound& operator*=(const ModBound& other) {
        montmul<N>(v.d, v.d, other.v.d, *mctx);
        return *this;
    }
};

// Version D: ModBound with the same parallel/reordered loop body as `par`.
// If the abstraction is truly lossless, this should match `par` exactly.
template<int N>
[[gnu::noinline]]
static uint64_t run_inner_bound(const MontCtx<N>& mctx, uint64_t iters,
                                uint64_t x0_seed, uint64_t c_seed) {
    auto c = ModBound<N>::from_raw(mctx, c_seed);
    auto y = ModBound<N>::from_raw(mctx, x0_seed);
    auto x = y;
    auto q = ModBound<N>::mont_one(mctx);

    for (uint64_t i = 0; i < iters; ++i) {
        ModBound<N> diff = x - y;
        y = y.sqr() + c;
        q *= diff;
    }

    uint64_t sink = 0;
    for (int i = 0; i < N; ++i) sink ^= y.v.d[i] ^ q.v.d[i];
    return sink;
}

// Version E: MontOps method-API on MontCtx<N>.  Stateless adapter, no
// per-value pointer, no TLS.  Should match `par`/`bnd` exactly if the
// abstraction is truly free.
template<int N>
[[gnu::noinline]]
static uint64_t run_inner_ops(const MontCtx<N>& mctx, uint64_t iters,
                              uint64_t x0_seed, uint64_t c_seed) {
    auto m = mont<N>(mctx);

    UInt<N> c{}; c.d[0] = c_seed;
    UInt<N> y{}; y.d[0] = x0_seed;
    UInt<N> x = y;
    UInt<N> q = m.one();

    for (uint64_t i = 0; i < iters; ++i) {
        UInt<N> diff = m.sub(x, y);
        y = m.add(m.sqr(y), c);
        q = m.mul(q, diff);
    }

    uint64_t sink = 0;
    for (int i = 0; i < N; ++i) sink ^= y.d[i] ^ q.d[i];
    return sink;
}

// Version C: reorder so the y-update (sqr chain) and the q-update (mul
// chain) operate on independent slices of the OoO graph.  Both read the
// "current" y at the top of the iteration; neither depends on the other
// within the iteration, only on the prior iteration's outputs.  This
// should roughly halve the critical-path latency — throughput goes from
// lat(A) + lat(B) per iter down to max(lat(A), lat(B)).
//
// Algorithmic note: this variant uses (x - y_k) instead of (x - y_{k+1})
// compared to the textbook formulation.  Still a valid quadratic iteration
// with the same cycle/collision statistics — it's just shifted by one step.
template<int N>
[[gnu::noinline]]
static uint64_t run_inner_parallel(const MontCtx<N>& mctx, uint64_t iters,
                                   uint64_t x0_seed, uint64_t c_seed) {
    mpn::limb_t c[N]  = {}; c[0]  = c_seed;
    mpn::limb_t y[N]  = {}; y[0]  = x0_seed;
    mpn::limb_t x[N]  = {}; x[0]  = x0_seed;
    mpn::limb_t q[N]  = {}; q[0]  = mctx.r_mod.d[0];
    for (int i = 1; i < N; ++i) q[i] = mctx.r_mod.d[i];

    mpn::limb_t y_new[N];
    mpn::limb_t diff[N];
    mpn::limb_t q_new[N];

    for (uint64_t i = 0; i < iters; ++i) {
        // diff = x - y_k                         (B chain)
        {
            mpn::limb_t bw = mpn::sub<N>(diff, x, y);
            mpn::cadd<N>(diff, diff, mctx.mod.d, bw);
        }
        // y_new = y_k^2 + c                      (A chain, independent of B)
        montsqr<N>(y_new, y, mctx);
        {
            uint8_t cy = mpn::add<N>(y_new, y_new, c);
            if (cy) mpn::sub<N>(y_new, y_new, mctx.mod.d);
            else    mpn::csub<N>(y_new, y_new, mctx.mod.d);
        }
        // q_new = q_k * diff                     (B chain cont.)
        montmul<N>(q_new, q, diff, mctx);

        // commit
        for (int k = 0; k < N; ++k) { y[k] = y_new[k]; q[k] = q_new[k]; }
    }

    uint64_t sink = 0;
    for (int i = 0; i < N; ++i) sink ^= y[i] ^ q[i];
    return sink;
}

template<int N>
static void bench_one() {
    auto n = make_modulus<N>();
    MontCtx<N> mctx;
    mctx.init(n);

    // Warmup
    (void)run_inner_mod<N>(mctx, 1 << 16, 2, 1);
    (void)run_inner_raw<N>(mctx, 1 << 16, 2, 1);
    (void)run_inner_parallel<N>(mctx, 1 << 16, 2, 1);
    (void)run_inner_bound<N>(mctx, 1 << 16, 2, 1);
    (void)run_inner_ops<N>(mctx, 1 << 16, 2, 1);

    constexpr uint64_t ITERS = 20'000'000;

    auto t0a = std::chrono::steady_clock::now();
    uint64_t sink_a = run_inner_mod<N>(mctx, ITERS, 2, 1);
    auto t1a = std::chrono::steady_clock::now();
    double ns_a = std::chrono::duration<double, std::nano>(t1a - t0a).count() / ITERS;

    auto t0b = std::chrono::steady_clock::now();
    uint64_t sink_b = run_inner_raw<N>(mctx, ITERS, 2, 1);
    auto t1b = std::chrono::steady_clock::now();
    double ns_b = std::chrono::duration<double, std::nano>(t1b - t0b).count() / ITERS;

    auto t0c = std::chrono::steady_clock::now();
    uint64_t sink_c = run_inner_parallel<N>(mctx, ITERS, 2, 1);
    auto t1c = std::chrono::steady_clock::now();
    double ns_c = std::chrono::duration<double, std::nano>(t1c - t0c).count() / ITERS;

    auto t0d = std::chrono::steady_clock::now();
    uint64_t sink_d = run_inner_bound<N>(mctx, ITERS, 2, 1);
    auto t1d = std::chrono::steady_clock::now();
    double ns_d = std::chrono::duration<double, std::nano>(t1d - t0d).count() / ITERS;

    auto t0e = std::chrono::steady_clock::now();
    uint64_t sink_e = run_inner_ops<N>(mctx, ITERS, 2, 1);
    auto t1e = std::chrono::steady_clock::now();
    double ns_e = std::chrono::duration<double, std::nano>(t1e - t0e).count() / ITERS;

    std::printf("  N=%-2d  mod=%6.2f  raw=%6.2f  par=%6.2f  bnd=%6.2f  ops=%6.2f   par/ops=%.2fx\n",
                N, ns_a, ns_b, ns_c, ns_d, ns_e, ns_c / ns_e);
    (void)sink_a; (void)sink_b; (void)sink_c; (void)sink_d; (void)sink_e;
}

int main() {
    std::printf("Pollard rho inner loop: y = y^2 + c; q *= (x - y)   (Montgomery)\n");
    std::printf("  'mod' = Mod<N> wrapper (TLS ctx per op)\n");
    std::printf("  'raw' = direct montsqr/montmul, same serial A;B shape\n");
    std::printf("  'par' = reordered raw: diff=x-y_k independent of y update\n");
    std::printf("  'bnd' = ModBound<N> (ctx pointer in value), reordered loop\n");
    std::printf("  'ops' = MontOps method API (mont(ctx).sqr/mul/add/sub), reordered\n");
    bench_one<1>();
    bench_one<2>();
    bench_one<3>();
    bench_one<4>();
    bench_one<6>();
    bench_one<8>();
    return 0;
}
