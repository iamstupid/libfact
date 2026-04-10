#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>

#include "zfactor/fixint/modular.h"
#include "hurchalla/montgomery_arithmetic/MontgomeryForm.h"

using namespace zfactor::fixint;

template<typename T>
inline void escape(const T& v) { asm volatile("" : : "g"(&v) : "memory"); }

struct Stats { double ns; };

template<typename Fn>
Stats run_bench(std::size_t items, std::size_t rounds, Fn&& fn) {
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < rounds; ++r) fn();
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return {ns / double(items * rounds)};
}

constexpr int COUNT = 256;
constexpr int ROUNDS = 15000;

void bench_u64() {
    std::mt19937_64 rng(42);
    uint64_t mod = rng() | 1;
    // Ensure mod > 1 and odd
    if (mod < 3) mod = 3;

    // === Hurchalla ===
    hurchalla::MontgomeryForm<uint64_t> hm(mod);
    using HV = decltype(hm)::MontgomeryValue;
    HV ha[COUNT], hb[COUNT], hout[COUNT];
    for (int i = 0; i < COUNT; i++) {
        ha[i] = hm.convertIn(rng() % mod);
        hb[i] = hm.convertIn(rng() % mod);
    }

    auto h_stats = run_bench(COUNT, ROUNDS, [&]() {
        for (int i = 0; i < COUNT; i++) {
            hout[i] = hm.multiply(ha[i], hb[i]);
            escape(hout[i]);
        }
    });

    // === zfactor N=1 ===
    UInt<1> zmod(mod);
    MontCtx<1> ctx; ctx.init(zmod);
    MontScope<1> scope(ctx);
    Mod<1> za[COUNT], zb[COUNT], zout[COUNT];
    rng.seed(42); rng(); // consume mod
    for (int i = 0; i < COUNT; i++) {
        za[i] = Mod<1>::from_uint(UInt<1>(rng() % mod));
        zb[i] = Mod<1>::from_uint(UInt<1>(rng() % mod));
    }

    auto z_stats = run_bench(COUNT, ROUNDS, [&]() {
        for (int i = 0; i < COUNT; i++) {
            zout[i] = za[i] * zb[i];
            escape(zout[i]);
        }
    });

    printf("64-bit montmul:   hurchalla %6.2f ns   zfactor %6.2f ns   ratio %.2fx\n",
           h_stats.ns, z_stats.ns, h_stats.ns / z_stats.ns);

    // Verify both produce same results
    rng.seed(42); rng();
    int err = 0;
    for (int i = 0; i < COUNT; i++) {
        uint64_t a_val = rng() % mod, b_val = rng() % mod;
        unsigned __int128 ref = (unsigned __int128)a_val * b_val % mod;
        uint64_t h_result = hm.convertOut(hm.multiply(hm.convertIn(a_val), hm.convertIn(b_val)));
        UInt<1> z_result = (Mod<1>::from_uint(UInt<1>(a_val)) * Mod<1>::from_uint(UInt<1>(b_val))).to_uint();
        if (h_result != (uint64_t)ref || z_result[0] != (uint64_t)ref) err++;
    }
    printf("  correctness: %d errors\n", err);
}

void bench_u128() {
    std::mt19937_64 rng(99);
    __uint128_t mod128 = ((__uint128_t)rng() << 64) | (rng() | 1);
    // Ensure odd and > 1
    mod128 |= 1;
    if (mod128 < 3) mod128 = 3;

    // === Hurchalla ===
    hurchalla::MontgomeryForm<__uint128_t> hm128(mod128);
    using HV128 = decltype(hm128)::MontgomeryValue;
    HV128 ha128[COUNT], hb128[COUNT], hout128[COUNT];
    for (int i = 0; i < COUNT; i++) {
        __uint128_t av = ((__uint128_t)rng() << 64) | rng();
        __uint128_t bv = ((__uint128_t)rng() << 64) | rng();
        av %= mod128; bv %= mod128;
        ha128[i] = hm128.convertIn(av);
        hb128[i] = hm128.convertIn(bv);
    }

    auto h128_stats = run_bench(COUNT, ROUNDS, [&]() {
        for (int i = 0; i < COUNT; i++) {
            hout128[i] = hm128.multiply(ha128[i], hb128[i]);
            escape(hout128[i]);
        }
    });

    // === zfactor N=2 ===
    UInt<2> zmod2;
    zmod2.d[0] = (uint64_t)mod128;
    zmod2.d[1] = (uint64_t)(mod128 >> 64);
    MontCtx<2> ctx2; ctx2.init(zmod2);
    MontScope<2> scope2(ctx2);
    Mod<2> za2[COUNT], zb2[COUNT], zout2[COUNT];
    rng.seed(99);
    for (int i = 0; i < COUNT; i++) {
        UInt<2> av, bv;
        av.d[0] = rng(); av.d[1] = rng();
        bv.d[0] = rng(); bv.d[1] = rng();
        while (mpn::cmp<2>(av.d, zmod2.d) >= 0) mpn::sub<2>(av.d, av.d, zmod2.d);
        while (mpn::cmp<2>(bv.d, zmod2.d) >= 0) mpn::sub<2>(bv.d, bv.d, zmod2.d);
        za2[i] = Mod<2>::from_uint(av);
        zb2[i] = Mod<2>::from_uint(bv);
    }

    auto z128_stats = run_bench(COUNT, ROUNDS, [&]() {
        for (int i = 0; i < COUNT; i++) {
            zout2[i] = za2[i] * zb2[i];
            escape(zout2[i]);
        }
    });

    printf("128-bit montmul:  hurchalla %6.2f ns   zfactor %6.2f ns   ratio %.2fx\n",
           h128_stats.ns, z128_stats.ns, h128_stats.ns / z128_stats.ns);
}

void bench_pow_u64() {
    uint64_t mod = 18446744073709551557ULL; // large prime < 2^64
    hurchalla::MontgomeryForm<uint64_t> hm(mod);

    UInt<1> zmod(mod);
    MontCtx<1> ctx; ctx.init(zmod);
    MontScope<1> scope(ctx);

    // Fermat test: base^(mod-1) should be 1
    uint64_t base = 2;
    uint64_t exp = mod - 1;

    // Hurchalla pow
    auto h_pow_stats = run_bench(1, 5000, [&]() {
        auto result = hm.pow(hm.convertIn(base), exp);
        escape(result);
    });

    // zfactor pow
    auto z_pow_stats = run_bench(1, 5000, [&]() {
        auto result = pow<1>(Mod<1>::from_uint(UInt<1>(base)), UInt<1>(exp));
        escape(result);
    });

    printf("64-bit modpow:    hurchalla %6.0f ns   zfactor %6.0f ns   ratio %.2fx\n",
           h_pow_stats.ns, z_pow_stats.ns, h_pow_stats.ns / z_pow_stats.ns);
}

int main() {
    printf("=== zfactor vs hurchalla/modular_arithmetic ===\n\n");
    bench_u64();
    bench_u128();
    bench_pow_u64();
    return 0;
}
