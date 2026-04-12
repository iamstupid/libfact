// Benchmark: our number-theoretic functions vs FLINT equivalents.
// Tests: gcd, jacobi, modinv at 128/192/256 bit.
#include "zfactor/fixint/gcd.h"
#include "zfactor/jacobi.h"
#include "zfactor/fixint/uint.h"

#include <gmp.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <random>

using namespace zfactor::fixint;
using Clock = std::chrono::steady_clock;

// ── Helpers ──

template<int N>
UInt<N> rand_uint(std::mt19937_64& rng) {
    UInt<N> r{};
    for (int i = 0; i < N; ++i) r.d[i] = rng();
    return r;
}

template<int N>
UInt<N> rand_odd(std::mt19937_64& rng) {
    auto r = rand_uint<N>(rng);
    r.d[0] |= 1;
    if (r.d[N-1] == 0) r.d[N-1] = 1;
    return r;
}

template<int N>
void uint_to_mpz(mpz_t out, const UInt<N>& x) {
    mpz_import(out, N, -1, 8, 0, 0, x.d);
}

// ── GCD benchmark ──

template<int N>
void bench_gcd(int iters) {
    std::mt19937_64 rng(42);
    std::vector<UInt<N>> as(iters), bs(iters);
    for (int i = 0; i < iters; ++i) { as[i] = rand_uint<N>(rng); bs[i] = rand_odd<N>(rng); }

    // Ours (lehmer_gcd)
    volatile uint64_t sink = 0;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto g = lehmer_gcd<N>(as[i], bs[i]);
        sink += g.d[0];
    }
    auto t1 = Clock::now();
    double ours_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;

    mpz_t ma, mb, mg;
    mpz_init(ma); mpz_init(mb); mpz_init(mg);
    auto t3 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        mpz_import(ma, N, -1, 8, 0, 0, as[i].d);
        mpz_import(mb, N, -1, 8, 0, 0, bs[i].d);
        mpz_gcd(mg, ma, mb);
        sink += mpz_get_ui(mg);
    }
    auto t4 = Clock::now();
    double gmp_ns = std::chrono::duration<double, std::nano>(t4 - t3).count() / iters;

    // GMP gcd without conversion overhead
    // Pre-convert all
    std::vector<mpz_t> mas(iters), mbs(iters);
    for (int i = 0; i < iters; ++i) {
        mpz_init(mas[i]); mpz_init(mbs[i]);
        mpz_import(mas[i], N, -1, 8, 0, 0, as[i].d);
        mpz_import(mbs[i], N, -1, 8, 0, 0, bs[i].d);
    }
    auto t5 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        mpz_gcd(mg, mas[i], mbs[i]);
        sink += mpz_get_ui(mg);
    }
    auto t6 = Clock::now();
    double gmp_pure_ns = std::chrono::duration<double, std::nano>(t6 - t5).count() / iters;
    for (int i = 0; i < iters; ++i) { mpz_clear(mas[i]); mpz_clear(mbs[i]); }

    std::printf("  gcd  N=%d: ours=%5.0fns  gmp(+conv)=%5.0fns  gmp(pure)=%5.0fns  ratio=%.2fx\n",
                N, ours_ns, gmp_ns, gmp_pure_ns, gmp_pure_ns / ours_ns);

    mpz_clear(ma); mpz_clear(mb); mpz_clear(mg);
    (void)sink;
}

// ── Jacobi benchmark ──

template<int N>
void bench_jacobi(int iters) {
    std::mt19937_64 rng(123);
    std::vector<UInt<N>> as(iters), ns(iters);
    for (int i = 0; i < iters; ++i) { as[i] = rand_uint<N>(rng); ns[i] = rand_odd<N>(rng); }

    volatile int sink = 0;

    // Ours
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        sink += zfactor::jacobi<N>(as[i], ns[i]);
    auto t1 = Clock::now();
    double ours_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;

    // GMP (mpz_jacobi)
    std::vector<mpz_t> mas(iters), mns(iters);
    for (int i = 0; i < iters; ++i) {
        mpz_init(mas[i]); mpz_init(mns[i]);
        mpz_import(mas[i], N, -1, 8, 0, 0, as[i].d);
        mpz_import(mns[i], N, -1, 8, 0, 0, ns[i].d);
    }
    auto t2 = Clock::now();
    for (int i = 0; i < iters; ++i)
        sink += mpz_jacobi(mas[i], mns[i]);
    auto t3 = Clock::now();
    double gmp_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;
    for (int i = 0; i < iters; ++i) { mpz_clear(mas[i]); mpz_clear(mns[i]); }

    std::printf("  jac  N=%d: ours=%5.0fns  gmp(pure)=%5.0fns  ratio=%.2fx\n",
                N, ours_ns, gmp_ns, gmp_ns / ours_ns);
    (void)sink;
}

// ── Modinv benchmark ──

template<int N>
void bench_modinv(int iters) {
    std::mt19937_64 rng(456);
    std::vector<UInt<N>> as(iters), ms(iters);
    for (int i = 0; i < iters; ++i) {
        as[i] = rand_uint<N>(rng);
        ms[i] = rand_odd<N>(rng);
        // Ensure a < m and gcd(a,m) likely 1
        if (mpn::cmp<N>(as[i].d, ms[i].d) >= 0) std::swap(as[i], ms[i]);
        ms[i].d[0] |= 1;
        if (ms[i].d[N-1] == 0) ms[i].d[N-1] = 1;
    }

    volatile uint64_t sink = 0;

    // Ours
    UInt<N> inv;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        modinv<N>(&inv, as[i], ms[i]);
        sink += inv.d[0];
    }
    auto t1 = Clock::now();
    double ours_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;

    // GMP (mpz_invert)
    std::vector<mpz_t> mas(iters), mms(iters);
    mpz_t minv;
    mpz_init(minv);
    for (int i = 0; i < iters; ++i) {
        mpz_init(mas[i]); mpz_init(mms[i]);
        mpz_import(mas[i], N, -1, 8, 0, 0, as[i].d);
        mpz_import(mms[i], N, -1, 8, 0, 0, ms[i].d);
    }
    auto t2 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        mpz_invert(minv, mas[i], mms[i]);
        sink += mpz_get_ui(minv);
    }
    auto t3 = Clock::now();
    double gmp_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;
    for (int i = 0; i < iters; ++i) { mpz_clear(mas[i]); mpz_clear(mms[i]); }
    mpz_clear(minv);

    std::printf("  inv  N=%d: ours=%5.0fns  gmp(pure)=%5.0fns  ratio=%.2fx\n",
                N, ours_ns, gmp_ns, gmp_ns / ours_ns);
    (void)sink;
}

int main() {
    std::printf("=== Number theory: ours vs GMP ===\n\n");

    int iters = 10000;
    bench_gcd<2>(iters);
    bench_gcd<3>(iters);
    bench_gcd<4>(iters);
    std::printf("\n");

    bench_jacobi<2>(iters);
    bench_jacobi<3>(iters);
    bench_jacobi<4>(iters);
    std::printf("\n");

    bench_modinv<2>(iters);
    bench_modinv<3>(iters);
    bench_modinv<4>(iters);
    return 0;
}
