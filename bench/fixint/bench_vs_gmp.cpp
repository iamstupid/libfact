#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <gmp.h>
#include "zfactor/fixint/modular.h"

using namespace zfactor::fixint;

template<typename T>
inline void escape(const T& v) { asm volatile("" : : "g"(&v) : "memory"); }

constexpr int COUNT = 256;
constexpr int ROUNDS = 15000;

// GMP Montgomery context
template<int N>
struct GmpMontCtx {
    mp_limb_t mod[N];
    mp_limb_t inv;       // -mod^{-1} mod 2^64
    mp_limb_t r2[2*N];   // R^2 mod n (2N limbs for mpn_redc_1 input)

    void init(const mp_limb_t* n) {
        for (int i = 0; i < N; i++) mod[i] = n[i];
        // Compute -n^{-1} mod 2^64
        mp_limb_t x = 1;
        for (int i = 0; i < 6; i++) x *= 2 - n[0] * x;
        inv = -x;
        // Compute R^2 mod n via mpn
        mp_limb_t t[2*N+1] = {};
        t[N] = 1;  // t = R = 2^(64N)
        // t mod n via division
        mp_limb_t q[N+1];
        mpn_tdiv_qr(q, t, 0, t, N+1, mod, N);
        // t[0..N-1] = R mod n. Square it for R^2.
        mp_limb_t rmod[N];
        for (int i = 0; i < N; i++) rmod[i] = t[i];
        // R^2 = rmod * rmod mod n
        mp_limb_t rr[2*N];
        mpn_mul_n(rr, rmod, rmod, N);
        mp_limb_t q2[N+1];
        mp_limb_t rem[N];
        mpn_tdiv_qr(q2, rem, 0, rr, 2*N, mod, N);
        // Store as 2N limbs (padded) for redc input
        for (int i = 0; i < N; i++) r2[i] = rem[i];
        for (int i = N; i < 2*N; i++) r2[i] = 0;
    }
};

// GMP montmul via mpn_mul_n + mpn_redc_1
template<int N>
inline void gmp_montmul(mp_limb_t* r, const mp_limb_t* a, const mp_limb_t* b,
                        const GmpMontCtx<N>& ctx) {
    mp_limb_t t[2*N];
    mpn_mul_n(t, a, b, N);
    // REDC using mpn_redc_1 (single-limb inverse, iterative)
    // mpn_redc_1 does: for i in 0..N-1: m=t[i]*inv, t += m*mod<<(i*64); result = t[N..2N-1]
    // It's not directly available in all GMP versions. Use manual CIOS.
    mp_limb_t buf[2*N+1];
    for (int i = 0; i < 2*N; i++) buf[i] = t[i];
    buf[2*N] = 0;
    for (int i = 0; i < N; i++) {
        mp_limb_t m = buf[i] * ctx.inv;
        buf[i+N] += mpn_addmul_1(buf+i, ctx.mod, N, m);
        // carry propagation
        if (buf[i+N] == 0 && m != 0) buf[i+N+1]++;
    }
    // Conditional subtract
    if (buf[2*N] || mpn_cmp(buf+N, ctx.mod, N) >= 0)
        mpn_sub_n(r, buf+N, ctx.mod, N);
    else
        for (int i = 0; i < N; i++) r[i] = buf[N+i];
}

// GMP to_mont
template<int N>
inline void gmp_to_mont(mp_limb_t* r, const mp_limb_t* a, const GmpMontCtx<N>& ctx) {
    // montmul(a, R^2)
    mp_limb_t t[2*N];
    mpn_mul_n(t, a, ctx.r2, N);
    mp_limb_t buf[2*N+1];
    for (int i = 0; i < 2*N; i++) buf[i] = t[i];
    buf[2*N] = 0;
    for (int i = 0; i < N; i++) {
        mp_limb_t m = buf[i] * ctx.inv;
        buf[i+N] += mpn_addmul_1(buf+i, ctx.mod, N, m);
        if (buf[i+N] == 0 && m != 0) buf[i+N+1]++;
    }
    if (buf[2*N] || mpn_cmp(buf+N, ctx.mod, N) >= 0)
        mpn_sub_n(r, buf+N, ctx.mod, N);
    else
        for (int i = 0; i < N; i++) r[i] = buf[N+i];
}

template<int N>
void bench_n() {
    std::mt19937_64 rng(42+N);
    mp_limb_t mod[N];
    for (int i = 0; i < N; i++) mod[i] = rng();
    mod[0] |= 1;
    if (N > 1) mod[N-1] |= (1ULL << 63);

    // GMP setup
    GmpMontCtx<N> gctx;
    gctx.init(mod);
    mp_limb_t ga[COUNT][N], gb[COUNT][N], gout[COUNT][N];
    for (int i = 0; i < COUNT; i++) {
        for (int j = 0; j < N; j++) { ga[i][j] = rng(); gb[i][j] = rng(); }
        // reduce mod
        mp_limb_t q[N+1], rem[N];
        mp_limb_t tmp[N+1] = {};
        for (int j = 0; j < N; j++) tmp[j] = ga[i][j];
        if (mpn_cmp(tmp, mod, N) >= 0) { mpn_tdiv_qr(q, rem, 0, tmp, N, mod, N); for(int j=0;j<N;j++) ga[i][j]=rem[j]; }
        for (int j = 0; j < N; j++) tmp[j] = gb[i][j];
        if (mpn_cmp(tmp, mod, N) >= 0) { mpn_tdiv_qr(q, rem, 0, tmp, N, mod, N); for(int j=0;j<N;j++) gb[i][j]=rem[j]; }
        gmp_to_mont<N>(ga[i], ga[i], gctx);
        gmp_to_mont<N>(gb[i], gb[i], gctx);
    }

    // Bench GMP montmul
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; r++)
        for (int i = 0; i < COUNT; i++) {
            gmp_montmul<N>(gout[i], ga[i], gb[i], gctx);
            escape(gout[i]);
        }
    auto t1 = std::chrono::steady_clock::now();
    double ns_gmp = std::chrono::duration<double,std::nano>(t1-t0).count()/(COUNT*ROUNDS);

    // Bench GMP mpn_mul_n alone (for reference)
    mp_limb_t wide[2*N];
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; r++)
        for (int i = 0; i < COUNT; i++) {
            mpn_mul_n(wide, ga[i], gb[i], N);
            escape(wide);
        }
    t1 = std::chrono::steady_clock::now();
    double ns_gmp_mul = std::chrono::duration<double,std::nano>(t1-t0).count()/(COUNT*ROUNDS);

    // zfactor setup
    UInt<N> zmod;
    for (int i = 0; i < N; i++) zmod.d[i] = mod[i];
    MontCtx<N> zctx; zctx.init(zmod);
    MontScope<N> scope(zctx);
    Mod<N> za[COUNT], zb[COUNT], zout[COUNT];
    rng.seed(42+N);
    for (int i = 0; i < N; i++) rng(); // consume mod
    for (int i = 0; i < COUNT; i++) {
        UInt<N> av, bv;
        for (int j = 0; j < N; j++) { av.d[j] = rng(); bv.d[j] = rng(); }
        while (mpn::cmp<N>(av.d, zmod.d) >= 0) mpn::sub<N>(av.d, av.d, zmod.d);
        while (mpn::cmp<N>(bv.d, zmod.d) >= 0) mpn::sub<N>(bv.d, bv.d, zmod.d);
        za[i] = Mod<N>::from_uint(av);
        zb[i] = Mod<N>::from_uint(bv);
    }

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; r++)
        for (int i = 0; i < COUNT; i++) {
            zout[i] = za[i] * zb[i];
            escape(zout[i]);
        }
    t1 = std::chrono::steady_clock::now();
    double ns_zf = std::chrono::duration<double,std::nano>(t1-t0).count()/(COUNT*ROUNDS);

    printf("N=%-2d  gmp_mul %6.2f  gmp_montmul %6.2f  zfactor %6.2f  zf/gmp %.2fx\n",
           N, ns_gmp_mul, ns_gmp, ns_zf, ns_zf/ns_gmp);
}

int main() {
    static_assert(sizeof(mp_limb_t) == sizeof(uint64_t));
    printf("=== zfactor vs GMP montmul ===\n\n");
    printf("%-4s  %-12s %-14s %-12s %-10s\n", "", "gmp mul", "gmp montmul", "zfactor", "ratio");
    bench_n<1>();
    bench_n<2>();
    bench_n<3>();
    bench_n<4>();
    bench_n<5>();
    bench_n<6>();
    bench_n<8>();
    return 0;
}
