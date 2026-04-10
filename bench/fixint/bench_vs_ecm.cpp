#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cstring>
#include <gmp.h>
#include "zfactor/fixint/modular.h"

using namespace zfactor::fixint;

// GMP-ECM's mulredc — generic loop version from x86_64/mulredc.s
extern "C" mp_limb_t mulredc(mp_limb_t* z, const mp_limb_t* x, const mp_limb_t* y,
                              const mp_limb_t* m, mp_size_t n, mp_limb_t inv_m);

template<typename T>
inline void escape(const T& v) { asm volatile("" : : "g"(&v) : "memory"); }

constexpr int COUNT = 256;
constexpr int ROUNDS = 15000;

template<int N>
void bench_n() {
    std::mt19937_64 rng(42+N);
    mp_limb_t mod[N];
    for (int i = 0; i < N; i++) mod[i] = rng();
    mod[0] |= 1;
    if (N > 1) mod[N-1] |= (1ULL << 63);

    // Compute neg_inv
    mp_limb_t x = 1;
    for (int i = 0; i < 6; i++) x *= 2 - mod[0] * x;
    mp_limb_t neg_inv = -x;

    // Prepare random operands (already reduced mod n)
    mp_limb_t ea[COUNT][N], eb[COUNT][N], eout[COUNT][N];
    for (int i = 0; i < COUNT; i++) {
        for (int j = 0; j < N; j++) { ea[i][j] = rng(); eb[i][j] = rng(); }
        // Crude reduction
        while (mpn_cmp(ea[i], mod, N) >= 0) mpn_sub_n(ea[i], ea[i], mod, N);
        while (mpn_cmp(eb[i], mod, N) >= 0) mpn_sub_n(eb[i], eb[i], mod, N);
    }

    // Bench ECM mulredc
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ROUNDS; r++)
        for (int i = 0; i < COUNT; i++) {
            mulredc(eout[i], ea[i], eb[i], mod, N, neg_inv);
            escape(eout[i]);
        }
    auto t1 = std::chrono::steady_clock::now();
    double ns_ecm = std::chrono::duration<double,std::nano>(t1-t0).count()/(COUNT*ROUNDS);

    // zfactor
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

    printf("N=%-2d  ecm_mulredc %6.2f ns  zfactor %6.2f ns  zf/ecm %.2fx\n",
           N, ns_ecm, ns_zf, ns_zf/ns_ecm);
}

int main() {
    static_assert(sizeof(mp_limb_t) == sizeof(uint64_t));
    printf("=== zfactor vs GMP-ECM mulredc (x86_64 asm) ===\n\n");
    bench_n<1>();
    bench_n<2>();
    bench_n<3>();
    bench_n<4>();
    bench_n<5>();
    bench_n<6>();
    bench_n<8>();
    return 0;
}
