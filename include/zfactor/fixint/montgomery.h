#pragma once

#include "uint.h"

namespace zfactor::fixint {

template<int N>
struct MontCtx {
    UInt<N> mod;
    UInt<N> r2_mod;
    UInt<N> r_mod;
    uint64_t pos_inv;  // n^{-1} mod 2^64
    uint64_t neg_inv;  // -n^{-1} mod 2^64

    void init(const UInt<N>& n) {
        mod = n;
        uint64_t x = 1;
        for (int i = 0; i < 6; ++i) x *= 2 - n[0] * x;
        pos_inv = x;
        neg_inv = -x;

        UInt<N> t(1);
        for (int i = 0; i < 64 * N; ++i) {
            uint8_t c = mpn::lshift1<N>(t.d, t.d);
            if (c) mpn::sub<N>(t.d, t.d, mod.d);
            else   mpn::csub<N>(t.d, t.d, mod.d);
        }
        r_mod = t;
        for (int i = 0; i < 64 * N; ++i) {
            uint8_t c = mpn::lshift1<N>(t.d, t.d);
            if (c) mpn::sub<N>(t.d, t.d, mod.d);
            else   mpn::csub<N>(t.d, t.d, mod.d);
        }
        r2_mod = t;
    }
};

// ============================================================================
// Positive-inverse REDC montmul for N=1.
// Pure __int128 — both GCC and Clang generate ~10 instructions.
// ============================================================================

__attribute__((always_inline))
inline void montmul_1(mpn::limb_t* __restrict__ r,
                      const mpn::limb_t* __restrict__ a,
                      const mpn::limb_t* __restrict__ b,
                      mpn::limb_t mod, mpn::limb_t pos_inv) {
    unsigned __int128 prod = (unsigned __int128)a[0] * b[0];
    uint64_t lo = (uint64_t)prod, hi = (uint64_t)(prod >> 64);
    uint64_t m = lo * pos_inv;
    uint64_t mn_hi = (uint64_t)(((unsigned __int128)m * mod) >> 64);
    uint64_t result = hi - mn_hi;
    if (hi < mn_hi) result += mod;
    r[0] = result;
}

// ============================================================================
// Positive-inverse REDC montmul for N=2.
// Pure __int128 — Clang generates optimal mulx chain.
// ============================================================================

__attribute__((always_inline))
inline void montmul_2(mpn::limb_t* __restrict__ r,
                      const mpn::limb_t* __restrict__ a,
                      const mpn::limb_t* __restrict__ b,
                      const mpn::limb_t* __restrict__ mod,
                      mpn::limb_t pos_inv) {
    using u128 = unsigned __int128;
    uint64_t a0 = a[0], a1 = a[1];
    uint64_t b0 = b[0], b1 = b[1];
    uint64_t n0 = mod[0], n1 = mod[1];

    // fullmul a*b → 4 limbs
    u128 p00 = (u128)a0 * b0;
    u128 p01 = (u128)a0 * b1;
    u128 p10 = (u128)a1 * b0;
    u128 p11 = (u128)a1 * b1;
    u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    u128 prod_lo = ((u128)(uint64_t)mid << 64) | (uint64_t)p00;
    u128 prod_hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);

    // Iterative m computation (positive inverse, subtraction-based)
    uint64_t plo0 = (uint64_t)prod_lo;
    uint64_t plo1 = (uint64_t)(prod_lo >> 64);

    // m0 = plo0 * pos_inv; adjust plo1 by subtracting hi(m0*n0) + lo(m0*n1)
    uint64_t m0 = plo0 * pos_inv;
    u128 mn0 = (u128)m0 * n0;
    u128 mn1_0 = (u128)m0 * n1;
    uint64_t adj = (uint64_t)(mn0 >> 64) + (uint64_t)mn1_0;
    uint64_t new_plo1 = plo1 - adj;

    // m1 = new_plo1 * pos_inv
    uint64_t m1 = new_plo1 * pos_inv;

    // mulhi(m, n): (m1:m0) * (n1:n0) >> 128
    u128 q00 = (u128)m0 * n0;
    u128 q01 = (u128)m0 * n1;
    u128 q10 = (u128)m1 * n0;
    u128 q11 = (u128)m1 * n1;
    u128 qmid = (q00 >> 64) + (uint64_t)q01 + (uint64_t)q10;
    u128 mn_hi = q11 + (q01 >> 64) + (q10 >> 64) + (qmid >> 64);

    // result = prod_hi - mn_hi
    u128 result = prod_hi - mn_hi;
    if (prod_hi < mn_hi)
        result += ((u128)n1 << 64) | n0;
    r[0] = (uint64_t)result;
    r[1] = (uint64_t)(result >> 64);
}

// ============================================================================
// CIOS montmul for N>=3 — fused multiply-reduce, plain loop.
// Uses __int128 addmul1 (generic path) — no inline asm.
// ============================================================================

template<int N>
inline void montmul_cios(mpn::limb_t* __restrict__ r,
                         const mpn::limb_t* __restrict__ a,
                         const mpn::limb_t* __restrict__ b,
                         const MontCtx<N>& ctx) {
    mpn::limb_t t[2 * N + 1] = {};
    for (int i = 0; i < N; ++i) {
        mpn::limb_t cy = mpn::addmul1<N>(t + i, a, b[i]);
        t[i + N] += cy;
        t[i + N + 1] += (t[i + N] < cy);
        mpn::limb_t m = t[i] * ctx.neg_inv;
        cy = mpn::addmul1<N>(t + i, ctx.mod.d, m);
        t[i + N] += cy;
        t[i + N + 1] += (t[i + N] < cy);
    }
    mpn::copy<N>(r, t + N);
    if (t[2 * N])
        mpn::sub<N>(r, r, ctx.mod.d);
    else
        mpn::csub<N>(r, r, ctx.mod.d);
}

// ============================================================================
// Public API
// ============================================================================

// LLVM IR-optimized montmul for N=3-8 (linked from montmul_ir.o)
#ifdef ZFACTOR_HAS_MONTMUL_IR
extern "C" {
void zfactor_montmul3(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
void zfactor_montmul4(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
void zfactor_montmul5(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
void zfactor_montmul6(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
void zfactor_montmul7(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
void zfactor_montmul8(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t);
}
#endif

template<int N>
inline void montmul(mpn::limb_t* r, const mpn::limb_t* a, const mpn::limb_t* b,
                    const MontCtx<N>& ctx) {
    if constexpr (N == 1) {
        montmul_1(r, a, b, ctx.mod.d[0], ctx.pos_inv);
    } else if constexpr (N == 2) {
        montmul_2(r, a, b, ctx.mod.d, ctx.pos_inv);
    } else {
#ifdef ZFACTOR_HAS_MONTMUL_IR
        if constexpr (N == 3) { zfactor_montmul3(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
        if constexpr (N == 4) { zfactor_montmul4(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
        if constexpr (N == 5) { zfactor_montmul5(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
        if constexpr (N == 6) { zfactor_montmul6(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
        if constexpr (N == 7) { zfactor_montmul7(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
        if constexpr (N == 8) { zfactor_montmul8(r, a, b, ctx.mod.d, ctx.neg_inv); return; }
#endif
        montmul_cios<N>(r, a, b, ctx);
    }
}

template<int N>
inline void mont_redc(mpn::limb_t* r, const mpn::limb_t* t_in, const MontCtx<N>& ctx) {
    mpn::limb_t t[2 * N + 1];
    mpn::copy<2 * N>(t, t_in);
    t[2 * N] = 0;
    for (int i = 0; i < N; ++i) {
        mpn::limb_t m = t[i] * ctx.neg_inv;
        mpn::limb_t cy = mpn::addmul1<N>(t + i, ctx.mod.d, m);
        t[i + N] += cy;
        t[i + N + 1] += (t[i + N] < cy);
    }
    mpn::copy<N>(r, t + N);
    if (t[2 * N])
        mpn::sub<N>(r, r, ctx.mod.d);
    else
        mpn::csub<N>(r, r, ctx.mod.d);
}

template<int N>
inline void montsqr(mpn::limb_t* r, const mpn::limb_t* a, const MontCtx<N>& ctx) {
    montmul<N>(r, a, a, ctx);
}

template<int N>
inline void to_mont(mpn::limb_t* r, const mpn::limb_t* a, const MontCtx<N>& ctx) {
    montmul<N>(r, a, ctx.r2_mod.d, ctx);
}

template<int N>
inline void from_mont(mpn::limb_t* r, const mpn::limb_t* a, const MontCtx<N>& ctx) {
    mpn::limb_t t[2 * N] = {};
    mpn::copy<N>(t, a);
    mont_redc<N>(r, t, ctx);
}

} // namespace zfactor::fixint
