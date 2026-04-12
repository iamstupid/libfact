#pragma once

#include "uint.h"

namespace zfactor::fixint {

template<int N>
struct MontCtx {
    UInt<N> mod;
    UInt<N> r2_mod;
    UInt<N> r_mod;
    uint64_t pos_inv;     // n^{-1} mod 2^64       (low limb)
    uint64_t neg_inv;     // -n^{-1} mod 2^64
    uint64_t pos_inv_hi;  // for N>=2: high limb of n^{-1} mod 2^128
                          // (used by Hurchalla-style montmul_2; otherwise 0)

    void init(const UInt<N>& n) {
        mod = n;
        uint64_t x = 1;
        for (int i = 0; i < 6; ++i) x *= 2 - n[0] * x;
        pos_inv = x;
        neg_inv = -x;

        // For N>=2 we also Hensel-lift the inverse to mod 2^128 so that
        // montmul_2 (Hurchalla REDC alternate) can compute m = u_lo * inv
        // as a full 128-bit product.  One Newton step on the 128-bit
        // value is enough since the 64-bit inverse is already correct.
        if constexpr (N >= 2) {
            using u128 = unsigned __int128;
            u128 x128 = (u128)pos_inv;            // low 64 bits ok, high bits = 0
            u128 n128 = ((u128)n[1] << 64) | n[0];
            // x128 *= 2 - n128 * x128   (mod 2^128); two iterations to converge
            for (int i = 0; i < 2; ++i) {
                x128 = x128 * (u128(2) - n128 * x128);
            }
            pos_inv_hi = (uint64_t)(x128 >> 64);
        } else {
            pos_inv_hi = 0;
        }

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

// Inline asm variant of montmul_1 — same algorithm, hand-scheduled for
// BMI2 mulx with the implicit-rdx contention made explicit.  Critical path
// is the same 12-cycle minimum (3 serial multiplies + reduce), but
// separating the load of `a` from the first mulx (rather than having mulx
// fold the memory operand inline as clang's __int128 emits) lets the load
// overlap with the previous iteration's tail, saving ~6 cycles per call
// vs the __int128 path on Zen4 in chained-latency benches.
//
// rdx is in the clobber list rather than bound via the "d" constraint
// because we overwrite it mid-body (rdx = m for the second mulx); a tied
// `+d` operand is awkward to express here, so we explicitly load `a_val`
// into rdx inside the asm and let the compiler put it anywhere.
__attribute__((always_inline))
inline void montmul_1_asm(mpn::limb_t* __restrict__ r,
                          const mpn::limb_t* __restrict__ a,
                          const mpn::limb_t* __restrict__ b,
                          mpn::limb_t mod, mpn::limb_t pos_inv) {
    uint64_t a_val = a[0];
    uint64_t b_val = b[0];
    uint64_t hi, lo, mod_corr;

    asm("movq %[a], %%rdx\n\t"                 // rdx = a_val (compiler picks any reg for [a])
        "mulxq %[b], %[lo], %[hi]\n\t"         // [hi:lo] = a*b
        "imulq %[inv], %[lo]\n\t"              // lo *= inv  → m
        "movq %[lo], %%rdx\n\t"                // rdx = m
        "mulxq %[mod], %[lo], %[lo]\n\t"       // lo = mn_hi  (mn_lo discarded — both dsts same)
        "subq %[lo], %[hi]\n\t"                // hi -= mn_hi   (CF = borrow)
        "leaq (%[hi], %[mod]), %[mc]\n\t"      // mc = hi + mod  (the "+mod" version)
        "cmovbq %[mc], %[hi]\n\t"              // if borrow, take the corrected version
        : [hi]  "=&r"(hi),
          [lo]  "=&r"(lo),
          [mc]  "=&r"(mod_corr)
        : [a]   "r"(a_val),
          [b]   "r"(b_val),
          [mod] "r"(mod),
          [inv] "r"(pos_inv)
        : "rdx", "cc");

    r[0] = hi;
    (void)mod_corr;
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
// Hurchalla-style "REDC alternate" for N=2 (128-bit modulus).
//
// Reference: Jeffrey Hurchalla, modular_arithmetic library, RedcStandard
// __uint128_t specialization.  See README_REDC.md in that repo for the
// algorithm derivation.  Uses positive inverse n^{-1} mod 2^128 (computed
// at MontCtx::init time and stored as pos_inv:pos_inv_hi).
//
// Algorithm vs standard CIOS:
//   1. u = a * b   (full 256-bit product, in 128-bit halves u_hi:u_lo)
//   2. m = u_lo * inv_n  (mod 2^128 — only the low 128 matter)
//   3. mn_hi = high(m * n)  (high 128 of the 256-bit product)
//   4. Precompute reg = u_hi + n  (the "+n corrected" candidate)
//   5. Parallel sub trick (in inline asm):
//        reg' = reg - mn_hi    (corrected version, always valid)
//        t_hi = u_hi - mn_hi   (uncorrected, may underflow)
//        result = (CF from t_hi sub) ? reg' : t_hi
//
// The two 128-bit subtracts are independent — they run in parallel on
// the OoO core.  No serial sub→cmov→add chain like standard reduction.
// ============================================================================

// NOTE on compiler dependence: the C++ __int128 form below compiles
// to optimal mulx + adc chains under clang (~13 cycles on Zen4 in
// chained-latency benches), but g++ <= 12 produces a measurably worse
// schedule on the same source (~7.3 ns vs clang's 4.22 ns at N=2).
//
// We provide TWO implementations:
//   * montmul_2_hurchalla        — pure C++ __int128 (this function)
//   * montmul_2_clang_lifted     — inline asm that mirrors clang's
//                                  instruction sequence verbatim
// The dispatch in `montmul<N>` picks the __int128 form under clang
// (where it's slightly faster than the asm — clang can schedule
// across iterations through the bare instructions but not through an
// inline-asm boundary) and the lifted asm under g++ (where it brings
// g++'s codegen up to clang's level, ~40% improvement on g++-12).
__attribute__((always_inline))
inline void montmul_2_hurchalla(mpn::limb_t* __restrict__ r,
                                const mpn::limb_t* __restrict__ a,
                                const mpn::limb_t* __restrict__ b,
                                const mpn::limb_t* __restrict__ mod,
                                mpn::limb_t pos_inv_lo,
                                mpn::limb_t pos_inv_hi) {
    using u128 = unsigned __int128;

    uint64_t a0 = a[0], a1 = a[1];
    uint64_t b0 = b[0], b1 = b[1];
    uint64_t n0 = mod[0], n1 = mod[1];

    // Step 1: u = a * b  (256-bit product split into u_hi:u_lo)
    u128 a0_b0 = (u128)a0 * b0;
    u128 a0_b1 = (u128)a0 * b1;
    u128 a1_b0 = (u128)a1 * b0;
    u128 a1_b1 = (u128)a1 * b1;
    u128 mid = (a0_b0 >> 64) + (uint64_t)a0_b1 + (uint64_t)a1_b0;
    uint64_t u_lo_0 = (uint64_t)a0_b0;
    uint64_t u_lo_1 = (uint64_t)mid;
    u128 u_hi = a1_b1 + (a0_b1 >> 64) + (a1_b0 >> 64) + (mid >> 64);

    // Step 2: m = u_lo * inv_n  (mod 2^128).  Only the low 128 matter,
    // so we can drop the u_lo_1 * inv_hi cross term entirely (it shifts
    // out at bit 128).  3 muls instead of 4.
    uint64_t m0, m1;
    {
        u128 lo0_inv0 = (u128)u_lo_0 * pos_inv_lo;
        // Cross terms only need their low halves (the high halves shift out).
        uint64_t lo0_inv1 = u_lo_0 * pos_inv_hi;
        uint64_t lo1_inv0 = u_lo_1 * pos_inv_lo;
        m0 = (uint64_t)lo0_inv0;
        m1 = (uint64_t)(lo0_inv0 >> 64) + lo0_inv1 + lo1_inv0;
    }

    // Step 3: mn_hi = high(m * n).  All 4 limb-pair muls needed because
    // even the lo*lo product contributes to the high half via carries.
    u128 mn_hi;
    {
        u128 m0_n0 = (u128)m0 * n0;
        u128 m0_n1 = (u128)m0 * n1;
        u128 m1_n0 = (u128)m1 * n0;
        u128 m1_n1 = (u128)m1 * n1;
        u128 mid2 = (m0_n0 >> 64) + (uint64_t)m0_n1 + (uint64_t)m1_n0;
        mn_hi = m1_n1 + (m0_n1 >> 64) + (m1_n0 >> 64) + (mid2 >> 64);
    }

    // Step 4: reg = u_hi + n  (precomputed for the parallel-sub trick)
    u128 n128 = ((u128)n1 << 64) | n0;
    u128 reg = u_hi + n128;

    // Step 5: parallel-sub trick — two independent 128-bit subtractions
    // and one cmov pair, in inline asm so the compiler can't merge them.
    uint64_t reg_lo = (uint64_t)reg;
    uint64_t reg_hi = (uint64_t)(reg >> 64);
    uint64_t uhi_lo = (uint64_t)u_hi;
    uint64_t uhi_hi = (uint64_t)(u_hi >> 64);
    uint64_t mn_hi_lo = (uint64_t)mn_hi;
    uint64_t mn_hi_hi = (uint64_t)(mn_hi >> 64);
    asm("subq %[mnhilo], %[reglo]\n\t"      // reg = (u_hi + n) - mn_hi   (lo)
        "sbbq %[mnhihi], %[reghi]\n\t"      //                              (hi)
        "subq %[mnhilo], %[uhilo]\n\t"      // t_hi = u_hi - mn_hi          (lo)
        "sbbq %[mnhihi], %[uhihi]\n\t"      //                              (hi)
        "cmovaeq %[uhilo], %[reglo]\n\t"    // if t_hi didn't underflow, take it
        "cmovaeq %[uhihi], %[reghi]\n\t"
        : [reglo] "+&r"(reg_lo), [reghi] "+&r"(reg_hi),
          [uhilo] "+&r"(uhi_lo), [uhihi] "+&r"(uhi_hi)
        : [mnhilo] "r"(mn_hi_lo), [mnhihi] "r"(mn_hi_hi)
        : "cc");

    r[0] = reg_lo;
    r[1] = reg_hi;
}

// ============================================================================
// montmul_2_clang_lifted: hand-written inline asm whose instruction
// sequence mirrors clang's codegen for `montmul_2_hurchalla` verbatim.
// This makes g++ produce the same schedule clang does, eliminating the
// compiler-quality gap.  See the long comment block above for context.
//
// The body computes the same Hurchalla "REDC alternate":
//   1. u = a*b              (256-bit, via 4 mulx fanout)
//   2. m = u_lo * inv       (mod 2^128, 1 mulx + 2 imul)
//   3. mn_hi = (m*n) >> 128 (4 mulx + 8-instruction carry sum)
//   4. reg = u_hi + n
//   5. parallel sub: (reg - mn_hi) vs (u_hi - mn_hi), cmov on borrow
// ============================================================================
__attribute__((always_inline))
inline void montmul_2_clang_lifted(mpn::limb_t* __restrict__ r,
                                   const mpn::limb_t* __restrict__ a,
                                   const mpn::limb_t* __restrict__ b,
                                   const mpn::limb_t* __restrict__ mod,
                                   mpn::limb_t pos_inv_lo,
                                   mpn::limb_t pos_inv_hi) {
    uint64_t a0 = a[0], a1 = a[1];
    uint64_t b0 = b[0], b1 = b[1];
    uint64_t n0 = mod[0], n1 = mod[1];
    uint64_t s1, s2, s3, s4, s5;

    asm(
        // ===== Step 1: a*b → 4-mul fanout (b0,b1 reused as p10.lo,p11.lo) =====
        "movq %[b0], %%rdx\n\t"
        "mulxq %[a0], %[s1], %[s2]\n\t"          // s2:s1 = b0*a0  (p00)
        "movq %[b1], %%rdx\n\t"
        "mulxq %[a0], %[s3], %[s4]\n\t"          // s4:s3 = b1*a0  (p01)
        "movq %[b0], %%rdx\n\t"
        "mulxq %[a1], %[b0], %[s5]\n\t"          // s5:b0 = b0*a1  (p10) — last use of input b0
        "movq %[b1], %%rdx\n\t"
        "mulxq %[a1], %[b1], %[a0]\n\t"          // a0:b1 = b1*a1  (p11) — last use of input b1, a0

        // After: s1=p00.lo, s2=p00.hi, s3=p01.lo, s4=p01.hi,
        //        b0=p10.lo, s5=p10.hi, b1=p11.lo, a0=p11.hi, a1 still=a1

        // ===== u_hi assembly (sum p00..p11 with carry chain through rdx) =====
        "xorl %%edx, %%edx\n\t"
        "addq %[s2], %[b0]\n\t"                  // b0 += p00.hi          → CF1
        "setb %%dl\n\t"                          // rdx = CF1 (saved)
        "addq %[s5], %[b1]\n\t"                  // b1 += p10.hi          → CF2
        "adcq $0, %[a0]\n\t"                     // a0 += CF2
        "addq %[s4], %[b1]\n\t"                  // b1 += p01.hi          → CF3
        "adcq $0, %[a0]\n\t"                     // a0 += CF3
        "addq %[s3], %[b0]\n\t"                  // b0 += p01.lo  → b0 = u[1]
        "adcq %%rdx, %[b1]\n\t"                  // b1 += rdx + CF4 → b1 = u[2]; CF5 pending

        // s1=u[0], b0=u[1], b1=u[2], a0=u[3] (CF5 still uncommitted)

        // ===== Step 2: m = u_lo * inv (mod 2^128) =====
        "movq %[s1], %%rdx\n\t"                  // rdx = u[0]
        "mulxq %[inv_lo], %%rdx, %[a1]\n\t"      // a1:rdx = u[0]*inv_lo  (lo0_inv0)
                                                  //   rdx = m0 (low half), a1 = hi half
        "adcq $0, %[a0]\n\t"                     // commit pending CF5 to a0 (mulx leaves flags alone)
        "imulq %[inv_hi], %[s1]\n\t"             // s1 = lo(u[0] * inv_hi) = lo0_inv1
        "imulq %[inv_lo], %[b0]\n\t"             // b0 = lo(u[1] * inv_lo) = lo1_inv0
        "xorl %k[s4], %k[s4]\n\t"                // s4 = 0 (carry capture for next chain)

        // ===== Step 3: m * n → mn_hi  (4 mulx + sum) =====
        "mulxq %[n0], %[s3], %[s3]\n\t"          // s3 = hi(m0*n0)  (same-dst mulx → hi half)
        "mulxq %[n1], %[inv_hi], %[inv_lo]\n\t"  // inv_lo:inv_hi = m0*n1
                                                  //   inv_hi=lo, inv_lo=hi  (input regs reused)
        "addq %[a1], %[s1]\n\t"                  // s1 += hi(lo0_inv0)
        "addq %[s1], %[b0]\n\t"                  // b0 = lo1_inv0 + s1 = m1
        "movq %[b0], %%rdx\n\t"                  // rdx = m1
        "mulxq %[n0], %[s5], %[a1]\n\t"          // a1:s5 = m1*n0   (s5=lo, a1=hi)
        "mulxq %[n1], %[b0], %%rdx\n\t"          // rdx:b0 = m1*n1  (b0=lo, rdx=hi)

        // sum into mn_hi = (rdx:b0)
        "addq %[s3], %[inv_hi]\n\t"              // inv_hi += hi(m0*n0)
        "setb %b[s4]\n\t"                        // s4 = carry
        "addq %[inv_lo], %[b0]\n\t"              // b0 += hi(m0*n1)
        "adcq $0, %%rdx\n\t"
        "addq %[a1], %[b0]\n\t"                  // b0 += hi(m1*n0)
        "adcq $0, %%rdx\n\t"
        "addq %[s5], %[inv_hi]\n\t"              // (carry-only) inv_hi += lo(m1*n0)
        "adcq %[s4], %[b0]\n\t"                  // b0 += saved carry
        "adcq $0, %%rdx\n\t"

        // Now: (rdx:b0) = mn_hi, (a0:b1) = u_hi

        // ===== Step 4: reg = u_hi + n (writes into n0,n1, the "+n" candidate) =====
        "addq %[b1], %[n0]\n\t"                  // n0 += u[2]
        "adcq %[a0], %[n1]\n\t"                  // n1 += u[3] + carry

        // ===== Step 5: parallel sub: (reg − mn_hi) vs (u_hi − mn_hi), cmov on borrow =====
        "subq %[b0], %[n0]\n\t"                  // n0 -= mn_hi.lo
        "sbbq %%rdx, %[n1]\n\t"                  // n1 -= mn_hi.hi
        "subq %[b0], %[b1]\n\t"                  // b1 -= mn_hi.lo
        "sbbq %%rdx, %[a0]\n\t"                  // a0 -= mn_hi.hi   (CF = borrow ⇔ uncorrected underflowed)
        "cmovaeq %[b1], %[n0]\n\t"               // if no borrow, take uncorrected
        "cmovaeq %[a0], %[n1]\n\t"

        : [a0]      "+&r"(a0),
          [a1]      "+&r"(a1),
          [b0]      "+&r"(b0),
          [b1]      "+&r"(b1),
          [n0]      "+&r"(n0),
          [n1]      "+&r"(n1),
          [inv_lo]  "+&r"(pos_inv_lo),
          [inv_hi]  "+&r"(pos_inv_hi),
          [s1]      "=&r"(s1),
          [s2]      "=&r"(s2),
          [s3]      "=&r"(s3),
          [s4]      "=&r"(s4),
          [s5]      "=&r"(s5)
        :
        : "rdx", "cc"
    );

    r[0] = n0;
    r[1] = n1;
}

// ============================================================================
// Hand-tuned N=3..4 CIOS (mulx + adcx/adox + lazy memory loads).
// Generated by tools/fixint/gen_montmul_adcx.py at build time.
//
// Algorithm summary: the standard CIOS template (montmul_inline_N) holds
// a[] and mod[] in registers throughout, costing 2N regs.  For N>=3 that
// pushes register pressure past the 16-GPR file and clang spills heavily
// (17 spills at N=3, 39 at N=4 — even before counting g++ overhead).
//
// The generated version below uses mulx-with-memory operands so a[] and
// mod[] never live in registers.  Combined with adcx/adox parallel CF/OF
// chains, the inner loop fits in (N+9) registers and produces zero spills
// for both clang and g++.
// ============================================================================
}  // namespace zfactor::fixint

#include "detail/montmul_adcx_generated.h"

namespace zfactor::fixint {


// ============================================================================
// CIOS montmul for N>=3 — fused multiply-reduce, plain loop.
// Uses __int128 addmul1 (generic path) — no inline asm.
// ============================================================================

// Generic CIOS fallback for N>8 (or when the inline kernels aren't built).
// For N=3..8 the dispatch in `montmul<N>` below uses the fully-unrolled
// inline kernels in detail/montmul_inline.h instead.
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

// Inline C++ CIOS for N=3..8 — fully unrolled, all temporaries in
// named locals so the compiler register-allocates them.  Lives in
// detail/montmul_inline.h, generated alongside the LLVM IR.
}  // namespace zfactor::fixint

#include "detail/montmul_inline.h"

namespace zfactor::fixint {

template<int N>
[[gnu::always_inline]] inline void montmul(mpn::limb_t* r,
                                           const mpn::limb_t* a,
                                           const mpn::limb_t* b,
                                           const MontCtx<N>& ctx) {
    if constexpr (N == 1) {
        montmul_1_asm(r, a, b, ctx.mod.d[0], ctx.pos_inv);
    } else if constexpr (N == 2) {
        // Compiler-dependent dispatch: clang's __int128 schedules better
        // around the inline-asm boundary (~6% faster on Zen4); g++'s
        // __int128 codegen is significantly worse, so the lifted-from-
        // clang inline asm wins by ~40% there.
#if defined(__clang__)
        montmul_2_hurchalla(r, a, b, ctx.mod.d, ctx.pos_inv, ctx.pos_inv_hi);
#else
        montmul_2_clang_lifted(r, a, b, ctx.mod.d, ctx.pos_inv, ctx.pos_inv_hi);
#endif
    } else if constexpr (N == 3) {
        montmul_3_adcx(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else if constexpr (N == 4) {
        montmul_4_adcx(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else if constexpr (N == 5) {
        montmul_5_adcx(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else if constexpr (N == 6) {
        montmul_6_adcx(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else if constexpr (N == 7) {
        montmul_inline_7(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else if constexpr (N == 8) {
        montmul_inline_8(r, a, b, ctx.mod.d, ctx.neg_inv);
    } else {
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

// ============================================================================
// MontCtx<N> value-API (no TLS, no per-value pointer).
//
// All operations take/return UInt<N> in Montgomery form.  The ctx is the
// natural owner of the modulus, so we put the operations on it as methods.
// Marked always_inline so the compiler hoists ctx-field loads (mod,
// pos_inv, neg_inv, r_mod) out of any tight loop that touches a single ctx.
//
// Naming convention: lowercase methods take values, return new values;
// _to suffix variants write into a result pointer for in-place fusion.
// ============================================================================

template<int N>
struct MontOps {
    const MontCtx<N>& c;

    [[gnu::always_inline]] UInt<N> one() const noexcept { return c.r_mod; }
    [[gnu::always_inline]] UInt<N> zero() const noexcept { return UInt<N>{}; }

    // Lift / drop between plain and Montgomery form.
    [[gnu::always_inline]] UInt<N> lift(const UInt<N>& a) const noexcept {
        UInt<N> r; to_mont<N>(r.d, a.d, c); return r;
    }
    [[gnu::always_inline]] UInt<N> drop(const UInt<N>& a) const noexcept {
        UInt<N> r; from_mont<N>(r.d, a.d, c); return r;
    }

    // Mont multiplication / squaring.
    [[gnu::always_inline]] UInt<N> mul(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r; montmul<N>(r.d, a.d, b.d, c); return r;
    }
    [[gnu::always_inline]] UInt<N> sqr(const UInt<N>& a) const noexcept {
        UInt<N> r; montsqr<N>(r.d, a.d, c); return r;
    }

    // Modular add / sub.  Both representatives are in [0, mod).
    [[gnu::always_inline]] UInt<N> add(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        uint8_t cy = mpn::add<N>(r.d, a.d, b.d);
        if (cy) mpn::sub<N>(r.d, r.d, c.mod.d);
        else    mpn::csub<N>(r.d, r.d, c.mod.d);
        return r;
    }
    [[gnu::always_inline]] UInt<N> sub(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        mpn::limb_t bw = mpn::sub<N>(r.d, a.d, b.d);
        mpn::cadd<N>(r.d, r.d, c.mod.d, bw);
        return r;
    }

    // In-place fused step variants — let the caller avoid named temporaries
    // for the common "y = y^2 + c" / "q = q * d" patterns.
    [[gnu::always_inline]] void sqr_inplace(UInt<N>& a) const noexcept {
        montsqr<N>(a.d, a.d, c);
    }
    [[gnu::always_inline]] void mul_inplace(UInt<N>& a, const UInt<N>& b) const noexcept {
        montmul<N>(a.d, a.d, b.d, c);
    }
    [[gnu::always_inline]] void add_inplace(UInt<N>& a, const UInt<N>& b) const noexcept {
        uint8_t cy = mpn::add<N>(a.d, a.d, b.d);
        if (cy) mpn::sub<N>(a.d, a.d, c.mod.d);
        else    mpn::csub<N>(a.d, a.d, c.mod.d);
    }
};

// Tiny adapter so callers can write `mont(ctx).sqr(a)` etc. — no storage,
// the helper is a stack object the optimiser deletes entirely.
template<int N>
[[gnu::always_inline]] inline MontOps<N> mont(const MontCtx<N>& c) noexcept {
    return MontOps<N>{c};
}

// ============================================================================
// montmul_outlined<N>: identical to montmul<N> but marked noinline so the
// inline-asm kernel sees an empty register state at call time.  Necessary
// when the caller has too many live locals for the kernel to allocate (e.g.
// EECM ed_add, which has 8+ live UInt<N> locals).  Costs one function call
// (~3 cycles on Zen4) per multiply.
// ============================================================================
template<int N>
[[gnu::noinline]] inline void montmul_outlined(mpn::limb_t* r,
                                               const mpn::limb_t* a,
                                               const mpn::limb_t* b,
                                               const MontCtx<N>* ctx) {
    montmul<N>(r, a, b, *ctx);
}

template<int N>
[[gnu::noinline]] inline void montsqr_outlined(mpn::limb_t* r,
                                               const mpn::limb_t* a,
                                               const MontCtx<N>* ctx) {
    montmul<N>(r, a, a, *ctx);
}

// ============================================================================
// MontOpsSlow<N>: same API as MontOps<N>, but routes mul/sqr through the
// generic CIOS loop (no inline asm).  Use for *setup-time* code where the
// fast inline-asm kernel's register pressure exceeds what the regalloc can
// handle in a long-lived function context (e.g. EECM curve setup, which
// has dozens of named locals plus a long sequence of mul calls).  Curve
// setup runs once per curve so we don't care about its speed.
// ============================================================================
template<int N>
struct MontOpsSlow {
    const MontCtx<N>& c;

    UInt<N> one()  const noexcept { return c.r_mod; }
    UInt<N> zero() const noexcept { return UInt<N>{}; }

    UInt<N> lift(const UInt<N>& a) const noexcept {
        // to_mont via slow path: t = (a, 0), then mont_redc<N> reduces.
        // We can just call montmul_cios with a*r2.
        UInt<N> r;
        montmul_cios<N>(r.d, a.d, c.r2_mod.d, c);
        return r;
    }
    UInt<N> drop(const UInt<N>& a) const noexcept {
        UInt<N> r;
        // from_mont = mont_redc applied to (a, 0...).
        // The generic from_mont in this header builds a 2N-limb t with low half = a
        // then runs mont_redc.  Replicate inline.
        mpn::limb_t t[2 * N] = {};
        mpn::copy<N>(t, a.d);
        mont_redc<N>(r.d, t, c);
        return r;
    }
    UInt<N> mul(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        montmul_cios<N>(r.d, a.d, b.d, c);
        return r;
    }
    UInt<N> sqr(const UInt<N>& a) const noexcept { return mul(a, a); }
    UInt<N> add(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        uint8_t cy = mpn::add<N>(r.d, a.d, b.d);
        if (cy) mpn::sub<N>(r.d, r.d, c.mod.d);
        else    mpn::csub<N>(r.d, r.d, c.mod.d);
        return r;
    }
    UInt<N> sub(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        mpn::limb_t bw = mpn::sub<N>(r.d, a.d, b.d);
        mpn::cadd<N>(r.d, r.d, c.mod.d, bw);
        return r;
    }
};

template<int N>
inline MontOpsSlow<N> mont_slow(const MontCtx<N>& c) noexcept {
    return MontOpsSlow<N>{c};
}

// ============================================================================
// MontOpsOut<N>: same API again, but routes mul/sqr through montmul_outlined
// (no-inline wrapper).  Use in hot paths whose register pressure is too high
// for the inline-asm kernel to allocate, but still want the fast kernel.
// EECM ed_add at N>=4 is the canonical case.
// ============================================================================
template<int N>
struct MontOpsOut {
    const MontCtx<N>& c;

    UInt<N> one()  const noexcept { return c.r_mod; }
    UInt<N> zero() const noexcept { return UInt<N>{}; }

    UInt<N> lift(const UInt<N>& a) const noexcept {
        UInt<N> r; montmul_outlined<N>(r.d, a.d, c.r2_mod.d, &c); return r;
    }
    UInt<N> drop(const UInt<N>& a) const noexcept {
        UInt<N> r;
        mpn::limb_t t[2 * N] = {};
        mpn::copy<N>(t, a.d);
        mont_redc<N>(r.d, t, c);
        return r;
    }
    UInt<N> mul(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r; montmul_outlined<N>(r.d, a.d, b.d, &c); return r;
    }
    UInt<N> sqr(const UInt<N>& a) const noexcept {
        UInt<N> r; montsqr_outlined<N>(r.d, a.d, &c); return r;
    }
    UInt<N> add(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        uint8_t cy = mpn::add<N>(r.d, a.d, b.d);
        if (cy) mpn::sub<N>(r.d, r.d, c.mod.d);
        else    mpn::csub<N>(r.d, r.d, c.mod.d);
        return r;
    }
    UInt<N> sub(const UInt<N>& a, const UInt<N>& b) const noexcept {
        UInt<N> r;
        mpn::limb_t bw = mpn::sub<N>(r.d, a.d, b.d);
        mpn::cadd<N>(r.d, r.d, c.mod.d, bw);
        return r;
    }
};

template<int N>
inline MontOpsOut<N> mont_out(const MontCtx<N>& c) noexcept {
    return MontOpsOut<N>{c};
}

} // namespace zfactor::fixint
