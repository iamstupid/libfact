// Top-level Edwards Curve Method (ECM) factorization entry point.
//
// Algorithm sketch:
//   for k in 1, 2, 3, ...
//     1. setup_curve<N>(k, ctx) — Z/6 starfish parameterization
//        - if it returns a factor, we're done
//     2. compute stage-1 scalar S = ∏_{p ≤ B1} p^⌊log_p(B1)⌋   (via zint)
//     3. wNAF + scalar mult: Q = [S] * P
//     4. test gcd(Q.Z mod n, n) — if non-trivial, we have a factor
//     5. (stage 2 — TODO)
//
// The stage-1 scalar S only depends on B1, so we cache S (and its wNAF) per
// thread.  The window size w is chosen by minimising 1 dbl_per_bit + add cost
// over the precompute table — for the bit-lengths we're dealing with (~B1
// from a few thousand to a few million primes => scalar of 10k–10M bits) the
// optimum is w = 5 or 6.
#pragma once

#include "zfactor/edwards.h"
#include "zfactor/eecm/curve_setup.h"
// scalar_mult.h no longer needed — stage 1 uses 128-bit accumulator batches
#include "zfactor/eecm/schedule_generated.h"
#include "zfactor/eecm/stage2.h"
#ifdef ZFACTOR_USE_FLINT
#include "zfactor/eecm/stage2_poly.h"
#endif
#include "zfactor/fixint/gcd.h"
#include "zfactor/sieve.h"

#include <cstdint>
#include <optional>
#include <vector>
#include <cmath>

namespace zfactor::ecm {

// Stage 1 multipliers: batch prime powers into 512-bit (8-limb) chunks.
// Precomputed once per B1, cached per thread, reused across curves.
struct Stage1Mults {
    uint64_t B1 = 0;
    std::vector<fixint::UInt<8>> mults;
};

inline const Stage1Mults& get_stage1_mults(uint64_t B1) {
    static thread_local Stage1Mults cache;
    if (cache.B1 == B1) return cache;
    cache.B1 = B1;
    cache.mults.clear();

    double log2B = std::log2((double)B1);
    double accum_bits = 0;
    cache.mults.push_back(fixint::UInt<8>(1));

    PrimeIter it(2, B1 + 1);
    while (uint64_t p = it.next()) {
        double log2p = std::log2((double)p);
        for (double bits_left = log2B; bits_left >= log2p; bits_left -= log2p) {
            if (accum_bits + log2p < 500) {
                accum_bits += log2p;
                // multiply accumulator by p (single-limb multiply)
                uint64_t carry = 0;
                for (int i = 0; i < 8; ++i) {
                    unsigned __int128 prod = (unsigned __int128)cache.mults.back().d[i] * p + carry;
                    cache.mults.back().d[i] = (uint64_t)prod;
                    carry = (uint64_t)(prod >> 64);
                }
            } else {
                accum_bits = log2p;
                fixint::UInt<8> v{};
                v.d[0] = p;
                cache.mults.push_back(v);
            }
        }
    }
    return cache;
}

// wNAF scalar mult for 8-limb scalar.  w=4: precomp 8 points, ~512 dbls + ~128 adds.
template<int N>
inline EdPoint<N> ed_scalar_wnaf8(const EdPoint<N>& Q, const fixint::UInt<8>& k,
                                   const EdCurve<N>& curve,
                                   const fixint::MontCtx<N>& ctx) {
    constexpr int W = 5;
    constexpr int HALF = 1 << (W - 1);  // 16
    constexpr int MASK = (1 << W) - 1;  // 31

    // Compute wNAF digits (simple shift loop — 512 iters × 8 limbs = trivial).
    uint64_t S[8];
    for (int i = 0; i < 8; ++i) S[i] = k.d[i];

    int8_t wnaf[520];
    int wnaf_len = 0;
    auto s_nonzero = [&]{ for (int i = 0; i < 8; ++i) if (S[i]) return true; return false; };

    while (s_nonzero()) {
        if (S[0] & 1) {
            int lo = (int)(S[0] & MASK);
            if (lo >= HALF) lo -= (1 << W);
            wnaf[wnaf_len++] = (int8_t)lo;
            // S -= lo
            if (lo >= 0) {
                uint64_t borrow = (uint64_t)lo;
                for (int i = 0; i < 8; ++i) {
                    uint64_t old = S[i]; S[i] -= borrow;
                    borrow = (S[i] > old) ? 1 : 0;
                }
            } else {
                uint64_t carry = (uint64_t)(-lo);
                for (int i = 0; i < 8; ++i) {
                    uint64_t old = S[i]; S[i] += carry;
                    carry = (S[i] < old) ? 1 : 0;
                }
            }
        } else {
            wnaf[wnaf_len++] = 0;
        }
        // right shift 1
        for (int i = 0; i < 7; ++i) S[i] = (S[i] >> 1) | (S[i+1] << 63);
        S[7] >>= 1;
    }

    // Precompute table: precomp[i] = [2i+1]*Q for i=0..HALF-1
    auto m = edwards_mont_ops<N>(ctx);
    EdPoint<N> precomp[HALF], precomp_neg[HALF];
    precomp[0] = Q;
    precomp_neg[0] = ed_neg<N>(Q, ctx);
    auto twoQ = ed_dbl<N>(Q, ctx);
    for (int i = 1; i < HALF; ++i) {
        precomp[i] = ed_add<N>(precomp[i-1], twoQ, curve, ctx);
        precomp_neg[i] = ed_neg<N>(precomp[i], ctx);
    }

    // Walk MSB-first
    EdPoint<N> R;
    R.X = m.zero(); R.Y = m.one(); R.Z = m.one(); R.T = m.zero();
    for (int j = wnaf_len - 1; j >= 0; --j) {
        R = ed_dbl<N>(R, ctx);
        int d = wnaf[j];
        if (d > 0) R = ed_add<N>(R, precomp[(d-1)/2], curve, ctx);
        else if (d < 0) R = ed_add<N>(R, precomp_neg[(-d-1)/2], curve, ctx);
    }
    return R;
}

// Outcome of one ECM trial (one curve, stage 1 only).
template<int N>
struct Stage1Result {
    bool factor_found = false;
    fixint::UInt<N> factor;   // if factor_found
};

// Run one curve attempt: stage 1, then (if no factor) stage 2.
template<int N>
inline Stage1Result<N> ecm_one_curve(uint64_t k, uint64_t B1, uint64_t B2,
                                     const fixint::MontCtx<N>& ctx) {
    Stage1Result<N> result;

    // 1. Set up curve.
    auto cs = setup_curve<N>(k, ctx);
    if (cs.factor_found) {
        result.factor_found = true;
        result.factor = cs.factor;
        return result;
    }

    // 2. Stage 1: multiply P0 by each batched 512-bit chunk of prime powers.
    const auto& mults = get_stage1_mults(B1);
    auto Q = cs.P0;
    for (const auto& m512 : mults.mults)
        Q = ed_scalar_wnaf8<N>(Q, m512, cs.curve, ctx);

    // 5. Test gcd(Q.X, n) for a stage-1 factor.  At a "successful" stage 1,
    // [s]*P ≡ identity (mod q) for the hidden factor q | n.  In twisted
    // Edwards extended coords the identity is (0:1:1:0), so X ≡ 0 (mod q)
    // is the right witness.
    auto m = fixint::mont(ctx);
    auto X_plain = m.drop(Q.X);
    if (X_plain.is_zero()) X_plain = ctx.mod;   // would-be gcd(0,n)=n
    auto g = fixint::gcd<N>(X_plain, ctx.mod);
    if (!g.is_zero() && fixint::mpn::cmp<N>(g.d, fixint::UInt<N>(1).d) > 0
                     && fixint::mpn::cmp<N>(g.d, ctx.mod.d) < 0) {
        result.factor_found = true;
        result.factor = g;
        return result;
    }

    // 6. Stage 2.
    if (B2 > B1) {
#ifdef ZFACTOR_USE_FLINT
        // FFT polynomial stage 2 wins when range ≥ ~5M (≥140-bit input).
        // Below that, BSGS inner product is faster due to lower constant.
        auto s2 = (B2 - B1 >= 4000000)
            ? ecm_stage2_poly<N>(Q, cs.curve, B1, B2, ctx)
            : ecm_stage2<N>(Q, cs.curve, B1, B2, ctx);
#else
        auto s2 = ecm_stage2<N>(Q, cs.curve, B1, B2, ctx);
#endif
        if (s2.factor_found) {
            result.factor_found = true;
            result.factor = s2.factor;
        }
    }
    return result;
}

// Escalating ECM driver.
//
// Schedule: step through factor-bit targets in increments of 16 bits
// (~5 digits), running the GMP-ECM-recommended number of curves at each
// level before escalating.  This is the standard approach: cheap levels
// first, then progressively heavier schedules.
//
// Starting point:
//   - Normal (n >= 2^132):  start at 66 bits (20 digits).  Assumes the
//     caller has already done trial division + Pollard rho, so factors
//     below ~20 digits are gone.
//   - Small inputs (n < 2^132):  start at max(20, bitlen(n)/4).  This
//     avoids overshooting — for a 64-bit n, start_bits = 20 and cap = 32.
//
// Cap: factor_bits <= floor(bitlen(n) / 2).  Beyond that the smallest
// factor exceeds sqrt(n) which means n is prime (or rho/GNFS is better).
template<int N>
inline std::optional<fixint::UInt<N>>
ecm(const fixint::UInt<N>& n) {
    fixint::MontCtx<N> ctx;
    ctx.init(n);

    int n_bits = (int)n.bit_length();
    int max_fb = (n_bits + 1) / 2;   // ceil: for 159-bit n, cap = 80

    // Starting point: 66 bits for large inputs, scaled down for small.
    int start_fb;
    if (n_bits >= 132)
        start_fb = 66;               // normal: 20 digits
    else
        start_fb = std::max(20, n_bits / 4);  // small: quarter of input

    // Clamp to schedule table range.
    if (start_fb < 20) start_fb = 20;
    if (max_fb < start_fb) max_fb = start_fb;

    uint64_t k_next = 2;   // curve parameter counter (monotonically increasing)

    for (int fb = start_fb; ; fb += 16) {
        if (fb > max_fb) fb = max_fb;   // always run the cap level

        const auto& row = schedule_for_bits(fb);
        int curves_this_level = (int)row.avg_curves;
        if (curves_this_level < 10) curves_this_level = 10;

        for (int i = 0; i < curves_this_level; ++i) {
            auto r = ecm_one_curve<N>(k_next++, row.B1, row.B2, ctx);
            if (r.factor_found) return r.factor;
        }

        if (fb >= max_fb) break;
    }
    return std::nullopt;
}

}  // namespace zfactor::ecm
