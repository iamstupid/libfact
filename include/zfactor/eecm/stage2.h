// ECM stage 2: baby-step giant-step with batch inversion.
//
// Algorithm (standard Brent-Suyama continuation):
//   1. Pick d = 2310 = 2·3·5·7·11 (a primorial).
//   2. Baby steps: compute y_j = affine-Y of [j]*Q for all j coprime to d,
//      1 ≤ j < d/2.  That's |J| = φ(2310)/2 = 240 points.
//      Normalize to affine using Montgomery's batch inversion trick.
//   3. Giant step: R_0 = [m0·d]*Q where m0 = ceil((B1+1)/d), then
//      R_{k+1} = R_k + [d]*Q.  Walk up to R_K where K = floor(B2/d).
//   4. For each giant R_k, accumulate ∏_{j∈J} (Y_gk - y_j · Z_gk)
//      into a running product.  Test gcd(product, n) periodically.
//
// Uses y-coordinate (not x): on twisted Edwards a=-1, negation is
// (-x, y), so y(P) = y(-P).  Comparing y detects BOTH p = m·d+j and
// p = m·d-j, covering all primes.  (x-coordinates would miss half.)
//
// Cost breakdown for B2=14M, N=3 (~7 ns/montmul):
//   Baby:  240 ed_adds + 1 batch inversion     ≈ 25 µs
//   Giant: 6000 ed_adds                         ≈ 624 µs
//   Inner: 6000 × 240 × (1 mul + 1 sub + 1 mul into acc) ≈ 20 ms
//   Total: ~21 ms  vs  94 ms naive  (~4.5× speedup)
#pragma once

#include "zfactor/edwards.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/sieve.h"

#include <cstdint>
#include <vector>
#include <optional>

namespace zfactor::ecm {

// d = 2310 = 2·3·5·7·11.  All j in [1, d/2) with gcd(j,d)=1.
// |J| = φ(2310)/2 = 240.
namespace detail_stage2 {

constexpr uint64_t D = 2310;

// Precompute the 240 residues j coprime to D, 1 ≤ j < D/2.
// Done at compile time.
consteval auto make_coprime_table() {
    struct R { int vals[240]; int count; };
    R r{};
    r.count = 0;
    for (int j = 1; j < (int)(D / 2); ++j) {
        // gcd(j, 2310) = 1 iff j is coprime to 2,3,5,7,11
        if (j % 2 && j % 3 && j % 5 && j % 7 && j % 11) {
            r.vals[r.count++] = j;
        }
    }
    return r;
}

constexpr auto COPRIME_TABLE = make_coprime_table();
static_assert(COPRIME_TABLE.count == 240);

}  // namespace detail_stage2


template<int N>
struct Stage2Result {
    bool factor_found = false;
    fixint::UInt<N> factor;
};


// Montgomery's batch inversion trick: given v[0..n-1] in Montgomery form,
// compute v[i]^{-1} mod m for all i, using 3(n-1) montmuls + 1 modinv.
// On failure (some v[i] shares a factor with m), returns the factor.
template<int N>
inline std::optional<fixint::UInt<N>>
batch_invert(fixint::UInt<N>* v, int n, const fixint::MontCtx<N>& ctx) {
    if (n == 0) return std::nullopt;
    // Use mont_slow: batch_invert runs once per curve (240 muls) so speed
    // doesn't matter; avoids register-pressure issues with inline-asm kernels.
    auto m = fixint::mont_slow(ctx);

    // prefix[i] = v[0] * v[1] * ... * v[i]
    std::vector<fixint::UInt<N>> prefix(n);
    prefix[0] = v[0];
    for (int i = 1; i < n; ++i)
        prefix[i] = m.mul(prefix[i - 1], v[i]);

    // Invert the full product.
    auto full_plain = m.drop(prefix[n - 1]);
    fixint::UInt<N> inv_or_factor;
    if (!fixint::modinv<N>(&inv_or_factor, full_plain, ctx.mod))
        return inv_or_factor;   // factor found!
    auto inv = m.lift(inv_or_factor);

    // Walk backwards: v[i]^{-1} = prefix[i-1] * inv,  inv *= v[i].
    for (int i = n - 1; i >= 1; --i) {
        auto vi_inv = m.mul(prefix[i - 1], inv);
        inv = m.mul(inv, v[i]);
        v[i] = vi_inv;
    }
    v[0] = inv;
    return std::nullopt;
}


// Scalar multiply Q by a uint64 scalar via binary method.
// (Reusable helper — no wNAF needed for one-off small scalars.)
template<int N>
inline EdPoint<N> ed_scalar_u64(const EdPoint<N>& Q, uint64_t k,
                                const EdCurve<N>& curve,
                                const fixint::MontCtx<N>& ctx) {
    auto m = edwards_mont_ops<N>(ctx);
    EdPoint<N> R;
    R.X = m.zero(); R.Y = m.one(); R.Z = m.one(); R.T = m.zero();
    EdPoint<N> Pi = Q;
    while (k) {
        if (k & 1) R = ed_add<N>(R, Pi, curve, ctx);
        k >>= 1;
        if (k) Pi = ed_dbl<N>(Pi, ctx);
    }
    return R;
}


// Inner accumulation loop for one giant step.  Marked noinline so the
// inline-asm montmul kernel sees a clean register state.  This is the
// hottest loop in stage 2: 240 iterations per giant, ~6000 giants.
//
// Uses y-coordinate: on twisted Edwards a=-1, negation is (-x, y),
// so y(P) = y(-P).  Comparing y detects both p = m·D+j and p = m·D-j.
template<int N>
[[gnu::noinline]]
fixint::UInt<N> stage2_inner_accum(fixint::UInt<N> acc,
                                   const fixint::UInt<N>& Y_g,
                                   const fixint::UInt<N>& Z_g,
                                   const fixint::UInt<N>* baby_y,
                                   int nj,
                                   const fixint::MontCtx<N>& ctx) {
    auto m = edwards_mont_ops<N>(ctx);
    for (int j = 0; j < nj; ++j) {
        auto diff = m.sub(Y_g, m.mul(baby_y[j], Z_g));
        acc = m.mul(acc, diff);
    }
    return acc;
}


template<int N>
inline Stage2Result<N> ecm_stage2(const EdPoint<N>& Q,
                                  const EdCurve<N>& curve,
                                  uint64_t B1, uint64_t B2,
                                  const fixint::MontCtx<N>& ctx) {
    Stage2Result<N> result;
    if (B2 <= B1) return result;
    using namespace detail_stage2;

    auto m = edwards_mont_ops<N>(ctx);

    // ====== 1. Baby steps: compute [j]*Q for j in coprime table ======
    // First build a table of ALL multiples [1]*Q, [2]*Q, ..., [D/2]*Q
    // via repeated addition.  Then extract the coprime subset.
    constexpr int HALF_D = (int)(D / 2);
    std::vector<EdPoint<N>> all_mult(HALF_D + 1);
    all_mult[0].X = m.zero(); all_mult[0].Y = m.one();
    all_mult[0].Z = m.one();  all_mult[0].T = m.zero();
    all_mult[1] = Q;
    if (HALF_D >= 2) all_mult[2] = ed_dbl<N>(Q, ctx);
    for (int i = 3; i <= HALF_D; ++i)
        all_mult[i] = ed_add<N>(all_mult[i - 1], Q, curve, ctx);

    // Extract coprime entries and batch-invert their Z coordinates.
    constexpr int NJ = COPRIME_TABLE.count;  // 240
    std::vector<fixint::UInt<N>> baby_y(NJ);     // affine y = Y/Z
    {
        // Collect Z values, batch invert, then multiply each Y by Z^{-1}.
        // Use y-coordinate: y(P) = y(-P) on twisted Edwards, detects ± both.
        std::vector<fixint::UInt<N>> z_vals(NJ);
        for (int i = 0; i < NJ; ++i)
            z_vals[i] = all_mult[COPRIME_TABLE.vals[i]].Z;

        auto maybe_factor = batch_invert<N>(z_vals.data(), NJ, ctx);
        if (maybe_factor) {
            result.factor_found = true;
            result.factor = *maybe_factor;
            return result;
        }
        // z_vals[i] now holds Z_i^{-1} in Montgomery form.
        for (int i = 0; i < NJ; ++i)
            baby_y[i] = m.mul(all_mult[COPRIME_TABLE.vals[i]].Y, z_vals[i]);
    }

    // ====== 2. Giant step setup ======
    // dQ = [D] * Q
    auto dQ = ed_scalar_u64<N>(Q, D, curve, ctx);

    // m0 = ceil((B1+1) / D),  starting giant = m0 * D
    uint64_t m0 = (B1 + D) / D;
    // Giant point R = [m0 * D] * Q
    auto R = ed_scalar_u64<N>(Q, m0 * D, curve, ctx);

    // K = floor(B2 / D) - m0 + 1  (number of giant steps)
    uint64_t mK = B2 / D;
    if (mK < m0) return result;   // B2 too small for even one giant

    // ====== 3. Inner loop: for each giant, accumulate product ======
    auto acc = m.one();
    int batch_count = 0;
    constexpr int GCD_BATCH = 128;

    auto check_acc = [&]() -> bool {
        auto plain = m.drop(acc);
        if (plain.is_zero()) plain = ctx.mod;
        auto g = fixint::gcd<N>(plain, ctx.mod);
        if (!g.is_zero()
            && fixint::mpn::cmp<N>(g.d, fixint::UInt<N>(1).d) > 0
            && fixint::mpn::cmp<N>(g.d, ctx.mod.d) < 0) {
            result.factor_found = true;
            result.factor = g;
            return true;
        }
        return false;
    };

    for (uint64_t mi = m0; mi <= mK; ++mi) {
        // For this giant R = [mi·D]*Q, accumulate
        //   ∏_{j ∈ J} (Y_R - y_j · Z_R)
        // Each term vanishes mod q iff [mi·D ± j]*Q = identity mod q.
        acc = stage2_inner_accum<N>(acc, R.Y, R.Z, baby_y.data(), NJ, ctx);

        batch_count++;
        if (batch_count >= GCD_BATCH) {
            if (check_acc()) return result;
            acc = m.one();
            batch_count = 0;
        }

        // Advance giant: R += dQ
        R = ed_add<N>(R, dQ, curve, ctx);
    }

    // Final batch check.
    if (batch_count > 0) {
        check_acc();
    }
    return result;
}

}  // namespace zfactor::ecm
