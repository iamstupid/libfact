// Edwards-curve scalar multiplication via signed-window NAF.
//
// Inputs:
//   * Base point P (extended Edwards coords, in Montgomery form)
//   * Curve coefficient k = 2d (Montgomery)
//   * Scalar S as a sequence of bits (MSB first or LSB first — we use LSB)
//
// Algorithm:
//   1. Precompute T_i = (2i+1) * P for i in 0..2^{w-2}-1.  Cost: 1 dbl + (2^{w-2}-1) adds.
//   2. Convert S to NAF_w: a sequence of digits d_j ∈ {0, ±1, ±3, ..., ±(2^{w-1}-1)}.
//   3. Walk d from high to low:  Q := 2*Q;  if d_j != 0:  Q := Q ± T_{|d_j|/2}.
//
// We implement the wNAF computation directly on a uint64-limb scalar (no
// dependency on zint at this layer — the *caller* uses zint to compute S
// and converts it to limbs).  This keeps scalar_mult.h compileable on its
// own without dragging zint headers everywhere.
#pragma once

#include "zfactor/edwards.h"
#include <cstdint>
#include <vector>

namespace zfactor::ecm {

// Compute the wNAF (signed window-NAF) of a non-negative integer represented
// as a little-endian uint64 limb sequence.  Output digits are in {0} ∪
// {±1, ±3, ..., ±(2^{w-1}-1)} and the highest-index nonzero entry is the MSB.
//
// Length of `out` is at most bit_length(s) + 1.
//
// Reference: HMV "Guide to Elliptic Curve Cryptography" Algorithm 3.35.
inline std::vector<int8_t>
compute_wnaf(const uint64_t* s, int s_limbs, int w) {
    // Bit-walking approach: O(bits) total, no shifting.
    // Read w+1 bits at position `pos`, emit signed digit, advance.
    int total_bits = 0;
    {
        // Find highest set bit.
        for (int i = s_limbs - 1; i >= 0; --i) {
            if (s[i]) { total_bits = i * 64 + 64 - std::countl_zero(s[i]); break; }
        }
    }
    if (total_bits == 0) return {};

    // We need a mutable copy only for the carry propagation when subtracting.
    // Instead, accumulate a carry bit as we walk.
    std::vector<uint64_t> S(s, s + s_limbs);
    // Pad one extra limb for potential carry.
    S.push_back(0);

    auto get_bit = [&](int pos) -> int {
        if (pos < 0 || pos >= (int)S.size() * 64) return 0;
        return (int)((S[pos >> 6] >> (pos & 63)) & 1);
    };
    auto get_bits = [&](int pos, int count) -> int {
        // Extract `count` bits starting at position `pos`.
        int val = 0;
        for (int i = 0; i < count; ++i)
            val |= get_bit(pos + i) << i;
        return val;
    };
    // Subtract/add a small value at bit position `pos` (affects bits pos..pos+w).
    auto sub_at = [&](int pos, int v) {
        // S -= v << pos.  Use 128-bit shift to handle cross-limb boundary.
        int limb = pos >> 6;
        int bit = pos & 63;
        unsigned __int128 wide = (unsigned __int128)(uint64_t)std::abs(v) << bit;
        uint64_t lo = (uint64_t)wide;
        uint64_t hi = (uint64_t)(wide >> 64);
        if (v >= 0) {
            // subtract
            uint64_t borrow = 0;
            for (int i = limb; i < (int)S.size(); ++i) {
                uint64_t sub = (i == limb) ? lo : (i == limb + 1) ? hi : 0;
                sub += borrow;
                borrow = (sub < borrow) || (S[i] < sub) ? 1 : 0;
                S[i] -= sub;
                if (!borrow && i > limb + 1) break;
            }
        } else {
            // add
            uint64_t carry = 0;
            for (int i = limb; i < (int)S.size(); ++i) {
                uint64_t add = (i == limb) ? lo : (i == limb + 1) ? hi : 0;
                S[i] += add;
                uint64_t c1 = (S[i] < add) ? 1 : 0;
                S[i] += carry;
                uint64_t c2 = (S[i] < carry) ? 1 : 0;
                carry = c1 + c2;
                if (!carry && i > limb + 1) break;
            }
        }
    };

    std::vector<int8_t> out(total_bits + 1, 0);
    const int half_window = 1 << (w - 1);
    const int mask = (1 << w) - 1;

    int pos = 0;
    while (pos <= total_bits) {
        if (!get_bit(pos)) {
            ++pos;
            continue;
        }
        int lo = get_bits(pos, w) & mask;
        if (lo >= half_window) lo -= (1 << w);
        out[pos] = (int8_t)lo;
        sub_at(pos, lo);
        pos += w;  // next w-1 bits are guaranteed zero by NAF property
    }

    // Trim trailing zeros.
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

// Scalar multiplication by S (given as wNAF digits with the LOW digit first
// — we walk it from the back/MSB to the front/LSB).  Returns the result point
// in extended-Edwards Montgomery form.
//
// `precomp` must already contain (2i+1)*P for i = 0..(half_window - 1).
// `precomp_neg` is parallel: (-(2i+1)) * P.  Cheaper than negating on the fly
// in the inner loop, since negation costs two sub_mod (3-4 ns each).
template<int N>
inline EdPoint<N>
scalar_mult_wnaf(const std::vector<int8_t>& wnaf,
                 const std::vector<EdPoint<N>>& precomp,
                 const std::vector<EdPoint<N>>& precomp_neg,
                 const EdCurve<N>& curve,
                 const fixint::MontCtx<N>& c) {
    // Identity in extended Edwards: (0, 1, 1, 0).  In Montgomery form:
    //   X = Mont(0) = 0, Y = Mont(1) = r_mod, Z = r_mod, T = 0.
    auto m = fixint::mont(c);
    EdPoint<N> Q;
    Q.X = m.zero();
    Q.Y = m.one();
    Q.Z = m.one();
    Q.T = m.zero();

    // Walk MSB-first.
    for (int j = (int)wnaf.size() - 1; j >= 0; --j) {
        Q = ed_dbl<N>(Q, c);
        int d = wnaf[j];
        if (d > 0) {
            Q = ed_add<N>(Q, precomp[(d - 1) / 2], curve, c);
        } else if (d < 0) {
            Q = ed_add<N>(Q, precomp_neg[(-d - 1) / 2], curve, c);
        }
    }
    return Q;
}

// Build the precomputation table T_i = (2i+1) * P for i in 0..half_window-1.
// half_window = 2^{w-1}.  Cost: 1 dbl + (half_window - 1) adds.
template<int N>
inline void
build_wnaf_precomp(std::vector<EdPoint<N>>& precomp,
                   std::vector<EdPoint<N>>& precomp_neg,
                   const EdPoint<N>& P,
                   int w,
                   const EdCurve<N>& curve,
                   const fixint::MontCtx<N>& c) {
    const int half_window = 1 << (w - 1);
    precomp.resize(half_window);
    precomp_neg.resize(half_window);
    precomp[0] = P;
    precomp_neg[0] = ed_neg<N>(P, c);
    if (half_window == 1) return;
    auto twoP = ed_dbl<N>(P, c);
    for (int i = 1; i < half_window; ++i) {
        precomp[i] = ed_add<N>(precomp[i - 1], twoP, curve, c);
        precomp_neg[i] = ed_neg<N>(precomp[i], c);
    }
}

}  // namespace zfactor::ecm
