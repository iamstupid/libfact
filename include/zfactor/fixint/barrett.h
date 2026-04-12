#pragma once

#include <cstdint>
#include "uint.h"

namespace zfactor::fixint {

// ============================================================================
// Barrett reduction context.
//
// Precomputes X = floor(B^(2N) / d) via recursive Newton iteration
// (GMP's mpn_invert2 algorithm). The inverse is stored in the implicit-B^N
// form: X = B^N + inv, with inv stored as N limbs.
//
// All multiplications use the stock mul<N> / mul_nm<N,M> / addmul1<N>
// primitives -- no schoolbook helper.
//
// REQUIREMENT: d must be normalized (top bit of d[N-1] set).
// ============================================================================

namespace detail_barrett {

// Computes the approximate reciprocal of a normalized N-limb value `ap`.
// Output: xp[0..N-1] holds X - B^N, where X = floor(B^(2N) / ap).
//   (X is in (B^N, 2*B^N], so X - B^N fits in N limbs.)
//
// This is GMP's mpn_invert2 algorithm:
//   1. Recursively compute the inverse of the top H limbs of `ap`.
//   2. Refine to full N-limb precision via one Newton step that uses
//      mul_nm<N,H> + mul<H>.
template<int M>
inline void invert_normalized(mpn::limb_t* xp, const mpn::limb_t* ap) {
    using limb_t = mpn::limb_t;

    if constexpr (M == 1) {
        // Bootstrap: q = floor((B^2 - 1) / A).
        // For normalized A in [B/2, B-1], q is in [B, 2B - 1].
        // xp[0] = q - B = q's low 64 bits (since q >= B).
        unsigned __int128 q = ~(unsigned __int128)0 / ap[0];
        xp[0] = static_cast<limb_t>(q);
        return;
    } else {
        // GMP uses L = (M-1)/2; the special case M=2 gives L=0 which doesn't
        // recurse, so we special-case L=1 there.
        constexpr int L = (M == 2) ? 1 : (M - 1) / 2;
        constexpr int H = M - L;
        static_assert(L >= 1 && H >= 1 && L < M && H < M);

        // Step 0: recursive call -- invert the top H limbs of ap.
        invert_normalized<H>(xp + L, ap + L);

        constexpr int TP_SIZE = M + H;
        limb_t tp[TP_SIZE];

        // Step 1: tp = ap * xp[L..M-1]   (M x H non-square multiply)
        mpn::mul_nm<M, H>(tp, ap, xp + L);

        // Step 2: tp[H..H+M-1] += ap   (so tp = ap * X_h, X_h = B^H + xp[L..M-1])
        uint8_t cy = mpn::add<M>(tp + H, tp + H, ap);

        // Step 3: overflow correction (cy=1 means ap*X_h >= B^(M+H), X_h too large)
        while (cy) {
            // Decrement xp[L..M-1] by 1
            (void)mpn::sub1<H>(xp + L, xp + L, 1);
            // tp -= ap (low M limbs), then propagate borrow through high H limbs
            uint8_t bw = mpn::sub<M>(tp, tp, ap);
            bw = mpn::sub1<H>(tp + M, tp + M, bw);
            cy = (cy >= bw) ? static_cast<uint8_t>(cy - bw) : 0;
        }

        // Step 4: negate tp[0..M-1]   (i.e., tp[0..M-1] := B^M - tp[0..M-1])
        {
            limb_t zero_m[M] = {};
            (void)mpn::sub<M>(tp, zero_m, tp);
        }

        // Step 5: up = tp[L..L+H-1] * xp[L..L+H-1]   (H x H multiply)
        limb_t up[2 * H];
        mpn::mul<H>(up, tp + L, xp + L);

        // Step 6: up[H..H+(H-L)-1] += tp[L..L+(H-L)-1]   (H-L limbs)
        // For L == H this is empty (H-L = 0).
        constexpr int HL = H - L;
        cy = 0;
        if constexpr (HL > 0) {
            cy = mpn::add<HL>(up + H, up + H, tp + L);
            // Propagate carry through the remaining L limbs of up's high half.
            if constexpr (L > 0) {
                cy = mpn::add1<L>(up + H + HL, up + H + HL, cy);
            }
        }

        // Step 7: xp[0..L-1] = up[2H-L..2H-1] + tp[H..H+L-1] + cy
        (void)mpn::addc<L>(xp, up + 2 * H - L, tp + H, cy);

        // Final verification: ensure xp represents the EXACT floor.
        // GMP's algorithm can produce a result that is 1 too small (or rarely
        // off by more), so we explicitly check and adjust here.  This makes
        // each recursion level return an exact inverse.
        for (int iter = 0; iter < 4; ++iter) {
            // Compute prod = ap * (B^M + xp)
            limb_t prod[2 * M + 1] = {};
            mpn::mul<M>(prod, ap, xp);
            uint8_t pc = mpn::add<M>(prod + M, prod + M, ap);
            prod[2 * M] = pc;

            // Check if ap * X > B^(2M), i.e., prod[2M] >= 1 with anything below nonzero
            bool too_large = (prod[2 * M] > 1);
            if (!too_large && prod[2 * M] == 1) {
                for (int i = 0; i < 2 * M; ++i)
                    if (prod[i] != 0) { too_large = true; break; }
            }
            if (too_large) {
                (void)mpn::sub1<M>(xp, xp, 1);
                continue;
            }

            // Check if ap * (X+1) = prod + ap <= B^(2M), i.e., X is too small
            limb_t pd[2 * M + 1];
            for (int i = 0; i <= 2 * M; ++i) pd[i] = prod[i];
            uint8_t pcy = mpn::add<M>(pd, pd, ap);
            pcy = mpn::add1<M>(pd + M, pd + M, pcy);
            pd[2 * M] += pcy;

            bool too_small = (pd[2 * M] == 0);
            if (!too_small && pd[2 * M] == 1) {
                too_small = true;
                for (int i = 0; i < 2 * M; ++i)
                    if (pd[i] != 0) { too_small = false; break; }
            }
            if (too_small) {
                (void)mpn::add1<M>(xp, xp, 1);
                continue;
            }
            break; // exact
        }
    }
}

} // namespace detail_barrett

template<int N>
struct BarrettCtx {
    UInt<N> d;
    UInt<N> inv;  // X - B^N where X = floor(B^(2N) / d). Implicit B^N top bit.

    void init(const UInt<N>& divisor) {
        d = divisor;
        detail_barrett::invert_normalized<N>(inv.d, d.d);
    }

    UInt<N> mod(const UInt<N>& a) const {
        UInt<N> q, r;
        compute_qr(q, r, a.d);
        return r;
    }

    void divrem(UInt<N>& q, UInt<N>& r, const UInt<N>& a) const {
        compute_qr(q, r, a.d);
    }

    // Wide Barrett reduction: input a is 2N limbs.
    // Returns a mod d.  Standard Barrett use case (reducing an N-by-N product).
    //
    // Algorithm:
    //   q1 = a[N-1 .. 2N-1]                          -- top (N+1) limbs of a
    //   q2 = q1 * (B^N + inv)                        -- (N+1)*(N+1) -> 2N+2 limbs
    //   q3 = q2[N+1 .. 2N+1]                         -- top N+1 limbs of q2
    //   r  = (a - q3*d) mod B^(N+1)                  -- N+1 limb arithmetic
    //   while r >= d: r -= d                         -- at most 2 iterations
    UInt<N> mod_wide(const mpn::limb_t a[2 * N]) const {
        constexpr int N1 = N + 1;
        const limb_t* q1 = a + (N - 1);

        // q2 = q1 * (B^N + inv) = q1*inv + q1*B^N
        limb_t q2[2 * N + 2] = {};
        mpn::mul_nm<N1, N>(q2, q1, inv.d);                  // q2 = q1*inv (2N+1 limbs)
        uint8_t cy = mpn::add<N1>(q2 + N, q2 + N, q1);      // q2 += q1*B^N
        q2[2 * N + 1] = cy;

        // q3 = q2[N+1 .. 2N+1]
        const limb_t* q3 = q2 + N + 1;

        // qd = q3*d  ((N+1)*N -> 2N+1 limbs)
        limb_t qd[2 * N + 1] = {};
        mpn::mul_nm<N, N1>(qd, d.d, q3);

        // r = (a - qd) mod B^(N+1)  -- working in N+1 limb precision
        limb_t r[N1];
        (void)mpn::sub<N1>(r, a, qd);

        // d extended to N+1 limbs (high limb 0)
        limb_t d_ext[N1] = {};
        mpn::copy<N>(d_ext, d.d);

        // Adjust: at most 3 iterations
        while (mpn::cmp<N1>(r, d_ext) >= 0)
            mpn::sub<N1>(r, r, d_ext);

        UInt<N> result;
        mpn::copy<N>(result.d, r);
        return result;
    }

private:
    using limb_t = mpn::limb_t;

    // Compute q = floor(a/d) and r = a mod d using the precomputed inverse.
    void compute_qr(UInt<N>& q, UInt<N>& r, const limb_t* a) const {
        // a * X = a * (B^N + inv) = a*inv + a*B^N
        limb_t prod[2 * N + 1];
        mpn::mul<N>(prod, a, inv.d);

        // Add a shifted by N limbs: prod[N..2N-1] += a (carry into prod[2N]).
        prod[2 * N] = mpn::add<N>(prod + N, prod + N, a);

        // Quotient estimate (single limb): for normalized d, q <= 2.
        limb_t q_est = prod[2 * N];

        // r = a - q_est * d   (single-limb scalar multiply via submul1)
        mpn::copy<N>(r.d, a);
        limb_t bw = mpn::submul1<N>(r.d, d.d, q_est);
        (void)bw; // q_est * d <= a, so the borrow at position N is 0

        // Adjust: at most 2 iterations. Track q_est increment locally
        // to avoid redoing a full add chain on q each time.
        while (mpn::cmp<N>(r.d, d.d) >= 0) {
            mpn::sub<N>(r.d, r.d, d.d);
            ++q_est;
        }

        mpn::set_zero<N>(q.d);
        q.d[0] = q_est;
    }
};

} // namespace zfactor::fixint
