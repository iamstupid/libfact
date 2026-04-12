#pragma once

// Pollard rho factoring — Brent's variant with batched GCD.
//
// Finds a non-trivial factor of a composite n.  Expected running time is
// O(n^(1/4)) iterations of the polynomial f(x) = x^2 + c mod n for n with
// smallest prime factor p (more precisely ~ sqrt(p) iterations).
//
// Design notes:
//   * Iteration runs entirely in Montgomery form.  Both the polynomial's
//     state y and the constant c stay in Montgomery form — we never
//     convert out of it inside the inner loop.
//   * Seeds are cheap: since the iteration just needs some arbitrary
//     quadratic-map starting point, we write x0_seed / c_seed directly
//     into the limbs without going through to_mont().  The "plain value"
//     represented by such a seed happens to be raw_seed · R^{-1} mod n
//     — still a perfectly valid starting position for the cycle walk.
//   * GCD skips Mont→plain conversion.  For odd n, gcd(R, n) = 1, so
//       gcd(q_mont, n) = gcd(q_plain · R mod n, n) = gcd(q_plain, n)
//     and we can gcd directly on the Montgomery representation.
//   * GCD scheduling: tiny segments don't justify a gcd call.  We let q
//     accumulate diffs across multiple Brent segments and only invoke
//     gcd once r reaches M/2 — by which point q holds ~M-1 terms.
//   * No backtracking on factor-found-with-collision.  If gcd(q, n) == n,
//     we return n and let the caller retry with a different c / x0.
//   * The caller owns a MontCtx<N> for n and passes it in; rho keeps no
//     MontScope, so the caller can retry with a single shared ctx.
//   * Preconditions: n is odd, composite, not a perfect power, and
//     mctx.init(n) has been called.

#include <algorithm>
#include <cstdint>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/montgomery.h"
#include "zfactor/fixint/gcd.h"

namespace zfactor {

namespace detail_rho {

template<int N>
inline bool is_one(const fixint::UInt<N>& v) noexcept {
    if (v.d[0] != 1) return false;
    for (int i = 1; i < N; ++i) if (v.d[i] != 0) return false;
    return true;
}

template<int N>
inline fixint::UInt<N> one_uint() noexcept {
    fixint::UInt<N> r{};
    r.d[0] = 1;
    return r;
}

// Build a Mont-form value directly from a raw seed — skips the to_mont
// REDC.  The "plain value" this represents is raw_seed · R^{-1} mod n,
// which is fine for anything we use it for (rho iteration seeds, constant
// c — these are arbitrary starting points for a quadratic chaos map).
template<int N>
inline fixint::UInt<N> raw_seed(uint64_t s) noexcept {
    fixint::UInt<N> r{};
    r.d[0] = s;
    return r;
}

} // namespace detail_rho

// Pollard rho with Brent's cycle detection and batched GCD.
//
// Returns a non-trivial factor d of n with 1 < d < n on success, or n
// itself on failure (caller should retry with a different c / x0).
//
// `mctx` must already have been initialized via `mctx.init(n)`.  All
// arithmetic goes through the MontOps method API — no TLS, no per-value
// pointer stash, the ctx fields hoist into registers around the hot loop.
//
// Inner-body shape (one Brent step):
//   diff = x - y_k          (B chain, reads y_k)
//   y    = y_k^2 + c        (A chain, also reads y_k — independent of B)
//   q    = q * diff         (B chain continuation)
//
// For r >= 4 we hand-unroll 4 steps so the y-update chain runs ahead of
// the q-mul chain by one full step, letting the OoO core overlap the
// sqr and the q*diff.  For r in {1, 2} the unrolled body cannot be
// filled, so we fall back to the simple loop.
template<int N>
inline fixint::UInt<N> pollard_rho_brent(
    const fixint::MontCtx<N>& mctx,
    uint64_t c_seed = 1,
    uint64_t x0_seed = 2
) {
    using namespace fixint;

    auto m = mont<N>(mctx);
    const UInt<N>& n = mctx.mod;

    const UInt<N> c = detail_rho::raw_seed<N>(c_seed);

    UInt<N> y = detail_rho::raw_seed<N>(x0_seed);
    UInt<N> x = y;
    UInt<N> q = m.one();

    UInt<N> g_uint = detail_rho::one_uint<N>();

    // GCD batch size: N=1 has cheap gcd; for larger N each gcd call is
    // many cycles, so we accumulate longer products before checking.
    constexpr int M = (N == 1) ? 128 : 512;
    int r = 1;

    while (detail_rho::is_one<N>(g_uint)) {
        x = y;
        for (int i = 0; i < r; ++i) y = m.add(m.sqr(y), c);

        int k = 0;
        while (k < r && detail_rho::is_one<N>(g_uint)) {
            int lim = std::min(M, r - k);

            if (r >= 4) {
                // lim is divisible by 4 here: it's min of two powers-of-2
                // both ≥ 4, so the unrolled body never has a tail.
                for (int i = 0; i < lim; i += 4) {
                    UInt<N> d0 = m.sub(x, y);
                    y = m.add(m.sqr(y), c);
                    q = m.mul(q, d0);
                    UInt<N> d1 = m.sub(x, y);
                    y = m.add(m.sqr(y), c);
                    q = m.mul(q, d1);
                    UInt<N> d2 = m.sub(x, y);
                    y = m.add(m.sqr(y), c);
                    q = m.mul(q, d2);
                    UInt<N> d3 = m.sub(x, y);
                    y = m.add(m.sqr(y), c);
                    q = m.mul(q, d3);
                }
            } else {
                // Head: r ∈ {1, 2}.  Body too short to bother unrolling.
                for (int i = 0; i < lim; ++i) {
                    UInt<N> diff = m.sub(x, y);
                    y = m.add(m.sqr(y), c);
                    q = m.mul(q, diff);
                }
            }

            // Defer the gcd until segments are long enough to amortize
            // it.  q keeps accumulating across the small-r segments;
            // by the time r reaches M/2 it carries ~M-1 terms.
            if (r >= M / 2) {
                // gcd on the raw Montgomery limbs — valid because
                // gcd(R, n) = 1 for odd n, so gcd(q_mont, n) == gcd(q_plain, n).
                g_uint = gcd<N>(q, n);
            }
            k += M;
        }
        r *= 2;

        // Safety rail on runaway segments.
        if (r > (1 << 30)) return n;
    }

    return g_uint;  // may equal n on collision; caller retries
}

} // namespace zfactor
