#pragma once

// Shanks's Square Form Factorization (SQUFOF), single-word variant.
//
// Gower-Wagstaff 2008.  Works on odd composite n that is not a perfect
// square; the sweet spot is n in roughly [2^40, 2^62].  For smaller n,
// trial division is cheaper; for larger n, rho/ECM/QS win.
//
// Sketch:
//   * Set D = k·n for some multiplier k (we sweep through a small list).
//   * Run the continued-fraction expansion of sqrt(D), tracking the
//     forms (P_i, Q_i, Q_{i-1}).  Invariants:
//       P_{i+1} = q·Q_i - P_i
//       Q_{i+1} = Q_{i-1} + q·(P_i - P_{i+1})
//     with q = floor((P_0 + P_i) / Q_i) and P_0 = floor(sqrt(D)).
//   * Stop the forward phase when Q at an even index is a perfect square
//     s².  Odd-index square Q's are "improper" and factor trivially
//     back to n, so we skip them.
//   * Reverse direction from (P, s, (D - P'²)/s) and iterate until
//     P stabilises (P_new == P) — the form is then ambiguous.
//   * gcd(P, n) is a non-trivial factor (probably).
//
// Multipliers {1, 3, 5, 7, 11, 3·5, 3·7, 3·11, 5·7, 5·11, 7·11, 3·5·7,
// 3·5·11, 3·7·11, 5·7·11, 3·5·7·11} give a near-certain hit for any
// composite n ≤ 2^62.  On failure we return 0; caller should fall back
// to rho or a larger algorithm.

#include <cmath>
#include <cstdint>

#include "zfactor/fixint/gcd.h"

namespace zfactor {

namespace detail_squfof {

// Fast u64 isqrt via double seed + one correction step.  Double sqrt
// handles values up to ~2^53 exactly; above that we may need ±1 fixup.
inline uint64_t isqrt_u64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
    // Bring r down if the double rounded up past floor(sqrt).
    while (r > 0 && r > n / r) --r;
    // Bring r up if the double rounded down.  Use overflow-safe test.
    uint64_t rp1 = r + 1;
    while (rp1 != 0 && rp1 <= n / rp1 && rp1 * rp1 <= n) {
        ++r;
        rp1 = r + 1;
    }
    return r;
}

// Mod-64 square residue filter.  Squares mod 64 live in {0, 1, 4, 9, 16,
// 17, 25, 33, 36, 41, 49, 57} — 12 of 64 values — so ~81% of random
// non-squares are rejected before we pay for sqrt().
inline bool is_square_u64(uint64_t n, uint64_t* root_out) {
    // Bit i set iff i is a square mod 64.
    constexpr uint64_t SQ_MOD_64_MASK =
        (uint64_t(1) << 0)  | (uint64_t(1) << 1)  | (uint64_t(1) << 4)  |
        (uint64_t(1) << 9)  | (uint64_t(1) << 16) | (uint64_t(1) << 17) |
        (uint64_t(1) << 25) | (uint64_t(1) << 33) | (uint64_t(1) << 36) |
        (uint64_t(1) << 41) | (uint64_t(1) << 49) | (uint64_t(1) << 57);
    if (((SQ_MOD_64_MASK >> (n & 63)) & 1u) == 0) return false;
    uint64_t r = isqrt_u64(n);
    if (r * r == n) { *root_out = r; return true; }
    return false;
}

// One SQUFOF attempt with a given multiplier k.  Returns a non-trivial
// factor of n on success, or 0 on failure (caller should try another k).
// Assumes n is odd, composite, and not a perfect square.
inline uint64_t squfof_with_k(uint64_t n, uint64_t k) {
    // D = k·n — avoid overflow
    if (k > 0 && n > UINT64_MAX / k) return 0;
    uint64_t D = k * n;
    if (D == 0) return 0;

    uint64_t P0 = isqrt_u64(D);
    if (P0 * P0 == D) {
        // kn is a perfect square; gcd(P0, n) is likely a factor
        uint64_t g = fixint::gcd_u64(P0, n);
        return (g > 1 && g < n) ? g : 0;
    }

    uint64_t P     = P0;
    uint64_t Qprev = 1;
    uint64_t Q     = D - P0 * P0;
    if (Q == 0) return 0;  // degenerate

    // Iteration bound — generous factor of 3 over the proved limit in
    // Gower-Wagstaff.
    uint64_t L = 2 * isqrt_u64(2 * P0);
    uint64_t B = 3 * L;
    if (B < 100) B = 100;

    // Parity convention: in our state indexing, after loop iteration
    // iloop=k the current Q holds Q_{k+2} in Gower-Wagstaff's notation
    // (because our initial (Qprev=1, Q=D-P_0^2) is their (Q_0, Q_1), so
    // iloop=0 computes Q_2).  Proper ambiguous forms live at even GW
    // indices → check at even iloop.
    uint64_t s = 0;
    uint64_t i;
    for (i = 0; i < B; ++i) {
        uint64_t q      = (P0 + P) / Q;
        uint64_t P_new  = q * Q - P;
        uint64_t Q_new  = Qprev + q * (P - P_new);

        Qprev = Q;
        Q     = Q_new;
        P     = P_new;

        if ((i & 1) != 0) continue;        // odd iloop → odd GW index → improper
        if (!is_square_u64(Q, &s)) continue;

        // Snapshot forward state so that a failed reverse phase can
        // resume the forward walk (some squares still give trivial gcd).
        uint64_t P_save     = P;
        uint64_t Q_save     = Q;
        uint64_t Qprev_save = Qprev;

        // Reverse direction: re-seed from the square form.
        uint64_t b   = (P0 - P) / s;
        uint64_t Prv = b * s + P;
        uint64_t Qrv_prev = s;
        uint64_t Qrv = (D - Prv * Prv) / s;
        if (Qrv == 0) {
            P = P_save; Q = Q_save; Qprev = Qprev_save;
            continue;
        }

        // Reverse cycle — terminate when the form is ambiguous (P stable).
        bool found = false;
        uint64_t rev_B = B + 100;
        for (uint64_t j = 0; j < rev_B; ++j) {
            uint64_t qr     = (P0 + Prv) / Qrv;
            uint64_t Prv_new = qr * Qrv - Prv;
            uint64_t Qrv_new = Qrv_prev + qr * (Prv - Prv_new);

            if (Prv_new == Prv) { found = true; break; }

            Qrv_prev = Qrv;
            Qrv      = Qrv_new;
            Prv      = Prv_new;
        }

        if (found) {
            uint64_t g = fixint::gcd_u64(Prv, n);
            if (g > 1 && g < n) return g;
        }

        // Reverse phase produced a trivial gcd or failed — keep walking.
        P = P_save; Q = Q_save; Qprev = Qprev_save;
    }
    return 0;
}

} // namespace detail_squfof

// Pollard-friendly multipliers (small smooth numbers).  Ordered by empirical
// hit rate; first hit wins.
inline constexpr uint64_t SQUFOF_MULTIPLIERS[] = {
    1,
    3, 5, 7, 11, 13,
    3 * 5, 3 * 7, 3 * 11, 5 * 7, 5 * 11, 7 * 11,
    3 * 5 * 7, 3 * 5 * 11, 3 * 7 * 11, 5 * 7 * 11,
    3 * 5 * 7 * 11
};
inline constexpr int SQUFOF_NUM_MULTIPLIERS =
    sizeof(SQUFOF_MULTIPLIERS) / sizeof(SQUFOF_MULTIPLIERS[0]);

// Returns a non-trivial factor of n, or 0 if SQUFOF fails with every
// multiplier in SQUFOF_MULTIPLIERS.  Caller is responsible for pre-checks:
// n should be odd, composite, and not a perfect power.
inline uint64_t squfof(uint64_t n) {
    if (n < 4) return 0;
    if ((n & 1) == 0) return 2;

    // Reject perfect squares up front — SQUFOF can return the trivial
    // factor sqrt(n) but the caller likely wants to know n was a square.
    {
        uint64_t r;
        if (detail_squfof::is_square_u64(n, &r)) return r;
    }

    for (int i = 0; i < SQUFOF_NUM_MULTIPLIERS; ++i) {
        uint64_t d = detail_squfof::squfof_with_k(n, SQUFOF_MULTIPLIERS[i]);
        if (d > 1 && d < n) return d;
    }
    return 0;
}

} // namespace zfactor
