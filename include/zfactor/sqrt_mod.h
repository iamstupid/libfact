#pragma once

// Modular square root mod an odd prime.
//
//   sqrt_mod_prime(a, p) returns x with 0 <= x < p and x^2 ≡ a (mod p),
//   or 0 if a is not a quadratic residue mod p (or a == 0).
//
// The caller is responsible for ensuring p is an odd prime.  We dispatch
// to the cheapest algorithm that fits p's residue class:
//
//   p ≡ 3 (mod 4)   →  closed form  x = a^((p+1)/4) mod p
//   p ≡ 5 (mod 8)   →  Atkin's formula (two powmods)
//   p ≡ 1 (mod 8)   →  Tonelli-Shanks
//
// Cipolla is also viable for the last case but Tonelli-Shanks is simpler
// and competitive for the typical small 2-adic valuations we care about.

#include <bit>
#include <cstdint>

#include "zfactor/jacobi.h"

namespace zfactor {

namespace detail_sqrt_mod {

// 64x64 → 64 modular multiply via __int128.  Plenty fast for our use
// case (factor base setup, occasional sqrt computations); not the hot
// loop of any algorithm.
inline uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) noexcept {
    return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % m);
}

inline uint64_t powmod_u64(uint64_t base, uint64_t exp, uint64_t mod) noexcept {
    if (mod == 1) return 0;
    uint64_t r = 1 % mod;
    base %= mod;
    while (exp > 0) {
        if (exp & 1u) r = mulmod_u64(r, base, mod);
        base = mulmod_u64(base, base, mod);
        exp >>= 1;
    }
    return r;
}

} // namespace detail_sqrt_mod

inline uint64_t sqrt_mod_prime(uint64_t a, uint64_t p) {
    using detail_sqrt_mod::mulmod_u64;
    using detail_sqrt_mod::powmod_u64;

    if (p == 2) return a & 1u;
    a %= p;
    if (a == 0) return 0;
    if (a == 1) return 1;

    // QR pre-check.  Tonelli-Shanks would loop forever on a non-residue,
    // and the closed-form cases would silently return garbage.
    if (jacobi_u64(a, p) != 1) return 0;

    // ---- Case 1: p ≡ 3 (mod 4) ----
    if ((p & 3u) == 3u) {
        return powmod_u64(a, (p + 1) >> 2, p);
    }

    // ---- Case 2: p ≡ 5 (mod 8) — Atkin ----
    //   v = (2a)^((p-5)/8)
    //   i = 2a·v^2     (then i^2 ≡ -1 mod p)
    //   x = a·v·(i-1)
    if ((p & 7u) == 5u) {
        uint64_t two_a = mulmod_u64(2, a, p);
        uint64_t v  = powmod_u64(two_a, (p - 5) >> 3, p);
        uint64_t v2 = mulmod_u64(v, v, p);
        uint64_t i  = mulmod_u64(two_a, v2, p);
        uint64_t i_minus_1 = i == 0 ? p - 1 : i - 1;
        return mulmod_u64(a, mulmod_u64(v, i_minus_1, p), p);
    }

    // ---- Case 3: p ≡ 1 (mod 8) — Tonelli-Shanks ----
    // Decompose p - 1 = Q·2^S with Q odd.
    uint64_t Q = p - 1;
    int S = std::countr_zero(Q);
    Q >>= S;

    // Find any quadratic non-residue z.  Linear search from 2 — average
    // ~2 trials since half of [2, p) are non-residues.
    uint64_t z = 2;
    while (jacobi_u64(z, p) != -1) ++z;

    int M = S;
    uint64_t c = powmod_u64(z, Q, p);
    uint64_t t = powmod_u64(a, Q, p);
    uint64_t R = powmod_u64(a, (Q + 1) >> 1, p);

    while (t != 1) {
        // Find the least i, 0 < i < M, with t^(2^i) ≡ 1 (mod p).
        // The invariant ord(t) | 2^(M-1) (established initially because
        // a is a QR, then maintained) guarantees i < M.
        int i = 0;
        uint64_t tt = t;
        while (tt != 1) {
            tt = mulmod_u64(tt, tt, p);
            ++i;
        }

        // b = c^(2^(M - i - 1))
        uint64_t b = c;
        for (int j = 0; j < M - i - 1; ++j)
            b = mulmod_u64(b, b, p);

        M = i;
        c = mulmod_u64(b, b, p);
        t = mulmod_u64(t, c, p);
        R = mulmod_u64(R, b, p);
    }
    return R;
}

} // namespace zfactor
