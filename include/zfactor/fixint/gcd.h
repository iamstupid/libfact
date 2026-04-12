#pragma once

#include <cstdint>
#include <bit>
#include <utility>
#include "uint.h"

#ifdef ZFACTOR_USE_GMP
#include <gmp.h>
#include <cstring>
#endif

namespace zfactor::fixint {

// ============================================================================
// Single-word binary GCD (Stein's algorithm)
// ============================================================================

inline uint64_t gcd_u64(uint64_t a, uint64_t b) {
    if (a == 0) return b;
    if (b == 0) return a;
    unsigned shift = std::countr_zero(a | b);
    a >>= std::countr_zero(a);
    do {
        b >>= std::countr_zero(b);
        if (a > b) std::swap(a, b);
        b -= a;
    } while (b != 0);
    return a << shift;
}

// ============================================================================
// Extended Euclidean GCD (single-word)
// Returns gcd(a, b) and sets *px, *py such that a*(*px) + b*(*py) = gcd.
// Requires a, b < 2^63 to avoid coefficient overflow.
// ============================================================================

inline uint64_t xgcd_u64(int64_t* px, int64_t* py, uint64_t a, uint64_t b) {
    if (b == 0) { *px = 1; *py = 0; return a; }
    if (a == 0) { *px = 0; *py = 1; return b; }
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    uint64_t old_r = a, r = b;
    while (r != 0) {
        uint64_t q = old_r / r;
        uint64_t tmp_r = old_r % r;
        old_r = r; r = tmp_r;
        int64_t tmp_s = old_s - static_cast<int64_t>(q) * s;
        old_s = s; s = tmp_s;
        int64_t tmp_t = old_t - static_cast<int64_t>(q) * t;
        old_t = t; t = tmp_t;
    }
    *px = old_s;
    *py = old_t;
    return old_r;
}

// ============================================================================
// Modular inverse: a^{-1} mod m.  Requires gcd(a, m) = 1.
// Handles full 64-bit range using __int128 internally.
// ============================================================================

inline uint64_t modinv_u64(uint64_t a, uint64_t m) {
    using i128 = __int128;
    i128 old_r = a % m, r = m;
    i128 old_s = 1, s = 0;
    while (r != 0) {
        i128 q = old_r / r;
        i128 tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s; old_s = s; s = tmp;
    }
    old_s %= static_cast<i128>(m);
    if (old_s < 0) old_s += m;
    return static_cast<uint64_t>(old_s);
}

// ============================================================================
// Multi-limb GCD.  When GMP is available (ZFACTOR_USE_GMP), delegates to
// mpz_gcd which is ~1.5-2× faster.  Otherwise falls back to Stein's.
// ============================================================================

#ifdef ZFACTOR_USE_GMP

template<int N>
inline UInt<N> gcd(const UInt<N>& a, const UInt<N>& b) {
    mpz_t ma, mb, mg;
    mpz_init(ma); mpz_init(mb); mpz_init(mg);
    mpz_import(ma, N, -1, 8, 0, 0, a.d);
    mpz_import(mb, N, -1, 8, 0, 0, b.d);
    mpz_gcd(mg, ma, mb);
    UInt<N> r{};
    size_t cnt;
    mpz_export(r.d, &cnt, -1, 8, 0, 0, mg);
    mpz_clear(ma); mpz_clear(mb); mpz_clear(mg);
    return r;
}

template<int N>
inline int modinv(UInt<N>* inv_or_factor, const UInt<N>& a, const UInt<N>& m) {
    mpz_t ma, mm, mr;
    mpz_init(ma); mpz_init(mm); mpz_init(mr);
    mpz_import(ma, N, -1, 8, 0, 0, a.d);
    mpz_import(mm, N, -1, 8, 0, 0, m.d);
    int ok = mpz_invert(mr, ma, mm);
    if (ok) {
        std::memset(inv_or_factor->d, 0, sizeof(inv_or_factor->d));
        size_t cnt;
        mpz_export(inv_or_factor->d, &cnt, -1, 8, 0, 0, mr);
    } else {
        mpz_gcd(mr, ma, mm);
        std::memset(inv_or_factor->d, 0, sizeof(inv_or_factor->d));
        size_t cnt;
        mpz_export(inv_or_factor->d, &cnt, -1, 8, 0, 0, mr);
    }
    mpz_clear(ma); mpz_clear(mm); mpz_clear(mr);
    return ok ? 1 : 0;
}

// Alias: lehmer_gcd just calls gcd when GMP is active.
template<int N>
inline UInt<N> lehmer_gcd(const UInt<N>& a, const UInt<N>& b) { return gcd<N>(a, b); }

#else

template<int N>
UInt<N> gcd(UInt<N> a, UInt<N> b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;

    unsigned shift_a = mpn::ctz<N>(a.d);
    unsigned shift_b = mpn::ctz<N>(b.d);
    unsigned shift = shift_a < shift_b ? shift_a : shift_b;

    mpn::rshift<N>(a.d, a.d, shift_a);

    do {
        mpn::rshift<N>(b.d, b.d, mpn::ctz<N>(b.d));
        if (mpn::cmp<N>(a.d, b.d) > 0) {
            UInt<N> t = a; a = b; b = t;
        }
        mpn::sub<N>(b.d, b.d, a.d);
    } while (!b.is_zero());

    mpn::lshift<N>(a.d, a.d, shift);
    return a;
}

// ============================================================================
// Multi-limb modular inverse a^{-1} mod m, using binary extended GCD.
// Returns:
//   1 with *inv_or_factor = a^{-1} mod m   if gcd(a, m) = 1
//   0 with *inv_or_factor = gcd(a, m)      if gcd(a, m) > 1
//                                          (a "stage-0 factor" find for ECM)
//
// REQUIRES m odd.  Even modulus produces incorrect results because the
// half-mod helper assumes m is odd to make `(x + m) / 2` integer when x
// is odd.  ECM always operates on odd composites so this is fine.
//
// Algorithm: Knuth TAOCP 4.5.2 alg X (binary version).  Loop invariants:
//   u = a*ua + m*va     v = a*ub + m*vb     u,v >= 0,  u + v shrinks each iter
// We only track ua/ub (the cofactor of a) since vb is not needed for the
// inverse output.
// ============================================================================

template<int N>
inline int modinv(UInt<N>* inv_or_factor, const UInt<N>& a_in, const UInt<N>& m_in) {
    using L = mpn::limb_t;

    // Trivial cases.
    if (a_in.is_zero()) {
        *inv_or_factor = m_in;   // gcd = m
        return 0;
    }

    // u, v are the running pair (initially u = m, v = a).
    UInt<N> u = m_in, v = a_in;
    // ua = cofactor of a in u; ub = cofactor of a in v.
    // u = a*ua + m*?,  v = a*ub + m*?
    // Initial: u = m → ua = 0; v = a → ub = 1.
    // We work mod m, so we keep ua, ub in [-(m-1), m-1] and reduce as needed.
    // Stored as (N+1)-limb signed magnitude using L array + sign bit.
    // For simplicity we use a UInt<N+1> sized array for the magnitude and a sign byte.

    // Use a linear, less optimal but correct version: store ua,ub mod m as UInt<N>
    // and add/subtract using the modulus.
    UInt<N> ua{};            // = 0
    UInt<N> ub(1);           // = 1
    // We need to also track integer-valued shifts.  For each rshift on u (or v)
    // we must halve ua (or ub) modulo m.  Halving modulo odd m is:
    //   if x even: x >>= 1
    //   else:      x = (x + m) >> 1   (works because m odd → x+m even)

    auto half_mod = [&](UInt<N>& x) {
        if (x.d[0] & 1) {
            // x = (x + m) / 2  — may overflow into the (N+1)-th limb, then >>1.
            L cy = mpn::add<N>(x.d, x.d, m_in.d);
            mpn::rshift<N>(x.d, x.d, 1);
            if (cy) x.d[N - 1] |= (L(1) << 63);
        } else {
            mpn::rshift<N>(x.d, x.d, 1);
        }
    };
    auto sub_mod = [&](UInt<N>& dst, const UInt<N>& a, const UInt<N>& b) {
        // dst = (a - b) mod m
        L bw = mpn::sub<N>(dst.d, a.d, b.d);
        if (bw) mpn::add<N>(dst.d, dst.d, m_in.d);
    };

    // Strip common factor of 2 — but m is odd so this never happens; just halve
    // u (or v) until odd, halving ua (resp. ub) along the way.
    while (!(u.d[0] & 1) && !u.is_zero()) {
        mpn::rshift<N>(u.d, u.d, 1);
        half_mod(ua);
    }
    while (!(v.d[0] & 1) && !v.is_zero()) {
        mpn::rshift<N>(v.d, v.d, 1);
        half_mod(ub);
    }

    while (!u.is_zero()) {
        if (mpn::cmp<N>(u.d, v.d) >= 0) {
            mpn::sub<N>(u.d, u.d, v.d);
            sub_mod(ua, ua, ub);
            while (!(u.d[0] & 1) && !u.is_zero()) {
                mpn::rshift<N>(u.d, u.d, 1);
                half_mod(ua);
            }
        } else {
            mpn::sub<N>(v.d, v.d, u.d);
            sub_mod(ub, ub, ua);
            while (!(v.d[0] & 1) && !v.is_zero()) {
                mpn::rshift<N>(v.d, v.d, 1);
                half_mod(ub);
            }
        }
    }

    // gcd is in v.  ub holds the cofactor of a (mod m) only if v == 1.
    if (mpn::cmp<N>(v.d, UInt<N>(1).d) != 0) {
        *inv_or_factor = v;
        return 0;
    }
    *inv_or_factor = ub;
    return 1;
}

// ============================================================================
// Lehmer-accelerated GCD for multi-limb values.
//
// Uses single-word cofactor accumulation (Collins' condition) on the top
// limbs to skip many full-precision operations.  Falls back to a full
// Euclidean step (via divrem) when the top-word approximation is degenerate.
// ============================================================================

namespace detail_gcd {

// Effective limbs (index of highest non-zero limb + 1)
template<int N>
inline int eff(const mpn::limb_t* a) {
    for (int i = N - 1; i >= 0; --i)
        if (a[i] != 0) return i + 1;
    return 0;
}

// Lehmer inner loop with Collins' stopping condition.
// Operates on double-word (128-bit) approximations for precision.
// Cofactors are single-word (64 bits); stops if they'd overflow.
// Returns number of steps taken (0 = no progress).
//
// After the loop:
//   even steps: a' = u0*a - v0*b,  b' = v1*b - u1*a
//   odd steps:  a' = v0*b - u0*a,  b' = u1*a - v1*b
using u128 = unsigned __int128;
inline int lehmer_step(u128 a1, u128 a2,
                       uint64_t& u0, uint64_t& u1,
                       uint64_t& v0, uint64_t& v1) {
    u0 = 1; u1 = 0; v0 = 0; v1 = 1;
    int steps = 0;

    for (;;) {
        if (a2 < v1) break;
        if ((a1 - a2) < ((u128)v0 + v1)) break;

        u128 q = a1 / a2;
        u128 r = a1 - q * a2;

        // Check cofactor overflow before committing
        u128 nu = (u128)u0 + q * u1;
        u128 nv = (u128)v0 + q * v1;
        if (nu > UINT64_MAX || nv > UINT64_MAX) break;

        a1 = a2; a2 = r;
        u0 = u1; u1 = static_cast<uint64_t>(nu);
        v0 = v1; v1 = static_cast<uint64_t>(nv);
        ++steps;
    }
    return steps;
}

// Apply Lehmer cofactor matrix to multi-limb (a, b).
// even steps: a' = u0*a - v0*b,  b' = v1*b - u1*a
// odd steps:  a' = v0*b - u0*a,  b' = u1*a - v1*b
// Returns true on success, false if overflow detected (caller should discard).
template<int N>
inline bool apply_matrix(mpn::limb_t* a, mpn::limb_t* b, int n,
                         uint64_t u0, uint64_t u1,
                         uint64_t v0, uint64_t v1,
                         int steps) {
    using limb_t = mpn::limb_t;
    // Compute: result = first*scalar1 - second*scalar2
    // Using addmul1_n into a zeroed buffer, then submul1_n to subtract.
    // Both reads of a and b finish before either is written, so no save needed.
    limb_t buf_a[N + 1] = {};
    limb_t buf_b[N + 1] = {};
    limb_t hi_a, hi_b;

    if (steps & 1) {
        // a' = v0*b - u0*a
        hi_a = mpn::addmul1_n(buf_a, b, v0, n);
        hi_a -= mpn::submul1_n(buf_a, a, u0, n);
        // b' = u1*a - v1*b
        hi_b = mpn::addmul1_n(buf_b, a, u1, n);
        hi_b -= mpn::submul1_n(buf_b, b, v1, n);
    } else {
        // a' = u0*a - v0*b
        hi_a = mpn::addmul1_n(buf_a, a, u0, n);
        hi_a -= mpn::submul1_n(buf_a, b, v0, n);
        // b' = v1*b - u1*a
        hi_b = mpn::addmul1_n(buf_b, b, v1, n);
        hi_b -= mpn::submul1_n(buf_b, a, u1, n);
    }

    // If the high limb didn't cancel to 0, the approximation was too coarse.
    if (hi_a != 0 || hi_b != 0)
        return false;

    mpn::copy<N>(a, buf_a);
    mpn::copy<N>(b, buf_b);
    return true;
}

} // namespace detail_gcd

template<int N>
UInt<N> lehmer_gcd(UInt<N> a, UInt<N> b) {
    if (a.is_zero()) return b;
    if (b.is_zero()) return a;

    // Ensure a >= b
    if (mpn::cmp<N>(a.d, b.d) < 0) {
        UInt<N> t = a; a = b; b = t;
    }

    for (;;) {
        if (b.is_zero()) return a;

        int na = detail_gcd::eff<N>(a.d);
        int nb = detail_gcd::eff<N>(b.d);

        // Finish with single-word GCD
        if (na <= 1 && nb <= 1) {
            UInt<N> r;
            r.d[0] = gcd_u64(a.d[0], b.d[0]);
            return r;
        }
        if (nb <= 1) {
            // Reduce a mod b (single limb) via divrem, then single-word finish
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a.d[0] = gcd_u64(r.d[0], b.d[0]);
            for (int i = 1; i < N; ++i) a.d[i] = 0;
            return a;
        }

        // Size mismatch: full Euclidean step (a, b) = (b, a mod b)
        if (na > nb + 1) {
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a = b;
            b = r;
            continue;
        }

        // Size mismatch by 1: reduce via divrem first
        if (na != nb) {
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a = b; b = r;
            continue;
        }

        // Extract top two limbs as 128-bit approximations.
        // This gives enough precision for Collins' condition to work.
        using u128 = unsigned __int128;
        u128 a1_128, a2_128;
        if (na >= 2) {
            a1_128 = ((u128)a.d[na - 1] << 64) | a.d[na - 2];
            a2_128 = ((u128)b.d[na - 1] << 64) | b.d[na - 2];
        } else {
            a1_128 = a.d[0];
            a2_128 = b.d[0];
        }
        if (a2_128 == 0) {
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a = b; b = r;
            continue;
        }

        // Run Lehmer inner loop on 128-bit approximations
        uint64_t u0, u1, v0, v1;
        int steps = detail_gcd::lehmer_step(a1_128, a2_128, u0, u1, v0, v1);

        if (steps == 0) {
            // No progress; do a full Euclidean step
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a = b; b = r;
            continue;
        }

        // Apply cofactor matrix to (a, b)
        bool ok = detail_gcd::apply_matrix<N>(a.d, b.d, na, u0, u1, v0, v1, steps);
        if (!ok) {
            // Approximation was too coarse; fall back to divrem
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, b.d);
            a = b; b = r;
            continue;
        }

        // Ensure a >= b
        if (mpn::cmp<N>(a.d, b.d) < 0) {
            UInt<N> t = a; a = b; b = t;
        }
    }
}

#endif  // ZFACTOR_USE_GMP

} // namespace zfactor::fixint
