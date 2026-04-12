#pragma once

#ifdef ZFACTOR_USE_GMP
#include <gmp.h>
#endif

// Jacobi symbol (a / n) for odd positive n.  Returns -1, 0, or +1.
//
// For prime n the Jacobi symbol coincides with the Legendre symbol, so it
// also serves as a quadratic-residue test for primes:
//   jacobi(a, p) ==  1   →   a is a non-zero quadratic residue mod p
//   jacobi(a, p) ==  0   →   p | a
//   jacobi(a, p) == -1   →   a is a non-residue mod p
//
// Algorithm: standard Cohen 1.4.10 (binary GCD-style with quadratic
// reciprocity).  Each iteration strips factors of 2, swaps a and n, and
// applies QR.  Loop terminates when a == 0; the answer is t if n == 1,
// else 0 (gcd(a, n) > 1).

#include <bit>
#include <cstdint>
#include <utility>

#include "zfactor/fixint/uint.h"

namespace zfactor {

namespace detail_jacobi {

template<int N>
inline bool is_one(const fixint::UInt<N>& x) noexcept {
    if (x.d[0] != 1) return false;
    for (int i = 1; i < N; ++i) if (x.d[i] != 0) return false;
    return true;
}

} // namespace detail_jacobi

// Single-word Jacobi symbol (a / n).  n must be odd and non-zero; result
// is undefined if n is even.
inline int jacobi_u64(uint64_t a, uint64_t n) noexcept {
    a %= n;
    int t = 1;
    while (a != 0) {
        // Strip factors of 2.  Each shift by 1 bit flips t iff
        // n mod 8 ∈ {3, 5}, so the cumulative flip depends on the parity
        // of the shift count.
        unsigned s = std::countr_zero(a);
        a >>= s;
        if ((s & 1u) && ((n & 7u) == 3u || (n & 7u) == 5u))
            t = -t;

        // Swap and apply quadratic reciprocity.
        std::swap(a, n);
        if ((a & 3u) == 3u && (n & 3u) == 3u)
            t = -t;

        a %= n;
    }
    return n == 1 ? t : 0;
}

// Multi-limb Jacobi symbol.
#ifdef ZFACTOR_USE_GMP
// Delegate to mpz_jacobi (~3-5× faster).
template<int N>
inline int jacobi(const fixint::UInt<N>& a, const fixint::UInt<N>& n) {
    mpz_t ma, mn;
    mpz_init(ma); mpz_init(mn);
    mpz_import(ma, N, -1, 8, 0, 0, a.d);
    mpz_import(mn, N, -1, 8, 0, 0, n.d);
    int r = mpz_jacobi(ma, mn);
    mpz_clear(ma); mpz_clear(mn);
    return r;
}
#else
template<int N>
inline int jacobi(fixint::UInt<N> a, fixint::UInt<N> n) {
    using namespace fixint;

    // Initial reduction a := a mod n.
    if (mpn::cmp<N>(a.d, n.d) >= 0) {
        UInt<N> q, r;
        mpn::divrem<N>(q.d, r.d, a.d, n.d);
        a = r;
    }

    int t = 1;
    while (!a.is_zero()) {
        unsigned s = mpn::ctz<N>(a.d);
        if (s > 0) {
            mpn::rshift<N>(a.d, a.d, s);
            uint64_t low8 = n.d[0] & 7u;
            if ((s & 1u) && (low8 == 3u || low8 == 5u))
                t = -t;
        }

        std::swap(a, n);
        if ((a.d[0] & 3u) == 3u && (n.d[0] & 3u) == 3u)
            t = -t;

        if (mpn::cmp<N>(a.d, n.d) >= 0) {
            UInt<N> q, r;
            mpn::divrem<N>(q.d, r.d, a.d, n.d);
            a = r;
        }
    }
    return detail_jacobi::is_one<N>(n) ? t : 0;
}
#endif  // ZFACTOR_USE_GMP

} // namespace zfactor
