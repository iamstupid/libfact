#pragma once

// Integer roots for multi-limb fixed-width unsigned integers.
//
//   isqrt<N>(n) — floor(sqrt(n))
//   icbrt<N>(n) — floor(cbrt(n))
//   is_square<N>(n)  — n is a perfect square
//   is_cube<N>(n)    — n is a perfect cube
//
// Both roots use Newton iteration starting from a power-of-two seed that
// is guaranteed >= the true root.  Newton from above is monotone decreasing
// and lands on the floor exactly when the iteration stops descending.
// Convergence is quadratic in the relative error, so for B-bit n we expect
// O(log B) iterations — about 7-9 divrems for N=8.

#include <cstdint>

#include "libdivide.h"
#include "zfactor/fixint/uint.h"

namespace zfactor::fixint {

namespace detail_iroot {

// Helper: compute n / 3 in place (multi-limb / single-word).  Used by icbrt's
// Newton step.  Uses __int128 per limb — ~25 cycles each, called twice per
// iteration, total noise next to the divrem cost.
template<int N>
inline UInt<N> div3(const UInt<N>& n) {
    UInt<N> result{};
    uint64_t r = 0;
    for (int i = N - 1; i >= 0; --i) {
        unsigned __int128 t = (static_cast<unsigned __int128>(r) << 64) | n.d[i];
        result.d[i] = static_cast<uint64_t>(t / 3);
        r            = static_cast<uint64_t>(t % 3);
    }
    return result;
}

// ---- modular verification helpers for is_square / is_cube ----
//
// Test primes for the modular check.  Their product is
//   3·5·7·11·13·17·19·23·29·31·37·41·43·47·53·59 ≈ 6.1·10¹⁷,
// so the false-positive rate of "x^q ≡ y (mod p_i) for all i" implying
// x^q == y is roughly k / 6e17, where k bounds the candidate space.
// For any practical factorization workload this is negligible.
inline constexpr uint32_t TEST_PRIMES[16] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59
};

// libdivide reciprocals for the test primes, lazily constructed.
inline const libdivide::divider<uint64_t>* test_dividers() {
    static const libdivide::divider<uint64_t> divs[16] = {
        libdivide::divider<uint64_t>( 3), libdivide::divider<uint64_t>( 5),
        libdivide::divider<uint64_t>( 7), libdivide::divider<uint64_t>(11),
        libdivide::divider<uint64_t>(13), libdivide::divider<uint64_t>(17),
        libdivide::divider<uint64_t>(19), libdivide::divider<uint64_t>(23),
        libdivide::divider<uint64_t>(29), libdivide::divider<uint64_t>(31),
        libdivide::divider<uint64_t>(37), libdivide::divider<uint64_t>(41),
        libdivide::divider<uint64_t>(43), libdivide::divider<uint64_t>(47),
        libdivide::divider<uint64_t>(53), libdivide::divider<uint64_t>(59),
    };
    return divs;
}

// n mod p via the half-limb trick (same shape as trial.h::mod_small).
// Numerator of every libdivide call stays < 2^64.
template<int N>
inline uint32_t mod_p(const mpn::limb_t* d, uint32_t p,
                     const libdivide::divider<uint64_t>& div) noexcept {
    uint64_t r = 0;
    const uint64_t pp = p;
    for (int i = N - 1; i >= 0; --i) {
        uint64_t lim = d[i];
        uint64_t t = (r << 32) | (lim >> 32);
        r = t - (t / div) * pp;
        t = (r << 32) | (lim & 0xFFFFFFFFu);
        r = t - (t / div) * pp;
    }
    return static_cast<uint32_t>(r);
}

} // namespace detail_iroot

// ============================================================================
// isqrt — floor(sqrt(n))
// ============================================================================
//
// Newton iteration: x_{k+1} = (x_k + n/x_k) / 2.
//
// Initial seed: x_0 = 2^ceil(B/2) where B = bit_length(n).  This is always
// >= sqrt(n), so the iteration is monotone decreasing.
//
// Termination: stop when x stops descending.  At that point x = floor(sqrt(n))
// (provable from x*(x+1) >= n >= x*x at the fixed point).
template<int N>
inline UInt<N> isqrt(const UInt<N>& n) {
    if (n.is_zero()) return UInt<N>{};

    unsigned B = mpn::bit_length<N>(n.d);
    unsigned half = (B + 1) / 2;

    UInt<N> x{};
    mpn::set_bit<N>(x.d, half);

    while (true) {
        UInt<N> q, r;
        mpn::divrem<N>(q.d, r.d, n.d, x.d);

        UInt<N> sum;
        mpn::add<N>(sum.d, x.d, q.d);

        UInt<N> y;
        mpn::rshift<N>(y.d, sum.d, 1);

        if (mpn::cmp<N>(y.d, x.d) >= 0) return x;
        x = y;
    }
}

// ============================================================================
// icbrt — floor(cbrt(n))
// ============================================================================
//
// Newton iteration: x_{k+1} = (2*x_k + n/x_k^2) / 3.
//
// Computes n/x^2 as (n/x)/x — two N×N divrems per step instead of one
// squaring + one division — which avoids needing a 2N-limb scratch buffer
// and is the same algorithmic cost.  AM-GM guarantees convergence from above
// when x_0 >= cbrt(n).
//
// Seed: x_0 = 2^ceil(B/3).
template<int N>
inline UInt<N> icbrt(const UInt<N>& n) {
    if (n.is_zero()) return UInt<N>{};

    unsigned B = mpn::bit_length<N>(n.d);
    unsigned third = (B + 2) / 3;

    UInt<N> x{};
    mpn::set_bit<N>(x.d, third);

    while (true) {
        // q1 = n / x
        UInt<N> q1, r1;
        mpn::divrem<N>(q1.d, r1.d, n.d, x.d);
        // q2 = q1 / x  ==  floor(n / x^2)
        UInt<N> q2, r2;
        mpn::divrem<N>(q2.d, r2.d, q1.d, x.d);

        // sum = 2*x + q2
        UInt<N> two_x;
        mpn::lshift<N>(two_x.d, x.d, 1);
        UInt<N> sum;
        mpn::add<N>(sum.d, two_x.d, q2.d);

        UInt<N> y = detail_iroot::div3<N>(sum);

        if (mpn::cmp<N>(y.d, x.d) >= 0) return x;
        x = y;
    }
}

// ============================================================================
// Perfect-power tests — modular verification, no full power-back
// ============================================================================
//
// We compute r = isqrt(n) (or icbrt) then verify r^q ≡ n (mod p_i) for the
// 16 small test primes.  Doing it this way:
//
//   * No multi-limb r^q computation — verification scales linearly in N
//     instead of quadratically.
//   * No 2N scratch buffer for the squared value.
//   * Same shape generalizes to any q without needing a fast q-th-power
//     kernel; only the "(rm * rm) % p" line changes.
//
// Correctness: if n is a true perfect q-th power then r is exact and
// r^q == n, so the modular comparisons hold for every p.  No false negatives.
// False-positive rate is bounded by k / ∏ p_i ≈ k / 6e17, where k is the
// candidate-space size — vanishing for any factorization workload.

template<int N>
inline bool is_square(const UInt<N>& n) {
    if (n.is_zero()) return true;
    UInt<N> r = isqrt<N>(n);
    const auto* divs = detail_iroot::test_dividers();
    for (int i = 0; i < 16; ++i) {
        uint32_t p  = detail_iroot::TEST_PRIMES[i];
        uint32_t rm = detail_iroot::mod_p<N>(r.d, p, divs[i]);
        uint32_t nm = detail_iroot::mod_p<N>(n.d, p, divs[i]);
        if ((static_cast<uint64_t>(rm) * rm) % p != nm) return false;
    }
    return true;
}

template<int N>
inline bool is_cube(const UInt<N>& n) {
    if (n.is_zero()) return true;
    UInt<N> r = icbrt<N>(n);
    const auto* divs = detail_iroot::test_dividers();
    for (int i = 0; i < 16; ++i) {
        uint32_t p  = detail_iroot::TEST_PRIMES[i];
        uint32_t rm = detail_iroot::mod_p<N>(r.d, p, divs[i]);
        uint32_t nm = detail_iroot::mod_p<N>(n.d, p, divs[i]);
        uint64_t r2m = (static_cast<uint64_t>(rm) * rm) % p;
        if ((r2m * rm) % p != nm) return false;
    }
    return true;
}

} // namespace zfactor::fixint
