#pragma once

// Baillie-PSW probable-prime test.
//
// Combines:
//   1. Trivial sieving (n < 2, n == 2, even n)
//   2. Strong Miller-Rabin with witness 2     (Montgomery powmod)
//   3. Strong Lucas test with Selfridge parameters (Lucas chain)
//
// No counterexample is known up to ~2^64; for any practical factorization
// workload (n < 2^1024) BPSW is the de facto "is prime" test.
//
// Caller passes a UInt<N>; we set up a Montgomery context internally.
//
// References:
//   * Baillie & Wagstaff, "Lucas pseudoprimes" (1980).
//   * Crandall & Pomerance, "Prime Numbers: A Computational Perspective" §3.6.
//   * Wikipedia, "Baillie-PSW primality test".

#include <cstdint>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/modular.h"
#include "zfactor/jacobi.h"

namespace zfactor {

namespace detail_bpsw {

// Halve a value in Montgomery form modulo n (n is odd).
//
// Trick: in any representative ring, dividing by 2 mod odd n is
//   (v + (v odd ? n : 0)) >> 1
// — and this works identically in Montgomery form because Montgomery is
// a bijection that respects the additive structure.  Carry from the +n
// is captured into the new top bit during the right shift.
template<int N>
inline fixint::Mod<N> mont_halve(const fixint::Mod<N>& a) {
    using namespace fixint;
    Mod<N> r = a;
    uint8_t cy = 0;
    if (r.v.d[0] & 1u) {
        cy = mpn::add<N>(r.v.d, r.v.d, ctx<N>().mod.d);
    }
    mpn::rshift<N>(r.v.d, r.v.d, 1);
    if (cy) r.v.d[N - 1] |= (uint64_t(1) << 63);
    return r;
}

// Convert a small signed integer to Mod<N> mod n.
//
// Reduces |v| modulo n first so this is safe even for tiny n (e.g. n=5
// with v=-7, where the naive "n - |v|" would underflow).  For the common
// case (n large, |v| < 100) the reduction is a no-op at runtime.
template<int N>
inline fixint::Mod<N> mod_from_signed_small(int64_t v, const fixint::UInt<N>& n) {
    using namespace fixint;
    uint64_t abs_v = static_cast<uint64_t>(v >= 0 ? v : -v);

    // |v| mod n as a single u64.  If n occupies more than one limb,
    // |v| < 100 is trivially smaller than n.
    bool n_single_limb = true;
    for (int i = 1; i < N; ++i) if (n.d[i] != 0) { n_single_limb = false; break; }
    uint64_t abs_mod = (n_single_limb && n.d[0] <= abs_v) ? (abs_v % n.d[0]) : abs_v;

    UInt<N> u{};
    if (abs_mod == 0) {
        // v ≡ 0 (mod n)
    } else if (v >= 0) {
        u.d[0] = abs_mod;
    } else {
        UInt<N> abs_uint{};
        abs_uint.d[0] = abs_mod;
        mpn::sub<N>(u.d, n.d, abs_uint.d);
    }
    return Mod<N>::from_uint(u);
}

// Mont(0) — special construction since Mod<N>::from_uint would call REDC
// on zero, which is fine but the all-zero limb pattern is the answer
// directly: zero is its own Montgomery form.
template<int N>
inline fixint::Mod<N> mont_zero() {
    fixint::Mod<N> z;
    fixint::mpn::set_zero<N>(z.v.d);
    return z;
}

// Test bit i of a multi-limb value.
template<int N>
inline bool test_bit(const fixint::UInt<N>& v, unsigned i) {
    return (v.d[i / 64] >> (i % 64)) & 1u;
}

// Strong Miller-Rabin with the single witness a = 2.
// Caller must have set up the Montgomery context for n; n must be odd >= 5.
template<int N>
inline bool strong_mr_base2(const fixint::UInt<N>& n) {
    using namespace fixint;

    // n - 1 = d_odd * 2^s
    UInt<N> n_minus_1 = n;
    mpn::sub1<N>(n_minus_1.d, n_minus_1.d, 1);
    unsigned s = mpn::ctz<N>(n_minus_1.d);
    UInt<N> d_odd = n_minus_1;
    mpn::rshift<N>(d_odd.d, d_odd.d, s);

    Mod<N> mont_one     = Mod<N>::one();
    Mod<N> mont_neg_one = Mod<N>::from_uint(n_minus_1);
    Mod<N> mont_two     = Mod<N>::from_uint(UInt<N>(2));

    Mod<N> x = pow<N>(mont_two, d_odd);

    if (x == mont_one || x == mont_neg_one) return true;
    for (unsigned i = 1; i < s; ++i) {
        x = x.sqr();
        if (x == mont_neg_one) return true;
        if (x == mont_one)     return false;  // composite
    }
    return false;
}

// Strong Lucas test with Selfridge (P, Q) parameters.
// Caller guarantees odd n >= 5 and Montgomery context set up.
template<int N>
inline bool strong_lucas_selfridge(const fixint::UInt<N>& n) {
    using namespace fixint;

    // ---- Selfridge: find smallest D in {5, -7, 9, -11, 13, -15, ...}
    // with jacobi(D, n) = -1.
    //
    // We fully reduce |D| mod n before building D_uint, so that
    // D_uint.is_zero() correctly means "n divides |D|" (i.e. n is a small
    // prime less than or equal to |D|).  Without this reduction, small n
    // like 3 or 5 would misclassify D = n as "jacobi == 0 with D_uint != 0".
    //
    // When jacobi == 0 and D_uint != 0, n shares a non-trivial factor
    // with the small integer |D mod n| (< 100), which (after trivial
    // sieving) proves n composite.
    bool n_single_limb = true;
    for (int i = 1; i < N; ++i) if (n.d[i] != 0) { n_single_limb = false; break; }

    int64_t D = 5;
    int sign = 1;
    bool found = false;
    for (int attempt = 0; attempt < 100; ++attempt, sign = -sign) {
        D = sign * (5 + 2 * attempt);
        uint64_t abs_D = static_cast<uint64_t>(D >= 0 ? D : -D);

        // |D mod n| as a single u64.  If n is huge (> one limb) then
        // |D| < 100 < n so no reduction needed.  If n fits in one limb,
        // reduce against n.d[0].
        uint64_t abs_D_mod_n = (n_single_limb && n.d[0] <= abs_D)
                                   ? (abs_D % n.d[0])
                                   : abs_D;

        if (abs_D_mod_n == 0) continue;  // n | |D| → n is a small prime, try next D

        UInt<N> D_uint{};
        if (D >= 0) {
            D_uint.d[0] = abs_D_mod_n;
        } else {
            UInt<N> abs_v{};
            abs_v.d[0] = abs_D_mod_n;
            mpn::sub<N>(D_uint.d, n.d, abs_v.d);
        }

        int j = jacobi<N>(D_uint, n);
        if (j == -1) { found = true; break; }
        if (j == 0)  return false;
    }
    if (!found) {
        // Failed Selfridge after 100 attempts — almost certainly a perfect
        // square (squares foil the Selfridge search).  Report composite.
        return false;
    }

    // ---- P = 1, Q = (1 - D) / 4 ----
    int64_t Q = (1 - D) / 4;

    // ---- n + 1 = d_lucas * 2^s ----
    UInt<N> n_plus_1;
    {
        UInt<N> one(1);
        mpn::add<N>(n_plus_1.d, n.d, one.d);
        // n + 1 cannot overflow N limbs here: if n = 2^(64N) - 1 then n is
        // odd and would have been caught by an earlier MR step or trivial
        // sieving (in particular such an n is divisible by 3 for N >= 1).
    }
    unsigned s_lucas = mpn::ctz<N>(n_plus_1.d);
    UInt<N> d_lucas = n_plus_1;
    mpn::rshift<N>(d_lucas.d, d_lucas.d, s_lucas);

    // ---- Lucas chain: compute (U_d, V_d, Q^d) mod n with P = 1 ----
    Mod<N> U      = Mod<N>::one();                          // U_1
    Mod<N> V      = Mod<N>::one();                          // V_1 = P = 1
    Mod<N> Qk     = mod_from_signed_small<N>(Q, n);         // Q^1
    Mod<N> Q_mont = Qk;                                     // Q (constant)
    Mod<N> D_mont = mod_from_signed_small<N>(D, n);
    Mod<N> mont_two = Mod<N>::from_uint(UInt<N>(2));

    unsigned bits = mpn::bit_length<N>(d_lucas.d);
    for (int i = int(bits) - 2; i >= 0; --i) {
        // Double: (U_k, V_k, Q^k) -> (U_{2k}, V_{2k}, Q^{2k})
        U = U * V;
        V = V.sqr() - mont_two * Qk;
        Qk = Qk.sqr();

        if (test_bit<N>(d_lucas, unsigned(i))) {
            // Increment by 1: (U_{2k}, V_{2k}, Q^{2k}) -> (U_{2k+1}, V_{2k+1}, Q^{2k+1})
            // P = 1, so U' = (U + V) / 2, V' = (D·U + V) / 2
            Mod<N> U_old = U;
            U = mont_halve(U + V);
            V = mont_halve(D_mont * U_old + V);
            Qk = Qk * Q_mont;
        }
    }

    // ---- Strong Lucas check ----
    Mod<N> zero = mont_zero<N>();

    if (U == zero) return true;
    if (V == zero) return true;
    for (unsigned k = 1; k < s_lucas; ++k) {
        // V_{2k}, Q^{2k} doubling
        V  = V.sqr() - mont_two * Qk;
        Qk = Qk.sqr();
        if (V == zero) return true;
    }
    return false;
}

} // namespace detail_bpsw

// BPSW core — runs strong MR base 2 followed by strong Lucas with
// Selfridge parameters.  Assumes:
//   * n is odd and >= 3
//   * the caller has set up a Montgomery context for n (via MontScope)
//
// This is the right entry point when the caller is already doing
// Montgomery arithmetic for the same modulus (e.g., a Pollard rho or
// ECM loop that wants to check whether a cofactor is prime without
// tearing down its own context).
template<int N>
inline bool bpsw_with_ctx(const fixint::UInt<N>& n) {
    if (!detail_bpsw::strong_mr_base2<N>(n)) return false;
    if (!detail_bpsw::strong_lucas_selfridge<N>(n)) return false;
    return true;
}

// BPSW probable-prime test.  Returns true if n is probably prime, false
// if n is definitely composite.  No counterexamples known below ~2^64;
// for n < 2^1024 the test is treated as deterministic in practice.
//
// Sets up its own Montgomery context — prefer bpsw_with_ctx if you
// already have an active MontScope for n.
template<int N>
inline bool bpsw(const fixint::UInt<N>& n) {
    using namespace fixint;

    // Trivial cases.  These must be handled before MontCtx::init since
    // the Montgomery setup requires odd n > 1.
    if (n.is_zero()) return false;
    {
        UInt<N> one(1);
        if (n == one) return false;
    }
    if (n == UInt<N>(2)) return true;
    if ((n.d[0] & 1u) == 0u) return false;  // even and != 2

    MontCtx<N> mctx;
    mctx.init(n);
    MontScope<N> scope(mctx);

    return bpsw_with_ctx<N>(n);
}

} // namespace zfactor
