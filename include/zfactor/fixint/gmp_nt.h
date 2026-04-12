// GMP-backed number theory: gcd, jacobi, modinv for multi-limb UInt<N>.
// ~2-5× faster than our pure C++ implementations at N=2..4.
// Single-word (_u64) versions stay in gcd.h / jacobi.h (already fast).
#pragma once

#include <gmp.h>
#include <cstring>
#include "uint.h"

namespace zfactor::fixint {

namespace detail_gmp_nt {

// UInt<N> → mpz (no allocation if fits in stack mpz)
template<int N>
inline void to_mpz(mpz_t z, const UInt<N>& x) {
    mpz_import(z, N, -1, 8, 0, 0, x.d);
}

// mpz → UInt<N>
template<int N>
inline void from_mpz(UInt<N>& x, const mpz_t z) {
    std::memset(x.d, 0, sizeof(x.d));
    size_t count;
    mpz_export(x.d, &count, -1, 8, 0, 0, z);
}

}  // namespace detail_gmp_nt


// GCD via mpz_gcd.
template<int N>
inline UInt<N> gcd(const UInt<N>& a, const UInt<N>& b) {
    mpz_t ma, mb, mg;
    mpz_init(ma); mpz_init(mb); mpz_init(mg);
    detail_gmp_nt::to_mpz<N>(ma, a);
    detail_gmp_nt::to_mpz<N>(mb, b);
    mpz_gcd(mg, ma, mb);
    UInt<N> r;
    detail_gmp_nt::from_mpz<N>(r, mg);
    mpz_clear(ma); mpz_clear(mb); mpz_clear(mg);
    return r;
}


// Jacobi symbol is provided directly in jacobi.h (zfactor:: namespace).

// Modular inverse via mpz_invert.
// Returns 1 with inverse if gcd(a,m)=1, 0 with gcd(a,m) otherwise.
// Same interface as the old modinv<N>().
template<int N>
inline int modinv(UInt<N>* inv_or_factor, const UInt<N>& a, const UInt<N>& m) {
    mpz_t ma, mm, mr;
    mpz_init(ma); mpz_init(mm); mpz_init(mr);
    detail_gmp_nt::to_mpz<N>(ma, a);
    detail_gmp_nt::to_mpz<N>(mm, m);
    int ok = mpz_invert(mr, ma, mm);
    if (ok) {
        detail_gmp_nt::from_mpz<N>(*inv_or_factor, mr);
    } else {
        // gcd(a, m) > 1 — return the gcd as the factor
        mpz_gcd(mr, ma, mm);
        detail_gmp_nt::from_mpz<N>(*inv_or_factor, mr);
    }
    mpz_clear(ma); mpz_clear(mm); mpz_clear(mr);
    return ok ? 1 : 0;
}

}  // namespace zfactor::fixint
