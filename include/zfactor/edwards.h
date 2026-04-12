// Twisted Edwards curve arithmetic for ECM.
//
// Curve form:  -X^2 + Y^2 = Z^2 + d * T^2,    T = XY/Z   (extended coords)
//              i.e. a = -1 in the BBLP08/HWCD08 sense.
//
// All operations are in Montgomery form mod n where n is the modulus we are
// trying to factor.  The curve coefficient is stored as `k = 2*d` (already in
// Montgomery form) so the addition formula folds the constant in.
//
// Reference cost (per the EFD):
//   dbl-2008-hwcd  (any a, here a=-1): 4M + 4S + 6add
//   add-2008-hwcd-4 (a=-1, with k=2d): 8M + 8add
//
// For ECM we never need a unified-completeness guarantee — we doubt and add
// repeatedly along the scalar chain, and at the end we test gcd(Z, n).  If
// any intermediate point has Z=0 we just hit a stage-1 factor.
#pragma once

#include "zfactor/fixint/montgomery.h"
#include "zfactor/fixint/modular.h"

namespace zfactor::ecm {

// Pick the right MontOps flavour for the curve arithmetic.  At low N (1..3)
// the inline-asm montmul kernel allocates fine inside ed_add even with
// 8+ live UInt<N> locals.  At N>=4 the kernel needs more GPRs than the
// surrounding context leaves free, so we route through `mont_out` which
// turns each montmul into a noinline call (~3-cycle overhead).
template<int N>
[[gnu::always_inline]]
inline auto edwards_mont_ops(const fixint::MontCtx<N>& c) {
    if constexpr (N <= 2) return fixint::mont(c);
    else                  return fixint::mont_out(c);
}

template<int N>
struct EdPoint {
    fixint::UInt<N> X, Y, Z, T;
};

// Curve constants: k = 2d in Montgomery form.  We carry the un-doubled d
// alongside in case we need it for re-derivation, but the hot path uses k.
template<int N>
struct EdCurve {
    fixint::UInt<N> k;   // 2 * d  (Montgomery)
    fixint::UInt<N> d;   // d      (Montgomery)
};

// dbl-2008-hwcd, a = -1 specialisation.
//
//   A = X1^2
//   B = Y1^2
//   C = 2 * Z1^2
//   D = -A                 (a = -1)
//   E = (X1+Y1)^2 - A - B
//   G = D + B
//   F = G - C
//   H = D - B
//   X3 = E * F
//   Y3 = G * H
//   T3 = E * H
//   Z3 = F * G
//
// Cost: 4M + 4S.
template<int N>
[[gnu::always_inline]]
inline EdPoint<N> ed_dbl(const EdPoint<N>& P,
                         const fixint::MontCtx<N>& c) {
    auto m = edwards_mont_ops<N>(c);
    auto A = m.sqr(P.X);
    auto B = m.sqr(P.Y);
    auto ZZ = m.sqr(P.Z);
    auto C = m.add(ZZ, ZZ);                      // 2 * Z1^2
    auto E = m.sub(m.sub(m.sqr(m.add(P.X, P.Y)), A), B);
    auto G = m.sub(B, A);                       // D + B with D=-A → B - A
    auto F = m.sub(G, C);
    auto H = m.sub(m.zero(), m.add(A, B));      // D - B with D=-A → -(A+B)
    EdPoint<N> R;
    R.X = m.mul(E, F);
    R.Y = m.mul(G, H);
    R.T = m.mul(E, H);
    R.Z = m.mul(F, G);
    return R;
}

// add-2008-hwcd-4, a = -1 specialisation, with k = 2*d folded in.
//
//   A = (Y1-X1) * (Y2-X2)
//   B = (Y1+X1) * (Y2+X2)
//   C = T1 * k * T2          (k = 2d)
//   D = Z1 * (2 * Z2)
//   E = B - A
//   F = D - C
//   G = D + C
//   H = B + A
//   X3 = E * F
//   Y3 = G * H
//   T3 = E * H
//   Z3 = F * G
//
// Cost: 8M.
template<int N>
[[gnu::always_inline]]
inline EdPoint<N> ed_add(const EdPoint<N>& P, const EdPoint<N>& Q,
                         const EdCurve<N>& curve,
                         const fixint::MontCtx<N>& c) {
    auto m = edwards_mont_ops<N>(c);
    auto A = m.mul(m.sub(P.Y, P.X), m.sub(Q.Y, Q.X));
    auto B = m.mul(m.add(P.Y, P.X), m.add(Q.Y, Q.X));
    auto C = m.mul(m.mul(P.T, curve.k), Q.T);
    auto D = m.mul(P.Z, m.add(Q.Z, Q.Z));
    auto E = m.sub(B, A);
    auto F = m.sub(D, C);
    auto G = m.add(D, C);
    auto H = m.add(B, A);
    EdPoint<N> R;
    R.X = m.mul(E, F);
    R.Y = m.mul(G, H);
    R.T = m.mul(E, H);
    R.Z = m.mul(F, G);
    return R;
}

// Negation: (X, Y, Z, T) → (-X, Y, Z, -T) on twisted Edwards.
template<int N>
[[gnu::always_inline]]
inline EdPoint<N> ed_neg(const EdPoint<N>& P,
                         const fixint::MontCtx<N>& c) {
    auto m = edwards_mont_ops<N>(c);
    EdPoint<N> R = P;
    R.X = m.sub(m.zero(), P.X);
    R.T = m.sub(m.zero(), P.T);
    return R;
}

}  // namespace zfactor::ecm
