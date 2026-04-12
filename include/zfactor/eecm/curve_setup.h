// Z/6Z-rational-torsion twisted Edwards curve generator.
//
// Ported from cado-nfs sieve/ecm/ec_parameterization.h::ec_parameterization_Z6
// (which references "Starfish on Strike" by Bernstein, Birkner, Lange, Peters,
// 2008).
//
// Workflow:
//   1. The "outer" curve is the integer Weierstrass curve  y^2 = x^3 - 9747 x + 285714
//      which has rank 1, generator P = (15, 378), and three 2-torsion points.
//   2. For parameter k = 1, 2, 3, ..., compute Q := [k]*P on the outer curve
//      using projective Weierstrass arithmetic (no inversions).  Result is
//      (T->x : T->y : T->z) over Z/nZ.
//   3. Plug (T->x, T->y, T->z) into the BBLP/SOS formulas to obtain a twisted
//      Edwards curve  -X^2 + Y^2 = Z^2 + d T^2  with rational Z/6 torsion,
//      together with a non-torsion base point (xE0:yE0:zE0:tE0).
//
// Two failure modes during setup:
//   * The Weierstrass smul hits a non-invertible doubling/addition denominator
//     mod n.  That denominator is a non-trivial factor of n -- we return it
//     immediately as a "stage-0 hit".
//   * The final divide-out at the end (16*u3*u5*u7*u2 has no inverse) likewise
//     is reported as a hit.  cado-nfs treats both the same way.
//
// All arithmetic is done in Montgomery form via MontOps<N>; only the
// final modular inverse is on raw integers.
#pragma once

#include "zfactor/edwards.h"
#include "zfactor/fixint/montgomery.h"
#include "zfactor/fixint/modular.h"
#include "zfactor/fixint/gcd.h"

namespace zfactor::ecm {

// Result of trying to set up a curve from parameter k:
//   factor_found = true  → *factor is a non-trivial factor of n (lucky!)
//   factor_found = false → P0 and curve are valid Edwards curve+point
template<int N>
struct CurveSetupResult {
    EdPoint<N> P0;
    EdCurve<N> curve;
    fixint::UInt<N> factor;
    bool factor_found;
};

namespace detail {

// Projective Weierstrass point on y^2*z = x^3 + a*x*z^2 + b*z^3.
template<int N>
struct WPoint {
    fixint::UInt<N> x, y, z;
};

// Doubling: standard projective formulas (Cohen 13.2.1.b), 7M+5S.  See
// cado-nfs `weierstrass_proj_dbl`.  Cost is irrelevant — once per curve.
template<int N>
inline WPoint<N> w_dbl(const WPoint<N>& P, const fixint::UInt<N>& a_mont,
                       const fixint::MontCtx<N>& c) {
    auto m = fixint::mont_slow(c);
    if (P.y.is_zero()) {
        // 2*P = O.  Represent O as (0:1:0).
        WPoint<N> O{m.zero(), m.one(), m.zero()};
        return O;
    }
    auto XX = m.sqr(P.x);
    auto ZZ = m.sqr(P.z);
    auto w = m.add(m.add(XX, XX), XX);              // 3*X^2
    w = m.add(w, m.mul(a_mont, ZZ));                 // 3*X^2 + a*Z^2
    auto s = m.mul(P.y, P.z);                        // s = Y*Z
    s = m.add(s, s);                                 // s = 2*Y*Z
    auto ss = m.sqr(s);
    auto sss = m.mul(s, ss);
    auto R = m.mul(P.y, s);                          // R = Y*s = 2*Y^2*Z
    auto RR = m.sqr(R);
    auto B = m.sub(m.sqr(m.add(P.x, R)), m.add(XX, RR));  // (X+R)^2 - X^2 - R^2 = 2XR
    auto h = m.sub(m.sqr(w), m.add(B, B));           // h = w^2 - 2*B
    WPoint<N> Q;
    Q.x = m.mul(s, h);                               // X3 = s*h
    Q.y = m.sub(m.mul(w, m.sub(B, h)), m.add(RR, RR)); // Y3 = w*(B - h) - 2*R^2
    Q.z = sss;
    return Q;
}

// Addition Q = P + R, projective Weierstrass.  Cohen 13.2.1.a, 12M+2S.
// Special cases (one of P,R is O, P == R, P == -R) are handled.
template<int N>
inline WPoint<N> w_add(const WPoint<N>& P, const WPoint<N>& R,
                       const fixint::UInt<N>& a_mont,
                       const fixint::MontCtx<N>& c) {
    auto m = fixint::mont_slow(c);
    if (P.z.is_zero()) return R;
    if (R.z.is_zero()) return P;
    auto Y1Z2 = m.mul(P.y, R.z);
    auto X1Z2 = m.mul(P.x, R.z);
    auto Z1Z2 = m.mul(P.z, R.z);
    auto u    = m.sub(m.mul(R.y, P.z), Y1Z2);
    auto v    = m.sub(m.mul(R.x, P.z), X1Z2);
    if (v.is_zero()) {
        // Same x-coord.  Either P = R (so do dbl) or P = -R (so result is O).
        if (u.is_zero()) return w_dbl(P, a_mont, c);
        WPoint<N> O{m.zero(), m.one(), m.zero()};
        return O;
    }
    auto uu = m.sqr(u);
    auto vv = m.sqr(v);
    auto vvv = m.mul(v, vv);
    auto Rr = m.mul(vv, X1Z2);
    auto A = m.sub(m.sub(m.mul(uu, Z1Z2), vvv), m.add(Rr, Rr));
    WPoint<N> S;
    S.x = m.mul(v, A);
    S.y = m.sub(m.mul(u, m.sub(Rr, A)), m.mul(vvv, Y1Z2));
    S.z = m.mul(vvv, Z1Z2);
    return S;
}

// Binary scalar multiplication on the projective Weierstrass curve.
// k must be a positive integer < 2^64.
template<int N>
inline WPoint<N> w_smul(WPoint<N> P, uint64_t k, const fixint::UInt<N>& a_mont,
                        const fixint::MontCtx<N>& c) {
    auto m = fixint::mont_slow(c);
    WPoint<N> R{m.zero(), m.one(), m.zero()};   // identity
    while (k) {
        if (k & 1) R = w_add(R, P, a_mont, c);
        k >>= 1;
        if (k) P = w_dbl(P, a_mont, c);
    }
    return R;
}

}  // namespace detail

// Try to set up a Z/6-torsion twisted Edwards curve from parameter k.
// Returns either a valid (P0, curve) pair or a non-trivial factor of n.
//
// k must be >= 1 (k = 0 is invalid per cado-nfs).  Recommended starting
// point is k = 1 followed by k = 2, 3, ... (cado-nfs notes that k = 1
// gives a special curve isomorphic to Brent-Suyama sigma=11).
template<int N>
inline CurveSetupResult<N> setup_curve(uint64_t k, const fixint::MontCtx<N>& c) {
    auto m = fixint::mont_slow(c);
    CurveSetupResult<N> out;
    out.factor_found = false;

    // Outer curve coefficient a = -9747 in Montgomery form.
    // We compute this each call (cheap relative to the smul that follows).
    fixint::UInt<N> a_plain;  // = (n - 9747) mod n  (i.e. -9747 mod n)
    fixint::UInt<N> tmp_9747(9747);
    {
        // a_plain = n - 9747  (assumes n > 9747, true for any meaningful ECM target)
        fixint::mpn::sub<N>(a_plain.d, c.mod.d, tmp_9747.d);
    }
    auto a_mont = m.lift(a_plain);

    // P = (15, 378, 1) in Montgomery form.
    detail::WPoint<N> P{m.lift(fixint::UInt<N>(15)),
                        m.lift(fixint::UInt<N>(378)),
                        m.one()};

    // T <- [k]*P.
    auto T = detail::w_smul(P, k, a_mont, c);

    if (T.z.is_zero()) {
        // [k]*P is the point at infinity — bad parameter, gcd would just be n.
        // cado-nfs treats this as "non-trivial gcd = n", caller should retry.
        out.factor = c.mod;
        out.factor_found = true;
        return out;
    }

    // From here on, we follow the formulas in cado-nfs verbatim.
    auto Tx = T.x, Ty = T.y, Tz = T.z;

    // U = 144 * (Tx + 3*Tz)
    auto u0 = m.lift(fixint::UInt<N>(144));
    auto U  = m.add(Tx, m.add(m.add(Tz, Tz), Tz));
    U = m.mul(U, u0);
    auto V = Ty;
    auto W = m.mul(m.lift(fixint::UInt<N>(2985984)), Tz);  // 12^6 * Tz

    // u0 = 96 * U
    u0 = m.lift(fixint::UInt<N>(96));
    u0 = m.mul(u0, U);

    auto u1 = m.sub(W, u0);          // u1 = W - 96*U
    auto u2 = m.sqr(u1);             // u1^2
    auto u3 = m.sqr(u0);             // u0^2

    // u4 = u2 - 5*u3
    auto five_u3 = m.add(m.add(u3, u3), m.add(u3, u3));  // 4*u3
    five_u3 = m.add(five_u3, u3);                        // 5*u3
    auto u4 = m.sub(u2, five_u3);

    auto u5 = m.mul(u4, m.sqr(u4));  // u4^3

    // u7 = 4*u1
    auto u7 = m.add(u1, u1);
    u7 = m.add(u7, u7);

    // ---- Edwards-form output ----
    // T->x = (u1 - u0) * (u1 + 5*u0) * (u2 + 5*u3)
    auto t1 = m.sub(u1, u0);                            // u1 - u0
    auto five_u0 = m.add(m.add(u0, u0), m.add(u0, u0)); // 4*u0
    five_u0 = m.add(five_u0, u0);
    auto t2 = m.add(u1, five_u0);                       // u1 + 5*u0
    auto Tedx = m.mul(t1, t2);
    auto t3 = m.add(u2, five_u3);                       // u2 + 5*u3
    Tedx = m.mul(Tedx, t3);

    // T->y = (u2 - 5*u3)^3   = u4^3 = u5
    auto Tedy = u5;

    // T->t = (u7*u0)^3
    auto u7u0 = m.mul(u7, u0);
    auto Tedt = m.mul(u7u0, m.sqr(u7u0));

    // u6 = Tedx * U^2
    auto u6 = m.mul(Tedx, m.sqr(U));

    // u2 = u0^3       (overwrite — this is our 'cube' variable)
    auto u0cube = m.mul(u3, u0);

    // u1 = 2 * u1 * u2 * V * W
    auto u1_new = m.mul(u1, u0cube);
    u1_new = m.mul(u1_new, V);
    u1_new = m.mul(u1_new, W);
    u1_new = m.add(u1_new, u1_new);

    // P0->z = (Tedy + Tedt) * u6
    // P0->x = (Tedy + Tedt) * u1_new
    // P0->y = (Tedy - Tedt) * u6
    // P0->t = (Tedy - Tedt) * u1_new
    auto sum_yt  = m.add(Tedy, Tedt);
    auto diff_yt = m.sub(Tedy, Tedt);
    out.P0.Z = m.mul(sum_yt,  u6);
    out.P0.X = m.mul(sum_yt,  u1_new);
    out.P0.Y = m.mul(diff_yt, u6);
    out.P0.T = m.mul(diff_yt, u1_new);

    // ---- Curve coefficient d ----
    //
    // Recover d directly from the constraint that P0 lies on the curve:
    //     -X^2 + Y^2 = Z^2 + d * T^2
    // ⇒  d = (Y^2 - X^2 - Z^2) / T^2.
    //
    // This sidesteps re-deriving d from the (u0..u7) intermediates -- the
    // cado-nfs sage block gives a formula in terms of α=u4/u3, β=u7/u0 and
    // r=V*W/U^2 that needs careful denominator-clearing.  Going through the
    // curve equation is one extra inverse but is bulletproof: by definition,
    // any valid (X,Y,Z,T) on the curve uniquely determines d when T ≠ 0.
    //
    // T = 0 means our base point hit a singular case (P0 is 2-torsion or the
    // identity); treat that as a parameter we should skip rather than crash.
    auto Y2 = m.sqr(out.P0.Y);
    auto X2 = m.sqr(out.P0.X);
    auto Z2 = m.sqr(out.P0.Z);
    auto T2 = m.sqr(out.P0.T);
    auto d_num = m.sub(m.sub(Y2, X2), Z2);

    auto T2_plain = m.drop(T2);
    if (T2_plain.is_zero()) {
        out.factor = c.mod;
        out.factor_found = true;
        return out;
    }
    fixint::UInt<N> T2_inv;
    int ok = fixint::modinv<N>(&T2_inv, T2_plain, c.mod);
    if (!ok) {
        out.factor = T2_inv;
        out.factor_found = true;
        return out;
    }
    auto T2_inv_mont = m.lift(T2_inv);
    auto d_mont = m.mul(d_num, T2_inv_mont);

    out.curve.d = d_mont;
    out.curve.k = m.add(d_mont, d_mont);   // k = 2d, used by ed_add
    return out;
}

}  // namespace zfactor::ecm
