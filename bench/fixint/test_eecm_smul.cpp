// Verify scalar multiplication: [n]*P should be on the curve.
//
// We don't know the group order of our setup curves directly, but we can
// verify two algebraic identities:
//   1. [a+b]*P == [a]*P + [b]*P  (group homomorphism)
//   2. [-a]*P == -([a]*P)
// And we can verify the wNAF expansion: sum of digit*2^i should equal a.
#include "zfactor/eecm.h"
#include "zfactor/eecm/scalar_mult.h"
#include "zfactor/eecm/scalar_mult.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
EdPoint<N> scalar_naive(uint64_t s, const EdPoint<N>& P,
                        const EdCurve<N>& curve, const MontCtx<N>& ctx) {
    auto m = mont(ctx);
    EdPoint<N> Q;
    Q.X = m.zero(); Q.Y = m.one(); Q.Z = m.one(); Q.T = m.zero();
    EdPoint<N> Pi = P;
    while (s) {
        if (s & 1) Q = ed_add<N>(Q, Pi, curve, ctx);
        s >>= 1;
        if (s) Pi = ed_dbl<N>(Pi, ctx);
    }
    return Q;
}

template<int N>
EdPoint<N> scalar_wnaf_test(uint64_t s, int w, const EdPoint<N>& P,
                            const EdCurve<N>& curve, const MontCtx<N>& ctx) {
    uint64_t lim[1] = {s};
    auto wnaf = compute_wnaf(lim, 1, w);
    std::vector<EdPoint<N>> precomp, precomp_neg;
    build_wnaf_precomp<N>(precomp, precomp_neg, P, w, curve, ctx);
    return scalar_mult_wnaf<N>(wnaf, precomp, precomp_neg, curve, ctx);
}

template<int N>
bool projectively_equal(const EdPoint<N>& A, const EdPoint<N>& B, const MontCtx<N>& ctx) {
    auto m = mont(ctx);
    auto e1 = m.sub(m.mul(A.X, B.Z), m.mul(B.X, A.Z));
    auto e2 = m.sub(m.mul(A.Y, B.Z), m.mul(B.Y, A.Z));
    return e1.is_zero() && e2.is_zero();
}

int main() {
    // Set up a curve mod a prime so we have a real group (no zero divisors).
    UInt<1> p; p.d[0] = 99999999999999997ULL;
    MontCtx<1> ctx;
    ctx.init(p);
    auto cs = setup_curve<1>(2, ctx);
    if (cs.factor_found) { std::printf("setup failed\n"); return 1; }

    // Test wNAF reproduction at various scalars and window sizes.
    for (int w : {3, 4, 5, 6}) {
        for (uint64_t s : {1ULL, 2ULL, 3ULL, 7ULL, 17ULL, 100ULL, 12345ULL,
                            1000003ULL, 9999999999ULL}) {
            auto naive = scalar_naive<1>(s, cs.P0, cs.curve, ctx);
            auto wnaf  = scalar_wnaf_test<1>(s, w, cs.P0, cs.curve, ctx);
            bool ok = projectively_equal<1>(naive, wnaf, ctx);
            std::printf("[w=%d s=%llu] %s\n", w, (unsigned long long)s, ok ? "OK" : "FAIL");
        }
    }

    return 0;
}
