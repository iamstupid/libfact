// Smoke test for the Z/6 curve setup.  Sets up a curve mod a small prime,
// verifies P0 lies on the resulting curve.
#include "zfactor/eecm/curve_setup.h"
#include "zfactor/fixint/montgomery.h"
#include "zfactor/fixint/modular.h"
#include <cstdio>
#include <cstdint>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
bool point_on_curve(const EdPoint<N>& P, const EdCurve<N>& curve,
                    const MontCtx<N>& c) {
    auto m = mont(c);
    // -X^2 + Y^2 == Z^2 + d * T^2  ?
    auto lhs = m.sub(m.sqr(P.Y), m.sqr(P.X));
    auto rhs = m.add(m.sqr(P.Z), m.mul(curve.d, m.sqr(P.T)));
    auto diff = m.sub(lhs, rhs);
    return diff.is_zero();
}

template<int N>
bool t_consistent(const EdPoint<N>& P, const MontCtx<N>& c) {
    // T*Z == X*Y for extended-coords points.
    auto m = mont(c);
    auto lhs = m.mul(P.T, P.Z);
    auto rhs = m.mul(P.X, P.Y);
    return m.sub(lhs, rhs).is_zero();
}

template<int N>
void test_one(uint64_t prime, uint64_t k) {
    UInt<N> mod{};
    mod.d[0] = prime;
    MontCtx<N> ctx;
    ctx.init(mod);

    auto res = setup_curve<N>(k, ctx);
    if (res.factor_found) {
        std::printf("N=%d p=%llu k=%llu: factor_found=%llu\n",
                    N, (unsigned long long)prime, (unsigned long long)k,
                    (unsigned long long)res.factor.d[0]);
        return;
    }
    bool oc = point_on_curve(res.P0, res.curve, ctx);
    bool tc = t_consistent(res.P0, ctx);
    std::printf("N=%d p=%llu k=%llu: on_curve=%d t_ok=%d\n",
                N, (unsigned long long)prime, (unsigned long long)k,
                int(oc), int(tc));
}

template<int N>
void test_dbl_add(uint64_t prime, uint64_t k) {
    UInt<N> mod{};
    mod.d[0] = prime;
    MontCtx<N> ctx;
    ctx.init(mod);

    auto res = setup_curve<N>(k, ctx);
    if (res.factor_found) return;
    auto P = res.P0;
    auto& curve = res.curve;

    // 2P should be on the curve, and T_2P * Z_2P = X_2P * Y_2P
    auto P2 = ed_dbl<N>(P, ctx);
    bool oc = point_on_curve(P2, curve, ctx);
    bool tc = t_consistent(P2, ctx);

    // 3P = 2P + P should also be on the curve
    auto P3 = ed_add<N>(P2, P, curve, ctx);
    bool oc3 = point_on_curve(P3, curve, ctx);
    bool tc3 = t_consistent(P3, ctx);

    // 4P via two doublings vs 2P + 2P
    auto P4a = ed_dbl<N>(P2, ctx);
    auto P4b = ed_add<N>(P2, P2, curve, ctx);
    // They should be projectively equal: X_a * Z_b == X_b * Z_a (and same for Y)
    auto m = mont(ctx);
    bool eq = m.sub(m.mul(P4a.X, P4b.Z), m.mul(P4b.X, P4a.Z)).is_zero()
           && m.sub(m.mul(P4a.Y, P4b.Z), m.mul(P4b.Y, P4a.Z)).is_zero();

    std::printf("N=%d p=%llu k=%llu: 2P on=%d t=%d  3P on=%d t=%d  4P_dbldbl==4P_addsame=%d\n",
                N, (unsigned long long)prime, (unsigned long long)k,
                int(oc), int(tc), int(oc3), int(tc3), int(eq));
}

int main() {
    // Small primes to keep things tractable.
    for (uint64_t k : {1, 2, 3, 5, 7, 11}) {
        test_one<1>(1000003ULL, k);
        test_one<1>(1000033ULL, k);
        test_one<1>(99999999999999997ULL, k);
    }
    std::printf("\n--- doubling/addition tests ---\n");
    for (uint64_t k : {1, 2, 3, 5, 7, 11}) {
        test_dbl_add<1>(1000003ULL, k);
        test_dbl_add<1>(99999999999999997ULL, k);
    }
    // 256-bit prime: 2^256 - 189
    {
        UInt<4> mod{};
        for (int i = 0; i < 4; ++i) mod.d[i] = ~uint64_t(0);
        mod.d[0] -= 188;
        MontCtx<4> ctx;
        ctx.init(mod);
        for (uint64_t k : {1, 2, 3, 5, 7, 11}) {
            auto res = setup_curve<4>(k, ctx);
            if (res.factor_found) {
                std::printf("N=4 256bit k=%llu: factor_found\n",
                            (unsigned long long)k);
                continue;
            }
            bool oc = point_on_curve(res.P0, res.curve, ctx);
            bool tc = t_consistent(res.P0, ctx);
            std::printf("N=4 256bit k=%llu: on_curve=%d t_ok=%d\n",
                        (unsigned long long)k, int(oc), int(tc));
        }
    }
    return 0;
}
