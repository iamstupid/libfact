// Diagnostic: run many curves against a known target and report which
// stage (1 or 2) finds the factor.  Helps tell whether stage 2 is broken
// or just statistically unlucky.
#include "zfactor/eecm.h"
#include "zfactor/eecm/scalar_mult.h"
#include <cstdio>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

int main() {
    // Use a smaller composite where success per curve is high so we can
    // tell stage 1 vs stage 2 hit rates clearly.
    // 1000003 is a 20-bit prime; pair with 10000019 (24-bit).
    UInt<1> n;
    n.d[0] = 1000003ULL * 10000019ULL;
    MontCtx<1> ctx;
    ctx.init(n);

    int B1_target_bits = 80;    // → B1=48700, B2=14M
    const auto& row = schedule_for_bits(B1_target_bits);
    std::printf("Target: 31-bit factor in n=%llu, B1=%llu, B2=%llu\n",
                (unsigned long long)n.d[0], (unsigned long long)row.B1,
                (unsigned long long)row.B2);

    int s1_hits = 0, s2_hits = 0, none = 0;
    int max_curves = 100;

    for (int i = 0; i < max_curves; ++i) {
        uint64_t k = 2 + i;
        auto cs = setup_curve<1>(k, ctx);
        if (cs.factor_found) { std::printf("k=%llu: setup factor!\n", (unsigned long long)k); continue; }

        const auto& chain = get_stage1_chain(row.B1);
        std::vector<EdPoint<1>> precomp, precomp_neg;
        build_wnaf_precomp<1>(precomp, precomp_neg, cs.P0, chain.w, cs.curve, ctx);
        auto Q = scalar_mult_wnaf<1>(chain.wnaf, precomp, precomp_neg, cs.curve, ctx);

        auto m = mont(ctx);
        auto Z_plain = m.drop(Q.Z);
        if (Z_plain.is_zero()) Z_plain = ctx.mod;
        auto g = zfactor::fixint::gcd<1>(Z_plain, ctx.mod);
        bool s1_hit = !g.is_zero() && g.d[0] > 1 && g.d[0] < n.d[0];
        if (s1_hit) {
            s1_hits++;
            std::printf("k=%llu: STAGE1 factor=%llu\n", (unsigned long long)k, (unsigned long long)g.d[0]);
            continue;
        }

        // Run stage 2 against Q.
        auto s2 = ecm_stage2<1>(Q, cs.curve, row.B1, row.B2, ctx);
        if (s2.factor_found) {
            s2_hits++;
            std::printf("k=%llu: STAGE2 factor=%llu\n", (unsigned long long)k, (unsigned long long)s2.factor.d[0]);
        } else {
            none++;
        }
    }
    std::printf("\nSummary: stage1=%d  stage2=%d  none=%d / %d curves\n",
                s1_hits, s2_hits, none, max_curves);

    // Cross-check via the high-level entry point.
    std::printf("\n--- via ecm<1>() ---\n");
    auto r = ecm<1>(n, B1_target_bits, max_curves);
    if (r) std::printf("HL: factor=%llu\n", (unsigned long long)r->d[0]);
    else   std::printf("HL: no factor\n");

    return 0;
}
