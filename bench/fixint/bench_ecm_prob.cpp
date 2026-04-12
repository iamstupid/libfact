// Measure actual per-curve success probability at 160-bit.
// Run many curves, count how many find the factor.
#include "zfactor/eecm.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
void test(const char* name, uint64_t* limbs, int total_bits, int ncurves) {
    UInt<N> n{};
    std::memcpy(n.d, limbs, N * 8);
    MontCtx<N> ctx;
    ctx.init(n);

    int fb = total_bits / 2;
    const auto& row = schedule_for_bits(fb);

    int found = 0;
    for (int k = 2; k < 2 + ncurves; ++k) {
        auto r = ecm_one_curve<N>((uint64_t)k, row.B1, row.B2, ctx);
        if (r.factor_found) found++;
    }
    double prob = (double)found / ncurves;
    double ec = found > 0 ? 1.0 / prob : 99999;
    std::printf("%s: %d/%d found (prob=%.4f ec=%.0f schedule=%u ratio=%.2f)\n",
                name, found, ncurves, prob, ec, row.avg_curves,
                row.avg_curves / ec);
}

int main() {
    {
        uint64_t l[] = {0x6078000000C62ECBULL, 0x0000000400000000ULL};
        test<2>("100bit", l, 100, 2000);
    }
    {
        uint64_t l[] = {0x8000000000B05D4FULL, 0x004000000000017BULL};
        test<2>("120bit", l, 120, 1000);
    }
    {
        uint64_t l[] = {0x0000000000B56CF9ULL, 0x0000000017BD0000ULL, 0x0000000040000000ULL};
        test<3>("160bit", l, 160, 1000);
    }
    return 0;
}
