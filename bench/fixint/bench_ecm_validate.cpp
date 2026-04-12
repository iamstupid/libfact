// Validate: factor test semiprimes with the new schedule.
// Reports curves used and wall time for each.
#include "zfactor/eecm.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

using namespace zfactor::fixint;
using namespace zfactor::ecm;
using Clock = std::chrono::steady_clock;

struct TestCase { int bits; int N; uint64_t limbs[4]; };
static const TestCase CASES[] = {
    // from gen_test_semiprimes.py
    { 60, 1, {0x040005E180AB6CABULL}},
    { 80, 2, {0x0017920000AD539BULL, 0x0000000000004000ULL}},
    {100, 2, {0x6078000000C62ECBULL, 0x0000000400000000ULL}},
    {120, 2, {0x8000000000B05D4FULL, 0x004000000000017BULL}},
    {140, 3, {0x0000000000B883A5ULL, 0x000000000005F8C0ULL, 0x0000000000000400ULL}},
    {160, 3, {0x0000000000B56CF9ULL, 0x0000000017BD0000ULL, 0x0000000040000000ULL}},
    {180, 3, {0x0000000000B2DD57ULL, 0x0000005EC0000000ULL, 0x0004000000000000ULL}},
    {200, 4, {0x0000000000B1D9E9ULL, 0x00017AD000000000ULL, 0x0000000000000000ULL, 0x0000000000000040ULL}},
};

template<int N>
void run_case(const TestCase& tc) {
    UInt<N> n{};
    std::memcpy(n.d, tc.limbs, N * 8);

    MontCtx<N> ctx;
    ctx.init(n);

    int fb = tc.bits / 2;  // known balanced semiprime
    const auto& row = schedule_for_bits(fb);
    int max_curves = (int)row.avg_curves * 4;  // give 4x budget

    uint64_t k_next = 2;
    auto t0 = Clock::now();

    for (int i = 0; i < max_curves; ++i) {
        auto r = ecm_one_curve<N>(k_next++, row.B1, row.B2, ctx);
        if (r.factor_found) {
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("%3dbit: FOUND curve %4d/%d (B1=%lu B2=%lu)  %7.1f ms  factor=%llu\n",
                        tc.bits, i + 1, (int)row.avg_curves, row.B1, row.B2,
                        ms, (unsigned long long)r.factor.d[0]);
            return;
        }
    }

    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%3dbit: NOT FOUND after %d curves (expected %d)  %.1f ms\n",
                tc.bits, max_curves, (int)row.avg_curves, ms);
}

int main() {
    std::printf("=== ECM validation with optimized schedule ===\n\n");
    for (const auto& tc : CASES) {
        switch (tc.N) {
            case 1: run_case<1>(tc); break;
            case 2: run_case<2>(tc); break;
            case 3: run_case<3>(tc); break;
            case 4: run_case<4>(tc); break;
        }
    }
    return 0;
}
