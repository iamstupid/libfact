// Calibration benchmark: measure stage 1 and stage 2 costs independently.
// Stage 1: sweep B1, measure scalar mult time.
// Stage 2: sweep B2, measure BSGS and FFT time (B1 fixed small).
// Output CSV to stdout for the optimizer.
#include "zfactor/eecm/curve_setup.h"
#include "zfactor/eecm/scalar_mult.h"
#include "zfactor/eecm/stage2.h"
#include "zfactor/eecm/stage2_poly.h"
#include "zfactor/eecm.h"
#include "zfactor/sieve.h"
#include <zint/zint.hpp>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace zfactor::fixint;
using namespace zfactor::ecm;
using Clock = std::chrono::steady_clock;

// Semiprimes from gen_test_semiprimes.py — one per N value.
// We only need a composite to run curves on; factor size doesn't matter
// for cost measurement (cost depends on N, B1, B2 only).
template<int N> UInt<N> get_composite();

template<> UInt<1> get_composite<1>() {
    UInt<1> n{}; n.d[0] = 0x040005E180AB6CABULL; return n; // 60bit
}
template<> UInt<2> get_composite<2>() {
    UInt<2> n{}; n.d[0]=0x8000000000B05D4FULL; n.d[1]=0x004000000000017BULL; return n; // 120bit
}
template<> UInt<3> get_composite<3>() {
    UInt<3> n{}; n.d[0]=0x0000000000B56CF9ULL; n.d[1]=0x0000000017BD0000ULL; n.d[2]=0x0000000040000000ULL; return n; // 160bit
}
template<> UInt<4> get_composite<4>() {
    UInt<4> n{}; n.d[0]=0x0000000000B1D9E9ULL; n.d[1]=0x00017AD000000000ULL; n.d[2]=0; n.d[3]=0x0000000000000040ULL; return n; // 200bit
}

// Measure stage 1 only: setup + scalar mult. Returns ms.
template<int N>
double measure_s1(uint64_t B1, const MontCtx<N>& ctx, uint64_t curve_k) {
    auto cs = setup_curve<N>(curve_k, ctx);
    if (cs.factor_found) return -1;

    const auto& chain = get_stage1_chain(B1);
    std::vector<EdPoint<N>> precomp, precomp_neg;
    build_wnaf_precomp<N>(precomp, precomp_neg, cs.P0, chain.w, cs.curve, ctx);

    auto t0 = Clock::now();
    auto Q = scalar_mult_wnaf<N>(chain.wnaf, precomp, precomp_neg, cs.curve, ctx);
    auto t1 = Clock::now();
    (void)Q;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Measure stage 2 only. B1 is used for the stage 1 scalar (needed to get Q),
// but we only time the stage 2 call. Returns ms.
template<int N, typename S2Func>
double measure_s2(uint64_t B1, uint64_t B2, const MontCtx<N>& ctx,
                  uint64_t curve_k, const EdCurve<N>& curve,
                  const EdPoint<N>& Q, S2Func s2func) {
    auto t0 = Clock::now();
    auto s2 = s2func(Q, curve, B1, B2, ctx);
    auto t1 = Clock::now();
    (void)s2;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Run stage 1 to get Q for stage 2 measurement.
template<int N>
EdPoint<N> get_Q(uint64_t B1, const MontCtx<N>& ctx, uint64_t curve_k,
                 EdCurve<N>& curve_out) {
    auto cs = setup_curve<N>(curve_k, ctx);
    curve_out = cs.curve;
    const auto& chain = get_stage1_chain(B1);
    std::vector<EdPoint<N>> precomp, precomp_neg;
    build_wnaf_precomp<N>(precomp, precomp_neg, cs.P0, chain.w, cs.curve, ctx);
    return scalar_mult_wnaf<N>(chain.wnaf, precomp, precomp_neg, cs.curve, ctx);
}

template<int N>
void bench_n() {
    auto n = get_composite<N>();
    MontCtx<N> ctx;
    ctx.init(n);

    // ── Stage 1 sweep ──
    uint64_t b1_vals[] = {1000, 3000, 10000, 30000, 100000, 300000,
                          1000000, 3000000, 10000000};
    for (auto B1 : b1_vals) {
        int nc = (B1 >= 1000000) ? 2 : 10;
        double total = 0;
        int count = 0;
        for (int k = 2; k < 2 + nc; ++k) {
            double ms = measure_s1<N>(B1, ctx, k);
            if (ms >= 0) { total += ms; count++; }
        }
        if (count > 0)
            std::printf("s1,%d,%lu,%.4f\n", N, B1, total / count);
    }

    // ── Stage 2 sweep ──
    const uint64_t S2_B1 = 1000;
    uint64_t b2_vals[] = {100000, 300000, 1000000, 3000000, 10000000,
                          30000000, 100000000, 300000000};

    for (auto B2 : b2_vals) {
        int nc = (B2 >= 10000000) ? 2 : 10;
        double total_bsgs = 0, total_fft = 0;
        int count = 0;
        for (int k = 2; k < 2 + nc; ++k) {
            EdCurve<N> curve;
            auto Q = get_Q<N>(S2_B1, ctx, k, curve);

            double ms_b = measure_s2<N>(S2_B1, B2, ctx, k, curve, Q, ecm_stage2<N>);
            total_bsgs += ms_b;

            double ms_f = measure_s2<N>(S2_B1, B2, ctx, k, curve, Q, ecm_stage2_poly<N>);
            total_fft += ms_f;

            count++;
        }
        if (count > 0) {
            std::printf("s2_bsgs,%d,%lu,%.4f\n", N, B2, total_bsgs / count);
            std::printf("s2_fft,%d,%lu,%.4f\n", N, B2, total_fft / count);
        }
    }
}

int main() {
    std::printf("type,N,param,ms\n");
    bench_n<2>();
    bench_n<4>();
    return 0;
}
