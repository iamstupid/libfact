// EECM benchmark: time-to-find-factor for various target sizes.
// Compares EECM stage 1+2 against pollard rho on the same composites.
#include "zfactor/eecm.h"
#include "zfactor/rho.h"

#include <cstdio>
#include <cstdint>
#include <chrono>

using namespace zfactor::fixint;
using namespace zfactor::ecm;

template<int N>
double time_eecm(const UInt<N>& n, int factor_bits, int max_curves) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto r = ecm<N>(n, factor_bits, max_curves);
    auto t1 = clock::now();
    if (!r) return -1.0;
    return std::chrono::duration<double>(t1 - t0).count();
}

template<int N>
double time_rho(const UInt<N>& n) {
    using clock = std::chrono::steady_clock;
    MontCtx<N> ctx;
    ctx.init(n);
    auto t0 = clock::now();
    auto factor = zfactor::pollard_rho_brent<N>(ctx, /*c=*/1, /*x0=*/2);
    auto t1 = clock::now();
    if (factor.is_zero() || factor == n) return -1.0;
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    struct Case {
        const char* name;
        UInt<1> n;
        int factor_bits;
        int max_curves;
    };
    // The interesting regime for ECM (over rho) is composites with a small
    // prime factor and a *huge* cofactor.  rho is O(p^{1/2}) where p is the
    // smallest factor, but its constant on multi-limb arithmetic is high.
    // We test a 1009 * (huge prime) at N=2 to make sure rho works hard.
    Case cases[] = {
        // 14-bit factor in a 27-bit composite (rho's sweet spot)
        {"1009*10007 (14 bit)",       UInt<1>(1009ULL * 10007ULL),    20,  20},
        // 20-bit factor in 40-bit composite
        {"100003*1000003 (20 bit)",   UInt<1>(100003ULL * 1000003ULL), 30, 30},
        // 20-bit factor in 44-bit composite
        {"1000003*10000019 (20 bit)", UInt<1>(0), 80, 50},
        // 31-bit factor
        {"31-bit Mersenne pair",      UInt<1>(0), 100, 100},
    };
    cases[2].n.d[0] = 1000003ULL * 10000019ULL;
    cases[3].n.d[0] = 9223372064772063217ULL;

    std::printf("%-32s %12s %12s\n", "case", "EECM (ms)", "rho (ms)");
    for (auto& c : cases) {
        double te = time_eecm<1>(c.n, c.factor_bits, c.max_curves);
        double tr = time_rho<1>(c.n);
        char eecm_buf[32], rho_buf[32];
        if (te < 0) std::snprintf(eecm_buf, 32, "no factor");
        else        std::snprintf(eecm_buf, 32, "%.3f", te * 1000);
        if (tr < 0) std::snprintf(rho_buf, 32, "no factor");
        else        std::snprintf(rho_buf, 32, "%.3f", tr * 1000);
        std::printf("%-32s %12s %12s\n", c.name, eecm_buf, rho_buf);
    }
    return 0;
}
