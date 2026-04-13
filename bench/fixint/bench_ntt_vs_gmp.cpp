// bench_ntt_vs_gmp.cpp
//
// Benchmark the new zint NTT dispatcher (separated-radix engines A/B/C +
// p50x4 fallback, via zint::ntt::big_multiply_u64) against GMP's
// mpn_mul, across a range of sizes that stresses every engine boundary.
//
// Correctness: for each size, the two products must be bit-for-bit
// identical. Aborts on mismatch.
//
// RNG: xoshiro256++ (zint::xoshiro256pp from third_party/zint/rng.hpp).
//
// Build (WSL): enable with -DZFACTOR_BENCH_GMP=ON. GMP headers and libgmp.a
// are located via ZFACTOR_GMP_ROOT (see top-level CMakeLists.txt wiring).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#include "zint/zint.hpp"
#include "zint/rng.hpp"
#include "zint/ntt/api.hpp"

// GMP. Wrap its mpn_* symbols in a private namespace to avoid macro clashes
// with zint's mpn code (same pattern as third_party/zint/bench/bench_vs_gmp.cpp).
#include <gmp.h>
namespace gmp {
    inline mp_limb_t mul(mp_limb_t* rp, const mp_limb_t* ap, mp_size_t an,
                         const mp_limb_t* bp, mp_size_t bn) {
        return __gmpn_mul(rp, ap, an, bp, bn);
    }
}
#undef mpn_mul
#undef mpn_mul_n

using u64 = zint::ntt::u64;
static_assert(sizeof(u64) == sizeof(mp_limb_t), "zint and GMP limb widths must match");

// ---- Timing helpers ----

static inline double now_ns() {
    using clk = std::chrono::high_resolution_clock;
    return (double)clk::now().time_since_epoch().count();
}

// Run `fn` until both the sample count and wall-clock budget are met,
// then return the median per-iter time in ns.
template<typename F>
static double bench_median(F&& fn, int min_iters, double min_ns) {
    std::vector<double> samples;
    samples.reserve(std::min(min_iters, 300));
    double total_ns = 0.0;
    for (int i = 0; i < 300 && (i < min_iters || total_ns < min_ns); ++i) {
        double t0 = now_ns();
        fn();
        double t1 = now_ns();
        double dt = t1 - t0;
        samples.push_back(dt);
        total_ns += dt;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Iteration count scales so total work n·log2(n)·iter ≈ 5e8 "limb-ops",
// clamped so tiny sizes don't run forever and huge sizes still get several
// samples for a stable median.
static int iters_for(std::size_t n) {
    constexpr double TARGET_WORK = 5e8;
    double logn = std::log2((double)std::max<std::size_t>(n, 2));
    double iters = TARGET_WORK / ((double)n * logn);
    if (iters < 1.0) iters = 1.0;
    if (iters > 200000.0) iters = 200000.0;
    return (int)iters;
}

static void budget_for(std::size_t n, int& min_iters, double& min_ns) {
    min_iters = iters_for(n);
    min_ns = (n <= 1024) ? 50e6 : 20e6;
}

// ---- Filled with xoshiro256++ ----

static void fill_random(u64* dst, std::size_t n, zint::xoshiro256pp& rng) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = rng.next();
    // Make sure top limb is non-zero so the size is faithfully exercised.
    if (n) dst[n - 1] |= (u64(1) << 63);
}

// ---- Verify bit-for-bit equality ----

static void verify_equal(const u64* x, const u64* y, std::size_t n,
                         std::size_t na, std::size_t nb) {
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            std::fprintf(stderr,
                "MISMATCH at size na=%zu nb=%zu, limb %zu: "
                "ntt=0x%016llx gmp=0x%016llx\n",
                na, nb, i,
                (unsigned long long)x[i], (unsigned long long)y[i]);
            std::exit(1);
        }
    }
}

// ---- One bench row ----

struct Row {
    std::size_t na;
    std::size_t nb;
    double ntt_ns;
    double gmp_ns;
};

static Row run_size(std::size_t na, std::size_t nb, zint::xoshiro256pp& rng) {
    std::vector<u64> a(na), b(nb), out_ntt(na + nb, 0), out_gmp(na + nb, 0);
    fill_random(a.data(), na, rng);
    fill_random(b.data(), nb, rng);

    // One correctness check up front, then time both.
    zint::ntt::big_multiply_u64(out_ntt.data(), na + nb, a.data(), na, b.data(), nb);
    // GMP requires ap >= bp in size; swap if needed.
    if (na >= nb) {
        gmp::mul((mp_limb_t*)out_gmp.data(),
                 (const mp_limb_t*)a.data(), (mp_size_t)na,
                 (const mp_limb_t*)b.data(), (mp_size_t)nb);
    } else {
        gmp::mul((mp_limb_t*)out_gmp.data(),
                 (const mp_limb_t*)b.data(), (mp_size_t)nb,
                 (const mp_limb_t*)a.data(), (mp_size_t)na);
    }
    verify_equal(out_ntt.data(), out_gmp.data(), na + nb, na, nb);

    int min_iters; double min_ns;
    budget_for(std::max(na, nb), min_iters, min_ns);

    double t_ntt = bench_median([&]{
        zint::ntt::big_multiply_u64(out_ntt.data(), na + nb, a.data(), na, b.data(), nb);
    }, min_iters, min_ns);

    double t_gmp = bench_median([&]{
        if (na >= nb) {
            gmp::mul((mp_limb_t*)out_gmp.data(),
                     (const mp_limb_t*)a.data(), (mp_size_t)na,
                     (const mp_limb_t*)b.data(), (mp_size_t)nb);
        } else {
            gmp::mul((mp_limb_t*)out_gmp.data(),
                     (const mp_limb_t*)b.data(), (mp_size_t)nb,
                     (const mp_limb_t*)a.data(), (mp_size_t)na);
        }
    }, min_iters, min_ns);

    return Row{na, nb, t_ntt, t_gmp};
}

// Label which p30x3 engine (or p50x4) handled the multiply. Pure presentation.
static const char* engine_label(std::size_t na, std::size_t nb) {
    using namespace zint::ntt;
    std::size_t min_n32 = 2 * (na + nb);
    idt NA = engine_ceil_size<EngineA>(min_n32);
    idt NB = engine_ceil_size<EngineB>(min_n32);
    idt NC = engine_ceil_size<EngineC>(min_n32);
    idt best = idt(~idt(0));
    int who = -1;
    if (NA && NA < best) { best = NA; who = 0; }
    if (NB && NB < best) { best = NB; who = 1; }
    if (NC && NC < best) { best = NC; who = 2; }
    switch (who) {
        case 0: return "A";
        case 1: return "B";
        case 2: return "C";
        default: return "p50x4";
    }
}

int main() {
    zint::xoshiro256pp rng(0xA5A5'5A5A'1234'5678ULL);

    // Size sweep in u64 limbs. For each size we do a balanced multiply
    // (na == nb) and one unbalanced (nb = na/2) to exercise both shapes.
    static const std::size_t sizes[] = {
        2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768,
        49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
        786432, 1048576, 1572864, 2097152, 3145728, 4194304, 6291456,
        8388608, 10485760,
        // edge of Engine C cap: na+nb = 40M u32 = 20M u64
    };

    std::printf("# bench_ntt_vs_gmp (xoshiro256++, seed=0xA5A5'5A5A'1234'5678)\n");
    std::printf("# na_u64  nb_u64  engine   NTT_ns      GMP_ns      ratio\n");

    for (std::size_t s : sizes) {
        Row bal = run_size(s, s, rng);
        std::printf(" %-8zu %-8zu %-7s  %10.0f  %10.0f  %6.2fx\n",
            bal.na, bal.nb, engine_label(bal.na, bal.nb),
            bal.ntt_ns, bal.gmp_ns, bal.gmp_ns / bal.ntt_ns);

        std::size_t s2 = s > 1 ? s / 2 : 1;
        Row unb = run_size(s, s2, rng);
        std::printf(" %-8zu %-8zu %-7s  %10.0f  %10.0f  %6.2fx\n",
            unb.na, unb.nb, engine_label(unb.na, unb.nb),
            unb.ntt_ns, unb.gmp_ns, unb.gmp_ns / unb.ntt_ns);
    }

    std::printf("# all sizes verified bit-for-bit equal to GMP\n");
    return 0;
}
