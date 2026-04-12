// Benchmark: polynomial multi-point evaluation via NTL and FLINT.
//
// We simulate the ECM stage 2 inner product:
//   f(x) = prod_{j in J} (x - baby_x[j])   (degree ~240)
//   Evaluate f at ~6000 giant-step x-coordinates.
//
// For ECM the modulus is composite, but both libraries assume prime.
// We use a large prime for the benchmark so the timings are representative
// of the arithmetic cost (modular multiplication speed).  The actual ECM
// code will use our own montmul; this benchmark tells us the *algorithmic*
// overhead of the subproduct tree vs naive.
//
// Build:
//   g++ -O2 -march=native -std=c++17 -I/tmp/ntl/include -I/tmp/flint/src \
//       bench_polyeval.cpp -L/tmp/ntl/lib -lntl -lgmp -lpthread \
//       -L/tmp/flint/.libs -lflint -lmpfr -o bench_polyeval

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

// ============= NTL =============
#include <NTL/ZZ_pX.h>
#include <NTL/vec_ZZ_p.h>
#include <NTL/ZZ.h>

// ============= FLINT =============
#include <fmpz.h>
#include <fmpz_vec.h>
#include <fmpz_mod.h>
#include <fmpz_mod_poly.h>

using Clock = std::chrono::steady_clock;

// ---------- NTL benchmark ----------
double bench_ntl(int deg, int npoints, const char* prime_hex) {
    using namespace NTL;

    ZZ p;
    p = conv<ZZ>(prime_hex);
    ZZ_p::init(p);

    // Random polynomial of given degree
    ZZ_pX f;
    random(f, deg + 1);

    // Random evaluation points
    vec_ZZ_p pts;
    pts.SetLength(npoints);
    for (int i = 0; i < npoints; ++i)
        random(pts[i]);

    // Warm up
    vec_ZZ_p result;
    eval(result, f, pts);

    // Benchmark
    auto t0 = Clock::now();
    int iters = 3;
    for (int i = 0; i < iters; ++i)
        eval(result, f, pts);
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// ---------- FLINT benchmark ----------
double bench_flint(int deg, int npoints, const char* prime_str) {
    fmpz_t p;
    fmpz_init(p);
    fmpz_set_str(p, prime_str, 10);

    flint_rand_t rng;
    flint_rand_init(rng);

    fmpz_mod_ctx_t ctx;
    fmpz_mod_ctx_init(ctx, p);

    // Build random polynomial
    fmpz_mod_poly_t f;
    fmpz_mod_poly_init(f, ctx);
    {
        fmpz_t coeff;
        fmpz_init(coeff);
        for (int i = 0; i <= deg; ++i) {
            fmpz_randm(coeff, rng, p);
            fmpz_mod_poly_set_coeff_fmpz(f, i, coeff, ctx);
        }
        fmpz_clear(coeff);
    }

    // Random evaluation points
    fmpz* xs = _fmpz_vec_init(npoints);
    fmpz* ys = _fmpz_vec_init(npoints);
    {
        for (int i = 0; i < npoints; ++i)
            fmpz_randm(xs + i, rng, p);
    }

    // Warm up
    fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npoints, ctx);

    // Benchmark
    auto t0 = Clock::now();
    int iters = 3;
    for (int i = 0; i < iters; ++i)
        fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npoints, ctx);
    auto t1 = Clock::now();

    _fmpz_vec_clear(xs, npoints);
    _fmpz_vec_clear(ys, npoints);
    fmpz_mod_poly_clear(f, ctx);
    fmpz_mod_ctx_clear(ctx);
    fmpz_clear(p);

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    // 192-bit prime (matches N=3 ECM target)
    const char* prime192 = "6277101735386680763835789423207666416102355444464034512899";
    // 256-bit prime (matches N=4 ECM target)
    const char* prime256 = "115792089237316195423570985008687907853269984665640564039457584007913129639747";

    struct TestCase {
        int deg;
        int npoints;
        const char* name;
    };

    TestCase cases[] = {
        {240,  6000,   "d=240  n=6000   (current BSGS)"},
        {1200, 1200,   "d=1200 n=1200   (balanced, same B2)"},
        {2400, 2400,   "d=2400 n=2400   (balanced, 4x B2)"},
        {4800, 4800,   "d=4800 n=4800   (balanced, 16x B2)"},
        {240,  60000,  "d=240  n=60000  (current BSGS, 10x B2)"},
        {6000, 6000,   "d=6000 n=6000   (balanced, 10x B2)"},
    };

    std::printf("Polynomial multi-point evaluation benchmark\n");
    std::printf("(subproduct tree, prime modulus)\n\n");

    for (auto& [deg, npts, name] : cases) {
        std::printf("--- %s ---\n", name);
        for (auto [prime, bits] : {std::pair{prime192, 192}, {prime256, 256}}) {
            double ntl_ms  = bench_ntl(deg, npts, prime);
            double flint_ms = bench_flint(deg, npts, prime);
            std::printf("  %d-bit:  NTL=%7.2f ms   FLINT=%7.2f ms   ratio=%.2fx\n",
                        bits, ntl_ms, flint_ms, ntl_ms / flint_ms);
        }
    }

    return 0;
}
