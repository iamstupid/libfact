// Benchmark: polynomial multiplication over Z/nZ.
//
// Compares FLINT's fmpz_mod_poly_mul vs zint bigint mul via Kronecker
// substitution.  This is the core inner loop of the subproduct tree
// used in polynomial stage 2.
//
// For ECM at N=3 (192-bit modulus), the subproduct tree multiplies
// polynomials of degree ~d/2 where d ranges from ~100 at the leaves
// to ~1200 at the root.  Each coefficient is a 192-bit integer.
//
// Build:
//   g++-14 -O2 -march=native -std=c++17 -I/tmp/ntl/include -I/tmp/flint/src \
//       -I include -I third_party \
//       bench/fixint/bench_polymul.cpp \
//       third_party/zint/asm/addmul_1_adx.S \
//       third_party/zint/asm/submul_1_adx.S \
//       third_party/zint/asm/mul_basecase_adx.S \
//       -L/tmp/flint -lflint -lgmp -lmpfr -lpthread \
//       -Wl,-rpath,/tmp/flint -mavx2 -mbmi2 -madx \
//       -o bench_polymul

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstring>

// FLINT
#include <fmpz.h>
#include <fmpz_vec.h>
#include <fmpz_mod.h>
#include <fmpz_mod_poly.h>

// zint
#include <zint/zint.hpp>

using Clock = std::chrono::steady_clock;

// ---------- FLINT poly mul ----------
double bench_flint_polymul(int deg, const char* prime_str) {
    fmpz_t p;
    fmpz_init(p);
    fmpz_set_str(p, prime_str, 10);

    flint_rand_t rng;
    flint_rand_init(rng);

    fmpz_mod_ctx_t ctx;
    fmpz_mod_ctx_init(ctx, p);

    fmpz_mod_poly_t a, b, c;
    fmpz_mod_poly_init(a, ctx);
    fmpz_mod_poly_init(b, ctx);
    fmpz_mod_poly_init(c, ctx);

    fmpz_t coeff;
    fmpz_init(coeff);
    for (int i = 0; i <= deg; ++i) {
        fmpz_randm(coeff, rng, p);
        fmpz_mod_poly_set_coeff_fmpz(a, i, coeff, ctx);
        fmpz_randm(coeff, rng, p);
        fmpz_mod_poly_set_coeff_fmpz(b, i, coeff, ctx);
    }
    fmpz_clear(coeff);

    // Warm up
    fmpz_mod_poly_mul(c, a, b, ctx);

    int iters = (deg < 500) ? 100 : (deg < 2000) ? 20 : 5;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        fmpz_mod_poly_mul(c, a, b, ctx);
    auto t1 = Clock::now();

    fmpz_mod_poly_clear(a, ctx);
    fmpz_mod_poly_clear(b, ctx);
    fmpz_mod_poly_clear(c, ctx);
    fmpz_mod_ctx_clear(ctx);
    flint_rand_clear(rng);
    fmpz_clear(p);

    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// ---------- zint Kronecker poly mul ----------
// Benchmark zint integer multiplication at the size that Kronecker
// substitution would produce for a degree-d poly with coeff_limbs-limb
// coefficients: each operand is (d+1) * (2*coeff_limbs+1) limbs.
double bench_zint_kronecker(int deg, int coeff_limbs) {
    int block = 2 * coeff_limbs + 1;
    uint32_t total_limbs = (uint32_t)((deg + 1) * block);

    // Allocate and fill with random data via zint's mpn allocator.
    auto* a = zint::mpn_alloc(total_limbs);
    auto* b = zint::mpn_alloc(total_limbs);
    auto* c = zint::mpn_alloc(2 * total_limbs);
    uint64_t seed = 0xDEADBEEF;
    for (uint32_t i = 0; i < total_limbs; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        a[i] = (zint::limb_t)seed;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        b[i] = (zint::limb_t)seed;
    }

    // Warm up
    zint::mpn_mul(c, a, total_limbs, b, total_limbs);

    int iters = (deg < 500) ? 100 : (deg < 2000) ? 20 : 5;
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        zint::mpn_mul(c, a, total_limbs, b, total_limbs);
    auto t1 = Clock::now();

    zint::mpn_free(a);
    zint::mpn_free(b);
    zint::mpn_free(c);

    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main() {
    const char* prime192 = "6277101735386680763835789423207666416102355444464034512899";
    const char* prime256 = "115792089237316195423570985008687907853269984665640564039457584007913129639747";

    struct Case { int deg; const char* name; };
    Case cases[] = {
        {100,  "deg=100  (tree leaf)"},
        {300,  "deg=300  (mid-tree)"},
        {600,  "deg=600  (near root)"},
        {1200, "deg=1200 (root)"},
        {2400, "deg=2400 (larger B2)"},
        {6000, "deg=6000 (big B2)"},
    };

    std::printf("Polynomial multiplication benchmark: FLINT vs zint Kronecker\n");
    std::printf("(single poly mul, microseconds)\n\n");

    std::printf("%-22s %12s %12s %12s %12s %10s\n",
                "case", "FLINT 192b", "zint 192b", "FLINT 256b", "zint 256b", "ratio 192");

    for (auto& [deg, name] : cases) {
        double f192 = bench_flint_polymul(deg, prime192);
        double z192 = bench_zint_kronecker(deg, 3);  // 192-bit = 3 limbs
        double f256 = bench_flint_polymul(deg, prime256);
        double z256 = bench_zint_kronecker(deg, 4);  // 256-bit = 4 limbs

        std::printf("%-22s %10.1f us %10.1f us %10.1f us %10.1f us %8.2fx\n",
                    name, f192, z192, f256, z256, f192 / z192);
    }

    return 0;
}
