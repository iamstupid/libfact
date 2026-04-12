// Test: does FLINT's fmpz_mod_poly_evaluate_fmpz_vec work on composite moduli?
// Specifically, does the subproduct tree crash when it hits a non-invertible element?
//
// We test with n = p * q (two known primes), build a polynomial whose
// evaluation points include multiples of p, and see what happens.

#include <cstdio>
#include <cstdint>
#include <cstring>

#include <gmp.h>
#include <fmpz.h>
#include <fmpz_vec.h>
#include <fmpz_mod.h>
#include <fmpz_mod_poly.h>

int main() {
    // n = p * q  (two 96-bit primes → 192-bit composite)
    const char* p_str = "79228162514264337593543950397";   // nextprime(2^96 + 100)
    const char* q_str = "79228162514264337593543951787";   // nextprime(2^96 + 1500)

    fmpz_t p, q, n;
    fmpz_init(p); fmpz_init(q); fmpz_init(n);
    fmpz_set_str(p, p_str, 10);
    fmpz_set_str(q, q_str, 10);
    fmpz_mul(n, p, q);

    char n_buf[256];
    fmpz_get_str(n_buf, 10, n);
    std::printf("n = %s\n", n_buf);
    std::printf("  = %s * %s\n", p_str, q_str);

    // --- Test 1: simple evaluate_fmpz_vec with composite modulus ---
    std::printf("\n=== Test 1: small poly, composite mod, no tricky points ===\n");
    {
        fmpz_mod_ctx_t ctx;
        fmpz_mod_ctx_init(ctx, n);

        fmpz_mod_poly_t f;
        fmpz_mod_poly_init(f, ctx);

        // f(x) = x^3 + 2x + 1
        fmpz_t c;
        fmpz_init(c);
        fmpz_one(c);   fmpz_mod_poly_set_coeff_fmpz(f, 0, c, ctx);
        fmpz_set_ui(c, 2); fmpz_mod_poly_set_coeff_fmpz(f, 1, c, ctx);
        fmpz_one(c);   fmpz_mod_poly_set_coeff_fmpz(f, 3, c, ctx);

        int npts = 5;
        fmpz* xs = _fmpz_vec_init(npts);
        fmpz* ys = _fmpz_vec_init(npts);

        for (int i = 0; i < npts; ++i)
            fmpz_set_ui(xs + i, i + 1);

        std::printf("  Calling fmpz_mod_poly_evaluate_fmpz_vec...\n");
        fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npts, ctx);
        std::printf("  OK! Results:\n");

        for (int i = 0; i < npts; ++i) {
            char buf[256];
            fmpz_get_str(buf, 10, ys + i);
            long x = i + 1;
            long expected = x*x*x + 2*x + 1;
            std::printf("    f(%ld) = %s  (expected %ld)\n", x, buf, expected);
        }

        _fmpz_vec_clear(xs, npts);
        _fmpz_vec_clear(ys, npts);
        fmpz_clear(c);
        fmpz_mod_poly_clear(f, ctx);
        fmpz_mod_ctx_clear(ctx);
    }

    // --- Test 2: larger poly to force subproduct tree (needs inverses?) ---
    std::printf("\n=== Test 2: degree 240, 1000 points, composite mod ===\n");
    {
        fmpz_mod_ctx_t ctx;
        fmpz_mod_ctx_init(ctx, n);

        flint_rand_t rng;
        flint_rand_init(rng);

        fmpz_mod_poly_t f;
        fmpz_mod_poly_init(f, ctx);
        {
            fmpz_t c;
            fmpz_init(c);
            for (int i = 0; i <= 240; ++i) {
                fmpz_randm(c, rng, n);
                fmpz_mod_poly_set_coeff_fmpz(f, i, c, ctx);
            }
            fmpz_clear(c);
        }

        int npts = 1000;
        fmpz* xs = _fmpz_vec_init(npts);
        fmpz* ys = _fmpz_vec_init(npts);

        for (int i = 0; i < npts; ++i)
            fmpz_randm(xs + i, rng, n);

        std::printf("  Calling fmpz_mod_poly_evaluate_fmpz_vec with %d points...\n", npts);
        fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npts, ctx);
        std::printf("  OK! (no crash)\n");

        // Verify a few by Horner
        std::printf("  Spot-checking 3 results via Horner...\n");
        {
            fmpz_t val, tmp;
            fmpz_init(val); fmpz_init(tmp);
            int ok = 0;
            for (int k = 0; k < 3; ++k) {
                fmpz_mod_poly_evaluate_fmpz(val, f, xs + k, ctx);
                if (fmpz_equal(val, ys + k)) ok++;
                else std::printf("    MISMATCH at point %d!\n", k);
            }
            std::printf("    %d/3 match\n", ok);
            fmpz_clear(val); fmpz_clear(tmp);
        }

        _fmpz_vec_clear(xs, npts);
        _fmpz_vec_clear(ys, npts);
        fmpz_mod_poly_clear(f, ctx);
        flint_rand_clear(rng);
        fmpz_mod_ctx_clear(ctx);
    }

    // --- Test 3: evaluation point that is a multiple of p (non-invertible) ---
    std::printf("\n=== Test 3: evaluation point = k*p (non-invertible element) ===\n");
    {
        fmpz_mod_ctx_t ctx;
        fmpz_mod_ctx_init(ctx, n);

        fmpz_mod_poly_t f;
        fmpz_mod_poly_init(f, ctx);

        // f(x) = x^2 + 1
        fmpz_t c;
        fmpz_init(c);
        fmpz_one(c);   fmpz_mod_poly_set_coeff_fmpz(f, 0, c, ctx);
        fmpz_one(c);   fmpz_mod_poly_set_coeff_fmpz(f, 2, c, ctx);

        int npts = 10;
        fmpz* xs = _fmpz_vec_init(npts);
        fmpz* ys = _fmpz_vec_init(npts);

        // Some normal points, and point 5 = 3*p (multiple of factor)
        for (int i = 0; i < npts; ++i)
            fmpz_set_ui(xs + i, i * 7 + 1);
        // Make xs[5] = 3 * p
        fmpz_mul_ui(xs + 5, p, 3);

        std::printf("  xs[5] = 3*p (a multiple of the factor)\n");
        std::printf("  Calling fmpz_mod_poly_evaluate_fmpz_vec...\n");
        fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npts, ctx);
        std::printf("  OK! (no crash with non-invertible eval point)\n");

        // Check: f(3p) = (3p)^2 + 1 mod n
        {
            fmpz_t expected, tmp;
            fmpz_init(expected); fmpz_init(tmp);
            fmpz_mul(tmp, xs + 5, xs + 5);   // (3p)^2
            fmpz_add_ui(tmp, tmp, 1);         // + 1
            fmpz_mod(expected, tmp, n);
            char buf1[256], buf2[256];
            fmpz_get_str(buf1, 10, ys + 5);
            fmpz_get_str(buf2, 10, expected);
            std::printf("    f(3p) computed = %s\n", buf1);
            std::printf("    f(3p) expected = %s\n", buf2);
            std::printf("    match: %s\n", fmpz_equal(ys + 5, expected) ? "YES" : "NO");
            fmpz_clear(expected); fmpz_clear(tmp);
        }

        _fmpz_vec_clear(xs, npts);
        _fmpz_vec_clear(ys, npts);
        fmpz_clear(c);
        fmpz_mod_poly_clear(f, ctx);
        fmpz_mod_ctx_clear(ctx);
    }

    // --- Test 4: large degree with many non-invertible differences ---
    // In subproduct tree, the tree nodes are prod(x - x_i).
    // Polynomial remainder uses these. If x_i - x_j shares a factor with n,
    // the tree division might fail internally.
    std::printf("\n=== Test 4: 500 points, many multiples of p mixed in ===\n");
    {
        fmpz_mod_ctx_t ctx;
        fmpz_mod_ctx_init(ctx, n);

        flint_rand_t rng;
        flint_rand_init(rng);

        fmpz_mod_poly_t f;
        fmpz_mod_poly_init(f, ctx);
        {
            fmpz_t c;
            fmpz_init(c);
            for (int i = 0; i <= 100; ++i) {
                fmpz_randm(c, rng, n);
                fmpz_mod_poly_set_coeff_fmpz(f, i, c, ctx);
            }
            fmpz_clear(c);
        }

        int npts = 500;
        fmpz* xs = _fmpz_vec_init(npts);
        fmpz* ys = _fmpz_vec_init(npts);

        // Every 10th point is a multiple of p
        for (int i = 0; i < npts; ++i) {
            if (i % 10 == 0)
                fmpz_mul_ui(xs + i, p, i + 1);
            else
                fmpz_randm(xs + i, rng, n);
        }

        std::printf("  50 of 500 points are multiples of p\n");
        std::printf("  Calling fmpz_mod_poly_evaluate_fmpz_vec...\n");
        fmpz_mod_poly_evaluate_fmpz_vec(ys, f, xs, npts, ctx);
        std::printf("  OK! (no crash)\n");

        // Spot-check via Horner
        int ok = 0, checked = 0;
        {
            fmpz_t val;
            fmpz_init(val);
            for (int k = 0; k < npts; k += 50) {
                fmpz_mod_poly_evaluate_fmpz(val, f, xs + k, ctx);
                if (fmpz_equal(val, ys + k)) ok++;
                else {
                    char buf1[256], buf2[256];
                    fmpz_get_str(buf1, 10, val);
                    fmpz_get_str(buf2, 10, ys + k);
                    std::printf("    MISMATCH at point %d: horner=%s vec=%s\n", k, buf1, buf2);
                }
                checked++;
            }
            fmpz_clear(val);
        }
        std::printf("  Spot-check: %d/%d match\n", ok, checked);

        _fmpz_vec_clear(xs, npts);
        _fmpz_vec_clear(ys, npts);
        fmpz_mod_poly_clear(f, ctx);
        flint_rand_clear(rng);
        fmpz_mod_ctx_clear(ctx);
    }

    fmpz_clear(p);
    fmpz_clear(q);
    fmpz_clear(n);

    std::printf("\nAll tests completed.\n");
    return 0;
}
