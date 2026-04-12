// Verify Kronecker substitution gives the same polynomial product as FLINT.
// Build a degree-10 poly over Z/pZ, multiply via both methods, compare.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include <fmpz.h>
#include <fmpz_vec.h>
#include <fmpz_mod.h>
#include <fmpz_mod_poly.h>

#include <zint/zint.hpp>

int main() {
    // Small prime for easy verification
    const char* prime_str = "6277101735386680763835789423207666416102355444464034512899";
    int coeff_limbs = 3;  // 192-bit
    int deg = 10;

    // --- FLINT reference ---
    fmpz_t p;
    fmpz_init(p);
    fmpz_set_str(p, prime_str, 10);

    fmpz_mod_ctx_t ctx;
    fmpz_mod_ctx_init(ctx, p);

    fmpz_mod_poly_t fa, fb, fc;
    fmpz_mod_poly_init(fa, ctx);
    fmpz_mod_poly_init(fb, ctx);
    fmpz_mod_poly_init(fc, ctx);

    // Large random coefficients (full 192-bit) for a realistic test.
    // Use deg from command line or default 1200.
    deg = 1200;
    std::vector<uint64_t> coeffs_a(deg + 1), coeffs_b(deg + 1);
    uint64_t seed = 0xDEADBEEF;
    auto next_rng = [&]() { seed = seed * 6364136223846793005ULL + 1; return seed; };
    for (int i = 0; i <= deg; ++i) { coeffs_a[i] = next_rng(); coeffs_b[i] = next_rng(); }
    // Set full-width random coefficients via fmpz
    {
        fmpz_t c;
        fmpz_init(c);
        flint_rand_t rng;
        flint_rand_init(rng);
        for (int i = 0; i <= deg; ++i) {
            fmpz_randm(c, rng, p);
            fmpz_mod_poly_set_coeff_fmpz(fa, i, c, ctx);
            fmpz_randm(c, rng, p);
            fmpz_mod_poly_set_coeff_fmpz(fb, i, c, ctx);
        }
        fmpz_clear(c);
        flint_rand_clear(rng);
    }
    fmpz_mod_poly_mul(fc, fa, fb, ctx);
    std::printf("FLINT poly mul done, result deg = %ld\n", fmpz_mod_poly_degree(fc, ctx));

    // --- Kronecker via zint ---
    // Extract FLINT's coefficients as limbs, pack into Kronecker blocks.
    int block = 2 * coeff_limbs + 1;  // 192-bit coeff → 7-limb block
    int total_limbs = (deg + 1) * block;
    auto* a_data = zint::mpn_alloc(total_limbs);
    auto* b_data = zint::mpn_alloc(total_limbs);
    auto* c_data = zint::mpn_alloc(2 * total_limbs);
    std::memset(a_data, 0, total_limbs * sizeof(zint::limb_t));
    std::memset(b_data, 0, total_limbs * sizeof(zint::limb_t));

    // Copy FLINT poly coefficients into Kronecker-packed limb arrays.
    {
        fmpz_t c;
        fmpz_init(c);
        for (int i = 0; i <= deg; ++i) {
            fmpz_mod_poly_get_coeff_fmpz(c, fa, i, ctx);
            // Extract up to coeff_limbs 64-bit limbs from fmpz.
            for (int j = 0; j < coeff_limbs; ++j) {
                a_data[i * block + j] = (zint::limb_t)fmpz_get_ui(c);
                fmpz_tdiv_q_2exp(c, c, 64);
            }
            fmpz_mod_poly_get_coeff_fmpz(c, fb, i, ctx);
            for (int j = 0; j < coeff_limbs; ++j) {
                b_data[i * block + j] = (zint::limb_t)fmpz_get_ui(c);
                fmpz_tdiv_q_2exp(c, c, 64);
            }
        }
        fmpz_clear(c);
    }

    zint::mpn_mul(c_data, a_data, total_limbs, b_data, total_limbs);
    std::printf("Kronecker mul done\n");

    // --- Reduce Kronecker coefficients mod p and compare against FLINT ---
    {
        fmpz_t kc, reduced, flint_c;
        fmpz_init(kc);
        fmpz_init(reduced);
        fmpz_init(flint_c);
        int mismatches = 0;
        for (int i = 0; i <= 2 * deg; ++i) {
            // Reconstruct fmpz from the Kronecker block.
            fmpz_zero(kc);
            for (int j = block - 1; j >= 0; --j) {
                fmpz_mul_2exp(kc, kc, 64);
                fmpz_add_ui(kc, kc, (unsigned long)c_data[i * block + j]);
            }
            fmpz_mod(reduced, kc, p);

            fmpz_mod_poly_get_coeff_fmpz(flint_c, fc, i, ctx);
            if (!fmpz_equal(reduced, flint_c)) mismatches++;
        }
        std::printf("%d mismatches out of %d coefficients\n", mismatches, 2*deg+1);
        fmpz_clear(kc);
        fmpz_clear(reduced);
        fmpz_clear(flint_c);
    }

    fmpz_mod_poly_clear(fa, ctx);
    fmpz_mod_poly_clear(fb, ctx);
    fmpz_mod_poly_clear(fc, ctx);
    fmpz_mod_ctx_clear(ctx);
    fmpz_clear(p);
    zint::mpn_free(a_data);
    zint::mpn_free(b_data);
    zint::mpn_free(c_data);
    return 0;
}
