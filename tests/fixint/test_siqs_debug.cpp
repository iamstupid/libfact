// Debug test: verify sieve roots are correct and smooth values exist.
#include <cstdio>
#include <cstdint>
#include "zfactor/fixint/uint.h"
#include "zfactor/siqs/params.h"
#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/poly.h"

using namespace zfactor;
using namespace zfactor::fixint;
using namespace zfactor::siqs;

int main() {
    // Small test: n = 1000000007 * 1000000009
    UInt<2> n;
    n.d[0] = 1000000007ULL * 1000000009ULL;
    n.d[1] = 0;
    // Actually compute properly
    {
        unsigned __int128 prod = (unsigned __int128)1000000007ULL * 1000000009ULL;
        n.d[0] = (uint64_t)prod;
        n.d[1] = (uint64_t)(prod >> 64);
    }
    printf("n = %lu * 2^64 + %lu (%u bits)\n", n.d[1], n.d[0], n.bit_length());

    uint32_t k = select_multiplier<2>(n);
    printf("multiplier k = %u\n", k);

    UInt<2> kn;
    {
        unsigned __int128 w = (unsigned __int128)n.d[0] * k;
        kn.d[0] = (uint64_t)w;
        unsigned __int128 w2 = (unsigned __int128)n.d[1] * k + (uint64_t)(w >> 64);
        kn.d[1] = (uint64_t)w2;
    }
    printf("kn bits = %u\n", kn.bit_length());

    SiqsParams params = get_params(kn.bit_length());
    printf("FB size target = %u\n", params.fb_size);

    FactorBase fb = FactorBase::build<2>(n, params, k);
    printf("FB built: %u primes, max = %u, med_B = %u\n",
           fb.fb_size, fb.prime[fb.fb_size-1], fb.med_B);
    printf("LP bound = %lu\n", fb.large_prime_max);

    // Print first 20 FB primes and their roots
    printf("\nFactor base (first 20):\n");
    for (uint32_t i = 0; i < std::min(fb.fb_size, 20u); i++) {
        printf("  [%u] p=%u, sqrt=%u, logp=%u\n", i, fb.prime[i], fb.modsqrt[i], fb.logp[i]);
    }

    // Generate a polynomial
    SiqsPoly<2> poly;
    uint32_t seed1 = 11111111, seed2 = 22222222;
    uint32_t sieve_len = 2 * params.num_blocks * BLOCKSIZE;
    new_poly_a<2>(poly, fb, kn, sieve_len / 2, seed1, seed2);
    init_poly_b<2>(poly, fb, kn, sieve_len);

    printf("\nPolynomial: A = %lu * 2^64 + %lu\n", poly.A.d[1], poly.A.d[0]);
    printf("           B = %lu * 2^64 + %lu\n", poly.B.d[1], poly.B.d[0]);
    printf("           s = %d factors\n", poly.s);
    printf("A-factor indices:");
    for (int j = 0; j < poly.s; j++) printf(" %u(p=%u)", poly.a_factors[j], fb.prime[poly.a_factors[j]]);
    printf("\n");

    // Verify: B^2 ≡ kn (mod A)
    UInt<4> B2;
    mpn::mul<2>(B2.d, poly.B.d, poly.B.d);
    UInt<4> A_wide = {};
    A_wide.d[0] = poly.A.d[0]; A_wide.d[1] = poly.A.d[1];
    UInt<4> Q_wide, R_wide;
    mpn::divrem<4>(Q_wide.d, R_wide.d, B2.d, A_wide.d);
    // R should equal kn mod A
    UInt<4> kn_wide = {};
    kn_wide.d[0] = kn.d[0]; kn_wide.d[1] = kn.d[1];
    UInt<4> Q2, R2;
    mpn::divrem<4>(Q2.d, R2.d, kn_wide.d, A_wide.d);
    bool b2_ok = (R_wide.d[0] == R2.d[0] && R_wide.d[1] == R2.d[1]);
    printf("B^2 mod A == kn mod A? %s\n", b2_ok ? "YES" : "NO");
    if (!b2_ok) {
        printf("  B^2 mod A = %lu, kn mod A = %lu\n", R_wide.d[0], R2.d[0]);
    }

    // Verify sieve roots: for a few FB primes, check that Q(root) ≡ 0 (mod p)
    printf("\nSieve root verification (first 10 non-A-factor primes):\n");
    int checked = 0;
    for (uint32_t i = 2; i < fb.fb_size && checked < 10; i++) {
        if (poly.root1[i] == -1) continue;
        uint32_t p = fb.prime[i];
        int32_t r1 = poly.root1[i];
        int32_t r2 = poly.root2[i];

        // Q(x) = A*x^2 + 2*B*x + C where x = r1 (raw sieve offset)
        // The roots should satisfy Q(x) ≡ 0 (mod p)
        // Actually, the sieve offset maps to polynomial variable as:
        //   sieve_location - M maps to x in the polynomial
        // But our roots are already sieve locations in [0, sieve_len).
        // So Q(root1 - M) should be divisible by p.

        // Easier check: (A*(r1-M) + B)^2 - kn ≡ 0 (mod p)
        // Which means A*(r1-M) + B ≡ ±sqrt(kn) (mod p)
        uint32_t M = sieve_len / 2;
        int32_t x = r1 - (int32_t)M;
        uint32_t a_modp = uint_mod_u32<2>(poly.A, p);
        uint32_t b_modp = uint_mod_u32<2>(poly.B, p);

        // val = A*x + B (mod p)
        int64_t ax = (int64_t)a_modp * x;
        int64_t val = ax + b_modp;
        uint32_t val_modp = (uint32_t)(((val % (int64_t)p) + p) % p);

        // Check val^2 ≡ kn (mod p)
        uint32_t val2 = (uint32_t)(((uint64_t)val_modp * val_modp) % p);
        uint32_t kn_modp = uint_mod_u32<2>(kn, p);

        printf("  p=%u r1=%d r2=%d x=%d: val=%u, val^2=%u, kn%%p=%u %s\n",
               p, r1, r2, x, val_modp, val2, kn_modp,
               val2 == kn_modp ? "OK" : "FAIL");
        checked++;
    }

    // Try to find smooth values manually by evaluating Q(x) for small x
    printf("\nSearching for smooth Q(x) values near x=0...\n");
    uint32_t M = sieve_len / 2;
    int found = 0;
    for (int x = -1000; x <= 1000 && found < 5; x++) {
        // Q(x) = A*x^2 + 2*B*x + C
        // Compute |Q(x)| and try to factor over FB
        uint32_t abs_x = (uint32_t)std::abs(x);

        // ax2 = A * x^2
        UInt<2> ax;
        {
            uint64_t carry = 0;
            for (int i = 0; i < 2; i++) {
                unsigned __int128 w = (unsigned __int128)poly.A.d[i] * abs_x + carry;
                ax.d[i] = (uint64_t)w;
                carry = (uint64_t)(w >> 64);
            }
        }
        UInt<2> ax2;
        {
            uint64_t carry = 0;
            for (int i = 0; i < 2; i++) {
                unsigned __int128 w = (unsigned __int128)ax.d[i] * abs_x + carry;
                ax2.d[i] = (uint64_t)w;
                carry = (uint64_t)(w >> 64);
            }
        }

        // Just check if Q(x) is small enough to be smooth
        // Q(x) = (Ax+B)^2/A - kn/A, approximately
        // For very small x, Q(x) ≈ C which is about sqrt(kn)
        // So log2(Q) ≈ bits(kn)/2 ≈ 31 for a 62-bit kn

        // Try to fully factor ax2 + 2*B*x + C over the FB
        // (Simplified: just compute Q mod each FB prime and check)
        uint32_t loc = M + x;  // sieve offset
        int total_logp = 0;
        for (uint32_t i = 2; i < fb.fb_size; i++) {
            if (poly.root1[i] == -1) continue;
            uint32_t p = fb.prime[i];
            uint32_t r1 = (uint32_t)poly.root1[i];
            uint32_t r2 = (uint32_t)poly.root2[i];
            if (loc % p == r1 % p || loc % p == r2 % p) {
                total_logp += fb.logp[i];
            }
        }
        if (total_logp > 20) {
            printf("  x=%d (loc=%u): total logp=%d\n", x, loc, total_logp);
            found++;
        }
    }
    if (found == 0) printf("  No smooth-ish values found!\n");

    return 0;
}
