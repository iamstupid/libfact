/*
 * Override FLINT's _flint_mpn_mul to dispatch to zint NTT for large
 * operands.  This is linked before libflint so the linker picks our
 * symbol first.  FLINT's Kronecker substitution does bit-pack → this
 * function → bit-unpack; we only swap the integer multiply kernel.
 *
 * For small operands we fall back to GMP's mpn_mul.
 */

#include <gmp.h>

/* C-linkage bridge to zint::mpn_mul, defined in the bridge .cpp file. */
extern void zint_mpn_mul_bridge(unsigned long long* rp,
                                const unsigned long long* ap, unsigned int an,
                                const unsigned long long* bp, unsigned int bn);

/* Threshold: use zint when both operands have >= this many limbs.
 * Below this, GMP's mpn_mul (Karatsuba/Toom) is faster because zint's
 * NTT has ~200-limb startup overhead. */
#define ZINT_MPN_MUL_THRESHOLD 256

mp_limb_t _flint_mpn_mul(mp_ptr r, mp_srcptr x, mp_size_t xn,
                          mp_srcptr y, mp_size_t yn)
{
    if (yn >= ZINT_MPN_MUL_THRESHOLD)
    {
        /* zint NTT path */
        zint_mpn_mul_bridge((unsigned long long*)r,
                            (const unsigned long long*)x, (unsigned int)xn,
                            (const unsigned long long*)y, (unsigned int)yn);
        return r[xn + yn - 1];
    }

    /* Small path: GMP */
    if (yn == 1)
    {
        r[xn] = mpn_mul_1(r, x, xn, y[0]);
        return r[xn];
    }
    mpn_mul(r, x, xn, y, yn);
    return r[xn + yn - 1];
}

void _flint_mpn_mul_n(mp_ptr r, mp_srcptr x, mp_srcptr y, mp_size_t n)
{
    if (n >= ZINT_MPN_MUL_THRESHOLD)
    {
        zint_mpn_mul_bridge((unsigned long long*)r,
                            (const unsigned long long*)x, (unsigned int)n,
                            (const unsigned long long*)y, (unsigned int)n);
        return;
    }
    mpn_mul_n(r, x, y, n);
}

mp_limb_t _flint_mpn_sqr(mp_ptr r, mp_srcptr x, mp_size_t n)
{
    if (n >= ZINT_MPN_MUL_THRESHOLD)
    {
        zint_mpn_mul_bridge((unsigned long long*)r,
                            (const unsigned long long*)x, (unsigned int)n,
                            (const unsigned long long*)x, (unsigned int)n);
        return r[2 * n - 1];
    }
    mpn_sqr(r, x, n);
    return r[2 * n - 1];
}
