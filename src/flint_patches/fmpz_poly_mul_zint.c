/*
 * Override FLINT's _fmpz_poly_mul with a zint-NTT-backed Kronecker
 * substitution for large inputs.  For small inputs we fall back to
 * FLINT's own classical/Karatsuba routines.
 *
 * This file is compiled into a static library that is linked BEFORE
 * libflint.so, so the linker picks our symbol first.
 *
 * The Kronecker substitution packs each fmpz coefficient into a fixed-
 * width block (2 * max_limbs + 1 limbs), multiplies the two packed
 * integers via zint::mpn_mul, then unpacks.  Coefficients are reduced
 * mod nothing — the caller (_fmpz_mod_poly_mul) handles the mod-p
 * reduction separately.
 */

#include <gmp.h>
#include "flint.h"
#include "fmpz.h"
#include "fmpz_vec.h"
#include "fmpz_poly.h"

/* Forward declaration of the original FLINT implementation.
 * We rename it via objcopy or just call the classical/KS fallbacks
 * directly.  For simplicity we re-implement the small-case dispatch
 * and only override the large-integer multiply. */

/* Declaration of zint's mpn_mul — C-linkage wrapper.
 * Defined in fmpz_poly_mul_zint_bridge.cpp */
extern void zint_mpn_mul_bridge(unsigned long long* rp,
                                const unsigned long long* ap, unsigned int an,
                                const unsigned long long* bp, unsigned int bn);

/* Kronecker threshold: use zint when both polys have >= this many terms
 * AND coefficients are >= this many limbs. */
#define ZINT_KS_LEN_THRESHOLD 32
#define ZINT_KS_BITS_THRESHOLD 128

static void
_fmpz_poly_mul_zint_kronecker(fmpz * res, const fmpz * poly1, slong len1,
                               const fmpz * poly2, slong len2)
{
    slong i, j;
    slong bits1 = _fmpz_vec_max_bits(poly1, len1);
    slong bits2 = _fmpz_vec_max_bits(poly2, len2);
    bits1 = FLINT_ABS(bits1);
    bits2 = FLINT_ABS(bits2);

    /* Max limbs per coefficient (input side). */
    slong limbs1 = (bits1 + 63) / 64;
    slong limbs2 = (bits2 + 63) / 64;
    slong max_limbs = FLINT_MAX(limbs1, limbs2);

    /* Block size in the packed integer: enough to hold the product of
     * two coefficients plus the carry from summing (min(len1,len2))
     * such products.  Conservative: 2*max_limbs + 1 limbs per block. */
    slong block = 2 * max_limbs + 1;

    slong total1 = len1 * block;
    slong total2 = len2 * block;
    slong total_out = total1 + total2;

    /* Allocate packed arrays (aligned for AVX2). */
    unsigned long long *a, *b, *c;
    a = (unsigned long long*)flint_calloc(total1, sizeof(unsigned long long));
    b = (unsigned long long*)flint_calloc(total2, sizeof(unsigned long long));
    c = (unsigned long long*)flint_calloc(total_out, sizeof(unsigned long long));

    /* Pack poly1 coefficients into a[]. */
    for (i = 0; i < len1; i++)
    {
        mp_srcptr limbs;
        slong nlimbs;
        int negative = 0;
        fmpz ci = poly1[i];

        if (!COEFF_IS_MPZ(ci))
        {
            if (ci >= 0)
                a[i * block] = (unsigned long long)ci;
            else
            {
                a[i * block] = (unsigned long long)(-ci);
                negative = 1;
            }
            nlimbs = (ci != 0) ? 1 : 0;
        }
        else
        {
            __mpz_struct * z = COEFF_TO_PTR(ci);
            nlimbs = z->_mp_size;
            if (nlimbs < 0) { nlimbs = -nlimbs; negative = 1; }
            limbs = z->_mp_d;
            for (j = 0; j < nlimbs && j < max_limbs; j++)
                a[i * block + j] = (unsigned long long)limbs[j];
        }

        if (negative)
        {
            /* Negate in-block: two's complement. */
            unsigned long long carry = 1;
            for (j = 0; j < block; j++)
            {
                unsigned long long v = ~a[i * block + j] + carry;
                carry = (v == 0 && carry) ? 1 : 0;
                a[i * block + j] = v;
            }
        }
    }

    /* Pack poly2 similarly. */
    for (i = 0; i < len2; i++)
    {
        mp_srcptr limbs;
        slong nlimbs;
        int negative = 0;
        fmpz ci = poly2[i];

        if (!COEFF_IS_MPZ(ci))
        {
            if (ci >= 0)
                b[i * block] = (unsigned long long)ci;
            else
            {
                b[i * block] = (unsigned long long)(-ci);
                negative = 1;
            }
            nlimbs = (ci != 0) ? 1 : 0;
        }
        else
        {
            __mpz_struct * z = COEFF_TO_PTR(ci);
            nlimbs = z->_mp_size;
            if (nlimbs < 0) { nlimbs = -nlimbs; negative = 1; }
            limbs = z->_mp_d;
            for (j = 0; j < nlimbs && j < max_limbs; j++)
                b[i * block + j] = (unsigned long long)limbs[j];
        }

        if (negative)
        {
            unsigned long long carry = 1;
            for (j = 0; j < block; j++)
            {
                unsigned long long v = ~b[i * block + j] + carry;
                carry = (v == 0 && carry) ? 1 : 0;
                b[i * block + j] = v;
            }
        }
    }

    /* Multiply packed integers via zint NTT. */
    zint_mpn_mul_bridge(c, a, (unsigned int)total1, b, (unsigned int)total2);

    /* Unpack: extract each block of `block` limbs from c[] as a signed
     * integer and store into res[]. */
    slong out_len = len1 + len2 - 1;
    for (i = 0; i < out_len; i++)
    {
        unsigned long long * blk = c + i * block;
        /* Check sign: if top bit of last limb is set, it's negative. */
        int negative = (blk[block - 1] >> 63) & 1;

        if (negative)
        {
            /* Negate block to get magnitude. */
            unsigned long long carry = 1;
            for (j = 0; j < block; j++)
            {
                unsigned long long v = ~blk[j] + carry;
                carry = (v == 0 && carry) ? 1 : 0;
                blk[j] = v;
            }
        }

        /* Find actual number of limbs. */
        slong nlimbs = block;
        while (nlimbs > 0 && blk[nlimbs - 1] == 0) nlimbs--;

        if (nlimbs == 0)
        {
            fmpz_zero(res + i);
        }
        else
        {
            /* Build an mpz_t from the limbs, then set the fmpz from it. */
            mpz_t tmp;
            mpz_init2(tmp, (mp_bitcnt_t)(nlimbs * 64));
            for (j = 0; j < nlimbs; j++)
                tmp->_mp_d[j] = (mp_limb_t)blk[j];
            tmp->_mp_size = negative ? -(int)nlimbs : (int)nlimbs;
            fmpz_set_mpz(res + i, tmp);
            mpz_clear(tmp);
        }
    }

    flint_free(a);
    flint_free(b);
    flint_free(c);
}

/* Override _fmpz_poly_mul: dispatch to zint for large cases,
 * fall back to FLINT's own routines for small. */
void
_fmpz_poly_mul(fmpz * res, const fmpz * poly1,
               slong len1, const fmpz * poly2, slong len2)
{
    slong bits1, bits2;

    if (len2 == 1)
    {
        _fmpz_vec_scalar_mul_fmpz(res, poly1, len1, poly2);
        return;
    }

    if (poly1 == poly2 && len1 == len2)
    {
        _fmpz_poly_sqr(res, poly1, len1);
        return;
    }

    bits1 = FLINT_ABS(_fmpz_vec_max_bits(poly1, len1));
    bits2 = FLINT_ABS(_fmpz_vec_max_bits(poly2, len2));

    /* Use zint Kronecker for large inputs. */
    if (len2 >= ZINT_KS_LEN_THRESHOLD && (bits1 + bits2) >= ZINT_KS_BITS_THRESHOLD)
    {
        _fmpz_poly_mul_zint_kronecker(res, poly1, len1, poly2, len2);
        return;
    }

    /* Small fallback: classical or Karatsuba. */
    if (len2 < 7)
        _fmpz_poly_mul_classical(res, poly1, len1, poly2, len2);
    else
        _fmpz_poly_mul_KS(res, poly1, len1, poly2, len2);
}
