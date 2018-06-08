#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <gmp.h>
#include "gmp_utils.h"

/*
 * This file contains functions that use GMP to perform some of the
 * precomputations that are used by the GPU based functions.  They all
 * fundamentally rely on the extended GCD algorithm for big integers,
 * which could be implemented relatively easily if we wanted to remove
 * GMP as a dependency.
 */

enum {
    ORDER  = -1,         /* least significant word first */
    SIZE   = 1,          /* one byte at a time */
    ENDIAN = 0,          /* native endianness */
    NAILS  = 0           /* how many nail bits */
};

/*
 * Convert the array in of n bytes to an mpz.
 */
static void
intmod_to_mpz(mpz_t out, const uint8_t *in, int n)
{
    mpz_import(out, n, ORDER, SIZE, ENDIAN, NAILS, in);
}

/*
 * Convert the mpz in to an array.
 *
 * The array out must have nbytes bytes; unused higher order bytes
 * are set to zero.
 */
static void
intmod_from_mpz(uint8_t out[], const mpz_t in, size_t nbytes)
{
    if (mpz_sizeinbase(in, 8) > nbytes) {
        // FIXME: Handle this error more gracefully.
        fprintf(stderr, "Failed to convert intmod\n");
        abort();
    }
    size_t cnt = 0;
    (void) mpz_export(out, &cnt, ORDER, SIZE, ENDIAN, NAILS, in);
    memset(out + cnt, 0, nbytes - cnt);
}

/*
 * Set s to be x^-1 (mod m), normalised so 0 <= s < m.
 */
static void
mpz_invmod(mpz_t s, const mpz_t x, const mpz_t m)
{
    mpz_t d;

    mpz_init(d);

    // d = 1 = sx + tm (ignoring t)
    mpz_gcdext(d, s, NULL, x, m);
    assert(mpz_cmp_ui(d, 1) == 0);
    while (mpz_cmp_ui(s, 0) < 0)
        mpz_add(s, s, m);

    mpz_clear(d);
}

/*
 * Given N an array of width digits, return ni satisfying ni * N = -1
 * (mod 2^DIGIT_BITS).
 *
 * Assumes N is odd (this is not checked).
 *
 * TODO: Currently assumes that digit_t is unsigned; can we relax this?
 */
template< typename digit_t >
digit_t
get_invmod(const uint8_t N[], int nbytes)
{
    // This assertion checks that the result of mpz_get_ui() is compatible with
    // the desired digit_t.
    static_assert(sizeof(digit_t) <= sizeof(unsigned long));
    static constexpr int DIGIT_BITS = 8 * sizeof(digit_t);
    mpz_t s, n, b;
    unsigned long ni;

    mpz_init(s);

    // b = 2^DIGIT_BITS
    mpz_init_set_ui(b, 1);
    mpz_mul_2exp(b, b, DIGIT_BITS);

    // n = N
    mpz_init(n);
    intmod_to_mpz(n, N, nbytes);

    mpz_invmod(s, n, b);

    // s = -s
    mpz_neg(s, s);
    // s % 2^DIGIT_BITS
    mpz_fdiv_r_2exp(s, s, DIGIT_BITS);
    // ni = (unsigned) s
    ni = mpz_get_ui(s);

    // Hence n * ni = -1 (mod 2^DIGIT_BITS)
    mpz_mul_ui(n, n, ni);
    mpz_add_ui(n, n, 1);
    mpz_fdiv_r_2exp(n, n, DIGIT_BITS);
    assert(mpz_cmp_ui(n, 0) == 0);

    mpz_clear(n);
    mpz_clear(b);
    mpz_clear(s);

    return (digit_t)ni;
}

/*
 * Given arrays R_mod, R_sqr_mod, and mod of width digits each, set
 *
 *   R_mod = 2^NBITS (mod "mod")
 *   R_sqr_mod = R_mod^2 (mod "mod")
 *
 * where NBITS := width * DIGIT_BITS = nbytes * 8.
 *
 * These values are used for Montgomery arithmetic; see 'modexp.cu'.
 */
void
get_R_and_Rsqr_mod(uint8_t R_mod[], uint8_t R_sqr_mod[], const uint8_t mod[], int nbytes)
{
    mpz_t n, r, r_mod;

    // n = N
    mpz_init(n);
    intmod_to_mpz(n, mod, nbytes);

    // r = 2^(width * DIGIT_BITS)
    mpz_init_set_ui(r, 1);
    mpz_mul_2exp(r, r, nbytes * 8);

    // r_modn = r % n
    mpz_init(r_mod);
    mpz_mod(r_mod, r, n);
    intmod_from_mpz(R_mod, r_mod, nbytes);

    // r = r^2
    mpz_mul_2exp(r, r, nbytes * 8);
    // r_modn = r^2 % n
    mpz_mod(r_mod, r, n);
    intmod_from_mpz(R_sqr_mod, r_mod, nbytes);

    mpz_clear(r_mod);
    mpz_clear(n);
    mpz_clear(r);
}

