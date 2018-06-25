#include <cstdio>
#include <cstdlib>
#include <cassert>
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
    ORDER  = -1,         /* least significant byte first */
    ENDIAN = 0,          /* native endianness */
    NAILS  = 0           /* how many nail bits */
};

/*
 * Convert the array in of n digit_t's to an mpz.
 */
template<typename digit_t>
static void
intmod_to_mpz(mpz_t out, const digit_t *in, int n)
{
    mpz_import(out, n, ORDER, sizeof(digit_t), ENDIAN, NAILS, in);
}

/*
 * Convert the mpz in to an array.
 *
 * The array out must have nelts digit_t's; unused higher order digits
 * are set to zero.
 */
template<typename digit_t>
static void
intmod_from_mpz(digit_t out[], const mpz_t in, size_t nelts)
{
    static constexpr int DIGIT_BITS = sizeof(digit_t) * 8;
    if (mpz_sizeinbase(in, 2) > nelts * DIGIT_BITS) {
        // FIXME: Handle this error more gracefully.
        fprintf(stderr, "Failed to convert intmod\n");
        abort();
    }
    size_t cnt = 0;
    (void) mpz_export(out, &cnt, ORDER, sizeof(digit_t), ENDIAN, NAILS, in);
    for (; cnt < nelts; ++cnt)
        out[cnt] = (digit_t)0UL;
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
 * Given N an array of nelts elements, return ni satisfying ni * N = -1
 * (mod 2^modbits).
 *
 * Assumes N is odd (this is not checked).
 * TODO: Mark where this assumption is used in the function (i.e. N must be
 * coprime to 2 for gcd).
 */
template< typename digit_t >
unsigned long
get_invmod(const digit_t N[], int nelts, int modbits)
{
    // This checks that the result of mpz_get_ui() is compatible with
    // the desired modbits.
    // FIXME: handle error more appropriately
    assert(modbits > 0 && (unsigned)modbits <= 8 * sizeof(unsigned long));
    mpz_t s, n, b;
    unsigned long ni;

    mpz_init(s);

    // b = 2^modbits
    mpz_init_set_ui(b, 1);
    mpz_mul_2exp(b, b, modbits);

    // n = N
    mpz_init(n);
    intmod_to_mpz(n, N, nelts);

    mpz_invmod(s, n, b);

    // TODO: Replace this calculation with the simpler: ni = 2^modbits - s
    // s = -s
    mpz_neg(s, s);
    // s % 2^modbits
    mpz_fdiv_r_2exp(s, s, modbits);
    // ni = (unsigned) s
    ni = mpz_get_ui(s);

    // Hence n * ni = -1 (mod 2^modbits)
    mpz_mul_ui(n, n, ni);
    mpz_add_ui(n, n, 1);
    mpz_fdiv_r_2exp(n, n, modbits);
    assert(mpz_cmp_ui(n, 0) == 0);

    mpz_clear(n);
    mpz_clear(b);
    mpz_clear(s);

    return ni;
}

/*
 * Given arrays R_mod, R_sqr_mod, and mod of width digits each, set
 *
 *   R_mod = 2^NBITS (mod "mod")
 *   R_sqr_mod = R_mod^2 (mod "mod")
 *
 * where NBITS := 8 * sizeof(digit_t) * NDIGITS.
 *
 * These values are used for Montgomery arithmetic; see 'modexp.cu'.
 */
template< int NDIGITS, typename digit_t >
void
get_R_and_Rsqr_mod(digit_t R_mod[NDIGITS], digit_t R_sqr_mod[NDIGITS], const uint8_t mod[], int nbytes)
{
    mpz_t n, r, r_mod;
    // TODO: Double-check that this is the right value for R. If mod is small,
    // then we might be able to choose R with lots of zeros in it, which might
    // be advantageous.
    static constexpr mp_bitcnt_t R_BITS = 8 * sizeof(digit_t) * NDIGITS;

    // n = N
    mpz_init(n);
    intmod_to_mpz(n, mod, nbytes);

    // r = 2^R_BITS
    mpz_init_set_ui(r, 1);
    mpz_mul_2exp(r, r, R_BITS);

    // r_modn = r % n
    mpz_init(r_mod);
    mpz_mod(r_mod, r, n);
    intmod_from_mpz(R_mod, r_mod, NDIGITS);

    // r = r^2
    mpz_mul_2exp(r, r, R_BITS);
    // r_modn = r^2 % n
    mpz_mod(r_mod, r, n);
    intmod_from_mpz(R_sqr_mod, r_mod, NDIGITS);

    mpz_clear(r_mod);
    mpz_clear(n);
    mpz_clear(r);
}
