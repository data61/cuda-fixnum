#pragma once

#include "functions/quorem_preinv.cu"

namespace cuFIXNUM {

namespace internal {

template< typename fixnum_ >
class monty {
public:
    typedef fixnum_ fixnum;
    typedef fixnum modnum;

    __device__ monty(fixnum modulus);

    __device__ void add(modnum &z, modnum x, modnum y) const {
        fixnum::add(z, x, y);
        if (fixnum::cmp(z, mod) >= 0)
            fixnum::sub(z, z, mod);
    }

    __device__ void neg(modnum &z, modnum x) const {
        fixnum::sub(z, mod, x);
    }

    __device__ void sub(modnum &z, modnum x, modnum y) const {
        fixnum my;
        neg(my, y);
        fixnum::add(z, x, my);
        if (fixnum::cmp(z, mod) >= 0)
            fixnum::sub(z, z, mod);
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ modnum one() const {
        return R_mod;
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ modnum zero() const {
        return fixnum::zero();
    }

    // FIXME: Get rid of this hack
    int is_valid;

    // Modulus for Monty arithmetic
    fixnum mod;
    // R_mod = 2^fixnum::BITS % mod
    modnum R_mod;
    // Rsqr = R^2 % mod
    modnum Rsqr_mod;

    // TODO: We save this after using it in the constructor; work out
    // how to make it available for later use. For example, it could
    // be used to reduce arguments to modexp prior to the main
    // iteration.
    quorem_preinv<fixnum> modrem;
};


template< typename fixnum >
__device__
monty<fixnum>::monty(fixnum modulus)
: mod(modulus), modrem(modulus)
{
    // mod must be odd > 1 in order to calculate R^-1 mod "mod".
    // FIXME: Handle these errors properly
    if (fixnum::two_valuation(modulus) != 0 //fixnum::get(modulus, 0) & 1 == 0
            || fixnum::cmp(modulus, fixnum::one()) == 0) {
        is_valid = 0;
        return;
    }
    is_valid = 1;

    fixnum Rsqr_hi, Rsqr_lo;

    // R_mod = R % mod
    modrem(R_mod, fixnum::one(), fixnum::zero());
    fixnum::sqr_wide(Rsqr_hi, Rsqr_lo, R_mod);
    // Rsqr_mod = R^2 % mod
    modrem(Rsqr_mod, Rsqr_hi, Rsqr_lo);
}

} // End namespace internal

} // End namespace cuFIXNUM
