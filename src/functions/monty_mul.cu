#pragma once

#include "functions/modinv.cu"
#include "functions/quorem_preinv.cu"

namespace cuFIXNUM {

template< typename fixnum >
class monty_mul {
public:
    __device__ monty_mul(fixnum modulus);

    /**
     * z <- x * y
     */
    __device__ void operator()(fixnum &z, fixnum x, fixnum y) const;

    /**
     * z <- x^2
     */
    __device__ void operator()(fixnum &z, fixnum x) const {
        // TODO: Implement my fancy squaring algo.
        (*this)(z, x, x);
    }

    // TODO: Might be worth specialising monty_mul for this case, since one of
    // the operands is known.
    __device__ void to_monty(fixnum &z, fixnum x) const {
        (*this)(z, x, Rsqr_mod);
    }

    // TODO: Might be worth specialising monty_mul for this case, since one of
    // the operands is known.
    __device__ void from_monty(fixnum &z, fixnum x) const {
        (*this)(z, x, fixnum::one());
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ fixnum one() const {
        return R_mod;
    }

private:
    typedef typename fixnum::digit digit;
    // TODO: Check whether we can get rid of this declaration
    static constexpr int WIDTH = fixnum::SLOT_WIDTH;

    // FIXME: Get rid of this hack
    int is_valid;

    // Modulus for Monty arithmetic
    fixnum mod;
    // R_mod = 2^fixnum::BITS % mod
    fixnum R_mod;
    // Rsqr = R^2 % mod
    fixnum Rsqr_mod;
    // inv_mod * mod = -1 % 2^digit::BITS.
    digit  inv_mod;

    // TODO: We save this after using it in the constructor; work out
    // how to make it available for later use. For example, it could
    // be used to reduce arguments to modexp prior to the main
    // iteration.
    quorem_preinv<fixnum> modrem;

    __device__ void normalise(fixnum &x, int msb, fixnum m) const;
};


template< typename fixnum >
__device__
monty_mul<fixnum>::monty_mul(fixnum modulus)
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

    // TODO: Tidy this up.
    modinv<fixnum> minv;
    fixnum im;
    minv(im, mod, digit::BITS);
    digit::neg(inv_mod, im);
    // TODO: Ugh.
    typedef typename fixnum::layout layout;
    // TODO: Can we avoid this broadcast?
    inv_mod = layout::shfl(inv_mod, 0);
    assert(1 + inv_mod * layout::shfl(mod, 0) == 0);
}

/*
 * z = x * y (mod) in Monty form.
 *
 * Spliced multiplication/reduction implementation of Montgomery
 * modular multiplication.  Specifically it is the CIOS (coursely
 * integrated operand scanning) splice.
 */
template< typename fixnum >
__device__ void
monty_mul<fixnum>::operator()(fixnum &z, fixnum x, fixnum y) const
{
    typedef typename fixnum::layout layout;
    // FIXME: Fix this hack!
    z = fixnum::zero();
    if (!is_valid) { return; }

    int L = layout::laneIdx();
    digit tmp;
    digit::mul_lo(tmp, x, inv_mod);
    digit::mul_lo(tmp, tmp, fixnum::get(y, 0));
    digit cy = digit::zero();

    for (int i = 0; i < WIDTH; ++i) {
        digit u;
        digit xi = fixnum::get(x, i);
        digit z0 = fixnum::get(z, 0);
        digit tmpi = fixnum::get(tmp, i);

        digit::mad_lo(u, z0, inv_mod, tmpi);

        digit::mad_lo_cy(z, cy, mod, u, z);
        digit::mad_lo_cy(z, cy, y, xi, z);

        assert(L || digit::is_zero(z));  // z[0] must be 0
        z = layout::shfl_down0(z, 1); // Shift right one word

        digit::add_cy(z, cy, z, cy);

        digit::mad_hi_cy(z, cy, mod, u, z);
        digit::mad_hi_cy(z, cy, y, xi, z);
    }
    // Resolve carries
    digit msw = fixnum::top_digit(cy);
    cy = layout::shfl_up0(cy, 1); // left shift by 1
    fixnum::add_cy(z, cy, z, cy);
    digit::add(msw, msw, cy);
    assert(msw == !!msw); // msw = 0 or 1.

    normalise(z, (int) msw, mod);
}

/*
 * Let X = x + msb * 2^64.  Then return X -= m if X > m.
 *
 * Assumes X < 2*m, i.e. msb = 0 or 1, and if msb = 1, then x < m.
 */
template< typename fixnum >
__device__ void
monty_mul<fixnum>::normalise(fixnum &x, int msb, fixnum m) const {
    fixnum r;
    digit br;

    // br = 0 ==> x >= m
    fixnum::sub_br(r, br, x, m);
    if (msb || digit::is_zero(br)) {
        // If the msb was set, then we must have had to borrow.
        assert(!msb || msb == br);
        x = r;
    }
}

} // End namespace cuFIXNUM
