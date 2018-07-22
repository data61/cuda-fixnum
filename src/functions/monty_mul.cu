#pragma once

#include "util/primitives.cu"
#include "functions/modinv.cu"
#include "functions/quorem_preinv.cu"

template< typename fixnum_impl >
class monty_mul {
public:
    typedef typename fixnum_impl::fixnum fixnum;

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
        (*this)(z, x, fixnum_impl::one());
    }

    /*
     * Return the Montgomery image of one.
     */
    __device__ fixnum one() const {
        return R_mod;
    }

private:
    typedef typename fixnum_impl::word_tp word_tp;
    // TODO: Check whether we can get rid of this declaration
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // Modulus for Monty arithmetic
    fixnum mod;
    // R_mod = 2^FIXNUM_BITS % mod
    fixnum R_mod;
    // Rsqr = R^2 % mod
    fixnum Rsqr_mod;
    // inv_mod * mod = -1 % 2^WORD_BITS.
    word_tp  inv_mod;

    // TODO: We save this after using it in the constructor; work out
    // how to make it available for later use. For example, it could
    // be used to reduce arguments to modexp prior to the main
    // iteration.
    quorem_preinv<fixnum_impl> modrem;

    __device__ void normalise(fixnum &x, int msb, fixnum m) const;
};


template< typename fixnum_impl >
__device__
monty_mul<fixnum_impl>::monty_mul(fixnum modulus)
: mod(modulus), modrem(modulus)
{
    // mod must be odd > 1 in order to calculate R^-1 mod "mod".
    // FIXME: Handle these errors properly
    assert(fixnum_impl::two_valuation(modulus) == 0);
    assert(fixnum_impl::cmp(modulus, fixnum_impl::one()) != 0);

    fixnum Rsqr_hi, Rsqr_lo;
    int L = fixnum_impl::slot_layout::laneIdx();

    // R_mod = R % mod
    modrem(R_mod, fixnum_impl::one(), fixnum_impl::zero());
    fixnum_impl::sqr_wide(Rsqr_hi, Rsqr_lo, R_mod);
    // Rsqr_mod = R^2 % mod
    modrem(Rsqr_mod, Rsqr_hi, Rsqr_lo);

    // TODO: Tidy this up.
    modinv<fixnum_impl> minv;
    minv(inv_mod, mod, fixnum_impl::WORD_BITS);
    inv_mod = -inv_mod;
    // TODO: Ugh.
    typedef typename fixnum_impl::slot_layout slot_layout;
    // TODO: Can we avoid this broadcast?
    inv_mod = slot_layout::shfl(inv_mod, 0);
    assert(1 + inv_mod * slot_layout::shfl(mod, 0) == 0);
}

/*
 * z = x * y (mod me->mod) in Monty form.
 *
 * Spliced multiplication/reduction implementation of Montgomery
 * modular multiplication.  Specifically it is the CIOS (coursely
 * integrated operand scanning) splice.
 */
template< typename fixnum_impl >
__device__ void
monty_mul<fixnum_impl>::operator()(fixnum &z, fixnum x, fixnum y) const
{
    typedef typename fixnum_impl::slot_layout slot_layout;

    int L = slot_layout::laneIdx();
    const word_tp tmp = x * fixnum_impl::get(y, 0) * inv_mod;
    word_tp cy = 0;
    z = 0;

    for (int i = 0; i < WIDTH; ++i) {
        word_tp u;
        word_tp xi = fixnum_impl::get(x, i);
        word_tp z0 = fixnum_impl::get(z, 0);
        word_tp tmpi = fixnum_impl::get(tmp, i);

        umad_lo(u, z0, inv_mod, tmpi);

        umad_lo_cc(z, cy, mod, u, z);
        umad_lo_cc(z, cy, y, xi, z);

        assert(L || !z);  // z[0] must be 0
        z = slot_layout::shfl_down0(z, 1); // Shift right one word

        z += cy;
        cy = z < cy;

        umad_hi_cc(z, cy, mod, u, z);
        umad_hi_cc(z, cy, y, xi, z);
    }
    // Resolve carries
    word_tp msw = fixnum_impl::top_digit(cy);
    cy = slot_layout::shfl_up0(cy, 1); // left shift by 1
    msw += fixnum_impl::add_cy(z, z, cy);
    assert(msw == !!msw); // msw = 0 or 1.

    normalise(z, (int) msw, mod);
}

/*
 * Let X = x + msb * 2^64.  Then return X -= m if X > m.
 *
 * Assumes X < 2*m, i.e. msb = 0 or 1, and if msb = 1, then x < m.
 */
template< typename fixnum_impl >
__device__ void
monty_mul<fixnum_impl>::normalise(fixnum &x, int msb, fixnum m) const {
    fixnum r;
    int br;

    // br = 0 ==> x >= m
    br = fixnum_impl::sub_br(r, x, m);
    if (msb || !br) {
        // If the msb was set, then we must have had to borrow.
        assert(!msb || msb == br);
        x = r;
    }
}
