#pragma once

#include "util/managed.cu"
#include "util/primitives.cu"
#include "util/gmp_utils.h"

template< typename fixnum_impl >
class monty_mul : public managed {
    typedef typename fixnum_impl::word_tp word_tp;
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // FIXME: These definitions assume that fixnum_impl is the default classical
    // implementation!! They should be set with a generic set method.

    // Modulus for Monty arithmetic
    word_tp mod[WIDTH];
    // R = 2^FIXNUM_BITS % mod
    word_tp R_mod[WIDTH];
    // Rsqr = R^2 % mod
    word_tp R_sqr_mod[WIDTH];
    // inv_mod * mod = -1 % 2^DIGIT_BITS.
    word_tp  inv_mod;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    monty_mul(const uint8_t *modulus, size_t modulus_bytes);

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
        (*this)(z, x, fixnum_impl::load(R_sqr_mod));
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
        return fixnum_impl::load(R_mod);
    }

private:
    __device__ void normalise(fixnum &x, int msb, fixnum m) const;
};


// TODO: These two functions should be in a utilities file. Also, they are
// specific to the default classical number representation.

/*
 * Return 1 if x is even, 0 if it is odd.
 */
template<typename digit_t>
int
is_even(digit_t x) { return !(x & 1); }

/*
 * Return 1 if the n digit number x is even or equal to one (1), 0
 * otherwise (ie. odd > 1).
 */
template<typename digit_t>
int
is_even_or_one(const digit_t *x, int n)
{
    // x is 0 or even:
    if (n == 0 || is_even(x[0])) return 1;
    // x is odd > 1
    if (x[0] > 1) return 0;
    // x = 1 iff we don't find a non-zero word
    int i = 1;
    for (; i < n; ++i)
        if (x[i]) break;
    return i == n;
}

template< typename fixnum_impl >
monty_mul<fixnum_impl>::monty_mul(const uint8_t *modulus, size_t modulus_bytes)
{
    // FIXME: It should probably be an error to provide a modulus that's too
    // big.
    size_t nbytes = std::min((size_t)fixnum_impl::FIXNUM_BYTES, modulus_bytes);
    // mod must be odd > 1 in order to calculate R^-1 mod "mod".
    // FIXME: Handle this error properly
    assert( ! is_even_or_one(modulus, nbytes));
    memset(mod, 0, sizeof(word_tp) * WIDTH);
    memcpy(mod, modulus, nbytes);
    get_R_and_Rsqr_mod<WIDTH>(R_mod, R_sqr_mod, modulus, nbytes);
    inv_mod = get_invmod(modulus, nbytes, fixnum_impl::WORD_BITS);
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
    const fixnum mod = fixnum_impl::load(this->mod);
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

        //UADDC(z, cy, z, cy)
        z += cy;
        cy = z < cy;

        umad_hi_cc(z, cy, mod, u, z);
        umad_hi_cc(z, cy, y, xi, z);
    }
    // Resolve carries
    word_tp msw = fixnum_impl::most_sig_dig(cy);
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
