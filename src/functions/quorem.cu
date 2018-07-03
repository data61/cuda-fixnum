#pragma once

#include <stdexcept>

#include "util/gmp_utils.h"
#include "util/managed.cu"

/*
 * Quotient and remainder via Barrett reduction.
 *
 * div: the divisor
 * (mu, mu_msw): floor(2^(2*NBITS) / div) where NBITS = FIXNUM_BITS
 * width: digits in div and mu.
 */
template< typename fixnum_impl >
class quorem : public managed {
    typedef typename fixnum_impl::word_tp word_tp;
    static constexpr int fixnum_impl::SLOT_WIDTH WIDTH;

    word_tp div[WIDTH];
    word_tp mu[WIDTH];
    word_tp mu_msw;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    quorem(const uint8_t *div, size_t nbytes);

    __device__ void operator()(fixnum &q, fixnum &r, fixnum A_hi, fixnum A_lo) const;
};

/*
 * Create a quorem object.
 *
 * Throws an exception if div does not have a sufficiently high bit switched
 * on, or if nbytes > FIXNUM_BYTES.
 */
template< typename fixnum_impl >
quorem<fixnum_impl>::quorem(const uint8_t *div, size_t nbytes)
{
    // Require at least one of the high 4 bits to be switched on. This
    // determines the maximum number of corrections needed at the end
    // of a reduction in quorem_rem.
    static constexpr word_tp MIN_MSW = (word_tp)1 << (WORD_BITS - 5);

    if (nbytes > FIXNUM_BYTES)
        throw std::exception("divisor is too big"); // TODO: More precise exception
    memcpy(this->div, div, nbytes);
    if (div[WIDTH - 1] < MIN_MSW)
        throw std::exception("divisor is too small"); // TODO: More precise exception
    get_mu<WIDTH>(mu, mu_msw, div);
}


/*
 * Return the quotient and remainder of A after division by this->div.
 *
 * Uses Barret reduction.  See HAC, Algo 14.42, and MCA, Algo 2.5.
 *
 * TODO: Explain how this implementation deviates from the algorithms
 * cited above, in particular how it relates to the expected number of
 * iterations of the "correction loop".  NB: It's possible this
 * "deviation" can be removed by "normalising" the relevant data (see
 * MCA, Section 1.4.1).
 */
template< typename fixnum_impl >
__device__ void
quorem<fixnum_impl>::operator()(fixnum &q, fixnum &r, fixnum A_hi, fixnum A_lo) const
{
    word_tp d, t, msw, hi, lo, br;

    int L = slot_layout::laneIdx();

    // (q, msw) = "A_hi * mu / 2^NBITS"
    // TODO: the lower half of the product, t, is unused, so we might
    // be able to use a mul_hi() function that only calculates an
    // approximate answer (see Short Product discussion at MCA,
    // Section 3.3 (from Section 2.4.1, p59)).
    fixnum_impl::mul_wide(q, t, A_hi, fixnum_impl::load(mu));
    msw = fixnum_impl::mad_cy(q, A_hi, mu_msw);

    // (hi, lo) = q*d
    d = fixnum_impl::load(div);
    fixnum_impl::mul_wide(hi, lo, q, d);
    msw = fixnum_impl::mad_cy(hi, d, msw);

    // q*d always fits in two fixnums, even though msw of q is
    // sometimes non-zero.
    assert(msw == 0);

    // (r, msw) = A - q*d
    br = fixnum_impl::sub_br(r, A_lo, lo);
    t = fixnum_impl::sub_br(msw, A_hi, hi);

    // A_hi >= hi
    assert(t == 0);

    // make br into a fixnum
    // FIXME: check why I can't just do "msw -= br" here.
    br = (L == 0) ? br : 0;
    t = fixnum_impl::sub_br(msw, msw, br);

    // msw >= br
    assert(t == 0);
    // msw < 2^64
    assert(L == 0 || msw == 0);
    msw = slot_layout::shfl(msw, 0);

    while (msw) {
        msw -= fixnum_impl::sub_br(r, r, d);
        fixnum_impl::incr_cy(q);
    }
    while ( ! fixnum_impl::sub_br(t, r, d)) {
        r = t;
        fixnum_impl::incr_cy(q);
    }
}

