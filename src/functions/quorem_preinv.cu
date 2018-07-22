#pragma once

#include "functions/quorem.cu"

/*
 * Quotient and remainder via Barrett reduction.
 *
 * div: the divisor
 * mu: floor(2^(2*NBITS) / div) where NBITS = FIXNUM_BITS (note: mu has an
 * implicit hi bit).
 */
template< typename fixnum_impl >
class quorem_preinv {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ quorem_preinv(fixnum div);

    // Assume clz(A) <= clz(div)
    __device__ void operator()(fixnum &q, fixnum &r, fixnum A_hi, fixnum A_lo) const;

    // Just return the remainder.
    __device__ void operator()(fixnum &r, fixnum A_hi, fixnum A_lo) const {
        fixnum q;
        (*this)(q, r, A_hi, A_lo);
    }

    // TODO: This should be somewhere more appropriate.
    __device__ static void reciprocal_approx(fixnum &mu, fixnum div);

private:
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // Note that mu has an implicit hi bit that is always on.
    fixnum div, mu;
    int lz;
};

// Assumes div has been normalised already.
// NB: Result has implicit hi bit on.
// TODO: This function should be generalised and made available more widely
template< typename fixnum_impl >
__device__ void
quorem_preinv<fixnum_impl>::reciprocal_approx(fixnum &mu, fixnum div)
{
    // Let B = 2^FIXNUM_BITS

    // Initial estimate is 2*B - div = B + (B - div)  (implicit hi bit)
    // TODO: Use better initial estimate: (48/17) - (32/17)*div (see
    // https://en.wikipedia.org/wiki/Division_algorithm#Newton-Raphson_division)
    fixnum_impl::neg(mu, div);

    // If we initialise mu = 2*B - div, then the error is 1.0 - mu*div/B^2 < 1/4.
    // In general, the error after iteration k = 0, 1, ... less than 1/(4^(2^k)).
    // We need an error less than 1/B^2, hence k >= log2(log2(B)).
    static constexpr uint32_t FIXNUM_BITS = 8 * fixnum_impl::FIXNUM_BYTES;
    // FIXME: For some reason this code doesn't converge as fast as it should.
    const int NITERS = ctz(FIXNUM_BITS); // TODO: Make ctz, hence NITERS, a constexpr
    int L = fixnum_impl::slot_layout::laneIdx();

    // TODO: Instead of calculating/using floor(B^2/div), calculate/use the
    // equivalent  floor((B^2 - 1)/div) - B  as described in the Möller & Granlund
    // paper; this should allow simplification because there's no implicit hi bit
    // in mu to account for.
    for (int i = 0; i < NITERS; ++i) {
        fixnum cy, br;
        fixnum a, b, c, d, e;

        // (hi, lo) = B^2 - mu*div. This is always positive.
        fixnum_impl::mul_wide(a, b, mu, div);
        cy = fixnum_impl::add_cy(a, a, div);  // account for hi bit of mu
        assert(cy == 0);  // cy will be 1 when mu = floor(B^2/div), which happens on the last iteration
        br = fixnum_impl::sub_br(b, fixnum_impl::zero(), b); // br == 0 iff b == 0.
        br = (L == 0) ? br : 0;
        fixnum_impl::neg(a, a);
        fixnum_impl::sub_br(a, a, br);

        // TODO: a + c is actually correct to within a single bit; investigate
        // whether using a mu that is off by one bit matters? If it does, we
        // should only do this correction on the last iteration.
        // TODO: Implement fused-multiply-add and use it here for "a*mu + b".
        fixnum_impl::mul_wide(c, d, a, mu);
        cy = fixnum_impl::add_cy(d, d, b);
        cy = (L == 0) ? cy : 0;
        cy = fixnum_impl::add_cy(c, c, cy);
        assert(cy == 0);

        // cy is the single extra bit that propogates to (a + c)
        fixnum_impl::mul_hi(e, mu, b);
        cy = fixnum_impl::add_cy(d, d, e);
        cy = (L == 0) ? cy : 0;

        // mu += a + c + cy_in
        cy = fixnum_impl::add_cy(a, a, cy);  assert(cy == 0);
        cy = fixnum_impl::add_cy(mu, mu, c); assert(cy == 0);
        cy = fixnum_impl::add_cy(mu, mu, a); assert(cy == 0);
    }
}


/*
 * Create a quorem_preinv object.
 *
 * Raise an error if div does not have a sufficiently high bit switched
 * on.
 */
template< typename fixnum_impl >
__device__
quorem_preinv<fixnum_impl>::quorem_preinv(fixnum div_)
    : div(div_)
{
    lz = quorem<fixnum_impl>::normalise_divisor(div);
    reciprocal_approx(mu, div);
}

/*
 * Return the quotient and remainder of A after division by div.
 *
 * Uses Barret reduction.  See HAC, Algo 14.42, and MCA, Algo 2.5.
 */
template< typename fixnum_impl >
__device__ void
quorem_preinv<fixnum_impl>::operator()(
    fixnum &q, fixnum &r, fixnum A_hi, fixnum A_lo) const
{
    fixnum t;
    int cy, L = fixnum_impl::slot_layout::laneIdx();

    // Normalise A
    // TODO: Rather than normalising A, we should incorporate the
    // normalisation factor into the algorithm at the appropriate
    // place.
    t = quorem<fixnum_impl>::normalise_dividend(A_hi, A_lo, lz);
    assert(t == 0);  // FIXME: check if t == 0 using cmp() and zero()

    // q = "A_hi * mu / 2^NBITS"
    // TODO: the lower half of the product, t, is unused, so we might
    // be able to use a mul_hi() function that only calculates an
    // approximate answer (see Short Product discussion at MCA,
    // Section 3.3 (from Section 2.4.1, p59)).
    fixnum_impl::mul_wide(q, t, A_hi, mu);
    cy = fixnum_impl::add_cy(q, q, A_hi); // mu has implicit hi bit
    assert(cy == 0);

    quorem<fixnum_impl>::quorem_with_candidate_quotient(q, r, A_hi, A_lo, div, q);

    // Denormalise r
    fixnum lo_bits = fixnum_impl::rshift(r, r, lz);
    assert(lo_bits == 0);
}
