#pragma once

#include "functions/quorem.cu"

template< typename fixnum_impl >
class chinese : public managed {
    typedef typename fixnum_impl::word_tp word_tp;
    static constexpr int WIDTH = fixnum_impl::WIDTH;

    // FIXME: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.
    word_tp p[WIDTH];
    word_tp q[WIDTH];
    word_tp c[WIDTH]; // c = p^-1 (mod q)

    quorem mod_q;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    chinese(const uint8_t *p, const uint8_t *q, const uint8_t *p_inv_modq, size_t nbytes);
};

template< typename fixnum_impl >
chinese<fixnum_impl>::chinese(
    const uint8_t *p, const uint8_t *q, const uint8_t *p_inv_modq, size_t nbytes)
    : mod_q(q, nbytes)
{
    if (nbytes > FIXNUM_BYTES)
        throw std::exception("parameters are too big"); // TODO: More precise exception
    memset(this->p, 0, FIXNUM_BYTES);
    memcpy(this->p, p, nbytes);
    // TODO: q is now stored here and in mod_q; need to work out how
    // to share q between them.
    memset(this->q, 0, FIXNUM_BYTES);
    memcpy(this->q, q, nbytes);
    memset(this->c, 0, FIXNUM_BYTES);
    memcpy(this->c, p_inv_modq, nbytes);
}

/*
 * CRT on Mp and Mq.
 *
 * Mp, Mq, p, q must all be WIDTH/2 digits long
 *
 * Source HAC, Note 14.75.
 */
template< typename fixnum_impl >
__device__ void
chinese<fixnum_impl>::operator()(fixnum &m, fixnum mp, fixnum mq) const
{
    // u = (mq - mp) * c (mod q)
    fixnum p, q, c;
    fixnum u = 0, t = 0, hi, lo;
    int br = fixnum_impl::sub_br(u, mq, mp);

    p = fixnum_impl::load(this->p);
    q = fixnum_impl::load(this->q);
    c = fixnum_impl::load(this->c);
    
    // TODO: It would be MUCH better to ensure that the mul_wide
    // and mod_q parts of this condition occur on the main
    // execution path to avoid long warp divergence.
    if (br) {
        // Mp > Mq
        // TODO: Can't I get this from u above?  Need a negation
        // function; maybe use "method of complements".
        br = fixnum_impl::sub_br(u, mp, mq);
        assert(br == 0);

        // TODO: Replace mul_wide with the equivalent mul_lo
        //digit_mul(hi, lo, u, c, width/2);
        fixnum_impl::mul_wide(hi, lo, u, c);
        assert(hi == 0);

        t = 0;
        //quorem_rem(mod_q, t, hi, lo, width/2);
        mod_q(t, hi, lo);

        br = fixnum_impl::sub_br(u, q, t);
        assert(br == 0);
    } else {
        // Mp < Mq
        // TODO: Replace mul_wide with the equivalent mul_lo
        //digit_mul(hi, lo, u, c, width/2);
        fixnum_impl::mul_wide(hi, lo, u, c);
        assert(hi == 0);

        u = 0;
        //quorem_rem(mod_q, u, hi, lo, width/2);
        mod_q(u, hi, lo);
    }
    // TODO: Replace mul_wide with the equivalent mul_lo
    //digit_mul(hi, lo, u, p, width/2);
    fixnum_impl::mul_wide(hi, lo, u, p);
    //shfl_up(hi, width/2, width);
    //t = (L < width/2) ? lo : hi;
    assert(hi == 0);
    t = lo;

    //digit_add(m, mp, t, width);
    fixnum_impl::add_cy(m, mp, t);
}
