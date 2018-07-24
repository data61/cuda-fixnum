#pragma once

#include "functions/quorem_preinv.cu"
#include "functions/multi_modexp.cu"

template< typename fixnum_impl >
class chinese {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ chinese(fixnum p, fixnum q);

    __device__ void operator()(fixnum &m, fixnum mp, fixnum mq) const;

private:
    // TODO: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.
    fixnum p, q, c;  // c = p^-1 (mod q)

    quorem_preinv<fixnum_impl> mod_q;
};

template< typename fixnum_impl >
__device__
chinese<fixnum_impl>::chinese(fixnum p_, fixnum q_)
    : p(p_), q(q_), mod_q(q)
{
    // TODO: q is now stored here and in mod_q; need to work out how
    // to share q between them.  Probably best just to provide quorem_preinv
    // with an accessor to the divisor.

    // TODO: Make modinv use xgcd and use modinv instead.
    // Use a^(q-2) = 1 (mod q)
    fixnum qm2, two = (fixnum_impl::slot_layout::laneIdx() == 0) ? 2 : 0;
    fixnum_impl::sub_br(qm2, q, two);
    multi_modexp<fixnum_impl> minv(q);
    minv(c, p, qm2);
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
    fixnum u = 0, t = 0, hi, lo;
    int br = fixnum_impl::sub_br(u, mq, mp);

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
