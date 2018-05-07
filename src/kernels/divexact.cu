#pragma once

template< typename fixnum_impl >
struct divexact {
    typedef typename fixnum_impl::fixnum fixnum;

    // FIXME: It would be better to set bi here.
    divexact() { }

    /*
     * q = a / b, assuming b divides a. bi must be 1/B (mod 2^(NBITS/2))
     * where NBITS := SLOT_WIDTH*FIXNUM_BITS.  bi is nevertheless treated as an
     * NBITS intmod, so its hi half must be all zeros.
     *
     * Source: MCA Algorithm 1.10.
     */
    __device__ void operator()(fixnum &q, fixnum a, fixnum b, fixnum bi) {
        fixnum t, w = 0;

        // NBITS := SLOT_WIDTH*FIXNUM_BITS
        // w <- a bi  (mod 2^(NBITS / 2))
        // FIXME: Original has half width:
        //digit_mullo(w, a, bi, width / 2);
        fixnum_impl::mul_lo(w, a, bi);
        // t <- b w    (mod 2^NBITS)
        fixnum_impl::mul_lo(t, b, w);
        // t <- a - b w (mod 2^NBITS)
        fixnum_impl::sub_br(t, a, t);
        // t <- bi (a - b w) (mod 2^NBITS)
        fixnum_impl::mul_lo(t, bi, t);
        // w <- w + bi (a - b w)
        fixnum_impl::add_cy(w, w, t);

        q = w;
    }
};
