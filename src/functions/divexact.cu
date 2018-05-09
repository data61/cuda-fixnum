#pragma once

#include "util/managed.cu"

template< typename fixnum_impl >
struct divexact : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    // FIXME: It would be better to set bi here.
    divexact() { }

    /*
     * q = a / b, assuming b divides a. bi must be 1/B (mod 2^(NBITS/2))
     * where NBITS := SLOT_WIDTH*FIXNUM_BITS.  bi is nevertheless treated as an
     * NBITS fixnum, so its hi half must be all zeros.
     *
     * Source: MCA Algorithm 1.10.
     */
    __device__ void operator()(fixnum &q, fixnum a, fixnum b, fixnum bi) {
        fixnum t, w = 0;

        // NBITS := SLOT_WIDTH*FIXNUM_BITS

        // w <- a bi  (mod 2^(NBITS / 2))

        // FIXME: This is wasteful since we only want the bottom half of the
        // result. Could we do something like:
        //
        //   create half_fixnum which is fixnum_impl< FIXNUM_BYTES / 2 > but
        //   with same slot_layout. Then use half_fixnum::mul_lo(w, a, bi)
        //
        fixnum_impl::mul_lo(w, a, bi);
        w = (slot_layout::laneIdx() < slot_layout::WIDTH / 2) ? w : 0;

        // TODO: Can use the "middle product" to speed this up a
        // bit. See MCA Section 1.4.5.
        // t <- b w (mod 2^NBITS)
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
