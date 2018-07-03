#pragma once

#include "util/managed.cu"
#include "functions/modinv.cu"

template< typename fixnum_impl >
struct divexact : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    /*
     * TODO: This should take the divisor as an argument and store its inverse
     * bi as an instance variable.
     */
    divexact() { }

    /*
     * q = a / b, assuming b divides a. Optional bi must be 1/B (mod 2^(NBITS/2))
     * where NBITS := FIXNUM_BITS.  bi is nevertheless treated as an
     * NBITS fixnum, so its hi half must be all zeros.
     *
     * Source: MCA Algorithm 1.10.
     */
    __device__ void operator()(fixnum &q, fixnum a, fixnum b, fixnum bi = 0) const {
        fixnum t, w = 0;

        // b must be odd
        // TODO: Handle even b.
        fixnum b0 = fixnum_impl::get(b, 0);
        assert(b0 & 1);

        // Calculate b inverse if it is not provided.
        if ( ! fixnum_impl::nonzero_mask(bi)) {
            modinv minv;
            minv(bi, b, FIXNUM_BITS/2);
        }

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
