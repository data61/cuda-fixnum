#pragma once

#include "functions/modinv.cu"

template< typename fixnum_impl >
class divexact {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ divexact(fixnum divisor) {
        b = divisor;

        // divisor must be odd
        // TODO: Handle even divisor. Should be easy: just make sure
        // the 2-part of the divisor and dividend are the same and
        // then remove them.
        fixnum b0 = fixnum_impl::get(b, 0);
        assert(b0 & 1);

        // Calculate b inverse
        modinv<fixnum_impl> minv;
        minv(bi, b, fixnum_impl::FIXNUM_BITS/2);
    }

    /*
     * q = a / b, assuming b divides a.
     *
     * Source: MCA Algorithm 1.10.
     */
    __device__ void operator()(fixnum &q, fixnum a) const {
        fixnum t, w = 0;

        // w <- a bi  (mod 2^(NBITS / 2))

        // FIXME: This is wasteful since we only want the bottom half of the
        // result. Could we do something like:
        //
        //   create half_fixnum which is fixnum_impl< FIXNUM_BYTES / 2 > but
        //   with same slot_layout. Then use half_fixnum::mul_lo(w, a, bi)
        //
        fixnum_impl::mul_lo(w, a, bi);
        // FIXME: This doesn't work when SLOT_WIDTH = 0
        //w = (fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH / 2) ? w : 0;

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

private:
    // Divisor
    fixnum b;
    // 1/b (mod 2^(NBITS/2)) where NBITS := FIXNUM_BITS.  bi is
    // nevertheless treated as an NBITS fixnum, so its hi half must be
    // all zeros.
    fixnum bi;
};
