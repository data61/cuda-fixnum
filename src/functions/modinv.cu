#pragma once

#include "util/primitives.cu"

/*
 * Calculate the modular inverse.
 * TODO: Only supports moduli of the form 2^k at the moment.
 */
template< typename fixnum_impl >
struct modinv {
    typedef typename fixnum_impl::fixnum fixnum;

    __host__ __device__ modinv() { }

    /*
     * Return x = 1/b (mod 2^k).  Must have 0 < k <= FIXNUM_BITS.
     *
     * Source: MCA Algorithm 1.10.
     *
     * TODO: Calculate this using the multiple inversion trick (MCA 2.5.1)
     */
    __device__ void operator()(fixnum &x, fixnum b, int k) const {
        static constexpr int WORD_BITS = fixnum_impl::WORD_BITS;
        // b must be odd
        fixnum b0 = fixnum_impl::get(b, 0);
        assert(k > 0 && k <= fixnum_impl::FIXNUM_BYTES * 8);

        fixnum binv = modinv_2k(b0);
        x = 0;
        fixnum_impl::set(x, binv, 0);
        if (k < WORD_BITS)
            x &= (1UL << k) - 1UL;
        if (k <= WORD_BITS)
            return;

        // Hensel lift x from (mod 2^WORD_BITS) to (mod 2^k)
        // FIXME: Double-check this condition on k!
        while (k >>= 1) {
            fixnum t;
            // TODO: Make multiplications faster by using the "middle
            // product" (see MCA 1.4.5 and 3.3.2).
            fixnum_impl::mul_lo(t, b, x);
            fixnum_impl::sub_br(t, fixnum_impl::one(), t);
            fixnum_impl::mul_lo(t, t, x);
            fixnum_impl::add_cy(x, x, t);
        }
    }
};

