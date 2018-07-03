#pragma once

#include "util/managed.cu"

// TODO: This obviously belongs somewhere else.
typedef unsigned long ulong;

/*
 * Calculate the modular inverse.
 * TODO: Only supports moduli of the form 2^k at the moment.
 */
template< typename fixnum_impl >
struct modinv : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    modinv() { }

    /*
     * Return x = 1/b (mod 2^k).  Must have 0 < k <= FIXNUM_BITS.
     *
     * Source: MCA Algorithm 1.10.
     *
     * TODO: Calculate this using the multiple inversion trick (MCA 2.5.1)
     */
    __device__ void operator()(fixnum &x, fixnum b, int k) const {
        // b must be odd
        fixnum b0 = fixnum_impl::get(b, 0);
        assert(b0 & 1);
        assert(k > 0 && k <= FIXNUM_BITS);

        fixnum binv = modinv_2k(b0);
        if (k < WORD_BITS)
            binv &= ((word_tp)1 << k) - 1;
        x = 0;
        fixnum_impl::set(x, binv, 0);
        if (k <= WORD_BITS)
            return;

        // FIXME: Double-check this condition on k!
        while (k >>= 1) {
            // TODO: Make multiplications faster by using the "middle
            // product" (see MCA 1.4.5 and 3.3.2).
            fixnum_impl::mul_lo(t, b, x);
            fixnum_impl::sub_br(t, fixnum_impl::one(), t);
            fixnum_impl::mul_lo(t, t, x);
            fixnum_impl::add_cy(x, x, t);
        }
    }

private:
    /*
     * Return 1/b (mod 2^ULONG_BITS) where b is odd.
     *
     * Source: MCA, Section 2.5.
     */
    __host__ __device__ __forceinline__
    static ulong
    modinv_2k(ulong b) {
        assert(b & 1);

        // TODO: Could jump into this list of iterations according to value of k
        // which would save several multiplications when k <= ULONG_BITS/2.
        ulong x;
        x = (2 - b * b) * b;
        x *= 2 - b * x;
        x *= 2 - b * x;
        x *= 2 - b * x;
        x *= 2 - b * x;
        return x;
    }

    /*
     * Return 1/b (mod n) where n is 2^k and b is odd. Require k > 0 && k < ULONG_BITS.
     */
    __host__ __device__ __forceinline__
    static ulong
    modinv_2k(ulong b, ulong k) {
        assert(k > 0 && k < ULONG_BITS);
        return modinv_2k(b) & ((1UL << k) - 1);
    }
};

