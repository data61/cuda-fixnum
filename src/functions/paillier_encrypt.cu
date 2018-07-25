#pragma once

#include "functions/quorem_preinv.cu"
#include "functions/multi_modexp.cu"

template< typename fixnum_impl >
class paillier_encrypt {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ paillier_encrypt(fixnum n_)
        : n(n_), n_sqr(square(n_)), pow(n_sqr, n_), mod_n2(n_sqr) { }

    /*
     * NB: In reality, the values r^n should be calculated out-of-band or
     * stock-piled and piped into an encryption function.
     */
    __device__ void operator()(fixnum &ctxt, fixnum m, fixnum r) const {
        // TODO: test this properly
        //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || m == 0);
        fixnum_impl::mul_lo(m, m, n);
        fixnum_impl::incr_cy(m);
        pow(r, r);
        fixnum c_hi, c_lo;
        fixnum_impl::mul_wide(c_hi, c_lo, m, r);
        mod_n2(ctxt, c_hi, c_lo);
    }

private:
    fixnum n;
    fixnum n_sqr;
    modexp<fixnum_impl> pow;
    quorem_preinv<fixnum_impl> mod_n2;

    // TODO: It is flipping stupid that this is necessary.
    __device__ fixnum square(fixnum n) {
        fixnum n2;
        // TODO: test this properly
        //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || n == 0);
        fixnum_impl::sqr_lo(n2, n);
        return n2;
    }
};

