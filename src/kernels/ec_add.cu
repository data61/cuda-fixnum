#pragma once

#include "set_const.cu"

// fixnum_impl is like a policy in a policy-based design
// (https://en.wikipedia.org/wiki/Policy-based_design).
template< typename fixnum_impl >
struct ec_add : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    set_const<fixnum_impl> *set_k;

    ec_add(/* ec params */ long k = 17)
    : set_k(set_const<fixnum_impl>::create(k)) { }

    ~ec_add() { delete set_k; }

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum k;
        (*set_k)(k);
        fixnum_impl::mul_lo(r, a, k);
        fixnum_impl::mul_lo(r, r, b);
        fixnum_impl::mul_lo(r, r, r);
        fixnum_impl::mul_lo(r, r, r);
    }
};
