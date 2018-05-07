#pragma once

#include "util/managed.cu"

template< typename fixnum_impl >
struct square : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a) {
        fixnum_impl::mul_lo(r, a, a);
    }
};
