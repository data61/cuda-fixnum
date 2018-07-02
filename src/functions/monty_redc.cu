#pragma once

#include "util/managed.cu"

template< typename fixnum_impl >
class monty_redc : public managed {
    typedef typename fixnum_impl::word_tp word_tp;
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // Modulus for Monty arithmetic
    word_tp mod[WIDTH];
    // inv_mod * mod = -1 % 2^DIGIT_BITS.
    word_tp  inv_mod;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &s, fixnum r) {
        static constexpr int T = slot_layout::toplaneIdx;
        int L = slot_layout::laneIdx();
        word_tp mod, inv_mod, x, msw;

        mod = this->mod[L];
        x = r;
        for (int i = 0; i < WIDTH; ++i) {
            word_tp u = slot_layout::shfl(r, i) * inv_mod;
            msw = fixnum_impl::addmuli(x, mod, u);

            r = (L == i) ? msw : r;
            x = slot_layout::shfl_down(x, 1);
            x = (L == T) ? slot_layout::shfl(s, i) : x;
        }
        msw = fixnum_impl::add_cy(s, x, r);
        digit_normalise(s, msw, mod);
    }
};
