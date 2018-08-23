#pragma once

template< typename fixnum >
class monty_redc {
    typedef typename fixnum::digit digit;

    // Modulus for Monty arithmetic
    fixnum mod;
    // inv_mod * mod = -1 % 2^digit::BITS.
    digit  inv_mod;

public:
    __device__ void operator()(fixnum &s, fixnum r_hi, fixnum r_lo) {
        static constexpr int T = fixnum::layout::toplaneIdx;
        static constexpr int WIDTH = fixnum::SLOT_WIDTH;
        int L = slot_layout::laneIdx();
        digit x, msw;

        x = r_lo;
        for (int i = 0; i < WIDTH; ++i) {
            digit u = fixnum::get(r_lo, i) * inv_mod;
            msw = fixnum::addmuli(x, mod, u);

            r_lo = (L == i) ? msw : r_lo;
            x = layout::shfl_down0(x, 1);
            r_hi_i = fixnum::get(r_hi, i);
            x = (L == T) ? r_hi_i : x;
        }
        fixnum::add_cy(s, msw, x, r_lo);
        normalise(s, msw, mod);
    }
};
