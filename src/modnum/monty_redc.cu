#pragma once

namespace cuFIXNUM {

template< typename fixnum >
class monty_redc {
    // Modulus for Monty arithmetic
    fixnum mod;
    // inv_mod * mod = -1 % 2^fixnum::BITS.
    fixnum inv_mod;

public:
    __device__ monty_redc(fixnum mod_)
    : mod(mod_) {
        modinv<fixnum> minv;
        minv(inv_mod, mod, fixnum::BITS);
        fixnum::neg(inv_mod);
    }

    __device__ void operator()(fixnum &r, fixnum a_hi, fixnum a_lo) {
        fixnum b, s_hi, s_lo;
        digit cy;

        fixnum::mul_lo(b, a_lo, inv_mod);

        // This section is essentially s = floor(mad_wide(b, mod, a) / R)

        // TODO: Can we employ the trick to avoid a multiplication because we
        // know b = am' (mod R)?
        fixnum::mul_wide(s_hi, s_lo, b, mod);
        // TODO: Only want the carry; find a cheaper way to determine that
        // without doing the full addition.
        fixnum::add_cy(s_lo, cy, s_lo, a_lo);
#ifndef NDEBUG
        // NB: b = am' (mod R) => a + bm = a + amm' = 2a (mod R). So surely
        // all I need to propagate is the top bit of a_lo?
        fixnum top_bit, dummy;
        fixnum::lshift(dummy, top_bit, a_lo, 1);
        assert(digit::cmp(cy, top_bit) == 0);
#endif
        fixnum::add(r, s_hi, a_hi);
        fixnum::add(r, r, cy);

        if (fixnum::cmp(r, mod) >= 0)
            sub(r, r, mod);
    }
};

} // End namespace cuFIXNUM
