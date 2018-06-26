#pragma once

#include "util/managed.cu"
#include "util/primitives.cu"
#include "functions/monty_mul.cu"

// FIXME: Factor out code in common with modexp
#include "functions/modexp.cu"

template< typename fixnum_impl >
class multi_modexp : public managed {
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // TODO: Generalise multi_modexp so that it can work with any modular
    // multiplication algorithm.
    const monty_mul<fixnum_impl> monty;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    multi_modexp(const uint8_t *mod, size_t modbytes)
    : monty(mod, modbytes) { }

    __device__ void operator()(fixnum &z, fixnum x, fixnum e) const;
};

/*
 * Left-to-right k-ary exponentiation (see [HAC, Algorithm 14.82]).
 *
 * Note: I don't immediately see how to use the "modified" variant
 * [HAC, Algo 14.83] since there the number of squarings depends on
 * the 2-adic valuation of the window value.
 */
template< typename fixnum_impl >
__device__ void
multi_modexp<fixnum_impl>::operator()(fixnum &z, fixnum x, fixnum e) const
{
    // TODO: WINDOW_MAX should be determined by the length of e, or --
    // better -- by experiment.  The number of multiplications for
    // window size k is 2^k + 1024 * (1 + 1/k):
    //
    // ? [[k, ceil(2^k + 1024 * (1 + 1/k))] | k <- [1 .. 8]]
    // % = [[1, 2050], [2, 1540], [3, 1374], [4, 1296], [5, 1261], [6, 1259], [7, 1299], [8, 1408]]
    //
    // So the minimum for 1024-bit numbers occurs with a window size
    // of 6 bits, though 5 bits is extremely close and would save 32
    // registers per thread.
    //
    // TODO: This enum should be integrated with the similar code in
    // monty_modexp above.
    enum {
        WINDOW_MAIN_BITS = 5,
        WINDOW_REM_BITS = 4,
        WINDOW_MAX = (1U << WINDOW_MAIN_BITS),
        WINDOW_MAIN_MASK = (1U << WINDOW_MAIN_BITS) - 1,
        WINDOW_REM_MASK = (1U << WINDOW_REM_BITS) - 1
    };

    /* G[t] = z^t, t >= 0 */
    fixnum G[WINDOW_MAX];
    monty.to_monty(z, x);
    G[0] = monty.one();
    for (int t = 1; t < WINDOW_MAX; ++t) {
        G[t] = G[t - 1];
        monty(G[t], G[t], z);
    }

    z = G[0];
    for (int i = slot_layout::WIDTH - 1; i >= 0; --i) {
        fixnum f = fixnum_impl::get(e, i);

        // TODO: The squarings are noops on the first iteration (i =
        // w-1) and should be removed.
        //
        // Window decomposition: 64 = 12 * 5 + 4
        int win;
        for (int j = 64 - WINDOW_MAIN_BITS; j >= 0; j -= WINDOW_MAIN_BITS) {
            // TODO: For some bizarre reason, it is significantly
            // faster to do this loop than it is to unroll the 5
            // statements manually.  Idem for the remainder below.
            // Investigate how this is even possible!
            for (int k = 0; k < WINDOW_MAIN_BITS; ++k)
                monty(z, z);
            win = (f >> j) & WINDOW_MAIN_MASK;
            monty(z, z, G[win]);
        }

        // Remainder
        for (int k = 0; k < WINDOW_REM_BITS; ++k)
            monty(z, z);
        win = f & WINDOW_REM_MASK;
        monty(z, z, G[win]);
    }
    monty.from_monty(z, z);
}

#if 0
/*
 * Left-to-right k-ary exponentiation (see [HAC, Algorithm 14.82]).
 *
 * This is just multi_modexp::operator() specialised to the case where the
 * exponents all fit in a single digit.  It is noticeably faster than
 * multi_modexp::operator() for the same exponents.
 *
 * TODO: This function and the one above obviously should be
 * refactored somehow.
 */
__device__ void
monty_multi_modexp_smallexp(const ModexpCommon *me, digit_t &z, const digit_t &e)
{
    int w = me->width, L = laneIdx(w);

    // TODO: WINDOW_MAX should be determined by the length of e, or --
    // better -- by experiment.  The number of multiplications with a
    // b-bit exponent and window size k is 2^k + b * (1 + 1/k):
    //
    // The minimum for 53- and 64-bit numbers occurs with a window
    // size of 3 bits (79 and 94 mults respectively).  Assuming
    // sizeof(digit_t) = 8, we have
    //
    //   64 = 21 * 3 + 1
    //
    // TODO: This enum should be integrated with the similar code in
    // monty_modexp above.
    enum {
        WINDOW_MAIN_BITS = 3,
        WINDOW_REM_BITS = 1,
        WINDOW_MAX = (1U << WINDOW_MAIN_BITS),
        WINDOW_MAIN_MASK = (1U << WINDOW_MAIN_BITS) - 1,
        WINDOW_REM_MASK = (1U << WINDOW_REM_BITS) - 1
    };

    /* G[t] = z^i, i >= 0 */
    digit_t G[WINDOW_MAX];
    G[0] = me->R_mod[L];  // R is 1 in Montyland.
    for (int i = 1; i < WINDOW_MAX; ++i) {
        G[i] = G[i - 1];
        monty_mul(me, G[i], z);
    }

    // The "first iteration" can be simplified to just z = G[win].
    int j = 64 - WINDOW_MAIN_BITS;
    int win = (e >> j) & WINDOW_MAIN_MASK;
    z = G[win];
    j -= WINDOW_MAIN_BITS;

    do {
        for (int k = 0; k < WINDOW_MAIN_BITS; ++k)
            monty_sqr(me, z);
        win = (e >> j) & WINDOW_MAIN_MASK;
        monty_mul(me, z, G[win]);
        j -= WINDOW_MAIN_BITS;
    } while (j >= 0);

    // Remainder = 1
    monty_sqr(me, z);
    win = e & WINDOW_REM_MASK;
    monty_mul(me, z, G[win]);
}

/*
 * Let m be the modulus and exponent of *this. This calculates
 *
 *   z <- z^e (mod m)
 *
 * As is customary, z^0 will give 1 for all z, including z = 0.
 *
 * FIXME: There is a bug in this function (or maybe in
 * intmodvector_create) which causes a failure when ewidth != zwidth.
 */
DEV_INL void
MultiModexp::apply(digit_t Z[], int zwidth, const digit_t E[], int ewidth) const
{
    int w = impl.width;
    // TODO: We require w == zwidth only to ensure that there's enough
    // space in Z to put the result. An alternative would be to return
    // a new IntmodVector of the right size; this would allow zwidth
    // to be smaller than w (with the top digits set to zero as is
    // done with E), at the expense of more memory management.
    assert(w == zwidth && ewidth <= w);

    int L = laneIdx(w);
    digit_t z = Z[L];
    digit_t e = L < ewidth ? E[L] : 0;
    int b = d_msb(e, ewidth);
    if (b >= 0) {
        to_monty(&impl, z);

        if (b < 64) {
            monty_multi_modexp_smallexp(&impl, z, shfl(e, 0, w));
        } else {
            monty_multi_modexp(&impl, z, e, ewidth);
        }
        from_monty(&impl, z);
    } else {
        // e = 0, so return 1.
        z = (L == 0);
    }
    Z[L] = z;
}
#endif
