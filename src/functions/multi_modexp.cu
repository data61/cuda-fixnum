#pragma once

#include "functions/monty_mul.cu"

template< typename fixnum_impl, int WINDOW_SIZE = 5 >
class multi_modexp {
    static_assert(WINDOW_SIZE >= 1 && WINDOW_SIZE <= 7,
                  "Invalid or unreasonable window size specified.");
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;
    static constexpr int WORD_BITS = fixnum_impl::WORD_BITS;

    // TODO: Generalise multi_modexp so that it can work with any modular
    // multiplication algorithm.
    const monty_mul<fixnum_impl> monty;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ multi_modexp(fixnum mod)
    : monty(mod) { }

    __device__ void operator()(fixnum &z, fixnum x, fixnum e) const;
};


// TODO: Finish this properly and use it to set the window length.
#if 0
static void
k_ary_window_params(
    int &win_quo, int &win_rem, int &win_size,
    int exp_bits, int word_bits)
{
    // Iterate through window sizes while nmults is decreasing; works because
    // 2^k + b(1 + 1/k) is convex.
    // TODO: These values should be determined by experiment on the hardware.
    // In particular, the choice below simply minimises the number of
    // multiplications; in many circumstances the trade-off with register
    // pressure would suggest using a smaller window size (basically because the
    // convex function above is quite flat around its minimum).

    // nmults <- 2(b + 1) = ceil(2^k + b(1 + 1/k)) at k = 1
    for (int k = 2, nmults = 2 * (exp_bits + 1); k < word_bits; ++k) {
        // nmults = ceil(2^k + b(1 + 1/k))
        int nmults_next = iceil(k * (1 << k) + exp_bits * (k + 1), k);
        // check for upturn
        if (nmults_next > nmults)
            break;
        nmults = nmults_next;
    }
    // k can't be bigger than a word.
    assert(k < word_bits);

    win_size = k;
    win_quo = word_bits / k;
    win_rem = word_bits % k;
}
#endif

/*
 * Left-to-right k-ary exponentiation (see [HAC, Algorithm 14.82]).
 *
 * Note: I don't immediately see how to use the "modified" variant
 * [HAC, Algo 14.83] since there the number of squarings depends on
 * the 2-adic valuation of the window value.
 */
template< typename fixnum_impl, int WINDOW_SIZE >
__device__ void
multi_modexp<fixnum_impl, WINDOW_SIZE>::operator()(fixnum &z, fixnum x, fixnum e) const
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
    static constexpr int WINDOW_MAIN_BITS = WINDOW_SIZE;
    static constexpr int WINDOW_REM_BITS = WORD_BITS % WINDOW_SIZE;
    static constexpr int WINDOW_MAX = (1U << WINDOW_MAIN_BITS);
    static constexpr int WINDOW_MAIN_MASK = (1U << WINDOW_MAIN_BITS) - 1;
    static constexpr int WINDOW_REM_MASK = (1U << WINDOW_REM_BITS) - 1;

    /* G[t] = z^t, t >= 0 */
    fixnum G[WINDOW_MAX];
    monty.to_monty(z, x);
    G[0] = monty.one();
    for (int t = 1; t < WINDOW_MAX; ++t) {
        G[t] = G[t - 1];
        monty(G[t], G[t], z);
    }

    z = G[0];
    for (int i = WIDTH - 1; i >= 0; --i) {
        fixnum f = fixnum_impl::get(e, i);

        // TODO: The squarings are noops on the first iteration (i =
        // w-1) and should be removed.
        //
        // Window decomposition: WORD_BITS = q * 5 + r
        int win;
        for (int j = WORD_BITS - WINDOW_MAIN_BITS; j >= 0; j -= WINDOW_MAIN_BITS) {
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
