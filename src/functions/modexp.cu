#pragma once

#include "functions/monty_mul.cu"

template< typename fixnum >
class modexp {
    typedef typename fixnum::digit digit;

    // TODO: Update the comment below
    // TODO: These should be determined by the exponent. Current choices
    // are optimal for 1024 bit exponents.
    //
    // For various exponent bit lengths, this GP code prints the best choice of
    // window size k for reducing multiplications, however it doesn't consider the
    // possibility of register usage (e.g. for 1024 bit exponents, 5 is a better
    // choice than the optimal 6, since 6 saves only two multiplications at a
    // cost of 32 registers).
    //
    // forstep(b=256,8200,256,
    //   my (W = [[k, ceil(2^k + b * (1 + 1/k))] | k <- [1 .. 16]],
    //       m = vecsort(W, 2)[1]);
    //   print(b, ": ", m))
    //
    // NB: The best window size exceeds MAX_WINDOW_BITS=8 when b ~ 18700 bits.
    // FIXME: The formula above differs slightly from MCA, Section 2.6.2. Work out
    // which is correct.

    // Decomposition of the exponent for use in the varying-width sliding-window
    // algorithm.  Allocated & deallocated once per thread block. Ref:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#per-thread-block-allocation
    // TODO: Consider storing the whole exp_wins array in shared memory.
    uint32_t *exp_wins;
    int exp_wins_len;
    int window_size;

    // TODO: Generalise modexp so that it can work with any modular
    // multiplication algorithm.
    const monty_mul<fixnum> monty;

    // Helper functions for decomposing the exponent into windows.
    __device__ uint32_t
    scan_window(int &hi_idx, fixnum &n, int max_window_bits);

    __device__ int
    scan_zero_window(int &hi_idx, fixnum &n);

    __device__ uint32_t
    scan_nonzero_window(int &hi_idx, fixnum &n, int max_window_bits);

public:
    /*
     * NB: It is assumed that the caller has reduced exp and mod using knowledge
     * of their properties (e.g. reducing exp modulo phi(mod), CRT, etc.).
     */
    __device__ modexp(fixnum mod, fixnum exp);

    __device__ ~modexp();

    __device__ void operator()(fixnum &z, fixnum x) const;
};


template< typename fixnum >
__device__ uint32_t
modexp<fixnum>::scan_nonzero_window(int &hi_idx, fixnum &n, int max_window_bits) {
    uint32_t bits_remaining = hi_idx + 1, win_bits;
    digit w, lsd = fixnum::bottom_digit(n);

    internal::min(win_bits, bits_remaining, max_window_bits);
    digit::rem_2exp(w, lsd, win_bits);
    fixnum::rshift(n, n, win_bits);
    hi_idx -= win_bits;

    return w;
}


template< typename fixnum >
__device__ int
modexp<fixnum>::scan_zero_window(int &hi_idx, fixnum &n) {
    int nzeros = fixnum::two_valuation(n);
    fixnum::rshift(n, n, nzeros);
    hi_idx -= nzeros;
    return nzeros;
}


template< typename fixnum >
__device__ uint32_t
modexp<fixnum>::scan_window(int &hi_idx, fixnum &n, int max_window_bits) {
    int nzeros;
    uint32_t window;
    nzeros = scan_zero_window(hi_idx, n);
    window = scan_nonzero_window(hi_idx, n, max_window_bits);
    // top half is the odd window, bottom half is nzeros
    // TODO: fix magic number
    return (window << 16) | nzeros;
}

template< typename fixnum >
__device__
modexp<fixnum>::modexp(fixnum mod, fixnum exp)
    : monty(mod)
{
    // sliding window decomposition
    int hi_idx;

    hi_idx = fixnum::msb(exp);
    // TODO: select window size properly
    window_size = 5;

    // Allocate exp_wins once per threadblock.
    __shared__ uint32_t *data;
    if (threadIdx.x == 0) {
        int max_windows;
        internal::ceilquo(max_windows, fixnum::BITS, window_size);
        data = (uint32_t *) malloc(max_windows * sizeof(uint32_t));
        // FIXME: Handle this error properly.
        assert(data != nullptr);
    }
    // Synchronise threads before using data.
    __syncthreads();

    exp_wins = data;
    uint32_t *ptr = exp_wins;
    while (hi_idx >= 0)
        *ptr++ = scan_window(hi_idx, exp, window_size);
    exp_wins_len = ptr - exp_wins;
}


template< typename fixnum >
__device__
modexp<fixnum>::~modexp()
{
    // Synchronise threads before freeing data.
    __syncthreads();
    if (threadIdx.x == 0)
        free(exp_wins);
}


template< typename fixnum >
__device__ void
modexp<fixnum>::operator()(fixnum &z, fixnum x) const
{
    static constexpr int WINDOW_MAX_BITS = 16;
    static constexpr int WINDOW_LEN_MASK = (1UL << WINDOW_MAX_BITS) - 1UL;
    // TODO: Actual maximum is 16 at the moment (see above), but it will very
    // rarely need to be more than 7. Consider storing G in shared memory to
    // remove the need for WINDOW_MAX_BITS altogether.
    static constexpr int WINDOW_MAX_BITS_REDUCED = 7;
    static constexpr int WINDOW_MAX_VAL_REDUCED = 1U << WINDOW_MAX_BITS_REDUCED;
    assert(window_size <= WINDOW_MAX_BITS_REDUCED);

    // We need to know that exp_wins_len > 0 when z is initialised just before
    // the main loop.
    if (exp_wins_len == 0) {
        z = fixnum::one();
        return;
    }

    // TODO: handle case of small exponent specially

    int window_max = 1U << window_size;
    /* G[t] = z^(2t + 1) t >= 0 (odd powers of z) */
    fixnum G[WINDOW_MAX_VAL_REDUCED / 2];
    monty.to_monty(z, x);
    G[0] = z;
    if (window_size > 1) {
        monty(z, z);
        for (int t = 1; t < window_max / 2; ++t) {
            G[t] = G[t - 1];
            monty(G[t], G[t], z);
        }
    }

    // Iterate over windows from most significant window to least significant
    // (i.e. reverse order from the order they're stored).
    const uint32_t *windows = exp_wins + exp_wins_len - 1;
    uint32_t win = *windows--;
    uint16_t two_val = win & WINDOW_LEN_MASK;
    uint16_t e = win >> WINDOW_MAX_BITS;

    z = G[e / 2];
    while (two_val-- > 0)
        monty(z, z);

    while (windows >= exp_wins) {
        two_val = window_size;
        while (two_val-- > 0)
            monty(z, z);

        win = *windows--;
        two_val = win & WINDOW_LEN_MASK;
        e = win >> WINDOW_MAX_BITS;

        monty(z, z, G[e / 2]);
        while (two_val-- > 0)
            monty(z, z);
    }
    monty.from_monty(z, z);
}
