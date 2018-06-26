#pragma once

#include "util/managed.cu"
#include "util/primitives.cu"
#include "functions/monty_mul.cu"

// FIXME: This belongs elsewhere. Perhaps I should have a "host word" typedef
// for this sort of thing?
typedef unsigned long ulong;

// FIXME: This is not the correct way to calculate ULONG_BITS.
// See https://stackoverflow.com/a/4589384
static constexpr int ULONG_BITS = sizeof(ulong) * 8;


template< typename fixnum_impl >
class modexp : public managed {
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

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

    static constexpr int WINDOW_BITS = 5, MAX_WINDOW_BITS = 8;
    static constexpr int WINDOW_MAX = (1U << WINDOW_BITS);
    static constexpr int WINDOW_LEN_MASK = (1U << MAX_WINDOW_BITS) - 1;

    static_assert(WINDOW_BITS > 0,
                  "window must have non-zero length");
    static_assert(WINDOW_BITS <= MAX_WINDOW_BITS,
                  "currently only support windows of length at most 8 bits");

    // Decomposition of the exponent for use in the varying-width sliding-window
    // algorithm.
    // TODO: Packing the data into a uint16_t is unnecessarily miserly,
    // complicates the implementation, and causes the limitation of MAX_WINDOW_BITS
    // above; change to something more general.
    int      exp_decomp_len;
    uint16_t *exp_decomp;

    // TODO: Generalise modexp so that it can work with any modular
    // multiplication algorithm.
    const monty_mul<fixnum_impl> monty;

    static void
    get_window_decomp(
        uint16_t **exp_decomp, int *exp_decomp_len,
        const ulong *exp, int explen, int e_msb = -1);

public:
    typedef typename fixnum_impl::fixnum fixnum;

    /*
     * NB: It is assumed that the caller has reduced exp and mod using knowledge
     * of their properties (e.g. reducing exp modulo phi(mod), CRT, etc.).
     */
    modexp(const uint8_t *mod, size_t modbytes, const uint8_t *exp, size_t expbytes);

    ~modexp() {
        if (exp_decomp_len)
            cuda_free(exp_decomp);
    }

    __device__ void operator()(fixnum &z, fixnum x) const;
};

// TODO: msb() and has_bit() are utility functions that belong somewhere else.

/*
 * x points to an array of ndigits longs. Return the index of the most
 * significant bit of x, or -1 if x is zero. Parameter start specifies the index
 * to start searching for a bit, which makes this function into something like
 * "find next bit". If start is negative, start from the highest bit position.
 * Must have start < ULONG_BITS * ndigits or you'll cause a segfault.
 */
static int
msb(const ulong *x, int ndigits, int start = -1)
{
    int i, r, total_bits = ULONG_BITS * ndigits;
    unsigned long d;
    assert(start < total_bits);

    if (start < 0)
        start = total_bits - 1;

    i = start / ULONG_BITS;
    r = start % ULONG_BITS;

    // Kill the top (ULONG_BITS - (r + 1)) bits of x[i].
    r = ULONG_BITS - (r + 1);
    d = (x[i] << r) >> r;

    // i is maximal such that x[i] != 0.
    if ( ! d) {
        for (--i; i >= 0 && ! x[i]; --i)
            ;
        // x = 0
        if (i < 0)
            return -1;
        d = x[i];
    }

    return ULONG_BITS - clz(d) + ULONG_BITS * i - 1;
}

/*
 * x points to an array of ndigits elements. Return 1 if the bit at index b is
 * on, 0 otherwise. Must have b < ULONG_BITS * ndigits or you'll cause a
 * segfault.
 */
static int
has_bit(const ulong *x, int b)
{
    int q, r;

    q = b / ULONG_BITS;
    r = b % ULONG_BITS;
    return !!(x[q] & (1UL << r));
}

/*
 * Return the decomposition of the exponent exp into windows suitable
 * for the sliding window modexp implementation.
 *
 * Writing exp in binary as (e_n...e_0)_2, put into each element of
 * *exp_decomp the values
 *
 *   (i - s + 1) and (e_i ... e_s)_2
 *
 * packed as two ubytes in a single uint16.  See HAC Algo 14.85 for
 * details.  The parameter e_msb should be the index of the most
 * significant bit of exp (can be omitted).
 */
template< typename fixnum_impl >
void
modexp<fixnum_impl>::get_window_decomp(
    uint16_t **exp_decomp, int *exp_decomp_len,
    const ulong *exp, int explen, int e_msb)
{
    int i, j, s, n = 0;
    uint16_t *decomp;

    i = e_msb < 0 ? msb(exp, explen) : e_msb;
    if (i < 0) {
        // exp = 0
        *exp_decomp = nullptr;
        *exp_decomp_len = 0;
        return;
    } else if (i == 0) {
        // i = 0 ==> exp = 1
        // TODO: This case should be integrated into the
        // code/loop below.
        *exp_decomp = new uint16_t[1];
        (*exp_decomp)[0] = 1 << MAX_WINDOW_BITS;
        *exp_decomp_len = 1;
        return;
    }

    *exp_decomp = new uint16_t[i];  // overestimate
    decomp = *exp_decomp;

    while (i >= 0) {
        if (has_bit(exp, i)) {
            uint8_t a = 0;
            s = max(i - WINDOW_BITS + 1, 0);
            while ( ! has_bit(exp, s))
                ++s;
            // Copy the (i - s + 1)-bit number whose least significant
            // bit is at s into a.
            // TODO: This is a brain-dead way of achieving this, but
            // it's easy to debug.
            for (j = 0; j < i - s + 1; ++j)
                a |= has_bit(exp, j + s) << j;
            decomp[n++] = j | a << MAX_WINDOW_BITS;
            i = s - 1;
        } else {
            s = 1;
            while (--i >= 0 && ! has_bit(exp, i)) {
                if (++s == (1 << MAX_WINDOW_BITS) - 1) {
                    --i;
                    break;
                }
            }
            decomp[n++] = s;
        }
    }
    *exp_decomp_len = n;
}


template< typename fixnum_impl >
modexp<fixnum_impl>::modexp(
    const uint8_t *mod, size_t modbytes,
    const uint8_t *exp_, size_t expbytes)
    : monty(mod, modbytes)
{
    ulong *exp;
    int expdigits;

    expdigits = iceil(expbytes, sizeof(ulong));
    exp = new ulong[expdigits];
    memset(exp, 0, expdigits * sizeof(ulong));
    memcpy(exp, exp_, expbytes);

    uint16_t *decomp;
    get_window_decomp(&decomp, &exp_decomp_len, exp, expdigits);

    if (exp_decomp_len) {
        int decomp_bytes = exp_decomp_len * sizeof(uint16_t);
        cuda_malloc_managed(&exp_decomp, decomp_bytes);
        memcpy(exp_decomp, decomp, decomp_bytes);
    }

    delete[] exp;
}


template< typename fixnum_impl >
__device__ void
modexp<fixnum_impl>::operator()(fixnum &z, fixnum x) const
{
    /* G[t] = z^(2t + 1) t >= 0 (odd powers of z) */
    fixnum G[WINDOW_MAX / 2];
    monty.to_monty(z, x);
    G[0] = z;
    if (WINDOW_BITS > 1) {
        monty(z, z);
        for (int t = 1; t < WINDOW_MAX / 2; ++t) {
            G[t] = G[t - 1];
            monty(G[t], G[t], z);
        }
    }

    z = monty.one();
    const int decomp_len = exp_decomp_len;
    const uint16_t *decomp = exp_decomp;
    for (int i = 0; i < decomp_len; ++i) {
        uint16_t win = *decomp++;
        uint8_t wlen = win & WINDOW_LEN_MASK;
        uint8_t e = win >> MAX_WINDOW_BITS;
        while (wlen-- > 0)
            monty(z, z);
        if (e)
            monty(z, z, G[e / 2]);
    }
    monty.from_monty(z, z);
}
