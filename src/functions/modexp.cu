#pragma once

#include "util/managed.cu"
#include "util/primitives.cu"

#include "functions/monty_mul.cu"

template< template<class> multiply, template<class> square, typename fixnum_impl >
class modexp : public managed {
    typedef typename fixnum_impl::word_tp word_tp;
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;

    // TODO: These should be determined by the exponent. Current choices
    // are optimal for 1024 bit exponents.
    static constexpr int WINDOW_BITS = 5, MAX_WINDOW_BITS = 8;
    static constexpr int WINDOW_MAX = (1U << WINDOW_BITS);
    static constexpr int WINDOW_LEN_MASK = (1U << MAX_WINDOW_BITS) - 1;

    static_assert(WINDOW_BITS > 0,
                  "window must have non-zero length");
    static_assert(WINDOW_BITS <= MAX_WINDOW_BITS,
                  "currently only support windows of length at most 8 bits");

    // Decomposition of the exponent for use in the varying-width
    // sliding-window algorithm.
    int      exp_decomp_len;
    uint16_t *exp_decomp;

    const multiply<fixnum_impl> mul;
    const square<fixnum_impl> sqr;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    modexp(const multiply<fixnum_impl> &mul_, const square<fixnum_impl> &sqr_, const uint8_t *exp, size_t explen);

    __device__ void operator()(fixnum &z, fixnum x) const;
};

/*
 * Sliding window parameters
 */

/*
 * Interpreting x as an INTMOD_BIT-integer, return 1 if the bit at
 * index b is on, 0 otherwise.
 */
static inline int
has_bit(const digit_t *x, int b)
{
    int q, r;
    assert(b >= 0 && b < INTMOD_BITS);

    q = b / DIGIT_BITS;
    r = b % DIGIT_BITS;
    return !!(x[q] & (1UL << r));
}

/*
 * Return the decomposition of the exponent e into windows suitable
 * for the sliding window modexp implementation.
 *
 * Writing e in binary as (e_n...e_0)_2, put into each element of
 * *exp_decomp the values
 *
 *   (i - s + 1) and (e_i ... e_s)_2
 *
 * packed as two ubytes in a single uint16.  See HAC Algo 14.85 for
 * details.  The parameter e_msb should be the index of the most
 * significant bit of e (can be omitted).
 */
static void
get_window_decomp(uint16_t **exp_decomp, int *exp_decomp_len, const digit_t *e, int width, int e_msb = -1)
{
    int i, j, s, n = 0;
    uint16_t *decomp;

    i = e_msb < 0 ? msb(e, width*DIGIT_BITS - 1) : e_msb;
    assert(i > 0);
    *exp_decomp = (uint16_t *) malloc(sizeof(uint16_t)*i);  // overestimate
    decomp = *exp_decomp;

    while (i >= 0) {
        if (has_bit(e, i)) {
            uint8_t a = 0;
            s = max(i - WINDOW_BITS + 1, 0);
            while ( ! has_bit(e, s))
                ++s;
            // Copy the (i - s + 1)-bit number whose least significant
            // bit is at s into a.
            // TODO: This is a brain-dead way of achieving this, but
            // it's easy to debug.
            for (j = 0; j < i - s + 1; ++j)
                a |= has_bit(e, j + s) << j;
            decomp[n++] = j | a << MAX_WINDOW_BITS;
            i = s - 1;
        } else {
            s = 1;
            while (--i >= 0 && ! has_bit(e, i)) {
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
modexp<fixnum_impl>::operator()(fixnum &z, fixnum x) {
    /* G[0] = z, G[t] = z^(2t + 1) t > 0 (odd powers of z) */
    fixnum G[WINDOW_MAX / 2];
    z = x;
    G[0] = z;
    if (WINDOW_BITS > 1) {
        sqr(z, z);
        for (int t = 1; t < WINDOW_MAX / 2; ++t) {
            G[t] = G[t - 1];
            mul(G[t], G[t], z);
        }
    }

    z = fixnum_impl::get(this->R_mod[L]);
    const int decomp_len = this->exp_decomp_len;
    const uint16_t *decomp = this->exp_decomp;
    for (int i = 0; i < decomp_len; ++i) {
        uint16_t win = *decomp++;
        uint8_t wlen = win & WINDOW_LEN_MASK;
        uint8_t e = win >> MAX_WINDOW_BITS;
        while (wlen-- > 0)
            sqr(z, z);
        if (e)
            mul(z, z, G[e / 2]);
    }
}

