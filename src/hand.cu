#ifndef CUDA_MPMA_HAND_CU
#define CUDA_MPMA_HAND_CU

#include "utils.cu"

/*
 * A "hand" encapsulates our notion of a fixed collection of digits in
 * a multiprecision number.
 *
 * A hand is stored in a warp register, and so implicitly consumes 32
 * times the size of the register.  The interface can support
 * different underlying representations however, including a "full
 * utilisation always handle carries" implementation, a "avoid carries
 * with nail bits" implementation, as well as more exotic
 * implementations using floating-point registers.
 *
 * A limb is an array of digits stored in memory. Chunks of digits are
 * loaded into hands in order for arithmetic to be performed on
 * numbers. So a limb can be thought of as a sequence of hands.
 *
 * The interface to hands and the interface to limbs is (largely) the
 * same. The interface for limbs is defined in terms of the one for
 * hands.
 */

class number {
public:
    //  +=  +  -  <<  >>  &=
    //  <  >

    number add(const number x, const number y);
    number sub(const number x);
    number neg();
    number mulwide(const number x);
    number mullo(const number x);
    number mulhi(const number x);

    int lt(number x, number y);
    int gt(number x, number y);

    // multiply-by-zero-or-one
    number keep_or_kill(int x);
};


template< typename digit, int width = warpSize >
class hand {
    digit d;

public:
    //  +=  +  -  <<  >>  &=
    //  <  >

    // multiply-by-zero-or-one
};

template< typename digit, typename subwarp = subwarp_data<> >
__device__ int
hand_resolve_cy(digit &r, int cy)
{
    constexpr digit DIGIT_MAX = ~(digit)0;
    int L = subwarp::laneIdx();
    uint32_t allcarries, p, g;
    int cy_hi;

    g = subwarp::ballot(cy);                  // carry generate
    p = subwarp::ballot(r == DIGIT_MAX);      // carry propagate
    allcarries = (p | g) + g;                 // propagate all carries
    cy_hi = allcarries < g;                   // detect final overflow
    allcarries = (allcarries ^ p) | (g << 1); // get effective carries
    r += (allcarries >> L) & 1;

    // return highest carry
    return cy_hi;
}

template< typename digit, typename subwarp = subwarp_data<> >
__device__ int
hand_add_cy(digit &r, digit a, digit b)
{
    int cy;

    r = a + b;
    cy = r < a;

    return hand_resolve_cy<digit, subwarp>(r, cy);
}


template< typename digit >
__device__ __forceinline__ void
hand_add(digit &r, digit a, digit b)
{
    r = a + b;
}


template< typename digit, int NAIL_BITS >
struct nail_data
{
    typedef typename digit digit;
    // FIXME: This doesn't work if digit is signed
    constexpr digit DIGIT_MAX = ~(digit)0;
    constexpr int DIGIT_BITS = sizeof(digit) * 8;
    constexpr int NON_NAIL_BITS = DIGIT_BITS - NAIL_BITS;
    constexpr digit NAIL_MASK = DIGIT_MAX << NON_NAIL_BITS;
    constexpr digit NON_NAIL_MASK = ~NAIL_MASK;
    constexpr digit NON_NAIL_MAX = NON_NAIL_MASK; // alias

    // A nail must fit in an int.
    static_assert(NAIL_BITS > 0 && NAIL_BITS < sizeof(int) * 8,
            "invalid number of nail bits");
};


// TODO: This is ugly
template< typename digit, int NAIL_BITS >
__device__ __forceinline__ int
hand_extract_nail(digit &r)
{
    typedef nail_data<digit, NAIL_BITS> nd;

    // split r into nail and non-nail parts
    nail = r >> nd::NON_NAIL_BITS;
    r &= nd::NON_NAIL_MASK;
    return nail;
}


/*
 * Current cost of nail resolution is 4 vote functions.
 */
template< typename digit, int NAIL_BITS >
__device__ int
hand_resolve_nails(digit &r)
{
    // TODO: Make this work with a general width
    constexpr int WIDTH = warpSize;
    // TODO: This is ugly
    typedef nail_data<digit, NAIL_BITS> nd;
    typedef subwarp_data<WIDTH> subwarp;

    int nail, nail_hi;
    nail = hand_extract_nail<digit, NAIL_BITS>(r);
    nail_hi = subwarp::shfl(nail, subwarp::toplaneIdx);

    nail = subwarp::shfl_up0(nail, 1);
    r += nail;

    // nail is 0 or 1 this time
    nail = hand_extract_nail<digit, NAIL_BITS>(r);

    return nail_hi + hand_resolve_cy(r, nail, nd::NON_NAIL_MAX);
}


/*
 * r = lo_half(a * b) with subwarp size width.
 *
 * The "lo_half" is the product modulo 2^(???), i.e. the same size as
 * the inputs.
 */
template< typename digit, int WIDTH = warpSize >
__device__ void
hand_mullo_cy(digit &r, digit a, digit b)
{
    typedef subwarp_data<WIDTH> subwarp;

    // TODO: This should be smaller, probably uint16_t (smallest
    // possible for addition).  Strangely, the naive translation to
    // the smaller size broke; to investigate.
    digit cy = 0;

    r = 0;
    for (int i = WIDTH - 1; i >= 0; --i) {
        digit aa = subwarp::shfl(a, i);

        // TODO: See if using umad.wide improves this.
        umad_hi_cc(r, cy, aa, b, r);
        r = subwarp::shfl_up0(r, 1);
        cy = subwarp::shfl_up0(cy, 1);
        umad_lo_cc(r, cy, aa, b, r);
    }
    cy = subwarp::shfl_up0(cy, 1);
    (void) hand_add_cy(r, r, cy); // FIXME: Should take a WIDTH
}


template< typename digit, int NAIL_BITS, int WIDTH = warpSize >
__device__ void
hand_mullo_nail(digit &r, digit a, digit b)
{
    // FIXME: We shouldn't need nail bits to divide the width
    static_assert(!(WIDTH % NAIL_BITS), "nail bits does not divide width");
    // FIXME: also need to check that digit has enough space for the
    // accumulated nails.

    typedef subwarp_data<WIDTH> subwarp;

    digit n = 0; // nails

    r = 0;
    for (int i = WIDTH - 1; i >= 0; --i) {
        // FIXME: Should this be NAIL_BITS/2? Because there are two
        // additions (hi & lo)? Maybe at most one of the two additions
        // will cause an overflow? For example, 0xff * 0xff = 0xfe01
        // so overflow is likely in the first case and unlikely in the
        // second...
        for (int j = 0; j < NAIL_BITS; ++j, --i) {
            digit aa = subwarp::shfl(a, i);

            // TODO: See if using umad.wide improves this.
            umad_hi(r, aa, b, r);
            r = subwarp::shfl_up0(r, 1);
            umad_lo(r, aa, b, r);
        }
        // FIXME: Supposed to shuffle up n by NAIL_BITS digits
        // too. Can this be avoided?
        n += hand_extract_nails(r);
    }
    n = subwarp::shfl_up0(n, 1);
    hand_add(r, r, n);
    hand_resolve_nails(r);
}

#endif
