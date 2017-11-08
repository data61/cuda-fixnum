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

#endif
