#pragma once

#include "util/primitives.cu"

/*
 * Quotient and remainder via long-division.
 *
 * Source: MCA Algo 1.6, HAC Algo 14.20.
 *
 * TODO: Implement Svoboda divisor preconditioning (using
 * Newton-Raphson iteration to calculate floor(beta^(n+1)/div)) (see
 * MCA Algo 1.7).
 */
template< typename fixnum_impl >
class quorem {
    static constexpr int WIDTH = fixnum_impl::SLOT_WIDTH;
    typedef typename fixnum_impl::word_tp word_tp;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(
        fixnum &q, fixnum &r,
        fixnum A, fixnum div) const;

    // TODO: These functions obviously belong somewhere else. The need
    // to be available to both quorem (here) and quorem_preinv.
    static __device__ int normalise_divisor(fixnum &div);
    static __device__ fixnum normalise_dividend(fixnum &u, int k);
    static __device__ fixnum normalise_dividend(fixnum &u_hi, fixnum &u_lo, int k);
    static __device__ void quorem_with_candidate_quotient(
        fixnum &quo, fixnum &rem,
        fixnum A_hi, fixnum A_lo, fixnum div, fixnum q);
};

template< typename fixnum_impl >
__device__ int
quorem<fixnum_impl>::normalise_divisor(fixnum &div) {
    static constexpr int FIXNUM_BITS = fixnum_impl::FIXNUM_BYTES * 8;
    int lz = FIXNUM_BITS - (fixnum_impl::msb(div) + 1);
    fixnum overflow = fixnum_impl::lshift(div, div, lz);
    assert(overflow == 0);
    return lz;
}

// TODO: Ideally the algos would be written to incorporate the
// normalisation factor, rather than "physically" normalising the
// dividend.
template< typename fixnum_impl >
__device__ typename fixnum_impl::fixnum
quorem<fixnum_impl>::normalise_dividend(fixnum &u, int k) {
    return fixnum_impl::lshift(u, u, k);
}

// TODO: Ideally the algos would be written to incorporate the
// normalisation factor, rather than "physically" normalising the
// dividend.
template< typename fixnum_impl >
__device__ typename fixnum_impl::fixnum
quorem<fixnum_impl>::normalise_dividend(fixnum &u_hi, fixnum &u_lo, int k) {
    fixnum hi_part = fixnum_impl::lshift(u_hi, u_hi, k);
    fixnum middle_part = fixnum_impl::lshift(u_lo, u_lo, k);
    int cy = fixnum_impl::add_cy(u_hi, u_hi, middle_part);
    assert(cy == 0);
    return hi_part;
}

template< typename fixnum_impl >
__device__ void
quorem<fixnum_impl>::quorem_with_candidate_quotient(
    fixnum &quo, fixnum &rem,
    fixnum A_hi, fixnum A_lo, fixnum div, fixnum q)
{
    fixnum hi, lo, msw, br, r, t;
    int L = fixnum_impl::slot_layout::laneIdx();

    // (hi, lo) = q*d
    fixnum_impl::mul_wide(hi, lo, q, div);

    // (msw, r) = A - q*d
    br = fixnum_impl::sub_br(r, A_lo, lo);
    t = fixnum_impl::sub_br(msw, A_hi, hi);
    assert(t == 0);  // A_hi >= hi

    // TODO: Could skip these two lines if we could pass br to the last
    // sub_br above as a "borrow in".
    // Make br into a fixnum
    br = (L == 0) ? br : 0;
    t = fixnum_impl::sub_br(msw, msw, br);
    assert(t == 0);  // msw >= br
    assert((L == 0 && msw < 4) || msw == 0); // msw < 4 (TODO: possibly should have msw < 3)
    // Broadcast
    msw = fixnum_impl::slot_layout::shfl(msw, 0);

    // NB: Could call incr_cy in the loops instead; as is, it will
    // incur an extra add_cy even when msw is 0 and r < d.
    fixnum q_inc = 0;
    while (msw) {
        msw -= fixnum_impl::sub_br(r, r, div);
        ++q_inc;
    }
    while ( ! fixnum_impl::sub_br(t, r, div)) {
        r = t;
        ++q_inc;
    }
    // TODO: Replace loops above with something like the one below,
    // which will reduce warp divergence a bit.
#if 0
    fixnum tmp, q_inc;
    while (1) {
        br = fixnum_impl::sub_br(tmp, r, div);
        if (msw == 0 && br == 1)
            break;
        msr -= br;
        ++q_inc;
        r = tmp;
    }
#endif

    q_inc = (L == 0) ? q_inc : 0;
    fixnum_impl::add_cy(q, q, q_inc);

    quo = q;
    rem = r;
}

#if 0
template< typename fixnum_impl >
__device__ void
quorem<fixnum_impl>::operator()(
    fixnum &q_hi, fixnum &q_lo, fixnum &r,
    fixnum A_hi, fixnum A_lo, fixnum div) const
{
    int k = normalise_divisor(div);
    fixnum t = normalise_dividend(A_hi, A_lo, k);
    assert(t == 0); // dividend too big.

    fixnum r_hi;
    (*this)(q_hi, r_hi, A_hi, div);

    // FIXME WRONG! r_hi is not a good enough candidate quotient!
    // Do div2by1 of (r_hi, A_lo) by div using that r_hi < div.
    // r_hi is now the candidate quotient
    fixnum qq = r_hi;
    if (fixnum_impl::cmp(A_lo, div) > 0)
        fixnum_impl::incr_cy(qq);

    quorem_with_candidate_quotient(q_lo, r, r_hi, A_lo, div, qq);

    word_tp lo_bits = fixnum_impl::rshift(r, r, k);
    assert(lo_bits == 0);
}
#endif

// TODO: Implement a specifically *parallel* algorithm for division,
// such as those of Takahashi.
template< typename fixnum_impl >
__device__ void
quorem<fixnum_impl>::operator()(
    fixnum &q, fixnum &r, fixnum A, fixnum div) const
{
    int n = fixnum_impl::most_sig_dig(div) + 1;
    assert(n >= 0); // division by zero.

    word_tp div_msw = fixnum_impl::get(div, n - 1);

    // TODO: Factor out the normalisation code.
    int k = clz(div_msw); // guaranteed to be >= 0, since div_msw != 0

    // div is normalised when its msw is >= 2^(WORD_BITS - 1),
    // i.e. when its highest bit is on, i.e. when the number of
    // leading zeros of msw is 0.
    if (k > 0) {
        fixnum h;
        // Normalise div by shifting it to the left.
        h = fixnum_impl::lshift(div, div, k);
        assert(h == 0);
        h = fixnum_impl::lshift(A, A, k);
        // FIXME: We should be able to handle this case.
        assert(h == 0);  // FIXME: check if h == 0 using cmp() and zero()
        div_msw <<= k;
    }

    int m = fixnum_impl::most_sig_dig(A) - n + 1;
    // FIXME: Just return div in this case
    assert(m >= 0); // dividend too small

    // TODO: Work out if we can just incorporate the normalisation factor k
    // into the subsequent algorithm, rather than actually modifying div and A.

    q = r = 0;

    // Set q_m
    word_tp qj;
    fixnum dj, tmp;
    // TODO: Urgh.
    typedef typename fixnum_impl::slot_layout slot_layout;
    dj = slot_layout::shfl_up0(div, m);
    int br = fixnum_impl::sub_br(tmp, A, dj);
    if (br) qj = 0; // dj > A
    else { qj = 1; A = tmp; }

    fixnum_impl::set(q, qj, m);

    word_tp dinv = uquorem_reciprocal(div_msw);
    for (int j = m - 1; j >= 0; --j) {
        word_tp a_hi, a_lo, hi, dummy;

        // (q_hi, q_lo) = floor((a_{n+j} B + a_{n+j-1}) / div_msw)
        // TODO: a_{n+j} is a_{n+j-1} from the previous iteration; hence I
        // should be able to get away with just one call to get() per
        // iteration.
        // TODO: Could normalise A on the fly here, one word at a time.
        a_hi = fixnum_impl::get(A, n + j);
        a_lo = fixnum_impl::get(A, n + j - 1);

        // TODO: uquorem_wide has a bad branch at the start which will
        // cause trouble when div_msw < a_hi is not universally true
        // across the warp. Need to investigate ways to alleviate that.
        uquorem_wide(qj, dummy, a_hi, a_lo, div_msw, dinv);

        dj = slot_layout::shfl_up0(div, j);
        hi = fixnum_impl::muli(tmp, qj, dj);
        assert(hi == 0);

        int iters = 0;
        fixnum AA;
        while (1) {
            br = fixnum_impl::sub_br(AA, A, tmp);
            if (!br)
                break;
            br = fixnum_impl::sub_br(tmp, tmp, dj);
            assert(br == 0);
            --qj;
            ++iters;
        }
        A = AA;
        assert(iters <= 2); // MCA, Proof of Theorem 1.3.
        fixnum_impl::set(q, qj, j);
    }
    // Denormalise A to produce r.
    tmp = fixnum_impl::rshift(r, A, k);
    assert(tmp == 0); // Above division should be exact.
}
