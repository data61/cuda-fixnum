
namespace internal {
    __constant__ int DIGITREMAP[WARPSIZE / 2][2][WARPSIZE - 1] = {
        {
            {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
            {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }
        },
        {
            {  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 16 },
            {  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 17 }
        },
        {
            {  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 16, 17 },
            {  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 18, 18 }
        },
        {
            {  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16, 17, 18 },
            {  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 19, 19, 19 }
        },
        {
            {  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 16, 17, 18, 19 },
            {  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 20, 20, 20, 20 }
        },
        {
            {  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 16, 17, 18, 19, 20 },
            {  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 21, 21, 21, 21, 21 }
        },
        {
            {  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 16, 17, 18, 19, 20, 21 },
            {  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 22, 22, 22, 22, 22, 22 }
        },
        {
            {  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 16, 17, 18, 19, 20, 21, 22 },
            {  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 23, 23, 23, 23, 23, 23, 23 }
        },
        {
            {  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8, 16, 17, 18, 19, 20, 21, 22, 23 },
            {  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 24, 24, 24, 24, 24, 24, 24, 24 }
        },
        {
            {  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
            { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 25, 25, 25, 25, 25, 25, 25, 25, 25 }
        },
        {
            { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 },
            { 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26 }
        },
        {
            { 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 },
            { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27 }
        },
        {
            { 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 },
            { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28 }
        },
        {
            { 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 },
            { 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29 }
        },
        {
            { 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
            { 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30 }
        },
        {
            { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
            { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31 }
        }
    };

    /*
     * Adapt "rediagonalisation" trick described in Figure 4 of Ozturk,
     * Guilford, Gopal (2013) "Large Integer Squaring on Intel
     * Architecture Processors".
     */
    template< typename layout >
    __device__ void
    sqr_wide(fixnum &ss, fixnum &rr, fixnum a)
    {
        int L = layout::laneIdx();

        fixnum r, s;
        r = fixnum::zero();
        s = fixnum::zero();
        digit cy = digit::zero();

        // WARNING: See section "Dynamic Indexing with Non-Uniform Access" here:
        // https://devblogs.nvidia.com/parallelforall/fast-dynamic-indexing-private-arrays-cuda/
        // Suggests that it might be wise to put tblmap_* in shared memory.
        for (int i = 0; i < layout::WIDTH / 2; ++i) {
            fixnum a1, a2;
            // ti = tblmap[i][L] is the lane for this iteration of squaring.
            a1 = get(a, tblmap_lhs[i][L]);
            a2 = get(a, tblmap_rhs[i][L]);

            digit::mad_lo_cc(s, a1, a2, s);

            fixnum s0 = get(s, 0);
            r = (L == i) ? s0 : r; // r[i] = s[0]
            s = layout::shfl_down0(s, 1);

            digit::addc_cc(s, s, cy);  // add carry from prev digit
            digit::addc(cy, 0, 0);     // cy = CC.CF
            digit::mad_hi_cy(s, cy, a1, a2, s);
        }

        cy = layout::shfl_up0(cy, 1);
        add(s, s, cy);

        fixnum overflow;
        lshift(s, s, digit::BITS + 1);  // s *= 2
        lshift(r, overflow, r, digit::BITS + 1);  // r *= 2
        // Propagate r overflow to s
        add_cy(s, cy, s, overflow); // really a logior, since s was just lshifted.
        assert(digit::is_zero(cy));

        // Calculate and add the squares on the diagonal. NB: by parallelising
        // the multiplication like this then shuffling the results, we lose the
        // ability to use a fused multiply-and-add call.
        digit::mul_wide(ai_sqr_hi, ai_sqr_lo, a, a);

        // NB: This version saves one shuffle at the expense of more complicated
        // lane calculations (~5 vs ~2 bitops). Alternative version is in a
        // comment below.
        int is_hi_half = L >= layout::WIDTH / 2;
        int lane_shift = (L << 1) & (layout::WIDTH - 1); // 2*L % width;
        t_lo = layout::shfl(ai_sqr_lo, lane_shift + is_hi_half);
        t_hi = layout::shfl(ai_sqr_hi, lane_shift + !is_hi_half);
        r_diag = (L & 1) ? t_hi : t_lo;
        t_hi = (L & 1) ? t_lo : t_hi;
        s_diag = layout::shfl(t_hi, L^1U); // switch odd and even lanes
#if 0
        t_lo = layout::shfl(ai_sqr_lo, L / 2);
        t_hi = layout::shfl(ai_sqr_hi, L / 2);
        r_diag = (L & 1) ? t_hi : t_lo;
        t_lo = layout::shfl(ai_sqr_lo, (L + layout::WIDTH) / 2);
        t_hi = layout::shfl(ai_sqr_hi, (L + layout::WIDTH) / 2);
        s_diag = (L & 1) ? t_hi : t_lo;
#endif

        add_cy(r, cy, r, r_diag);
        add(s, s, s_diag);
        add(s, s, cy);
    }
}
