#pragma once

#include <math.h>
#include <type_traits>

#include "util/primitives.cu"
#include "slot_layout.cu"

/*
 * This is an archetypal implementation of a fixnum instruction
 * set. It defines the de facto interface for such implementations.
 *
 * All methods are defined for the device. It is someone else's
 * problem to get the data onto the device.
 */
template< int FIXNUM_BYTES_, typename word_tp_ = uint32_t >
class default_fixnum_impl {
    static_assert(FIXNUM_BYTES_ > 0,
            "Fixnum bytes must be positive.");
    static_assert(FIXNUM_BYTES_ % sizeof(word_tp_) == 0,
            "Fixnum word size must divide fixnum bytes.");
    static_assert(std::is_integral< word_tp_ >::value,
            "word_tp must be integral.");
    static constexpr int WORD_BITS = 8 * sizeof(word_tp_);

public:
    typedef word_tp_ word_tp;
    static constexpr int FIXNUM_BYTES = FIXNUM_BYTES_;
    static constexpr int SLOT_WIDTH = FIXNUM_BYTES_ / sizeof(word_tp_);
    // FIXME: slot_layout should not be exposed by this interface.
    typedef slot_layout< SLOT_WIDTH > slot_layout;
    typedef word_tp fixnum;

    /***************************
     * Representation functions.
     */

    /*
     * Set r using bytes, interpreting bytes as a base-256 unsigned
     * integer. Return the number of bytes used. If nbytes >
     * FIXNUM_BYTES, then the last nbytes - FIXNUM_BYTES are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int from_bytes(fixnum *r, const uint8_t *bytes, int nbytes) {
        uint8_t *s = reinterpret_cast< uint8_t * >(r);
        int n = min(nbytes, FIXNUM_BYTES);
        memcpy(s, bytes, n);
        memset(s + n, 0, FIXNUM_BYTES - n);
        return n;
    }

    /*
     * Set bytes using r, converting r to a base-256 unsigned
     * integer. Return the number of bytes written. If nbytes <
     * FIXNUM_BYTES, then the last FIXNUM_BYTES - nbytes are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int to_bytes(uint8_t *bytes, int nbytes, const fixnum *r) {
        int n = min(nbytes, FIXNUM_BYTES);
        memcpy(bytes, r, n);
        return n;
    }

    /*
     * load/set the value from ptr corresponding to this thread (lane) in
     * slot number idx.
     */
    __device__ static fixnum &load(fixnum *ptr, int idx = 0) {
        int off = idx * slot_layout::WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }

    /*
     * Return digit at index idx.
     *
     * FIXME: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static fixnum get(fixnum var, int idx) {
        return slot_layout::shfl(var, idx);
    }

    /*
     * Return most significant digit.
     *
     * FIXME: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static fixnum most_sig_dig(fixnum var) {
        return slot_layout::shfl(var, slot_layout::toplaneIdx);
    }

    /***********************
     * Arithmetic functions.
     */

    __device__ static int add_cy(fixnum &r, fixnum a, fixnum b) {
        int cy;
        r = a + b;
        cy = r < a;
        return resolve_carries(r, cy);
    }

    __device__ static int sub_br(fixnum &r, fixnum a, fixnum b) {
        int br;
        r = a - b;
        br = r > a;
        return resolve_borrows(r, br);
    }

    __device__ static fixnum zero() {
        return 0;
    }

    __device__ static fixnum one() {
        return (slot_layout::laneIdx() == 0);
    }

    __device__ static int incr_cy(fixnum &r) {
        return add_cy(r, r, one());
    }

    __device__ static int decr_br(fixnum &r) {
        return sub_br(r, r, one());
    }


    /*
     * r = lo_half(a * b)
     *
     * The "lo_half" is the product modulo 2^(8*FIXNUM_BYTES),
     * i.e. the same size as the inputs.
     */
    __device__ static void mul_lo(fixnum &r, fixnum a, fixnum b) {
        // TODO: This should be smaller, probably uint16_t (smallest
        // possible for addition).  Strangely, the naive translation to
        // the smaller size broke; to investigate.
        fixnum cy = 0;

        r = 0;
        for (int i = slot_layout::WIDTH - 1; i >= 0; --i) {
            fixnum aa = slot_layout::shfl(a, i);

            // TODO: See if using umad.wide improves this.
            umad_hi_cc(r, cy, aa, b, r);
            // TODO: Could use rotate here, which is slightly
            // cheaper than shfl_up0...
            r = slot_layout::shfl_up0(r, 1);
            cy = slot_layout::shfl_up0(cy, 1);
            umad_lo_cc(r, cy, aa, b, r);
        }
        cy = slot_layout::shfl_up0(cy, 1);
        add_cy(r, r, cy);
    }

    /*
     * (s, r) = a * b
     *
     * r is the "lo half" (see mul_lo above) and s is the
     * corresponding "hi half".
     */
    __device__ static void mul_wide(fixnum &s, fixnum &r, fixnum a, fixnum b) {
        // TODO: See if we can get away with a smaller type for cy.
        fixnum cy = 0;
        int L = slot_layout::laneIdx();

        // TODO: Rewrite this using rotates instead of shuffles;
        // should be simpler and faster.
        r = s = 0;
        for (int i = slot_layout::WIDTH - 1; i >= 0; --i) {
            fixnum aa = slot_layout::shfl(a, i), t;

            // TODO: Review this code: it seems to have more shuffles than
            // necessary, and besides, why does it not use digit_addmuli?
            umad_hi_cc(r, cy, aa, b, r);

            t = slot_layout::shfl(cy, slot_layout::toplaneIdx);
            // TODO: Is there a way to avoid this add?  Definitely need to
            // propagate the carry at least one place, but maybe not more?
            // Previous (wrong) version: "s = (L == 0) ? s + t : s;"
            t = (L == 0) ? t : 0;
            add_cy(s, s, t);

            // shuffle up hi words
            s = slot_layout::shfl_up(s, 1);
            // most sig word of lo words becomes least sig of hi words
            t = slot_layout::shfl(r, slot_layout::toplaneIdx);
            s = (L == 0) ? t : s;

            r = slot_layout::shfl_up0(r, 1);
            cy = slot_layout::shfl_up0(cy, 1);
            umad_lo_cc(r, cy, aa, b, r);
        }
        // TODO: This carry propgation from r to s is a bit long-winded.
        // Can we simplify?
        // NB: cy_hi <= width.  TODO: Justify this explicitly.
        fixnum cy_hi = slot_layout::shfl(cy, slot_layout::toplaneIdx);
        cy = slot_layout::shfl_up0(cy, 1);
        cy = add_cy(r, r, cy);
        cy_hi += cy;  // Can't overflow since cy_hi <= width.
        assert(cy_hi >= cy);
        // TODO: Investigate: replacing the following two lines with
        // simply "s = (L == 0) ? s + cy_hi : s;" produces no detectible
        // errors. Can I prove that (MAX_UINT64 - s[0]) < width?
        cy = (L == 0) ? cy_hi : 0;
        cy = add_cy(s, s, cy);
        assert(cy == 0);
    }

    /*
     * Return a mask of width bits whose ith bit is set if and only if
     * the ith digit of r is nonzero. In particular, result is zero
     * iff r is zero.
     */
    __device__ static uint32_t nonzero_mask(fixnum r) {
        return slot_layout::ballot(r != 0);
    }

    /*
     * Return -1, 0, or 1, depending on whether x is less than, equal
     * to, or greater than y.
     */
    __device__ static int cmp(fixnum x, fixnum y) {
        fixnum r;
        int br = sub_br(r, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    /*
     * Return the index of the most significant bit of x, or -1 if x is
     * zero.
     */
    __device__ static int msb(fixnum x) {
        // FIXME: Should be able to get this value from limits or numeric_limits
        // or whatever.
        enum { UINT32_BITS = 8 * sizeof(uint32_t) };
        static_assert(UINT32_BITS == 32, "uint32_t isn't 32 bits");

        uint32_t a = nonzero_mask(x);
        // b is the index of the first non-zero word, or -1 if x is zero.
        int b = UINT32_BITS - (clz(a) + 1);
        if (b < 0) return b;
        word_tp y = slot_layout::shfl(x, b);
        int c = clz(y);
        return WORD_BITS - (c + 1) + WORD_BITS * b;
    }

private:
    __device__ static int resolve_carries(fixnum &r, int cy) {
        // FIXME: Can't call std::numeric_limits<fixnum>::max() on device.
        //static constexpr fixnum FIXNUM_MAX = std::numeric_limits<fixnum>::max();
        static constexpr fixnum FIXNUM_MAX = ~(fixnum)0;
        static constexpr int WIDTH = slot_layout::WIDTH;
        int L = slot_layout::laneIdx();
        uint32_t allcarries, p, g;
        int cy_hi;

        g = slot_layout::ballot(cy);              // carry generate
        p = slot_layout::ballot(r == FIXNUM_MAX); // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        // FIXME: Unify these two expressions to remove the conditional;
        // the simple expression is not correct when WIDTH != warpSize
        //cy_hi = allcarries < g;                   // detect final overflow
        cy_hi = (WIDTH == 32) ? (allcarries < g) : ((allcarries >> WIDTH) & 1);
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        r += (allcarries >> L) & 1;

        // return highest carry
        return cy_hi;
    }

    __device__ static int resolve_borrows(fixnum &r, int cy) {
        // FIXME: This is at best a half-baked attempt to adapt
        // the carry propagation code above to the case of
        // subtraction.
        // FIXME: Use std::numeric_limits<fixnum>::min
        static constexpr fixnum FIXNUM_MIN = 0;
        int L = slot_layout::laneIdx();
        uint32_t allcarries, p, g;
        int cy_hi;

        g = ~slot_layout::ballot(cy);             // carry generate
        p = ~slot_layout::ballot(r == FIXNUM_MIN);// carry propagate
        allcarries = (p & g) - g;                 // propagate all carries
        // FIXME: This is not correct when WIDTH != warpSize
        cy_hi = allcarries > g;                   // detect final underflow
        allcarries = (allcarries ^ p) | (g >> 1); // get effective carries
        r -= (allcarries >> L) & 1;

        // return highest carry
        return cy_hi;
    }
};
