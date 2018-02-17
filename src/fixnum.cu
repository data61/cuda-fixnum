#pragma once

#include "slot_layout.cu"

template< int FIXNUM_BYTES_, typename register_tp = uint32_t >
class my_fixnum_impl {
    // TODO: static_assert's restricting FIXNUM_BYTES
    // FIXME: What if FIXNUM_BYTES < sizeof(register_tp)?

    typedef slot_layout< FIXNUM_BYTES_ / sizeof(register_tp) > slot_layout;

public:
    typedef register_tp fixnum;
    static constexpr int FIXNUM_BYTES = FIXNUM_BYTES_;
    static constexpr int STORAGE_BYTES = FIXNUM_BYTES;
    // FIXME: Not obviously the right thing to do:
    static constexpr int THREADS_PER_FIXNUM = slot_layout::SLOT_WIDTH;

    // FIXME: This probably belongs in map or array or something
    __device__ static int get_fn_idx() {
        int blk_tid_offset = blockDim.x * blockIdx.x;
        int tid_in_blk = threadIdx.x;
        int fn_idx = (blk_tid_offset + tid_in_blk) / slot_layout::SLOT_WIDTH;
        return fn_idx;
    }

    // Assume bytes represents a number \sum_{i=0}^{nbytes-1} bytes[i] * 256^i
    // If nbytes > FIXNUM_BYTES, then the last (nbytes - FIXNUM_BYTES) are ignored.
    // If nbytes = 0, then r is assigned 0.
    __device__ static void from_bytes(fixnum &r, const uint8_t *bytes, int nbytes) {
        int L = slot_layout::laneIdx();
        int nregs = nbytes / sizeof(register_tp);
        int lastsz = nbytes % sizeof(register_tp);
        const uint8_t *r_bytes = bytes + L * sizeof(register_tp);

        r = 0;

        // This will cause warp divergence when nbytes is not
        // divisible by sizeof(register_tp).
        if (L < nregs) {
            // Recall [CCPG, Section 4] that the nVidia GPU architecture
            // is little-endian, so this cast/dereference is safe from
            // endian issues.
            // FIXME: Not sure why this static cast fails
            //register_tp d = *static_cast< const register_tp * >(r_bytes);
            register_tp d = *(const register_tp *)r_bytes;

            // With more sophisticated fixnum implementations (e.g. with
            // nail bits) we might have to manipulate d to obtain the
            // correct value of r.
            r = d;
        } else if (lastsz && L == nregs) {
            // Construct r from the leftover bytes one-by-one.
            for (int i = 0; i < lastsz; ++i)
                r |= r_bytes[i] << (8*i); // 8 is "bits in byte"
        }
    }

    // Translate a to a byte array represented by \sum_{i=0}^{nbytes-1} bytes[i] * 256^i
    // Assumes bytes points to (at least) FIXNUM_BYTES of space.
    __device__ static void to_bytes(uint8_t *bytes, fixnum a) {
        int L = slot_layout::laneIdx();
        register_tp *out = static_cast< register_tp * >(bytes);
        // With more sophisticated fixnum implementations (e.g. with
        // nail bits) we might have to manipulate a to obtain the
        // correct value of out[L].
        out[L] = a;
    }

    // load the value from ptr corresponding to this thread (lane).
    template< typename T >
    __device__ static T &load(T *ptr, int fn_idx) {
        int off = fn_idx * slot_layout::SLOT_WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }

    __device__ static void add_cy(fixnum &r, int &cy_out, fixnum a, fixnum b) {
        int cy_in;

        r = a + b;
        cy_in = r < a;
        cy_out = resolve_carries(r, cy_in);
    }

    __device__ static int sub_br(fixnum &r, int &br_out, fixnum a, fixnum b) {
        int br_in;

        r = a - b;
        br_in = r > a;
        br_out = resolve_borrows(r, br_in);
    }

    __device__ static void incr_cy(fixnum &r, int &cy_out) {
        fixnum one = (slot_layout::laneIdx() == 0);
        add_cy(r, cy_out, r, one);
    }

    __device__ static void decr_br(fixnum &r, int &br_out) {
        fixnum one = (slot_layout::laneIdx() == 0);
        sub_br(r, br_out, r, one);
    }

    /*
     * r = lo_half(a * b)
     *
     * The "lo_half" is the product modulo 2^(???), i.e. the same size as
     * the inputs.
     */
    __device__ static void mul_lo(fixnum &r, fixnum a, fixnum b) {
        // TODO: This should be smaller, probably uint16_t (smallest
        // possible for addition).  Strangely, the naive translation to
        // the smaller size broke; to investigate.
        fixnum cy = 0;
        int c;

        r = 0;
        for (int i = slot_layout::SLOT_WIDTH - 1; i >= 0; --i) {
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
        add_cy(r, c, r, cy);
        //assert(c == 0);
    }

private:
    __device__ static int resolve_carries(fixnum &r, int cy) {
        // FIXME: Use std::numeric_limits<fixnum>::max
        static constexpr fixnum FIXNUM_MAX = ~(fixnum)0;
        int L = slot_layout::laneIdx();
        uint32_t allcarries, p, g;
        int cy_hi;

        g = slot_layout::ballot(cy);              // carry generate
        p = slot_layout::ballot(r == FIXNUM_MAX); // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        // FIXME: This is not correct when WIDTH != warpSize
        cy_hi = allcarries < g;                   // detect final overflow
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
