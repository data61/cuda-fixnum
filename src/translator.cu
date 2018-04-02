#pragma once

#include <stdint.h>

/**
 * Translate to and from arrays of bytes.
 *
 * TODO: translation with nail bits, 2's complement representation,
 * floats, etc.
 */

/**
 * Interprets the device side data as a plain contiguous block of register_tp.
 *
 * TODO: Implement translation with nail bits.
 */
template< typename register_tp >
struct plain_translate {
    // The implementation of a fixnum instruction set will rely on the
    // memory layout guarantees provided by the translator, which has
    // transformed the input byte stream into an array on the device.

    
    static constexpr int STORAGE_BYTES = FIXNUM_BYTES;
    
    // Assume bytes represents a number \sum_{i=0}^{nbytes-1} bytes[i] * 256^i
    // If nbytes > FIXNUM_BYTES, then the last (nbytes - FIXNUM_BYTES) are ignored.
    // If nbytes = 0, then r is assigned 0.
    //
    // Call with slot_layout::laneIdx() for idx.
    __device__ static void from_bytes(fixnum &r, const uint8_t *bytes, int nbytes, int idx) {
        // FIXME: Remove this restriction on alignment of bytes.
        assert(((uintptr_t)bytes & 0x7) == 0);

        int nregs = nbytes / sizeof(register_tp);
        int lastsz = nbytes % sizeof(register_tp);
        const uint8_t *r_bytes = bytes + idx * sizeof(register_tp);

        r = 0;

        // This will cause warp divergence when nbytes is not
        // divisible by sizeof(register_tp).
        if (idx < nregs) {
            // Recall [CCPG, Section 4] that the nVidia GPU architecture
            // is little-endian, so this cast/dereference is safe from
            // endian issues.
            register_tp d = *reinterpret_cast< const register_tp * >(r_bytes);

            // With more sophisticated fixnum implementations (e.g. with
            // nail bits) we might have to manipulate d to obtain the
            // correct value of r.
            r = d;
        } else if (lastsz && idx == nregs) {
            // TODO: Explain how this condition covers all the
            // remaining cases that we treat; it's not immediately
            // obvious.

            // Construct r from the leftover bytes one-by-one.
            for (int i = 0; i < lastsz; ++i)
                r |= r_bytes[i] << (8*i); // 8 is "bits in byte"
        }
    }

    // Translate a to a byte array represented by \sum_{i=0}^{nbytes-1} bytes[i] * 256^i
    // Assumes bytes points to (at least) FIXNUM_BYTES of space.
    __device__ static void to_bytes(uint8_t *bytes, fixnum a, int idx) {
        // FIXME: Remove this restriction on alignment of bytes.
        assert(((uintptr_t)bytes & 0x7) == 0);

        register_tp *out = reinterpret_cast< register_tp * >(bytes);
        // With more sophisticated fixnum implementations (e.g. with
        // nail bits) we might have to manipulate a to obtain the
        // correct value of out[idx].
        out[idx] = a;
    }
};