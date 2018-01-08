#pragma once

#include "slot_layout.cu"

template< int FIXNUM_BYTES, typename register_tp = uint32_t >
class my_fixnum_impl {
    // TODO: static_assert's restricting FIXNUM_BYTES
    // FIXME: What if FIXNUM_BYTES < sizeof(register_tp)?
    
    typedef slot_layout< FIXNUM_BYTES / sizeof(register_tp) > slot_layout;

public:
    typedef register_tp fixnum;
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
    
    __device__ static void add_cy(fixnum &r, /*int &cy,*/ fixnum a, fixnum b) {
        r = a + b + slot_layout::laneIdx();
    }

    __device__ static void mul_lo(fixnum &r, fixnum a, fixnum b) {
        r = a * b;
    }
};


#if 0
template<
  int FIXNUM_BYTES,
  typename Tdigit_tp = uint32_t,
  typename slot_layout = slot_layout<FIXNUM_BYTES / sizeof(digit_tp)> >
class my_fixnum_impl {
public:
    static constexpr int fixnum_bytes() { return FIXNUM_BYTES; }
    // Could be bigger with nail bits for example
    static constexpr int storage_bytes() { return FIXNUM_BYTES; }

    // FIXME: Would rather this type were not exposed
    typedef Tdigit_tp digit_tp;

    // TODO: This should translate the device representation into the
    // equivalent byte array for host consumption.
    class retrieve {
        uint8_t *m_dest;
        size_t m_dest_space;

    public:
        retrieve(uint8_t *dest, size_t dest_space)
            : m_dest(dest), m_dest_space(dest_space) {  }

        __host__ host_pre_hook() {  }
        __device__ device_pre_hook(const fixnum_array<my_fixnum_impl> *a) {  }

        __device__ operator()(digit_tp s) {
        }

        __device__ device_post_hook() {  }
        __host__ host_post_hook() {  }
    };
    
  
};
#endif