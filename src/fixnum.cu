#pragma once

template< int FIXNUM_BYTES, typename register_tp = uint32_t >
class my_fixnum_impl {
    // TODO: static_assert's restricting FIXNUM_BYTES
    
    typedef slot_layout< FIXNUM_BYTES / sizeof(register_tp) > slot_layout;

public:
    typedef typename register_tp fixnum;

    // FIXME: This probably belongs in map or array or something
    __device__ static int get_fn_idx() {
        int blk_tid_offset = blockDim.x * blockIdx.x;
        int tid_in_blk = threadIdx.x;
        int fn_idx = (blk_tid_offset + tid_in_blk) / slot_layout::SLOT_WIDTH;
        return fn_idx;
    }

    __device__ static void from_bytes(fixnum &s, const uint8_t *bytes, size_t nbytes) {
    }

    // load the value from ptr corresponding to this thread (lane).
    template< typename T >
    __device__ static T &load(T *ptr, int fn_idx) {
        int off = fn_idx * slot_layout::SLOT_WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }
    
    __device__ static void add_cy(fixnum &s, /*int &cy,*/ fixnum a, fixnum b) {
        s = a + b + slot_layout::laneIdx();
    }

    __device__ static void mul_lo(fixnum &s, fixnum a, fixnum b) {
        s = a * b;
    }
};


#if 0

// TODO: Should be a friend of fixnum_impl?
template< typename fixnum_impl >
struct set_const : function<fixnum_impl, set_const> {
    // It would be neater to do
    //   __device__ digit_tp digits[SLOT_WIDTH];
    // but __device__ is not allowed here (F.3.3.1).
    
    //digit_tp *digits;
    register_tp digits[SLOT_WIDTH]; // managed?

    template< typename T >
    set_const(T init) {
        constexpr int n = sizeof(T);
        static_assert(n < FIXNUM_BYTES, "Initialiser too large");

        cuda_malloc(&digits, storage_bytes());
        cuda_memset(digits, 0, FIXNUM_BYTES);
        // FIXME: Assumes endianness of host and device are the same
        cuda_memcpy_to_device(digits, &init, n);
    }

    ~set_const() {
        cuda_free(digits);
    }

    __device__ operator()(digit_tp &s) {
        int L = slot_layout::laneIdx();
        s[L] = digits[L];
    }
};
 

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