#pragma once

/*
template< typename fixnum_impl >
class function {
public:
    template< typename... Args >
    __device__ void call(int idx, Args... args);
    
    template< typename... Args >
    __device__ void operator()(int nelts, Args... args) {
        call(fn_idx, fixnum_impl::load(args)...);
    }
};
*/

template<
  int FIXNUM_BYTES,
  typename digit_tp = uint32_t,
  typename slot_layout = slot_layout<FIXNUM_BYTES / sizeof(digit_tp)> >
class my_fixnum_impl {
public:
    typedef digit_tp digit_tp;

    // FIXME: This probably belongs in map or array or something
    static int get_fn_idx() {
        int blk_tid_offset = blockDim.x * blockIdx.x;
        int tid_in_blk = threadIdx.x;
        int fn_idx = (blk_tid_offset + tid_in_blk) / slot_layout::SLOT_WIDTH;
        return fn_idx;
    }

    template< typename T >
    static T &load(int fn_idx, T *ptr) {
        int off = fn_idx * slot_layout::SLOT_WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }

    struct add_cy {
        // FIXME: Only used in dispatch()
        typedef my_fixnum_impl fixnum_impl;

        __device__ operator()(digit_tp &s, /*int &cy,*/ digit_tp a, digit_tp b) {
            // do the thing
            s = a + b + slot_layout::laneIdx();
        }
    };
    
private:
    // Could be bigger with nail bits for example
    static constexpr int STORAGE_BYTES = FIXNUM_BYTES;
};
