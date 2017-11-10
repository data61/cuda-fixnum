// -*- compile-command: "nvcc -D__STRICT_ANSI__ -ccbin clang-3.8 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <iostream>

#include "cuda_wrap.h"

using namespace std;

#define constexpr

// parameterised by
// hand implementation, which determines #bits per fixnum
//   and which is itself parameterised by
// subwarp data, which determines a SIMD decomposition of a fixnum
//
// TODO: Copy over functionality and documentation from IntmodVector.
template< typename hand_impl >
class fixnum_array {
public:
    template< typename T >
    static fixnum_array *create(size_t nelts, T init = 0) {
        fixnum_array *a;

        cuda_malloc_managed(&a, sizeof(*a));
        a->nelts = nelts;
        if (nelts > 0) {
            size_t nbytes = nelts * hand_impl::HAND_BYTES;
            cuda_malloc(&a->ptr, nbytes);
            // FIXME: Obviously should use zeros and init somehow
            cuda_memset(a->ptr, 42, nbytes);
        }
    }

    static fixnum_array *create(const uint8_t *data, size_t len, size_t bytes_per_elt);

    ~fixnum_array() {
        if (a->nelts > 0)
            cuda_free(a->ptr);
        cuda_free(a);
    }

    void retrieve(uint8_t **dest, size_t *dest_len, size_t *nelts) {
        size_t nbytes;
        *nelts = this->nelts;
        nbytes = *nelts * hand_impl::HAND_BYTES;
        *dest = new uint8_t[nbytes];
        cuda_memcpy_from_device(*dest, a->ptr, nbytes);
    }

private:
    // FIXME: This shouldn't be public; the create function that uses
    // it should be templatised.
    typedef typename hand_impl::digit value_tp;

    value_tp *ptr;
    int nelts;

    fixnum_array();
    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);
};

// FIXME: Passing this to map as an object probably makes inlining
// impossible in most circumstances.
template< typename T, typename subwarp_data >
struct device_op {
    int _x;

    device_op(int x) : _x(x) { }

    // A fixnum is represented by a register across a subwarp. This
    // thread is responsible for the Lth registers of the arguments,
    // where L is the lane index.
    //
    // This function should be available from the hand_impl; this sort
    // of function should be implemented in terms of the hand_impl
    // functions.
    __device__ void
    operator()(T &s, T &cy, T a, T b) {
        s = a + b;
        cy = s < a;
    }
};


template< size_t N >
ostream &
operator<<(ostream &os, const myarray<N> &arr) {
    os << "( " << arr[0];
    for (int i = 1; i < N; ++i)
        os << ", " << arr[i];
    os << " )" << flush;
    return os;
}

int main(int argc, char *argv[]) {
    long n = 16;
    if (argc > 1)
        n = atol(argv[1]);

    // hand_impl determines how a operations map to a warp
    //
    // bits_per_fixnum should somehow capture the fact that a warp can
    // be divided into subwarps
    //
    // n is the number of fixnums in the array; eventually only allow
    // initialisation via a byte array or whatever
    typedef fixnum_array< full_hand<uint32_t, 4> > fixnum_array;
    auto arr1 = fixnum_array::create(n);
    auto arr2 = fixnum_array::create(n);
    // device_op should be able to start operating on the appropriate
    // memory straight away
    device_op fn(7);
    // FIXME: How do I return cy without allocating a gigantic array
    // where each element is only 0 or 1?  Could return the carries in
    // the device_op fn?
    decltype(arr1) res;
    fixnum_array::map(fn, res, arr1, arr2);

    cout << "res = " << res << endl;
    return 0;
}
