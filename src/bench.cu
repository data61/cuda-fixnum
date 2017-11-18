// -*- compile-command: "nvcc -D__STRICT_ANSI__ -ccbin clang-3.8 -std=c++11 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <iostream>
#include <cstring>
#include <cassert>

#include "fixnum_array.h"

using namespace std;

// FIXME: Ignore this idea of feeding in new operations for now; just
// use a fixed set of operations determined by hand_impl
//
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


template< typename H >
ostream &
operator<<(ostream &os, const fixnum_array<H> *arr) {
    constexpr int nbytes = H::FIXNUM_BYTES;
    uint8_t num[nbytes];
    int nelts = arr->length();

    os << "( ";
    if (nelts > 0) {
        (void) arr->retrieve_into(num, nbytes, 0);
        os << (int)num[0];
        for (int i = 1; i < nelts; ++i) {
            (void) arr->retrieve_into(num, nbytes, i);
            os << ", " << (int)num[0];
        }
    }
    os << " )" << flush;
    return os;
}


int main(int argc, char *argv[]) {
    long n = 16;
    if (argc > 1)
        n = atol(argv[1]);

    // hand_impl determines how operations map to a warp
    //
    // bits_per_fixnum should somehow capture the fact that a warp can
    // be divided into subwarps
    //
    // n is the number of fixnums in the array; eventually only allow
    // initialisation via a byte array or whatever
    typedef full_hand<uint32_t, 8> hand_impl;
    typedef fixnum_array< hand_impl > fixnum_array;
    auto arr1 = fixnum_array::create(n, 5);
    auto arr2 = fixnum_array::create(n, 7);

    cout << "slot width: " << hand_impl::SLOT_WIDTH << endl;
    cout << "nslots:     " << hand_impl::NSLOTS << endl;

    cout << "bytes of things: " << endl;
    cout << "digit:  " << hand_impl::DIGIT_BYTES << endl;
    cout << "fixnum: " << hand_impl::FIXNUM_BYTES << endl;
    cout << "hand:   " << hand_impl::HAND_BYTES << endl;

    // device_op should be able to start operating on the appropriate
    // memory straight away
    //device_op fn(7);

    // FIXME: How do I return cy without allocating a gigantic array
    // where each element is only 0 or 1?  Could return the carries in
    // the device_op fn?
    //fixnum_array::map(fn, res, arr1, arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    arr1->add_cy(arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    arr1->mullo(arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    delete arr1;
    delete arr2;

    return 0;
}
