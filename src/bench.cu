// -*- compile-command: "nvcc -D__STRICT_ANSI__ -std=c++11 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

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
template< typename hand_impl >
struct device_op {
    typedef typename hand_impl::digit digit;
    int _x;

    device_op(int x) : _x(x) { }

    // A fixnum is represented by a register across a subwarp. This
    // thread is responsible for the Lth registers of the arguments,
    // where L is the lane index.
    //
    // This function should be available from the hand_impl; this sort
    // of function should be implemented in terms of the hand_impl
    // functions.
    __device__ void operator()(digit &s, digit a, digit b) {
        hand_impl::add_cy(s, a, b);
        hand_impl::mullo(s, s, b);
    }
};

template< typename fixnum >
struct ec_add {

    ec_add(/* ec params */) { }

    __device__ void operator()(fixnum &s, fixnum a, fixnum b) {
        fixnum::mullo(s, a, b);
    }
};


template< int FIXNUM_BYTES >
ostream &
operator<<(ostream &os, const fixnum_array<FIXNUM_BYTES> *arr) {
    uint8_t num[FIXNUM_BYTES];
    int nelts = arr->length();

    os << "( ";
    if (nelts > 0) {
        (void) arr->retrieve_into(num, FIXNUM_BYTES, 0);
        os << (int)num[0];
        for (int i = 1; i < nelts; ++i) {
            (void) arr->retrieve_into(num, FIXNUM_BYTES, i);
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

    // n is the number of fixnums in the array; eventually only allow
    // initialisation via a byte array or whatever
    typedef my_fixnum_impl<16> fixnum_impl;
    typedef fixnum_array<fixnum_imlp> fixnum_array;
    auto res = fixnum_array::create(n);
    auto arr1 = fixnum_array::create(n, 5);
    auto arr2 = fixnum_array::create(n, 7);

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    // device_op should be able to start operating on the appropriate
    // memory straight away
    //device_op fn(7);

    // FIXME: How do I return cy without allocating a gigantic array
    // where each element is only 0 or 1? Need fixnum_array to have
    // C-array semantics, hence allowing other C arrays to be used
    // alongside, so pass an int* to collect the carries.
    fixnum_array::map(fixnum_impl::add_cy(), res, arr1, arr2);

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    delete res;
    delete arr1;
    delete arr2;

    return 0;
}
