// -*- compile-command: "nvcc -D__STRICT_ANSI__ -std=c++11 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <iostream>
#include <cstring>
#include <cassert>

#include "fixnum_array.h"

using namespace std;

// From: https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

// TODO: functions should probably all be 'managed memory'
template< typename fixnum_impl, typename Func >
struct function : public Managed {
    // Make this available to derived classes
    typedef typename fixnum_impl::fixnum fixnum;

    template< typename... Args >
    __device__ void operator()(Args... args) {
        static_cast< Func * >(this)->call(args);
    }
};

template< typename fixnum_impl >
struct ec_add : public function<fixnum_impl, ec_add> {
    ec_add(/* ec params */) { }

    __device__ void call(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_lo(r, a, b);
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
    typedef fixnum_array<fixnum_impl> fixnum_array;
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
    fixnum_array::map(ec_add(), res, arr1, arr2);

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    delete res;
    delete arr1;
    delete arr2;

    return 0;
}
