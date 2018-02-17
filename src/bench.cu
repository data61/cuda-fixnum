// -*- compile-command: "nvcc -Wno-deprecated-declarations -std=c++11 -Xcompiler -Wall,-Wextra -g -G -gencode arch=compute_50,code=sm_50 -o bench bench.cu" -*-

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>

#include "fixnum.cu"
#include "fixnum_array.h"

using namespace std;

template< typename fixnum_impl >
struct set_const /*: public Managed*/ {
    // FIXME: The repetition of this is dumb and annoying
    typedef typename fixnum_impl::fixnum fixnum;

    uint8_t *bytes;
    int nbytes;

    // FIXME: Assumes endianness of host and device are the same (LE).
    template< typename T >
    set_const(T init)
    : set_const(&init, sizeof(T)) { }

    set_const(const void *bytes_, int nbytes_) : nbytes(nbytes_) {
        cuda_malloc(&bytes, nbytes);
        cuda_memcpy_to_device(bytes, bytes_, nbytes);
    }

    ~set_const() {
        // FIXME: Why does this break?
        //cuda_free(bytes);
    }

    __device__ void operator()(fixnum &s) {
        fixnum_impl::from_bytes(s, bytes, nbytes);
    }
};

// TODO: Consider having functions inherit from fixnum_impl; this
// would be like making the function a host and fixnum_impl the policy
// in a policy-based design
// (https://en.wikipedia.org/wiki/Policy-based_design).
template< typename fixnum_impl >
struct ec_add /*: public Managed*/ {
    typedef typename fixnum_impl::fixnum fixnum;

    set_const<fixnum_impl> set_k;

    ec_add(/* ec params */ long k = 17) : set_k(k) { }

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum k;
        set_k(k);
        fixnum_impl::mul_lo(r, a, k);
        fixnum_impl::mul_lo(r, r, b);
        fixnum_impl::mul_lo(r, r, r);
        fixnum_impl::mul_lo(r, r, r);
    }
};

template< typename fixnum_impl >
struct increments /*: public Managed*/ {
    typedef typename fixnum_impl::fixnum fixnum;
    long k;

    increments(long k_ = 17) : k(k_) { }

    __device__ void operator()(fixnum &r, fixnum a) {
        r = a;
        for (long i = 0; i < k; ++i)
            fixnum_impl::incr_cy(r);
    }
};

static string fixnum_as_str(const uint8_t *fn, int nbytes) {
    ostringstream ss;

    for (int i = nbytes - 1; i >= 0; --i) {
        // These IO manipulators are forgotten after each use;
        // i.e. they don't apply to the next output operation (whether
        // it be in the next loop iteration or in the conditional
        // below.
        ss << setfill('0') << setw(2) << hex;
        ss << (int)fn[i] << flush;
        if (i && !(i & 3))
            ss << ' ';
    }
    return ss.str();
}

template< typename fixnum_impl >
ostream &
operator<<(ostream &os, const fixnum_array<fixnum_impl> *fn_arr) {
    constexpr int fn_bytes = fixnum_impl::FIXNUM_BYTES;
    constexpr size_t bufsz = 4096;
    size_t nbytes;
    uint8_t arr[bufsz];
    int nelts;

    fn_arr->retrieve_all(arr, bufsz, &nbytes, &nelts);
    if (nelts < 0) {
        os << "( insufficient space to retrieve array )" << endl;
        return os;
    }

    os << "( ";
    if (nelts > 0) {
        os << fixnum_as_str(arr, fn_bytes);
        for (int i = 1; i < nelts; ++i) {
            os << ", " << fixnum_as_str(arr + i*fn_bytes, fn_bytes);
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
    auto arr1 = fixnum_array::create(n, ~0UL);
    auto arr2 = fixnum_array::create(n, 245);

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

    //fixnum_array::map(ec_add<fixnum_impl>(), res, arr1, arr2);
    fixnum_array::map(increments<fixnum_impl>(1), res, arr1);

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    delete res;
    delete arr1;
    delete arr2;

    return 0;
}
