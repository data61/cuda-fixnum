// -*- compile-command: "nvcc -ccbin clang-3.8 -Wno-deprecated-declarations -std=c++11 -lineinfo -Xcompiler -Wall,-Wextra -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>

#include "fixnum.cu"
#include "fixnum_array.h"

using namespace std;

static string fixnum_as_str(const uint8_t *fn, int nbytes) {
    ostringstream ss;

    for (int i = nbytes - 1; i >= 0; --i) {
        // These IO manipulators are forgotten after each use;
        // i.e. they don't apply to the next output operation (whether
        // it be in the next loop iteration or in the conditional
        // below.
        ss << setfill('0') << setw(2) << hex;
        ss << (int)fn[i];
        if (i && !(i & 3))
            ss << ' ';
    }
    return ss.str();
}

template< typename fixnum_impl >
ostream &operator<<(ostream &os, const fixnum_array<fixnum_impl> *fn_arr) {
    constexpr int fn_bytes = fixnum_impl::FIXNUM_BYTES;
    constexpr size_t bufsz = 4096;
    uint8_t arr[bufsz];
    int nelts;

    fn_arr->retrieve_all(arr, bufsz, &nelts);
    os << "( ";
    if (nelts < fn_arr->length()) {
        os << "insufficient space to retrieve array";
    } else if (nelts > 0) {
        os << fixnum_as_str(arr, fn_bytes);
        for (int i = 1; i < nelts; ++i)
            os << ", " << fixnum_as_str(arr + i*fn_bytes, fn_bytes);
    }
    os << " )" << flush;
    return os;
}

// TODO: Check whether the synchronize calls are necessary here (they
// are clearly sufficient).
struct managed {
    void *operator new(size_t bytes) {
        void *ptr;
        cuda_malloc_managed(&ptr, bytes);
        cuda_device_synchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cuda_device_synchronize();
        cuda_free(ptr);
    }
};

template< typename fixnum_impl >
class set_const : public managed {
public:
    // FIXME: The repetition of this is dumb and annoying
    typedef typename fixnum_impl::fixnum fixnum;

    static set_const *create(const uint8_t *konst, int nbytes) {
        set_const *sc = new set_const;
        fixnum_impl::from_bytes(sc->konst, konst, nbytes);
        return sc;
    }

    template< typename T >
    static set_const *create(T init) {
        auto bytes = reinterpret_cast<const uint8_t *>(&init);
        return create(bytes, sizeof(T));
    }

    __device__ void operator()(fixnum &s) {
        int L = fixnum_impl::slot_layout::laneIdx();
        s = konst[L];
    }

private:
    typename fixnum_impl::fixnum konst[fixnum_impl::SLOT_WIDTH];
};

// fixnum_impl is like a policy in a policy-based design
// (https://en.wikipedia.org/wiki/Policy-based_design).
template< typename fixnum_impl >
struct ec_add : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    set_const<fixnum_impl> *set_k;

    ec_add(/* ec params */ long k = 17)
    : set_k(set_const<fixnum_impl>::create(k)) { }

    ~ec_add() { delete set_k; }

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum k;
        (*set_k)(k);
        fixnum_impl::mul_lo(r, a, k);
        fixnum_impl::mul_lo(r, r, b);
        fixnum_impl::mul_lo(r, r, r);
        fixnum_impl::mul_lo(r, r, r);
    }
};

template< typename fixnum_impl >
struct increments : public managed {
    typedef typename fixnum_impl::fixnum fixnum;
    long k;

    increments(long k_ = 17) : k(k_) { }

    __device__ void operator()(fixnum &r, fixnum a) {
        r = a;
        for (long i = 0; i < k; ++i)
            fixnum_impl::incr_cy(r);
    }
};

template< typename fixnum_impl >
struct square : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a) {
        fixnum_impl::mul_lo(r, a, a);
    }
};

template< typename fixnum_impl >
struct sum : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a) {
        fixnum_impl::add_cy(r, a, a);
    }
};

template< int fn_bytes, typename word_tp = uint32_t >
void bench(size_t nelts) {
    typedef my_fixnum_impl<fn_bytes, word_tp> fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    uint8_t *input = new uint8_t[fn_bytes * nelts];
    for (size_t i = 0; i < fn_bytes * nelts; ++i)
        input[i] = (i * 17 + 11) % 256;

    fixnum_array *res, *in;
    in = fixnum_array::create(input, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    typedef square<fixnum_impl> square;
    auto fn = unique_ptr<square>(new square);
    // typedef sum<fixnum_impl> sum;
    // auto fn = unique_ptr<sum>(new sum);

    clock_t c = clock();
    fixnum_array::map(fn.get(), res, in);
    c = clock() - c;
    double total_MiB = fn_bytes * (double)nelts / (1 << 20);
    cout << nelts << " elts, "
         << setw(3) << fn_bytes << " (" << sizeof(word_tp) << ") bytes per element: "
         << total_MiB * (double)CLOCKS_PER_SEC / c
         << " MiB/s (take with grain of salt)"
         << endl;

    delete in;
    delete res;
    delete[] input;
}

int main(int argc, char *argv[]) {
    long n = 16, m = 1;
    if (argc > 1)
        n = atol(argv[1]);

    if (argc > 2)
        m = atol(argv[2]);

    typedef my_fixnum_impl<16> fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    auto res = fixnum_array::create(n);
    auto arr1 = fixnum_array::create(n, ~0UL);
    auto arr2 = fixnum_array::create(n, 245);

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    // FIXME: How do I return cy without allocating a gigantic array
    // where each element is only 0 or 1? Need fixnum_array to have
    // C-array semantics, hence allowing other C arrays to be used
    // alongside, so pass an int* to collect the carries.

    auto fn1 = new ec_add<fixnum_impl>();
    fixnum_array::map(fn1, res, arr1, arr2);
    auto fn2 = new increments<fixnum_impl>(1);
    fixnum_array::map(fn2, res, arr1);

    delete fn1;
    delete fn2;

    cout << "res  = " << res << endl;
    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    cout << endl << "uint32:" << endl;
    bench<8>(m);
    bench<16>(m);
    bench<32>(m);
    bench<64>(m);
    bench<128>(m);

    cout << endl << "uint64:" << endl;
    bench<8, uint64_t>(m);
    bench<16, uint64_t>(m);
    bench<32, uint64_t>(m);
    bench<64, uint64_t>(m);
    bench<128, uint64_t>(m);
    bench<256, uint64_t>(m);

    delete res;
    delete arr1;
    delete arr2;

    return 0;
}
