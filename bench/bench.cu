// -*- compile-command: "nvcc -I../src -ccbin clang-3.8 -Wno-deprecated-declarations -std=c++11 -lineinfo -Xcompiler -Wall,-Wextra -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <memory>
#include <iostream>
#include <cstring>
#include <cassert>

#include "fixnum/default.cu"
#include "array/fixnum_array.h"
#include "kernels/square.cu"
#include "kernels/ec_add.cu"

using namespace std;

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
    auto fn2 = new square<fixnum_impl>();
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
