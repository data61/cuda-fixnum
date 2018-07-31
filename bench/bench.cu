#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"

using namespace std;

template< typename fixnum >
struct my_modexp {
    __device__ void operator()(fixnum &z, fixnum x, fixnum e, fixnum m) {
        fixnum zz;
        modexp<fixnum> me(m, e);
        me(zz, x);
        z = zz;
    };
};

template< typename fixnum >
struct mul_lo {
    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum s;
        fixnum::mul_lo(s, a, b);
        r = s;
    }
};

template< typename fixnum >
struct mul_wide {
    __device__ void operator()(fixnum &s, fixnum &r, fixnum a, fixnum b) {
        fixnum rr, ss;
        fixnum::mul_wide(ss, rr, a, b);
        s = ss;
        r = rr;
    }
};

template< template <typename> class Func, typename fixnum, typename ...Args >
void bench_func(const char *fn_name, Args ...args) {
    typedef fixnum_array<fixnum> fixnum_array;

    // warm up
    fixnum_array::template map<Func>(args...);

    clock_t c = clock();
    fixnum_array::template map<Func>(args...);
    c = clock() - c;

    int nelts = std::min( { args->length()... } ); // should be all the same
    double secinv = (double)CLOCKS_PER_SEC / c;
    printf("%9s:  %6.4f  %9.1f\n",
           fn_name, 1/secinv, nelts * 1e-3 * secinv);

}

template< int fn_bytes, typename word_fixnum >
void bench(size_t nelts) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    uint8_t *input = new uint8_t[fn_bytes * nelts];
    for (size_t i = 0; i < fn_bytes * nelts; ++i)
        input[i] = (i * 17 + 11) % 256;

    fixnum_array *res, *in;
    in = fixnum_array::create(input, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    double total_MiB = fixnum::BYTES * (double)nelts / (1 << 20);
    printf("fixnum: %4d bits (%2d-bit digits); #elts: %5de3, total data: %.2f MiB\n",
           fixnum::BITS, fixnum::digit::BITS, (int)(nelts * 1e-3), total_MiB);
    printf("            seconds   Kops/s\n");
    bench_func<mul_lo, fixnum>("mul_lo", res, in, in);
    bench_func<mul_wide, fixnum>("mul_wide", res, res, in, in);
    //bench_func<mul_mod>("mul_mod", res, in, in);
    bench_func<my_modexp, fixnum>("modexp", res, in, in, in);

    puts("");

    delete in;
    delete res;
    delete[] input;
}

int main(int argc, char *argv[]) {
    long m = 1;
    if (argc > 1)
        m = atol(argv[1]);

    bench<8, u32_fixnum>(m);
    bench<16, u32_fixnum>(m);
    bench<32, u32_fixnum>(m);
    bench<64, u32_fixnum>(m);
    bench<128, u32_fixnum>(m);

    puts("");
    bench<8, u64_fixnum>(m);
    bench<16, u64_fixnum>(m);
    bench<32, u64_fixnum>(m);
    bench<64, u64_fixnum>(m);
    bench<128, u64_fixnum>(m);
    bench<256, u64_fixnum>(m);

    return 0;
}
