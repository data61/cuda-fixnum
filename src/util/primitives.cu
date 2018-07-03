#pragma once

#include <stdint.h>
#include <type_traits>

/*
 * Low-level/primitive functions.
 */

// TODO: Understand circumstances in which I might want to make this
// "#define ASM asm __volatile__".
//#define ASM asm

// hi * 2^32 + lo = a * b
__device__ __forceinline__ void
umul(uint32_t &hi, uint32_t &lo, uint32_t a, uint32_t b) {
    // TODO: Measure performance difference between this and the
    // equivalent:
    //   mul.hi.u32 %0, %2, %3
    //   mul.lo.u32 %1, %2, %3
    asm ("{\n\t"
         " .reg .u64 tmp;\n\t"
         " mul.wide.u32 tmp, %2, %3;\n\t"
         " mov.b64 { %0, %1 }, tmp;\n\t"
         "}"
         : "=r"(hi), "=r"(lo)
         : "r"(a), "r"(b));
}

__device__ __forceinline__ void
umul(uint64_t &r, uint32_t a, uint32_t b) {
    asm ("mul.wide.u32 %0, %1, %2;"
         : "=l"(r)
         : "r"(a), "r"(b));
}

// hi * 2^64 + lo = a * b
__device__ __forceinline__ void
umul(uint64_t &hi, uint64_t &lo, uint64_t a, uint64_t b) {
    asm ("mul.hi.u64 %0, %2, %3;\n\t"
         "mul.lo.u64 %1, %2, %3;"
         : "=l"(hi), "=l"(lo)
         : "l"(a), "l"(b));
}

// r = a * b + c
__device__ __forceinline__ void
umad(uint64_t &r, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.wide.u32 %0, %1, %2, %3;"
         : "=l"(r)
         : "r"(a), "r"(b), "r"(c));
}

// (hi, lo) = a * b + c
__device__ __forceinline__ void
umad(uint32_t &hi, uint32_t &lo, uint32_t a, uint32_t b, uint32_t c) {
    asm ("{\n\t"
         " .reg .u64 tmp;\n\t"
         " mad.wide.u32 tmp, %2, %3, %4;\n\t"
         " mov.b64 { %0, %1 }, tmp;\n\t"
         "}"
         : "=r"(hi), "=r"(lo)
         : "r"(a), "r"(b), "r"(c));
}

// (hi, lo) = a * b + c
__device__ __forceinline__ void
umad(uint64_t &hi, uint64_t &lo, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
         "madc.hi.u64 %0, %2, %3, 0;"
         : "=l"(hi), "=l"(lo)
         : "l"(a), "l" (b), "l"(c));
}

// lo = a * b + c (mod 2^32)
__device__ __forceinline__ void
umad_lo(uint32_t &lo, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.lo.u32 %0, %1, %2, %3;"
         : "=r"(lo)
         : "r"(a), "r" (b), "r"(c));
}

__device__ __forceinline__ void
umad_hi(uint32_t &hi, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.hi.u32 %0, %1, %2, %3;"
         : "=r"(hi)
         : "r"(a), "r" (b), "r"(c));
}

// lo = a * b + c (mod 2^64)
__device__ __forceinline__ void
umad_lo(uint64_t &lo, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.lo.u64 %0, %1, %2, %3;"
         : "=l"(lo)
         : "l"(a), "l" (b), "l"(c));
}

__device__ __forceinline__ void
umad_hi(uint64_t &hi, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.hi.u64 %0, %1, %2, %3;"
         : "=l"(hi)
         : "l"(a), "l" (b), "l"(c));
}

// as above but with carry in cy
__device__ __forceinline__ void
umad_lo_cc(uint32_t &lo, uint32_t &cy, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.lo.cc.u32 %0, %2, %3, %4;\n\t"
         "addc.u32 %1, %1, 0;"
         : "=r"(lo), "+r"(cy)
         : "r"(a), "r" (b), "r"(c));
}

__device__ __forceinline__ void
umad_hi_cc(uint32_t &hi, uint32_t &cy, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.hi.cc.u32 %0, %2, %3, %4;\n\t"
         "addc.u32 %1, %1, 0;"
         : "=r"(hi), "+r"(cy)
         : "r"(a), "r" (b), "r"(c));
}

__device__ __forceinline__ void
umad_lo_cc(uint64_t &lo, uint64_t &cy, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
         "addc.u64 %1, %1, 0;"
         : "=l"(lo), "+l"(cy)
         : "l"(a), "l" (b), "l"(c));
}

__device__ __forceinline__ void
umad_hi_cc(uint64_t &hi, uint64_t &cy, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.hi.cc.u64 %0, %2, %3, %4;\n\t"
         "addc.u64 %1, %1, 0;"
         : "=l"(hi), "+l"(cy)
         : "l"(a), "l" (b), "l"(c));
}


/*
 * Count Leading Zeroes in x.
 *
 * Use __builtin_clz{,l,ll}(x) or CUDA ASM depending on context.
 */
__host__ __device__ __forceinline__
int
clz(uint32_t x) {
#ifdef __CUDA_ARCH__
    int n;
    asm ("clz.b32 %0, %1;" : "=r"(n) : "r"(x));
    return n;
#else
    static_assert(sizeof(unsigned int) == sizeof(uint32_t),
            "attempted to use wrong __builtin_clz{,l,ll}()");
    return __builtin_clz(x);
#endif
}

__host__ __device__ __forceinline__
int
clz(uint64_t x) {
#ifdef __CUDA_ARCH__
    int n;
    asm ("clz.b64 %0, %1;" : "=r"(n) : "l"(x));
    return n;
#else
    static_assert(sizeof(unsigned long) == sizeof(uint64_t),
            "attempted to use wrong __builtin_clz{,l,ll}()");
    return __builtin_clzl(x);
#endif
}

/*
 * Count Trailing Zeroes in x.
 *
 * Use __builtin_ctz{,l,ll}(x) or CUDA ASM depending on context.
 */
__host__ __device__ __forceinline__
int
ctz(uint32_t x) {
#ifdef __CUDA_ARCH__
    int n;
    asm ("ctz.b32 %0, %1;" : "=r"(n) : "r"(x));
    return n;
#else
    static_assert(sizeof(unsigned int) == sizeof(uint32_t),
                  "attempted to use wrong __builtin_ctz{,l,ll}()");
    return __builtin_ctz(x);
#endif
}

__host__ __device__ __forceinline__
int
ctz(uint64_t x) {
#ifdef __CUDA_ARCH__
    int n;
    asm ("ctz.b64 %0, %1;" : "=r"(n) : "l"(x));
    return n;
#else
    static_assert(sizeof(unsigned long) == sizeof(uint64_t),
                  "attempted to use wrong __builtin_ctz{,l,ll}()");
    return __builtin_ctzl(x);
#endif
}

/*
 * GCD of x and y.
 *
 * Source: HAC Algorithm 14.54. Some ideas taken from
 * https://lemire.me/blog/2013/12/26/fastest-way-to-compute-the-greatest-common-divisor/
 */
template< typename int_type >
__host__ __device__
int_type
gcd(int_type u, int_type v)
{
    typedef typename std::make_unsigned<int_type>::type uint_type;

    if (u == 0) return v;
    if (v == 0) return u;

    // TODO: Strip zero bits as described by https://gmplib.org/manual/Binary-GCD.html#Binary-GCD

    int g = ctz(static_cast<uint_type>(u | v));
    u >>= ctz(static_cast<uint_type>(u));
    do {
        v >>= ctz(static_cast<uint_type>(v));
        // TODO: Check whether removing the branch helps much here.
        // Code would be something like
        //   setp.gt.u32 %p, %u, %v
        //   { .reg .u32 tmp
        //     mov.b32  tmp, %v
        //     selp.b32 %v, %v, %u, %p
        //     selp.b32 %u, %u, tmp, %p }
        if (u > v) {
            int_type t = v;
            v = u;
            u = t;
        }
        v = v - u;
    } while (v != 0);

    return u << g;
}


/*
 * Extended GCD of x and y.
 *
 * Source: HAC Algorithm 14.61. Some ideas due to
 * https://lemire.me/blog/2013/12/26/fastest-way-to-compute-the-greatest-common-divisor/
 *
 * FIXME: Not quite sure why, but when x > y, the cofactors returned are not the
 * optimal ones. If it's important to get the optimal cofactors, make sure x < y.
 */
template< typename int_type >
__host__ __device__
int_type
xgcd(
    int_type &a, int_type &b,
    int_type x, int_type y)
{
    static_assert(std::is_signed<int_type>::value == true,
                  "template type must be signed");
    typedef typename std::make_unsigned<int_type>::type uint_type;

    // TODO: Specialise to case of m odd to avoid some calculations as described
    // in HAC 14.64. See also errata: http://cacr.uwaterloo.ca/hac/errata/errata.html

    // TODO: Strip zero bits as described by https://gmplib.org/manual/Binary-GCD.html#Binary-GCD
    int_type u, v, g;
    int_type A = 1, B = 0, C = 0, D = 1;

    if (x == 0) { a = C; b = D; return y; }
    if (y == 0) { a = A; b = B; return x; }

    g = ctz(static_cast<uint_type>(x | y));
    x >>= g;
    y >>= g;
    u = x;
    v = y;

    auto reduce = [x, y] (int_type &t, int_type &G, int_type &H) {
        // TODO: Try speeding this up by treating ctz(t) bits at a time.
        while ( ! (t & 1)) {
            t >>= 1;
            // -((G | H) & 1) is 0 if G and H are even, and 0xffffffff... if G
            // or H is odd, because -1 in two's complement is all bits on.
            int_type odd_mask = -((G | H) & 1);

            G = (G + (odd_mask & y)) >> 1;
            H = (H - (odd_mask & x)) >> 1;
        }
    };

    do {
        reduce(u, A, B);
        reduce(v, C, D);
        // TODO: Find a way to avoid this branch.
        if (u >= v) {
            u -= v;
            A -= C;
            B -= D;
        } else {
            v -= u;
            C -= A;
            D -= B;
        }
    } while (u != 0);

    a = C;
    b = D;
    return v << g;
}

/*
 * Return 1 if x = 2^n for some n, 0 otherwise. (Caveat: Returns 1 for x = 0
 * which is not a binary power.)
 */
template< typename uint_type >
__host__ __device__ __forceinline__
int
is_binary_power(uint_type x) {
    static_assert(std::is_unsigned<uint_type>::value == true,
                  "template type must be unsigned");
    return ! (x & (x - 1));
}


/*
 * y >= x such that y = 2^n for some n. NB: This really is "inclusive"
 * next, i.e. if x is a binary power we just return it.
 */
__host__ __device__ __forceinline__
uint32_t
next_binary_power(uint32_t x) {
    enum { UINT32_BITS = 32 };
    return is_binary_power(x) ? x : (1 << (UINT32_BITS - clz(x)));
}


/*
 * ceiling(n / d) for integers.
 */
__host__ __device__ __forceinline__
int
iceil(int n, int d) {
    return (n + d - 1) / d;
}
