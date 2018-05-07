#pragma once

#include <stdint.h>

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
umad_hi_cc(uint32_t &lo, uint32_t &cy, uint32_t a, uint32_t b, uint32_t c) {
    asm ("mad.hi.cc.u32 %0, %2, %3, %4;\n\t"
         "addc.u32 %1, %1, 0;"
         : "=r"(lo), "+r"(cy)
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
umad_hi_cc(uint64_t &lo, uint64_t &cy, uint64_t a, uint64_t b, uint64_t c) {
    asm ("mad.hi.cc.u64 %0, %2, %3, %4;\n\t"
         "addc.u64 %1, %1, 0;"
         : "=l"(lo), "+l"(cy)
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
 * Return 1 if x = 2^n for some n, 0 otherwise.
 */
__host__ __device__ __forceinline__
int
is_binary_power(uint32_t x) {
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
