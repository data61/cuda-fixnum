#ifndef CUDA_MPMA_UTILS_CUH
#define CUDA_MPMA_UTILS_CUH

/*
 * PTX Assembler primitives
 */

// TODO: Understand circumstances in which I might want to make this
// "#define ASM asm __volatile__".
//#define ASM asm

// hi * 2^32 + lo = a * b
__device__ __forceinline__ void
umul(uint32_t &hi, uint32_t &lo, uint32_t a, uint32_t b)
{
    asm ("{\n\t"
         " .reg .u64 tmp;\n\t"
         " mul.wide.u32 tmp, %2, %3;\n\t"
         " mov.b64 { %0, %1 }, tmp;\n\t"
         "}"
         : "=r"(hi), "=r"(lo)
         : "r"(a), "r"(b));
}

__device__ __forceinline__ void
umul(uint64_t &r, uint32_t a, uint32_t b)
{
    asm ("mul.wide.u32 %0, %1, %2;"
         : "=l"(r)
         : "r"(a), "r"(b));
}

// hi * 2^64 + lo = a * b
__device__ __forceinline__ void
umul(uint64_t &hi, uint64_t &lo, uint64_t a, uint64_t b)
{
    asm ("mul.hi.u64 %0, %2, %3;\n\t"
         "mul.lo.u64 %1, %2, %3;"
         : "=l"(hi), "=l"(lo)
         : "l"(a), "l"(b));
}

// r = a * b + c
__device__ __forceinline__ void
umad(uint64_t &r, uint32_t a, uint32_t b, uint32_t c)
{
    asm ("mad.wide.u32 %0, %1, %2, %3;"
         : "=l"(r)
         : "r"(a), "r"(b), "r"(c));
}

// (hi, lo) = a * b + c
__device__ __forceinline__ void
umad(uint32_t &hi, uint32_t &lo, uint32_t a, uint32_t b, uint32_t c)
{
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
umad(uint64_t &hi, uint64_t &lo, uint64_t a, uint64_t b, uint64_t c)
{
    asm ("mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
         "madc.hi.u64 %0, %2, %3, 0;"
         : "=l"(hi), "=l"(lo)
         : "l"(a), "l" (b), "l"(c));
}

// lo = a * b + c (mod 2^64)
__device__ __forceinline__ void
umad_lo(uint64_t &lo, uint64_t a, uint64_t b, uint64_t c)
{
    asm ("mad.lo.u64 %0, %1, %2, %3;"
         : "=l"(lo)
         : "l"(a), "l" (b), "l"(c));
}

__device__ __forceinline__ void
umad_hi(uint64_t &hi, uint64_t a, uint64_t b, uint64_t c)
{
    asm ("mad.hi.u64 %0, %1, %2, %3;"
         : "=l"(lo)
         : "l"(a), "l" (b), "l"(c));
}

// as above but with carry in cy
__device__ __forceinline__ void
umad_lo_cc(uint64_t &lo, uint64_t &cy, uint64_t a, uint64_t b, uint64_t c)
{
    asm ("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
         "addc.u64 %1, %1, 0;"
         : "=l"(lo), "+l"(cy)
         : "l"(a), "l" (b), "l"(c));
}

__device__ __forceinline__ void
umad_hi_cc(uint64_t &lo, uint64_t &cy, uint64_t a, uint64_t b, uint64_t c)
{
    asm ("mad.hi.cc.u64 %0, %2, %3, %4;\n\t"
         "addc.u64 %1, %1, 0;"
         : "=l"(lo), "+l"(cy)
         : "l"(a), "l" (b), "l"(c));
}


/*
 * SUBWARPS:
 *
 * Most of these functions operate on the level of a "subwarp" (NB:
 * this is not standard terminology).  A *warp* is a logical block of
 * 32 threads executed in lock-step by the GPU (thus obviating the
 * need for explicit synchronisation). For any w > 1 that divides 32,
 * a warp can be partitioned into 32/w subwarps of w threads.  The
 * functions below take a parameter "width" which specifies the
 * subwarp size, and which thereby specifies the size of the numbers
 * on which they operate.
 *
 * More specifically, for a width w, the numbers are w digit_t's long,
 * with the ith digit_t in lane i of the subwarp.  So a subwarp of
 * size w is operating on numbers at most 2^(64 * w), since digit_t's
 * are 64 bits.
 *
 * The term "warp" should be reserved for subwarps of width 32
 * (=warpSize).
 */

/*
 * Return the lane index within the subwarp.
 *
 * The lane index is the thread index modulo the width of the subwarp.
 * width must divide warpSize (= 32) and be at least 2.
 */
__device__ __forceinline__ int
laneIdx(int width = warpSize)
{
    assert(width > 1 && !(warpSize & (width - 1)));
    // threadIdx.x % width = threadIdx.x & (width - 1) since width = 2^n
    return threadIdx.x & (width - 1);
}

/*
 * Return the lane index and the top lane of the current subwarp.
 *
 * The top lane of a subwarp is the one with index width - 1.
 */
__device__ void
laneIdx_and_topLane(int &L, int &T, int width)
{
    T = width - 1;
    L = threadIdx.x & T;
}

/*
 * Return the thread index within the warp where the subwarp
 * containing this lane begins.  Examples:
 *
 * - width 16: subwarp offset is 0 for threads 0-15, and 16 for
 *    threads 16-31
 *
 * - width 8: subwarp offset is 0 for threads 0-7, 8 for threads 8-15,
 *    16 for threads 16-23, and 24 for threads 24-31.
 *
 * The subwarp offset at thread T in a subwarp of width w is given by
 * floor(T/w)*w.
 *
 * Useful in conjunction with subwarpMask() and __ballot().
 */
__device__ __forceinline__ int
subwarpOffset(int width)
{
    // Thread index within the (full) warp.
    int T = threadIdx.x & (warpSize - 1);

    // Recall: x mod y = x - y*floor(x/y), so
    //
    //   subwarpOffset = width * floor(threadIdx/width)
    //                 = threadIdx - (threadIdx % width)
    //                 = threadIdx - (threadIdx & (width - 1))
    //
    // since width = 2^n.
    return T - (T & (width - 1));
}

/*
 * Return a mask which selects the first width bits of a number.
 *
 * Useful in conjunction with subwarpOffset() and __ballot().
 */
__device__ __forceinline__ uint32_t
subwarpMask(int width)
{
    return (1UL << width) - 1UL;
}

/*
 * Wrapper for notation consistency.
 */
__device__ __forceinline__ uint32_t
ballot(int tst)
{
    return __ballot(tst);
}

/*
 * Like ballot(tst) but restrict the result to the containing subwarp
 * of size width.
 */
__device__ __forceinline__ uint32_t
ballot(int tst, int width)
{
    uint32_t b = __ballot(tst);
    b >>= subwarpOffset(width);
    return b & subwarpMask(width);
}


/*
 * Wrappers for notation consistency.
 */
__device__ __forceinline__ uint32_t
shfl(const uint32_t var, int srcLane, int width)
{
    return __shfl(var, srcLane, width);
}

__device__ __forceinline__ void
shfl_up(uint32_t &var, unsigned int delta, int width)
{
    return __shfl_up(var, delta, width);
}

__device__ __forceinline__ void
shfl_down(uint32_t &var, unsigned int delta, int width)
{
    return __shfl_down(var, delta, width);
}

/*
 * The next three functions extend the usual shuffle functions to 64bit
 * parameters.  See CUDA C Programming Guide, B.14:
 *
 *   http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
 */
__device__ uint64_t
shfl(const uint64_t &var, int srcLane, int width)
{
    int hi, lo;
    uint64_t res;

    asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl(hi, srcLane, width);
    lo = __shfl(lo, srcLane, width);
    asm("mov.b64 %0, { %1, %2 };" : "=l"(res) : "r"(lo), "r"(hi));
    return res;
}

__device__ void
shfl_up(uint64_t &var, unsigned int delta, int width)
{
    int hi, lo;

    asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl_up(hi, delta, width);
    lo = __shfl_up(lo, delta, width);
    asm("mov.b64 %0, { %1, %2 };" : "=l"(var) : "r"(lo), "r"(hi));
}

__device__ void
shfl_down(uint64_t &var, unsigned int delta, int width)
{
    int hi, lo;

    asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl_down(hi, delta, width);
    lo = __shfl_down(lo, delta, width);
    asm("mov.b64 %0, { %1, %2 };" : "=l"(var) : "r"(lo), "r"(hi));
}



#endif
