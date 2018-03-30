#pragma once

// For some reason the warpSize value provided by CUDA is not
// considered a constant value, so cannot be used in constexprs or
// template parameters or static_asserts. Hence we must use WARPSIZE
// instead.
static constexpr int WARPSIZE = 32;

/*
 * SUBWARPS:
 *
 * Most of these functions operate on the level of a "subwarp" (NB:
 * this is not standard terminology).  A *warp* is a logical block of
 * 32 threads executed in lock-step by the GPU (thus obviating the
 * need for explicit synchronisation). For any w > 1 that divides 32,
 * a warp can be partitioned into 32/w subwarps of w threads.  The
 * struct below takes a parameter "width" which specifies the subwarp
 * size, and which thereby specifies the size of the numbers on which
 * its functions operate.
 *
 * The term "warp" should be reserved for subwarps of width 32
 * (=warpSize).
 *
 * TODO: All of the warp vote and warp shuffle functions will be
 * deprecated in CUDA 9.0 in favour of versions that take a mask
 * selecting relevant lanes in the warp on which to act (see CUDA
 * Programming Guide, B.15). Create an interface that encapsulates
 * both.
 *
 * TODO: Work out if using __forceinline__ in these definitions could
 * actually achieve anything.
 */

template<int width = WARPSIZE>
struct slot_layout
{
    static_assert(width > 0 && !(WARPSIZE & (width - 1)),
        "slot width must be a positive divisor of warpSize (=32)");

    static constexpr int SLOT_WIDTH = width;
    /*
     * Return the lane index within the slot.
     *
     * The lane index is the thread index modulo the width of the slot.
     */
    static __device__ __forceinline__
    int
    laneIdx() {
        // threadIdx.x % width = threadIdx.x & (width - 1) since width = 2^n
        return threadIdx.x & (width - 1);

        // TODO: Replace above with?
        // int L;
        // asm ("mov.b32 %0, %laneid;" : "=r"(L));
        // return L;
    }

    /*
     * Index of the top lane of the current slot.
     *
     * The top lane of a slot is the one with index width - 1.
     */
    static constexpr int toplaneIdx = width - 1;

    /*
     * Mask which selects the first width bits of a number.
     *
     * Useful in conjunction with offset() and __ballot().
     */
    static constexpr uint32_t mask = (1UL << width) - 1UL;

    /*
     * Return the thread index within the warp where the slot
     * containing this lane begins.  Examples:
     *
     * - width 16: slot offset is 0 for threads 0-15, and 16 for
     *    threads 16-31
     *
     * - width 8: slot offset is 0 for threads 0-7, 8 for threads 8-15,
     *    16 for threads 16-23, and 24 for threads 24-31.
     *
     * The slot offset at thread T in a slot of width w is given by
     * floor(T/w)*w.
     *
     * Useful in conjunction with mask() and __ballot().
     */
    static __device__ __forceinline__
    int
    offset() {
        // Thread index within the (full) warp.
        int T = threadIdx.x & (WARPSIZE - 1);

        // Recall: x mod y = x - y*floor(x/y), so
        //
        //   slotOffset = width * floor(threadIdx/width)
        //                 = threadIdx - (threadIdx % width)
        //                 = threadIdx - (threadIdx & (width - 1))
        //                 // TODO: Do use this last formulation!
        //                 = set bottom log2(width) bits of threadIdx to zero
        //
        // since width = 2^n.
        return T - (T & (width - 1));
    }

#if 0
    // TODO: Check if it is worth putting this in a specialisation of
    // width = warpSize. Note that, in the implementation below,
    // slotMask() will be a compile-time constant of 0xfff... so
    // the '&' instruction will be removed; however it's not clear
    // that the compiler will work out that slotOffset()
    // is always 0 when width = warpSize.
    /*
     * Wrapper for notation consistency.
     */
    static __device__ __forceinline__
    uint32_t
    ballot(int tst) {
        return __ballot(tst);
    }
#endif

    /*
     * Like ballot(tst) but restrict the result to the containing slot
     * of size width.
     */
    static __device__ __forceinline__
    uint32_t
    ballot(int tst) {
        uint32_t b = __ballot(tst);
        b >>= offset();
        return b & mask;
    }

    /*
     * Wrappers for notation consistency.
     */
    static __device__ __forceinline__
    uint32_t
    shfl(const uint32_t var, int srcLane) {
        return __shfl(var, srcLane, width);
    }

    static __device__ __forceinline__
    uint32_t
    shfl_up(uint32_t var, unsigned int delta) {
        return __shfl_up(var, delta, width);
    }

    static __device__ __forceinline__
    uint32_t
    shfl_down(uint32_t var, unsigned int delta) {
        return __shfl_down(var, delta, width);
    }

    // NB: Assumes delta <= width + L. (There should be no reason for
    // it ever to be more than width.)
    static __device__ __forceinline__
    uint32_t
    rotate_up(uint32_t var, unsigned int delta) {
        // FIXME: Doesn't necessarily work for width != 32
        int L = laneIdx();
        int srcLane = (L + width - delta) & (width - 1);
        return __shfl(var, srcLane, width);
    }

    static __device__ __forceinline__
    uint32_t
    rotate_down(uint32_t var, unsigned int delta) {
        // FIXME: Doesn't necessarily work for width != 32
        int L = laneIdx();
        int srcLane = (L + delta) & (width - 1);
        return __shfl(var, srcLane, width);
    }

    /*
     * The next three functions extend the usual shuffle functions to 64bit
     * parameters.  See CUDA C Programming Guide, B.14:
     *
     *   http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
     *
     * TODO: These are now available in CUDA 9.
     */
    static __device__ __forceinline__
    uint64_t
    shfl(const uint64_t &var, int srcLane) {
        uint64_t res;
        uint32_t hi, lo;

        asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
        hi = shfl(hi, srcLane);
        lo = shfl(lo, srcLane);
        asm("mov.b64 %0, { %1, %2 };" : "=l"(res) : "r"(lo), "r"(hi));
        return res;
    }

    static __device__ __forceinline__
    uint64_t
    shfl_up(uint64_t var, unsigned int delta) {
        uint64_t res;
        uint32_t hi, lo;

        asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
        hi = shfl_up(hi, delta);
        lo = shfl_up(lo, delta);
        asm("mov.b64 %0, { %1, %2 };" : "=l"(res) : "r"(lo), "r"(hi));
        return res;
    }

    static __device__ __forceinline__
    uint64_t
    shfl_down(uint64_t var, unsigned int delta) {
        uint64_t res;
        uint32_t hi, lo;

        asm("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
        hi = shfl_down(hi, delta);
        lo = shfl_down(lo, delta);
        asm("mov.b64 %0, { %1, %2 };" : "=l"(res) : "r"(lo), "r"(hi));
        return res;
    }

    /*
     * Like shfl_up but set bottom variable to zero.
     */
    template< typename T >
    static __device__ __forceinline__
    T
    shfl_up0(T var, unsigned int delta) {
        T res = shfl_up(var, delta);
        //return res & -(T)(laneIdx() > 0);
        return laneIdx() > 0 ? res : 0;
    }

    /*
     * Like shfl_down but set top variable to zero.
     */
    template< typename T >
    static __device__ __forceinline__
    T
    shfl_down0(T var, unsigned int delta) {
        T res = shfl_down(var, delta);
        //return res & -(T)(laneIdx() < toplaneIdx());
        return laneIdx() < toplaneIdx ? res : 0;
    }

private:
    slot_layout();
};
