// -*- compile-command: "nvcc -D__STRICT_ANSI__ -ccbin clang-3.8 -std=c++11 -Xcompiler -Wall,-Wextra -g -G -lineinfo -gencode arch=compute_50,code=sm_50 -o bench bench.cu -lstdc++" -*-

#include <iostream>
#include <cstring>
#include <cassert>

#include "cuda_wrap.h"
#include "hand.cu"

using namespace std;

template< typename H >
class fixnum_array;

// TODO: Passing both Hand and Func is ugly; Hand should be implicit
// in Func.
template< typename Hand, typename Func, typename... Args >
__global__ void
dispatch(Func fn, int nelts, Args... args) {
    int blk_tid_offset = blockDim.x * blockIdx.x;
    int tid_in_blk = threadIdx.x;
    int fn_idx = (blk_tid_offset + tid_in_blk) / Hand::SLOT_WIDTH;

    if (fn_idx < nelts) {
        int off = fn_idx * Hand::SLOT_WIDTH;

        //dest->ptr + off, dest->ptr + off, src->ptr + off);
        fn(off, args...);
    }
}


// parameterised by
// hand implementation, which determines #bits per fixnum
//   and which is itself parameterised by
// subwarp data, which determines a SIMD decomposition of a fixnum
//
// TODO: Copy over functionality and documentation from IntmodVector.
template< typename hand_impl >
class fixnum_array {
public:
    template< typename T >
    static fixnum_array *create(size_t nelts, T init = 0) {
        fixnum_array *a = new fixnum_array;
        a->nelts = nelts;
        if (nelts > 0) {
            size_t nbytes = nelts * hand_impl::FIXNUM_BYTES;
            int c = static_cast<int>(init);
            cuda_malloc(&a->ptr, nbytes);
            // FIXME: Obviously should use zeros and init somehow
            cuda_memset(a->ptr, c, nbytes);
        }
        return a;
    }

    static fixnum_array *create(const uint8_t *data, size_t len, size_t bytes_per_elt);

    ~fixnum_array() {
        if (nelts > 0)
            cuda_free(ptr);
    }

    int length() const {
        return nelts;
    }

    size_t retrieve_into(uint8_t *dest, size_t dest_space, int idx) const {
        size_t nbytes = hand_impl::FIXNUM_BYTES;
        if (dest_space < nbytes || idx < 0 || idx > nelts) {
            // FIXME: This is not the right way to handle an
            // "insufficient space" error or an "index out of bounds"
            // error.
            return 0;
        }
        // clear all of dest
        // TODO: Is this necessary? Should it be optional?
        memset(dest, 0, dest_space);
        cuda_memcpy_from_device(dest, ptr + idx * hand_impl::SLOT_WIDTH, nbytes);
        return nbytes;
    }

    void retrieve(uint8_t **dest, size_t *dest_len, int idx) const {
        *dest_len = hand_impl::FIXNUM_BYTES;
        *dest = new uint8_t[*dest_len];
        retrieve_into(&dest, *dest_len, idx);
    }

    void retrieve_all(uint8_t **dest, size_t *dest_len, size_t *nelts) const {
        size_t nbytes;
        *nelts = this->nelts;
        nbytes = *nelts * hand_impl::FIXNUM_BYTES;
        *dest = new uint8_t[nbytes];
        // FIXME: This won't correctly zero-pad each element
        memset(dest, 0, nbytes);
        cuda_memcpy_from_device(*dest, ptr, nbytes);
    }

    int add_cy(const fixnum_array *other) {
        // FIXME: Return correct carry
        int cy = 0;
        typename hand_impl::add_cy fn;
        apply_to_all(fn, other);
        return cy;
    }

    void mullo(const fixnum_array *other) {
        typename hand_impl::mullo fn;
        apply_to_all(fn, other);
    }

private:
    // FIXME: This shouldn't be public; the create function that uses
    // it should be templatised.
    typedef typename hand_impl::digit value_tp;

    value_tp *ptr;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);

    // TODO: This whole function shouldn't be a template; most of the
    // code is independent of Func.
    template< typename Func >
    void apply_to_all(Func fn, const fixnum_array *src, clock_t *t = 0) {
        // TODO: Set this to the number of threads on a single SM on the host GPU.
        constexpr int BLOCK_SIZE = 192;

        // dest and src must be the same length
        assert(nelts == src->nelts);
        // BLOCK_SIZE must be a multiple of warpSize
        static_assert(!(BLOCK_SIZE % WARPSIZE),
                "block size must be a multiple of warpSize");

        // FIXME: Check this calculation
        //int fixnums_per_block = (BLOCK_SIZE / warpSize) * hand_impl::NSLOTS;
        constexpr int fixnums_per_block = BLOCK_SIZE / hand_impl::SLOT_WIDTH;

        // FIXME: nblocks could be too big for a single kernel call to handle
        int nblocks = iceil(src->nelts, fixnums_per_block);

        if (t) *t = clock();
        // nblocks > 0 iff src->nelts > 0
        if (nblocks > 0) {
            cudaStream_t stream;
            cuda_check(cudaStreamCreate(&stream), "create stream");
            // FIXME: how do I attach the function?
            //stream_attach(stream, fn);
//            cuda_stream_attach_mem(stream, src->ptr);
//            cuda_stream_attach_mem(stream, ptr);
            cuda_check(cudaStreamSynchronize(stream), "stream sync");

            dispatch<hand_impl><<< nblocks, BLOCK_SIZE, 0, stream >>>(fn, nelts, ptr, ptr, src->ptr);

            cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
            cuda_check(cudaStreamSynchronize(stream), "stream sync");
            cuda_check(cudaStreamDestroy(stream), "stream destroy");
        }
        if (t) *t = clock() - *t;
    }
};


// FIXME: Ignore this idea of feeding in new operations for now; just
// use a fixed set of operations determined by hand_impl
//
// FIXME: Passing this to map as an object probably makes inlining
// impossible in most circumstances.
template< typename T, typename subwarp_data >
struct device_op {
    int _x;

    device_op(int x) : _x(x) { }

    // A fixnum is represented by a register across a subwarp. This
    // thread is responsible for the Lth registers of the arguments,
    // where L is the lane index.
    //
    // This function should be available from the hand_impl; this sort
    // of function should be implemented in terms of the hand_impl
    // functions.
    __device__ void
    operator()(T &s, T &cy, T a, T b) {
        s = a + b;
        cy = s < a;
    }
};


template< typename H >
ostream &
operator<<(ostream &os, const fixnum_array<H> *arr) {
    constexpr int nbytes = H::FIXNUM_BYTES;
    uint8_t num[nbytes];
    int nelts = arr->length();

    os << "( ";
    if (nelts > 0) {
        (void) arr->retrieve_into(num, nbytes, 0);
        os << (int)num[0];
        for (int i = 1; i < nelts; ++i) {
            (void) arr->retrieve_into(num, nbytes, i);
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

    // hand_impl determines how operations map to a warp
    //
    // bits_per_fixnum should somehow capture the fact that a warp can
    // be divided into subwarps
    //
    // n is the number of fixnums in the array; eventually only allow
    // initialisation via a byte array or whatever
    typedef fixnum_array< full_hand<uint32_t, 4> > fixnum_array;
    auto arr1 = fixnum_array::create(n, 5);
    auto arr2 = fixnum_array::create(n, 7);

    // device_op should be able to start operating on the appropriate
    // memory straight away
    //device_op fn(7);

    // FIXME: How do I return cy without allocating a gigantic array
    // where each element is only 0 or 1?  Could return the carries in
    // the device_op fn?
    //fixnum_array::map(fn, res, arr1, arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    arr1->add_cy(arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    arr1->mullo(arr2);

    cout << "arr1 = " << arr1 << endl;
    cout << "arr2 = " << arr2 << endl;

    delete arr1;
    delete arr2;

    return 0;
}
