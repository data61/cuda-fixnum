#include "cuda_wrap.h"
#include "hand.cu"

#include "fixnum_array.h"

// TODO: Passing both Hand and Func is ugly; Hand should be implicit
// in Func.
template< typename Func, typename... Args >
__global__ void
dispatch(Func fn, int nelts, Args... args) {

    // TODO NEXT: This stuff should go in some pre/post-hook code in
    // some parent class for device functions
    
    int blk_tid_offset = blockDim.x * blockIdx.x;
    int tid_in_blk = threadIdx.x;
    int fn_idx = (blk_tid_offset + tid_in_blk) / Hand::SLOT_WIDTH;

    if (fn_idx < nelts) {
        //dest->ptr + off, dest->ptr + off, src->ptr + off);
        Hand::call(fn, fn_idx, args...);
    }

    // NEXT: just use this
    
    fn(nelts, args...);
}

template< typename hand_impl >
template< typename T >
fixnum_array<hand_impl> *
fixnum_array<hand_impl>::create(size_t nelts, T init) {
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


template< typename hand_impl >
fixnum_array<hand_impl>::~fixnum_array() {
    if (nelts > 0)
        cuda_free(ptr);
}

template< typename hand_impl >
int
fixnum_array<hand_impl>::length() const {
    return nelts;
}

template< typename hand_impl >
size_t
fixnum_array<hand_impl>::retrieve_into(uint8_t *dest, size_t dest_space, int idx) const {
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

template< typename hand_impl >
void
fixnum_array<hand_impl>::retrieve(uint8_t **dest, size_t *dest_len, int idx) const {
    *dest_len = hand_impl::FIXNUM_BYTES;
    *dest = new uint8_t[*dest_len];
    retrieve_into(*dest, *dest_len, idx);
}

template< typename hand_impl >
void
fixnum_array<hand_impl>::retrieve_all(uint8_t **dest, size_t *dest_len, size_t *nelts) const {
    size_t nbytes;
    *nelts = this->nelts;
    nbytes = *nelts * hand_impl::FIXNUM_BYTES;
    *dest = new uint8_t[nbytes];
    // FIXME: This won't correctly zero-pad each element
    memset(dest, 0, nbytes);
    cuda_memcpy_from_device(*dest, ptr, nbytes);
}

// TODO: Currently restricted to a single return value of type
// fixnum_array. We might want to return multiple values, possible of
// plain arrays (e.g. add_cy could return a fixnum_array and an int
// array).
template< typename Func, typename... Args >
void
mapcar(Func fn, Args... args) {
    // TODO: Set this to the number of threads on a single SM on the host GPU.
    constexpr int BLOCK_SIZE = 192;

    // BLOCK_SIZE must be a multiple of warpSize
    static_assert(!(BLOCK_SIZE % WARPSIZE),
            "block size must be a multiple of warpSize");

    // FIXME: check that arrays are all the same length. Or find the minimum
    // length and use that?
    //assert(nelts == src->nelts);

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
//         cuda_stream_attach_mem(stream, src->ptr);
//         cuda_stream_attach_mem(stream, ptr);
        cuda_check(cudaStreamSynchronize(stream), "stream sync");

        dispatch<<< nblocks, BLOCK_SIZE, 0, stream >>>(fn, nelts, args...);

        cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
        cuda_check(cudaStreamSynchronize(stream), "stream sync");
        cuda_check(cudaStreamDestroy(stream), "stream destroy");
    }
    if (t) *t = clock() - *t;
}
