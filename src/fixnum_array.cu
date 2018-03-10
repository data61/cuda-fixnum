#include <algorithm> // for min

#include "cuda_wrap.h"
#include "fixnum_array.h"
#include "primitives.cu"

template< typename fixnum_impl >
struct set_const;

template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts) {
    fixnum_array *a = new fixnum_array;
    a->nelts = nelts;
    if (nelts > 0) {
        size_t nbytes = nelts * fixnum_impl::STORAGE_BYTES;
        cuda_malloc(&a->ptr, nbytes);
    }
    return a;
}

template< typename fixnum_impl >
template< typename T >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts, T init) {
    fixnum_array *a = new fixnum_array;
    a->nelts = nelts;
    if (nelts > 0) {
        size_t nbytes = nelts * fixnum_impl::STORAGE_BYTES;
        cuda_malloc(&a->ptr, nbytes);
        if (init)
            fixnum_array::map(set_const<fixnum_impl>(init), a);
        else
            cuda_memset(a->ptr, 0, nbytes);
    }
    return a;
}

template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(const uint8_t *data, size_t len, size_t bytes_per_elt) {
    fixnum_array *a = new fixnum_array;
    size_t nelts = len / bytes_per_elt;
    a->nelts = nelts;
    if (nelts > 0) {
        size_t nbytes = nelts * fixnum_impl::STORAGE_BYTES;
        cuda_malloc(&a->ptr, nbytes);
        // FIXME: Finish this?!  Need a more intelligent way to handle
        // copying to (and from) the device.
    }
    return a;
}


template< typename fixnum_impl >
fixnum_array<fixnum_impl>::~fixnum_array() {
    if (nelts > 0)
        cuda_free(ptr);
}

template< typename fixnum_impl >
int
fixnum_array<fixnum_impl>::length() const {
    return nelts;
}

template< typename fixnum_impl >
size_t
fixnum_array<fixnum_impl>::retrieve_into(uint8_t *dest, size_t dest_space, int idx) const {
    constexpr size_t nbytes = fixnum_impl::FIXNUM_BYTES;
    if (dest_space < nbytes || idx < 0 || idx > nelts) {
        // FIXME: This is not the right way to handle an
        // "insufficient space" error or an "index out of bounds"
        // error.
        return 0;
    }
    // clear all of dest
    // TODO: Is this necessary? Should it be optional?
    memset(dest, 0, dest_space);
    cuda_memcpy_from_device(dest, ptr + idx * fixnum_impl::SLOT_WIDTH, nbytes);
    return nbytes;
}

// FIXME: Caller should provide the memory
// FIXME: Should be delegating to fixnum_impl to interpret the raw data
template< typename fixnum_impl >
void
fixnum_array<fixnum_impl>::retrieve_all(uint8_t *dest, size_t dest_space, size_t *dest_len, int *nelts) const {
    size_t space_needed = this->nelts * fixnum_impl::STORAGE_BYTES;

    *nelts = -1; // FIXME: don't mix error values and return parameters!
    *dest_len = 0;
    if (space_needed > dest_space) return;

    *nelts = this->nelts;
    if ( ! *nelts) return;

    *dest_len = space_needed;
    // FIXME: This won't correctly zero-pad each element in general
    memset(dest, 0, *dest_len);
    cuda_memcpy_from_device(dest, ptr, *dest_len);
}


template< template <typename> class Func, typename fixnum_impl, typename... Args >
__global__ void
dispatch(Func<fixnum_impl> fn, int nelts, Args... args) {
    int fn_idx = fixnum_impl::get_fn_idx();
    if (fn_idx < nelts)
        fn(fixnum_impl::load(args, fn_idx)...);
}

template< typename fixnum_impl >
template< template <typename> class Func, typename... Args >
void
fixnum_array<fixnum_impl>::map(Func<fixnum_impl> fn, Args... args) {
    // TODO: Set this to the number of threads on a single SM on the host GPU.
    constexpr int BLOCK_SIZE = 192;

    // BLOCK_SIZE must be a multiple of warpSize
    static_assert(!(BLOCK_SIZE % WARPSIZE),
            "block size must be a multiple of warpSize");

    int nelts = std::min( { args->length()... } );

    // FIXME: Check this calculation
    //int fixnums_per_block = (BLOCK_SIZE / warpSize) * fixnum_impl::NSLOTS;
    constexpr int fixnums_per_block = BLOCK_SIZE / fixnum_impl::THREADS_PER_FIXNUM;

    // FIXME: nblocks could be too big for a single kernel call to handle
    int nblocks = iceil(nelts, fixnums_per_block);

//    if (t) *t = clock();

    // nblocks > 0 iff nelts > 0
    if (nblocks > 0) {
        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "create stream");
        // FIXME: how do I attach the function?
        //stream_attach(stream, fn);
//         cuda_stream_attach_mem(stream, src->ptr);
//         cuda_stream_attach_mem(stream, ptr);
        cuda_check(cudaStreamSynchronize(stream), "stream sync");

        dispatch<<< nblocks, BLOCK_SIZE, 0, stream >>>(fn, nelts, args->ptr...);

        cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
        cuda_check(cudaStreamSynchronize(stream), "stream sync");
        cuda_check(cudaStreamDestroy(stream), "stream destroy");
    }

//    if (t) *t = clock() - *t;
}
