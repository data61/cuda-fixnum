#include <algorithm> // for min

#include "cuda_wrap.h"
#include "fixnum_array.h"
#include "primitives.cu"

template< typename fixnum_impl >
struct set_const;

template< typename fixnum_impl >
template< typename T >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts, T init) {
    fixnum_array *a = new fixnum_array;
    a->nelts = nelts;
    if (nelts > 0) {
        size_t nbytes = nelts * fixnum_impl::STORAGE_BYTES;
        cuda_malloc(&a->ptr, nbytes);
        fixnum_array::map(set_const<fixnum_impl>(init), a);
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

template< typename fixnum_impl >
void
fixnum_array<fixnum_impl>::retrieve(uint8_t **dest, size_t *dest_len, int idx) const {
    *dest_len = fixnum_impl::FIXNUM_BYTES;
    *dest = new uint8_t[*dest_len];
    retrieve_into(*dest, *dest_len, idx);
}

template< typename fixnum_impl >
void
fixnum_array<fixnum_impl>::retrieve_all(uint8_t **dest, size_t *dest_len, size_t *nelts) const {
    size_t nbytes;
    *nelts = this->nelts;
    nbytes = *nelts * fixnum_impl::FIXNUM_BYTES;
    *dest = new uint8_t[nbytes];
    // FIXME: This won't correctly zero-pad each element
    memset(dest, 0, nbytes);
    cuda_memcpy_from_device(*dest, ptr, nbytes);
}


template< template <typename> class Func, typename fixnum_impl, typename... Args >
__global__ void
dispatch(function< Func<fixnum_impl> > fn, int nelts, Args... args) {
    int fn_idx = fixnum_impl::get_fn_idx();
    if (fn_idx < nelts)
        fn(fixnum_impl::load(args, fn_idx)...);
}

template< typename fixnum_impl >
template< template <typename> class Func, typename... Args >
void
fixnum_array<fixnum_impl>::map(function< Func<fixnum_impl> > fn, Args... args) {
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
