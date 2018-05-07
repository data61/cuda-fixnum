#include <algorithm> // for min

#include "cuda_wrap.h"
#include "fixnum_array.h"
#include "primitives.cu"


static constexpr int CHUNK_BYTES = 1 << 20;
static constexpr int NCHUNKS = 4;
static int next_chunk = 0;
__device__ static uint8_t CHUNKS[NCHUNKS][CHUNK_BYTES];

template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts) {
    return new fixnum_array(nullptr, nelts, 0);
}

template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(const uint8_t *src, size_t nelts, size_t bytes_per_elt) {
    return new fixnum_array(src, nelts, bytes_per_elt);
}


template< typename fixnum_impl >
fixnum_array<fixnum_impl>::~fixnum_array() {
    //
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
static uint8_t *
copy_to_device(fixnum_array<fixnum_impl> *arg, int nbytes, int offset) {
    //assert(nbytes <= CHUNK_BYTES);
    uint8_t *chunk = CHUNKS[next_chunk++];
    if (arg->ptr != nullptr) {
        cuda_memcpy_to_device(chunk, arg->ptr + offset, nbytes);
    }
    return chunk;
}

template< typename fixnum_impl >
template< template <typename> class Func, typename Arg >
void
fixnum_array<fixnum_impl>::map(Func<fixnum_impl> fn, Arg arg) {
    // TODO: Set this to the number of threads on a single SM on the host GPU.
    constexpr int BLOCK_SIZE = 192;

    // BLOCK_SIZE must be a multiple of warpSize
    static_assert(!(BLOCK_SIZE % WARPSIZE),
            "block size must be a multiple of warpSize");

    int nelts = args->length();

    // FIXME: Check this calculation
    //int fixnums_per_block = (BLOCK_SIZE / warpSize) * fixnum_impl::NSLOTS;
    constexpr int fixnums_per_block = BLOCK_SIZE / fixnum_impl::THREADS_PER_FIXNUM;

    // FIXME: nblocks could be too big for a single kernel call to handle
    int nblocks = iceil(nelts, fixnums_per_block);

    // nblocks > 0 iff nelts > 0
    if (nblocks > 0) {
        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "create stream");

        uint8_t *dargs[] = { copy_to_device(args, , 0)... };

        dispatch<<< nblocks, BLOCK_SIZE, 0, stream >>>(fn, nelts, dargs);

        cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
        cuda_check(cudaStreamSynchronize(stream), "stream sync");
        cuda_check(cudaStreamDestroy(stream), "stream destroy");
    }

    int elements_per_call;
    for (int i = 0; i < nelts; i += elements_per_call) {
        // get next stream
        // copy arg to chunk
        // call kernel
        // retrieve result from chunk
    }
}
