#include <algorithm> // for min

#include "cuda_wrap.h"
#include "fixnum_array.h"
#include "primitives.cu"

// Notes: Read programming guide Section K.3
// - Can prefetch unified memory
// - Can advise on location of unified memory

// TODO: Can I use smart pointers? unique_ptr?

// TODO: Refactor these three constructors
template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts) {
    fixnum_array *a = new fixnum_array;
    a->nelts = nelts;
    if (nelts > 0) {
        size_t nwords = nelts * FIXNUM_STORAGE_WORDS;
        cuda_malloc_managed(&a->ptr, nwords * sizeof(word_tp));
    }
    return a;
}

template< typename fixnum_impl >
struct set_const;

template< typename fixnum_impl >
template< typename T >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(size_t nelts, T init) {
    fixnum_array *a = create(nelts);
    auto fn = set_const<fixnum_impl>::create(init);
    map(fn, a);
    delete fn;
    return a;
}

template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::create(const uint8_t *data, size_t total_bytes, size_t bytes_per_elt) {
    size_t nelts = total_bytes / bytes_per_elt;
    fixnum_array *a = create(nelts);

    word_tp *p = a->ptr;
    const uint8_t *d = data;
    for (size_t i = 0; i < nelts; ++i) {
        fixnum_impl::from_bytes(p, d, bytes_per_elt);
        p += FIXNUM_STORAGE_WORDS;
        d += bytes_per_elt;
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
    if (idx < 0 || idx > nelts) {
        // FIXME: This is not the right way to handle an "index out of
        // bounds" error.
        return 0;
    }
    fixnum_impl::to_bytes(dest, dest_space, ptr + idx * FIXNUM_STORAGE_WORDS);
    // FIXME: Return correct value.
    return dest_space;
}

// FIXME: Can return fewer than nelts elements.
template< typename fixnum_impl >
void
fixnum_array<fixnum_impl>::retrieve_all(uint8_t *dest, size_t dest_space, int *dest_nelts) const {
    const word_tp *p = ptr;
    uint8_t *d = dest;
    int max_dest_nelts = dest_space / fixnum_impl::FIXNUM_BYTES;
    *dest_nelts = std::min(nelts, max_dest_nelts);
    for (int i = 0; i < *dest_nelts; ++i) {
        fixnum_impl::to_bytes(d, fixnum_impl::FIXNUM_BYTES, p);
        p += FIXNUM_STORAGE_WORDS;
        d += fixnum_impl::FIXNUM_BYTES;
    }
}

template< template <typename> class Func, typename fixnum_impl, typename... Args >
__global__ void
dispatch(Func<fixnum_impl> *fn, int nelts, Args... args) {
    int idx = fixnum_impl::slot_idx();
    if (idx < nelts)
        (*fn)(fixnum_impl::get(args, idx)...);
}

template< typename fixnum_impl >
template< template <typename> class Func, typename... Args >
void
fixnum_array<fixnum_impl>::map(Func<fixnum_impl> *fn, Args... args) {
    // TODO: Set this to the number of threads on a single SM on the host GPU.
    constexpr int BLOCK_SIZE = 192;

    // BLOCK_SIZE must be a multiple of warpSize
    static_assert(!(BLOCK_SIZE % WARPSIZE),
            "block size must be a multiple of warpSize");

    int nelts = std::min( { args->length()... } );

    // FIXME: Check this calculation
    constexpr int fixnums_per_block = BLOCK_SIZE / fixnum_impl::SLOT_WIDTH;

    // FIXME: nblocks could be too big for a single kernel call to handle
    int nblocks = iceil(nelts, fixnums_per_block);

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

        // FIXME: Only synchronize when retrieving data from array
        cuda_device_synchronize();
    }
}
