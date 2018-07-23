// for printing arrays
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
// for min
#include <algorithm>

#include "util/cuda_wrap.h"
#include "util/primitives.cu"
#include "functions/set_const.cu"
#include "fixnum_array.h"

// TODO: The only device function in this file is the dispatch kernel
// mechanism, which could arguably be placed elsewhere, thereby
// allowing this file to be compiled completely for the host.

// Notes: Read programming guide Section K.3
// - Can prefetch unified memory
// - Can advise on location of unified memory

// TODO: Can I use smart pointers? unique_ptr?

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
    // FIXME: Should handle this error more appropriately
    if (total_bytes == 0 || bytes_per_elt == 0)
        return nullptr;

    size_t nelts = iceil(total_bytes, bytes_per_elt);
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

// TODO: Find a way to return a wrapper that just modifies the requested indices
// on the fly, rather than copying the whole array. Hard part will be making it
// work with map/dispatch.
template< typename fixnum_impl >
fixnum_array<fixnum_impl> *
fixnum_array<fixnum_impl>::rotate(int i) {
    fixnum_array *a = create(length());
    word_tp *p = a->ptr;
    if (i < 0) {
        int j = -i;
        i += nelts * iceil(j, nelts);
        assert(i >= 0 && i < nelts);
        i = nelts - i;
    } else if (i >= nelts)
        i %= nelts;
    int pivot = i * FIXNUM_STORAGE_WORDS;
    int nwords = nelts * FIXNUM_STORAGE_WORDS;
    std::copy(ptr, ptr + nwords - pivot, p + pivot);
    std::copy(ptr + nwords - pivot, ptr + nwords, p);
    return a;
}


template< typename fixnum_impl >
int
fixnum_array<fixnum_impl>::set(int idx, const uint8_t *data, size_t nbytes) {
    // FIXME: Better error handling
    if (idx < 0 || idx >= this->nelts)
        return -1;

    int off = idx * FIXNUM_STORAGE_WORDS;
    return fixnum_impl::from_bytes(this->ptr + off, data, nbytes);
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
    return fixnum_impl::to_bytes(dest, dest_space, ptr + idx * FIXNUM_STORAGE_WORDS);
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

namespace {
    std::string
    fixnum_as_str(const uint8_t *fn, int nbytes) {
        std::ostringstream ss;

        for (int i = nbytes - 1; i >= 0; --i) {
            // These IO manipulators are forgotten after each use;
            // i.e. they don't apply to the next output operation (whether
            // it be in the next loop iteration or in the conditional
            // below.
            ss << std::setfill('0') << std::setw(2) << std::hex;
            ss << static_cast<int>(fn[i]);
            if (i && !(i & 3))
                ss << ' ';
        }
        return ss.str();
    }
}

template< typename fixnum_impl >
std::ostream &
operator<<(std::ostream &os, const fixnum_array<fixnum_impl> *fn_arr) {
    constexpr int fn_bytes = fixnum_impl::FIXNUM_BYTES;
    constexpr size_t bufsz = 4096;
    uint8_t arr[bufsz];
    int nelts;

    fn_arr->retrieve_all(arr, bufsz, &nelts);
    os << "( ";
    if (nelts < fn_arr->length()) {
        os << "insufficient space to retrieve array";
    } else if (nelts > 0) {
        os << fixnum_as_str(arr, fn_bytes);
        for (int i = 1; i < nelts; ++i)
            os << ", " << fixnum_as_str(arr + i*fn_bytes, fn_bytes);
    }
    os << " )" << std::flush;
    return os;
}


template< template <typename> class Func, typename fixnum_impl, typename... Args >
__global__ void
dispatch(int nelts, Args... args) {
    // Get the slot index for the current thread.
    int blk_tid_offset = blockDim.x * blockIdx.x;
    int tid_in_blk = threadIdx.x;
    int idx = (blk_tid_offset + tid_in_blk) / fixnum_impl::SLOT_WIDTH;

    if (idx < nelts) {
        Func<fixnum_impl> fn;
        fn(fixnum_impl::load(args, idx)...);
    }
}

template< typename fixnum_impl >
template< template <typename> class Func, typename... Args >
void
fixnum_array<fixnum_impl>::map(Args... args) {
    // TODO: Set this to the number of threads on a single SM on the host GPU.
    constexpr int BLOCK_SIZE = 192;

    // FIXME: WARPSIZE should come from slot_layout
    constexpr int WARPSIZE = 32;
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
//         cuda_stream_attach_mem(stream, src->ptr);
//         cuda_stream_attach_mem(stream, ptr);
        cuda_check(cudaStreamSynchronize(stream), "stream sync");

        dispatch<Func, fixnum_impl ><<< nblocks, BLOCK_SIZE, 0, stream >>>(nelts, args->ptr...);

        cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
        cuda_check(cudaStreamSynchronize(stream), "stream sync");
        cuda_check(cudaStreamDestroy(stream), "stream destroy");

        // FIXME: Only synchronize when retrieving data from array
        cuda_device_synchronize();
    }
}
