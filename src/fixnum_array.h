#pragma once

#include <stdint.h>

// From: https://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

// parameterised by
// hand implementation, which determines #bits per fixnum
//   and which is itself parameterised by
// subwarp data, which determines a SIMD decomposition of a fixnum
//
// TODO: Copy over functionality and documentation from IntmodVector.
template< typename fixnum_impl >
class fixnum_array {
public:
    static fixnum_array *create(size_t nelts);
    // src points to nelts*bytes_per_elt bytes.
    static fixnum_array *create(const uint8_t *src, size_t nelts, size_t bytes_per_elt);

    ~fixnum_array();

    int length() const;

    size_t retrieve_into(uint8_t *dest, size_t dest_space, int idx) const;
    void retrieve_all(uint8_t *dest, size_t dest_space, size_t *dest_len, int *nelts) const;

    // Use:
    // fixnum_array::map(ec_add<fixnum_impl>(a, b), res, arr1, arr2);
    template< template <typename> class Func, typename... Args >
    static void map(Func<fixnum_impl> fn, Args... args);

private:
    uint8_t *ptr;
    int nelts;
    int bytes_per_elt;

    fixnum_array(uint8_t *ptr_, int nelts_, int bytes_per_elt_)
    : ptr(ptr_), nelts(nelts_), bytes_per_elt(bytes_per_elt_) {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);
};

#include "fixnum_array.cu"
