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
    template< typename T >
    static fixnum_array *create(size_t nelts, T init);
    // NB: If bytes_per_elt doesn't divide len, the last len % bytes_per_elt
    // bytes are *dropped*.
    static fixnum_array *create(const uint8_t *data, size_t len, size_t bytes_per_elt);

    ~fixnum_array();

    int length() const;

    size_t retrieve_into(uint8_t *dest, size_t dest_space, int idx) const;
    void retrieve_all(uint8_t *dest, size_t dest_space, size_t *dest_len, int *nelts) const;

    // Use:
    // fixnum_array::map(ec_add<fixnum_impl>(a, b), res, arr1, arr2);
    template< template <typename> class Func, typename... Args >
    static void map(Func<fixnum_impl> fn, Args... args);

private:
    // FIXME: this is not the purpose of the fixnum definition, it
    // will not be a register type in general!
    typedef typename fixnum_impl::fixnum fixnum;
    fixnum *ptr;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);
};

#include "fixnum_array.cu"
