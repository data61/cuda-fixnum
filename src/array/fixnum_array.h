#pragma once

#include <iostream>
#include <stdint.h>

// TODO: Copy over functionality and documentation from IntmodVector.
template< typename fixnum_impl >
class fixnum_array {
public:
    static fixnum_array *create(size_t nelts);
    template< typename T >
    static fixnum_array *create(size_t nelts, T init);
    // NB: If bytes_per_elt doesn't divide len, the last len % bytes_per_elt
    // bytes are *dropped*.
    static fixnum_array *create(const uint8_t *data, size_t total_bytes, size_t bytes_per_elt);

    ~fixnum_array();

    int length() const;

    int set(int idx, const uint8_t *data, size_t len);
    size_t retrieve_into(uint8_t *dest, size_t dest_space, int idx) const;
    void retrieve_all(uint8_t *dest, size_t dest_space, int *nelts) const;

    // Use:
    // fixnum_array::map(ec_add<fixnum_impl>(a, b), res, arr1, arr2);
    template< template <typename> class Func, typename... Args >
    static void map(Func<fixnum_impl> *fn, Args... args);

private:
    static constexpr int FIXNUM_STORAGE_WORDS = fixnum_impl::SLOT_WIDTH;

    typedef typename fixnum_impl::word_tp word_tp;
    word_tp *ptr;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);
};

template< typename fixnum_impl >
std::ostream &
operator<<(std::ostream &os, const fixnum_array<fixnum_impl> *fn_arr);

#include "fixnum_array.cu"
