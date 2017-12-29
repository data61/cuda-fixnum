#pragma once

#include <stdint.h>

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
    static fixnum_array *create(size_t nelts, T init = 0);
    static fixnum_array *create(const uint8_t *data, size_t len, size_t bytes_per_elt);

    ~fixnum_array();

    int length() const;

    size_t retrieve_into(uint8_t *dest, size_t dest_space, int idx) const;
    void retrieve(uint8_t **dest, size_t *dest_len, int idx) const;
    void retrieve_all(uint8_t **dest, size_t *dest_len, size_t *nelts) const;

private:
    // FIXME: This shouldn't be public; the create function that uses
    // it should be templatised.
    typedef typename hand_impl::digit value_tp;

    value_tp *ptr;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);

    template< typename Func >
    friend const fixnum_array *
    mapcar(Func fn, std::initializer_list<const fixnum_array *> args);
};

#include "fixnum_array.cu"
