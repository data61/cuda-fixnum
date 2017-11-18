#pragma once

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

    int add_cy(const fixnum_array *other) {
        // FIXME: Return correct carry
        int cy = 0;
        typename hand_impl::add_cy fn;
        apply_to_all(fn, other);
        return cy;
    }

    void mullo(const fixnum_array *other) {
        typename hand_impl::mullo fn;
        apply_to_all(fn, other);
    }


private:
    // FIXME: This shouldn't be public; the create function that uses
    // it should be templatised.
    typedef typename hand_impl::digit value_tp;

    value_tp *ptr;
    int nelts;

    fixnum_array() {  }

    fixnum_array(const fixnum_array &);
    fixnum_array &operator=(const fixnum_array &);

    // TODO: This whole function shouldn't be a template; most of the
    // code is independent of Func.
    template< typename Func >
    void apply_to_all(Func fn, const fixnum_array *src, clock_t *t = 0);
};

#include "fixnum_array.cu"
