#pragma once

#include "util/managed.cu"

template< typename fixnum_impl >
class set_const : public managed {
public:
    // FIXME: The repetition of this typedef in all the kernels is
    // dumb and annoying
    typedef typename fixnum_impl::fixnum fixnum;

    static set_const *create(const uint8_t *konst, int nbytes) {
        set_const *sc = new set_const;
        fixnum_impl::from_bytes(sc->konst, konst, nbytes);
        return sc;
    }

    template< typename T >
    static set_const *create(T init) {
        auto bytes = reinterpret_cast<const uint8_t *>(&init);
        return create(bytes, sizeof(T));
    }

    __device__ void operator()(fixnum &s) {
        int L = fixnum_impl::slot_layout::laneIdx();
        s = konst[L];
    }

private:
    typename fixnum_impl::fixnum konst[fixnum_impl::SLOT_WIDTH];
};
