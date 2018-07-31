# cuda-fixnum
`cuda-fixnum` is a fixed-precision SIMD library that targets CUDA. It provides the apparatus necessary to easily create efficient functions that operate on vectors of _n_-bit integers, where _n_ can be much larger than the size of a usual machine or device register.  Currently supported values of _n_ are 32, 64, 128, 256, 512, 1024, and 2048 (larger values will be possible in a forthcoming release).

The primary use case for fast arithmetic of numbers in the range covered by `cuda-fixnum` is in cryptography and computational number theory. As such, special attention is given to support modular arithmetic; this is used in an example implementation of the Paillier additively homomorphic encryption scheme and of elliptic curve scalar multiplication.  Future releases will provide additional support for operations useful to implementing Ring-LWE-based somewhat homomorphic encryption schemes.

Finally, the library is designed to be _fast_. Through exploitation of warp-synchronous programming, vote functions, and deferred carry handling, the primitives of the library are currently competitive with the state-of-the-art in the literature for modular multiplication and modular exponentiation on GPUs.  The design of the library allows transparent substitution of the underlying arithmetic, allowing the user to select whichever performs best on the available hardware. Moreover, several algorithms, both novel and from the literature, will incorporated shortly that will improve performance by a further 25-50%.

The library is currently at the alpha stage of development.  It has many rough edges, but most features are present and it is performant enough to be competitive.  Comments, questions and contributions are welcome!

## Example

To get a feel for what it's like to use the library, let's consider a simple example. Here is an [implementation](cuda-fixnum/src/functions/paillier_encrypt.cu) of encryption in the [Paillier cryptosystem](https://en.wikipedia.org/wiki/Paillier_cryptosystem):
```cuda
#include "functions/quorem_preinv.cu"
#include "functions/modexp.cu"

template< typename fixnum >
class paillier_encrypt {
public:
    __device__ paillier_encrypt(fixnum n_)
        : n(n_), n_sqr(square(n_)), pow(n_sqr, n_), mod_n2(n_sqr) { }

    __device__ void operator()(fixnum &ctxt, fixnum m, fixnum r) const {
        fixnum::mul_lo(m, m, n);
        fixnum::incr_cy(m);
        pow(r, r);
        fixnum c_hi, c_lo;
        fixnum::mul_wide(c_hi, c_lo, m, r);
        mod_n2(ctxt, c_hi, c_lo);
    }

private:
    fixnum n;
    fixnum n_sqr;
    modexp<fixnum> pow;
    quorem_preinv<fixnum> mod_n2;

    __device__ fixnum square(fixnum n) {
        fixnum n2;
        fixnum::sqr_lo(n2, n);
        return n2;
    }
};
```
A few features will be common to most user-defined functions such as the one above: They will be templates that rely on a `fixnum`, which will be instantiated with one of the fixnum arithmetic implemententations provided, usually the [`warp_fixnum`](cuda-fixnum/src/fixnum/warp_fixnum.cu).  Functions in the `fixnum` class are static and (usually) return their results in the first one or two parameters. Complicated functions that might perform precomputation, such as [modular exponentiation (`modexp`)](cuda-fixnum/src/functions/modexp.cu) and [quotient & remainder with precomputed inverse (`quorem_preinv`)](cuda-fixnum/src/functions/quorem_preinv.cu) are instance variables in the object that are initialised in the constructor.

Although it is not (yet) the focus of this project to help optimise host-device communication, the [`fixnum_array`](cuda-fixnum/src/array/fixnum_array.h) facility is provided to make it easy to apply user-defined functions to data originating in the host. Using `fixnum_array` will often look like this:
```C++
void host_function() {
    ...
    typedef warp_fixnum<32, u64_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    fixnum_array *ctxts, *ptxts, *rnds, *pkeys;

    int nelts = ...;  // ~ 1e6
    int message_bytes = ...; // <= 32

    ptxts = fixnum_array::create(input_array, message_bytes, nelts);
    rands = fixnum_array::create(random_data, message_bytes, nelts);
    pkeys = fixnum_array::create_constant(public_key, pkey_bytes, nelts); // same value repeated
    ctxts = fixnum_array::create(ptxts->length());

    // TODO: Explain how to handle the fact that paillier_encrypt has a constructor.
    fixnum_array::template map<paillier_encrypt>(ctxts, ptxts, rands, pkeys);

    ctxts->retrieve_all(byte_buffer, buflen);
   ...
}
```
