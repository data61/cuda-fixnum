#pragma once

#include "utils/managed.cu"
#include "functions/quorem_preinv.cu"
#include "functions/divexact.cu"
#include "functions/chinese.cu"
#include "functions/modexp.cu"


template< typename fixnum_impl >
class paillier_decrypt_mod;


template< typename fixnum_impl >
class paillier_decrypt : public managed {
    typedef typename fixnum_impl::word_tp word_tp;

    // Secret key is (p, q).
    word_tp p[WIDTH];
    word_tp q[WIDTH];

    // Remainder after division by p^2 and q^2
    quorem_preinv<fixnum_impl> mod_p2, mod_q2;
    paillier_decrypt_mod<fixnum_impl> decrypt_modp, decrypt_modq;
    chinese<fixnum_impl> crt;

public:
    typedef typename fixnum_impl::fixnum fixnum;

    paillier_decrypt(const uint8_t *p, const uint8_t *q, size_t nbytes);

    __device__ void operator()(fixnum &ptxt, fixnum ctxt_hi, fixnum ctxt_lo) const;
};

/**
 * Decrypt the ciphertext c = (c_hi, c_lo) and put the resulting plaintext in m.
 *
 * m, c_hi and c_lo must be PLAINTEXT_DIGITS long.
 */
template< typename fixnum_impl >
__device__ void
paillier_decrypt<fixnum_impl>::operator()(fixnum &ptxt, fixnum ctxt_hi, fixnum ctxt_lo) const
{
    fixnum mp, mq;

    // TODO: Does it make more sense to put the mod_[pq]2 stuff in the
    // decrypt_mod objects?
    mp = mq = 0;
    // mp = c_hi * 2^n + c_lo (mod p^2)
    mod_p2(mp, ctxt_hi, ctxt_lo);
    // mq = c_hi * 2^n + c_lo (mod q^2)
    mod_q2(mq, ctxt_hi, ctxt_lo);

    decrypt_modp(mp, mp);
    decrypt_modq(mq, mq);

    crt(ptxt, mp, mq);
}


template< typename fixnum_impl >
class paillier_decrypt_mod : public managed {
    typedef typename fixnum_impl::word_tp word_tp;

    // FIXME: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.

    // Precomputation of
    //   L((1 + n)^(p - 1) mod p^2)^-1 (mod p)
    // for CRT, where p = mod, n = pq is the public key, and L(x) = (x-1)/p.
    word_tp h[WIDTH];
    word_tp mod[WIDTH];

    divexact div;

    // FIXME: comment above about width applies here too.
    // Remainder after division by mod.
    quorem_preinv<fixnum_impl> rem_mod;

    // Modexp for x |--> x^(mod - 1) (mod mod^2)
    modexp<fixnum_impl> pow;
    
public:
    typedef typename fixnum_impl::fixnum fixnum;

    paillier_decrypt_mod(const uint8_t *mod, size_t nbytes);

    __device__ void operator()(fixnum &mp, fixnum c) const;
};


template< typename fixnum_impl >
paillier_decrypt_mod<fixnum_impl>::paillier_decrypt_mod(const uint8_t *mod, size_t nbytes)
    : rem_mod(mod, nbytes)
{
    pow = modexp(/* mod - 1, nbytes */);
}

/*
 * Decrypt ciphertext c and put the result in mp.
 *
 * c should be the ciphertext reduced modulo p^2.  Decryption mod p of
 * c is put in the (bottom half of) mp.
 */
template< typename fixnum_impl >
__device__ void
paillier_decrypt_mod<fixnum_impl>::operator()(fixnum &mp, fixnum c) const
{
    fixnum u, m, h, hi, lo;
    m = fixnum_impl::load(this->mod);
    h = fixnum_impl::load(this->h);
    
    pow(u, c);
    fixnum_impl::decr_br(u);
    // TODO: Get divexact to precompute the inverse to m rather than doing it every time.
    div(u, u, m);
    // Check that the high half of u is now zero.
    assert(fixnum_impl::laneIdx() < WIDTH/2 || u == 0);

    // TODO: make use of the fact that u and h are half-width.
    fixnum_impl::mul_wide(hi, lo, u, h);
    assert(hi == 0);
    mp = 0;
    rem_mod(mp, hi, lo);
}
