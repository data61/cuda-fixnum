#pragma once

#include "functions/quorem_preinv.cu"
#include "functions/divexact.cu"
#include "functions/chinese.cu"
#include "functions/multi_modexp.cu"


template< typename fixnum_impl >
class paillier_decrypt_mod;

template< typename fixnum_impl >
class paillier_decrypt {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ paillier_decrypt(fixnum p, fixnum q)
        : n(prod(p, q))
        , crt(p, q)
        , decrypt_modp(p, n)
        , decrypt_modq(q, n) {  }

    __device__ void operator()(fixnum &ptxt, fixnum ctxt_hi, fixnum ctxt_lo) const;

private:
    // We only need this in the constructor to initialise decrypt_mod[pq], but we
    // store it here because it's the only way to save the computation and pass
    // it to the constructors of decrypt_mod[pq].
    fixnum n;

    // Secret key is (p, q).
    paillier_decrypt_mod<fixnum_impl> decrypt_modp, decrypt_modq;

    // TODO: crt and decrypt_modq both compute and hold quorem_preinv(q); find a
    // way to share them.
    chinese<fixnum_impl> crt;

    // TODO: It is flipping stupid that this is necessary. Also, it has to be
    // called twice.
    __device__ fixnum prod(fixnum p, fixnum q) {
        fixnum n;
        // TODO: These don't work when SLOT_WIDTH = 0
        //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || p == 0);
        //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || q == 0);
        fixnum_impl::mul_lo(n, p, q);
        return n;
    }
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
    decrypt_modp(mp, ctxt_hi, ctxt_lo);
    decrypt_modq(mq, ctxt_hi, ctxt_lo);
    crt(ptxt, mp, mq);
}


template< typename fixnum_impl >
class paillier_decrypt_mod {
public:
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ paillier_decrypt_mod(fixnum p, fixnum n);

    __device__ void operator()(fixnum &mp, fixnum c_hi, fixnum c_lo) const;

private:
    // FIXME: These all have width = WIDTH/2, so this is a waste of
    // space, and (worse) the operations below waste cycles.

    // Precomputation of
    //   L((1 + n)^(p - 1) mod p^2)^-1 (mod p)
    // for CRT, where n = pq is the public key, and L(x) = (x-1)/p.
    fixnum h;

    // We only need this in the constructor to initialise mod_p2 and pow, but we
    // store it here because it's the only way to save the computation and pass
    // it to the constructors of mod_p2 and pow.
    fixnum p_sqr;

    // Exact division by p
    divexact<fixnum_impl> div_p;
    // Remainder after division by p.
    quorem_preinv<fixnum_impl> mod_p;
    // Remainder after division by p^2.
    quorem_preinv<fixnum_impl> mod_p2;

    // Modexp for x |--> x^(p - 1) (mod p^2)
    modexp<fixnum_impl> pow;

    // TODO: It is flipping stupid that this is necessary.
    __device__ fixnum square(fixnum p) {
        fixnum p2;
        // TODO: This doesn't work when SLOT_WIDTH = 0
        //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || p == 0);
        fixnum_impl::sqr_lo(p2, p);
        return p2;
    }
    __device__ fixnum sub1(fixnum p) {
        fixnum pm1;
        fixnum_impl::sub_br(pm1, p, fixnum_impl::one());
        return pm1;
    }
};


template< typename fixnum_impl >
__device__
paillier_decrypt_mod<fixnum_impl>::paillier_decrypt_mod(fixnum p, fixnum n)
    : p_sqr(square(p))
    , div_p(p)
    , mod_p(p)
    , mod_p2(p_sqr)
    , pow(p_sqr, sub1(p))
{
    int cy;
    fixnum t = n;
    cy = fixnum_impl::incr_cy(t);
    // n is the product of primes, and 2^(2^k) - 1 has (at least) k factors,
    // hence n is less than 2^FIXNUM_BITS - 1, hence incrementing n shouldn't
    // overflow.
    assert(cy == 0);
    // TODO: Check whether reducing t is necessary.
    mod_p2(t, 0, t);
    pow(t, t);
    fixnum_impl::decr_br(t);
    div_p(t, t);

    // TODO: Make modinv use xgcd and use modinv instead.
    // Use a^(p-2) = 1 (mod p)
    fixnum pm2, two = (fixnum_impl::slot_layout::laneIdx() == 0) ? 2 : 0;
    fixnum_impl::sub_br(pm2, p, two);
    multi_modexp<fixnum_impl> minv(p);
    minv(h, t, pm2);
}

/*
 * Decrypt ciphertext (c_hi, c_lo) and put the result in mp.
 *
 * Decryption mod p of c is put in the (bottom half of) mp.
 */
template< typename fixnum_impl >
__device__ void
paillier_decrypt_mod<fixnum_impl>::operator()(fixnum &mp, fixnum c_hi, fixnum c_lo) const
{
    fixnum c, u, hi, lo;
    // mp = c_hi * 2^n + c_lo (mod p^2)  which is nonzero because p != q
    mod_p2(c, c_hi, c_lo);

    pow(u, c);
    fixnum_impl::decr_br(u);
    div_p(u, u);
    // Check that the high half of u is now zero.
    // TODO: This doesn't work when SLOT_WIDTH = 0
    //assert(fixnum_impl::slot_layout::laneIdx() < fixnum_impl::SLOT_WIDTH/2 || u == 0);

    // TODO: make use of the fact that u and h are half-width.
    fixnum_impl::mul_wide(hi, lo, u, h);
    assert(hi == 0);
    mod_p(mp, hi, lo);
}
