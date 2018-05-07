from cffi import FFI
from nose.tools import *
from nose_parameterized import parameterized
from collections import deque
import math
import pickle

ffi = FFI()
lib = ffi.dlopen('libpaillier.so')
print('Loaded lib {0}'.format(lib))

# Copypasta from 'src/paillier.h'
ffi.cdef('''
void cuda_finalise();
typedef int clock_t;

typedef struct IntmodVector IntmodVector;
IntmodVector *intmodvector_create(const uint8_t *in, int bytes_per_block, int nblocks);
IntmodVector *intmodvector_copy(const IntmodVector *v);
void intmodvector_destroy(IntmodVector *v);
void intmodvector_retrieve(const IntmodVector *v, uint8_t **out, int *bytes_per_block, int *nblocks);
int intmodvector_retrieve_into(
    const IntmodVector *v, uint8_t *out, size_t outbytes, int *bytes_per_block, int *nblocks);

typedef struct Modexp Modexp;
Modexp *modexp_create(const uint8_t *mod, int modbytes, const uint8_t *exp, int expbytes);
void modexp_destroy(Modexp *me);
void modexp_apply(const Modexp *me, IntmodVector *Z, clock_t *t);
// FIXME: Test this function
//void modexp_reset_exponent(Modexp *me, const digit_t *exp, int explen);

typedef struct MultiModexp MultiModexp;
MultiModexp *multimodexp_create(const uint8_t *mod, int modbytes);
void multimodexp_destroy(MultiModexp *me);
void multimodexp_apply(const MultiModexp *me, IntmodVector *Z, const IntmodVector *E, clock_t *t);

typedef struct Encrypt Encrypt;
Encrypt *encrypt_create(const uint8_t *key, int keylen, int only_obfuscate);
void encrypt_destroy(Encrypt *enc);
void encrypt_apply(const Encrypt *enc, IntmodVector *Z, const IntmodVector *R, clock_t *t);

typedef struct Decrypt Decrypt;
Decrypt *decrypt_create(const uint8_t *p, const uint8_t *q, int nbytes);
void decrypt_destroy(Decrypt *dec);
void decrypt_apply(const Decrypt *dec, IntmodVector *Z, clock_t *t);
''')

def int_bytes(N):
    return math.ceil(N.bit_length() / 8)

def int_to_array(N, block_bytes = 0):
    nbytes = int_bytes(N)
    if block_bytes == 0:
        block_bytes = nbytes
    if nbytes > block_bytes:
        raise ValueError('number ({} bytes) is too big for requested width ({} bytes)'.format(nbytes, block_bytes))
    return N.to_bytes(block_bytes, byteorder='little')

def seq_to_bytearray(intseq, block_bytes):
    return b''.join(int_to_array(N, block_bytes) for N in intseq);

def to_intmodvector(intseq, block_bytes = 0):
    if block_bytes == 0:
        block_bytes = max(int_bytes(n) for n in intseq)
    v = lib.intmodvector_create(seq_to_bytearray(intseq, block_bytes), block_bytes, len(intseq))
    assert v != ffi.NULL, 'block_bytes was {}'.format(block_bytes)
    return v

def from_intmodvector(ivec):
    x = ffi.new('uint8_t **')
    nblocks = ffi.new('int *')
    intbytes = ffi.new('int *')
    lib.intmodvector_retrieve(ivec, x, intbytes, nblocks)
    y = x[0]  # y is now a "uint8_t *"
    intbytes = intbytes[0]
    nblocks = nblocks[0]

    # TODO: I feel like there ought to be a more idiomatic way to
    # iterate over subarrays in python, something like CL's array
    # cursor.
    L = [int.from_bytes(y[i : i + intbytes], byteorder='little')
         for i in range(0, nblocks * intbytes, intbytes)]
    return L, intbytes, nblocks

# TODO: Implement "interface coherence/behaviour" tests more thoroughly.
class TestFixture:
    def setup(self):
        self.intbytes = 32
        self.mod = 123
        self.exps = [7, 8, 9, 10, 11, 12]
        self.bases = [0, 1, 2, 3, 4, 5]
        self.expected = [pow(b, e, self.mod) for b, e in zip(self.bases, self.exps)]

        self.B = to_intmodvector(self.bases, self.intbytes)
        self.E = to_intmodvector(self.exps, self.intbytes)

    def teardown(self):
        lib.intmodvector_destroy(self.B)
        lib.intmodvector_destroy(self.E)

class TestCreate(TestFixture):
    def test_create(self):
        u = lib.intmodvector_copy(self.E);
        L, intbytes, nblocks = from_intmodvector(u)
        assert nblocks == len(self.exps) and len(L) == nblocks
        assert intbytes == self.intbytes
        # Modify u
        intbytes = 128
        exp = int_to_array(1234567, intbytes)
        me = lib.modexp_create(exp, intbytes, exp, intbytes)
        t = ffi.new('int *')
        lib.modexp_apply(me, u, t)
        assert L == self.exps
        lib.intmodvector_destroy(u)
        lib.modexp_destroy(me)
        L, intbytes, nblocks = from_intmodvector(self.E)
        assert L == self.exps


@parameterized(pickle.load(open('multimodexp_input', 'rb')))
def test_multimodexp(mod, width, bases, exps, expected):
    intbytes = width * 8
    B = to_intmodvector(bases, intbytes)
    E = to_intmodvector(exps, intbytes)
    M = int_to_array(mod, intbytes)
    me = lib.multimodexp_create(M, intbytes)
    assert me != ffi.NULL

    t = ffi.new('int *')
    lib.multimodexp_apply(me, B, E, t)
    lib.multimodexp_destroy(me)

    out, w, n = from_intmodvector(B)
    tmp, w, n = from_intmodvector(E)
    assert out == expected, 'Failure with exps width = {}'.format(w)
    lib.intmodvector_destroy(B)
    lib.intmodvector_destroy(E)

    # FIXME: I think this leaks 'out'

@parameterized(pickle.load(open('modexp_input', 'rb')))
def test_modexp(mod, width, bases, exp, expected):
    intbytes = width * 8
    E = int_to_array(exp)
    M = int_to_array(mod, intbytes)
    B = to_intmodvector(bases, intbytes)
    me = lib.modexp_create(M, intbytes, E, len(E))
    if exp < 2:
        eq_(me, ffi.NULL)
    else:
        assert me != ffi.NULL
        t = ffi.new('int *')
        lib.modexp_apply(me, B, t)
        lib.modexp_destroy(me)

        out, w, n = from_intmodvector(B)
        eq_(out, expected)
        lib.intmodvector_destroy(B)

    # FIXME: I think this leaks 'out'

@parameterized(pickle.load(open('encrypt_input', 'rb')))
def test_encrypt_decrypt(key, width, plaintxt, masks, expected):
    p, q = key
    if p == q or p < 5:
        return
    intbytes = 128

    P = to_intmodvector(plaintxt, 2*intbytes)
    R = to_intmodvector(masks, intbytes)
    key = int_to_array(p*q, intbytes)

    enc = lib.encrypt_create(key, intbytes, 0)
    assert enc != ffi.NULL
    t = ffi.new('int *')
    lib.encrypt_apply(enc, P, R, t)
    lib.encrypt_destroy(enc)

    out, w, n = from_intmodvector(P)
    assert n == len(plaintxt)
    eq_(out, expected)
    lib.intmodvector_destroy(R)

    halfkey_bytes = intbytes // 2
    halfkey_bits = halfkey_bytes * 8;
    if p.bit_length() < halfkey_bits or q.bit_length() < halfkey_bits:
        return
    p = int_to_array(p, halfkey_bytes)
    q = int_to_array(q, halfkey_bytes)
    dec = lib.decrypt_create(p, q, halfkey_bytes)
    assert dec != ffi.NULL
    lib.decrypt_apply(dec, P, t)
    out, w, n = from_intmodvector(P)
    eq_(out, plaintxt)
    lib.decrypt_destroy(dec)
    lib.intmodvector_destroy(P)

