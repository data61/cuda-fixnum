import sys
import pickle
import functools
import itertools
import operator
import multiprocessing
import gmpy2
from gmpy2 import mpz
from collections import deque

NCPUS = multiprocessing.cpu_count()

def generate_interesting_numbers():
    nums = list(range(12))
    digbits = 64
    width = 32
    intmodbits = width * digbits
    for i in range(-2, 5):
        b = (i * digbits - 2) % intmodbits
        nums += [(1 << b) + k for k in range(-4, 4)]
        nums += [1 << (b + k) for k in range(6)]
#        nums += [1 << k for k in range(0, intmodbits, 29)]
#        nums += [(1 << k) - 1 for k in range(0, intmodbits, 29)]

    return nums

def print_msg(name, ntests):
    print('Generating about {} test cases for {} (ncpus={})...'.format(ntests, name, NCPUS))

def my_pow(base, exp, mod):
    return pow(base, exp, mod)

def do_pickle(fname, tests):
    print('\nGenerated {} test cases'.format(len(tests)))
    with open(fname, 'wb') as f:
        pickle.dump(tests, f)

def gen_modexp(nums):
    pool = multiprocessing.Pool(processes = NCPUS)
    print_msg('modexp', len(nums)**2 // 2)
    res = []
    width = 32
    for mod in nums:
        if mod == 0 or mod < 2 or not (mod & 1):
            continue
        # FIXME: ugly
        tmpnums = [n % mod for n in nums]

        for e in tmpnums:
            fn = functools.partial(my_pow, exp = e, mod = mod)
            expected = pool.map(fn, tmpnums)
            #expected = [fn(b) for b in tmpnums]

            # TODO: bases is the same for every element: store it just once.
            # TODO: for some reason, pickling the deque exponents doesn't store
            # the fact that it has been rotated.
            res.append([mod, width, tmpnums, e, list(expected)])
            print('.', end='', flush=True)
    do_pickle('modexp_input', res)

def gen_multimodexp(nums):
    pool = multiprocessing.Pool(processes = NCPUS)
    print_msg('multimodexp', len(nums)**2 // 2)
    width = 32
    res = []
    for mod in nums:
        if mod < 2 or not (mod & 1):
            continue
        if mod == 0:
            # Can't divide by zero
            continue
        # FIXME: ugly
        tmpnums = [n % mod for n in nums]
        exponents = deque(tmpnums)

        for i in range(len(nums)):
            #expected = [pow(b, e, mod) for b, e in zip(tmpnums, exponents)]
            fn = functools.partial(my_pow, mod = mod)
            expected = pool.starmap(fn, zip(tmpnums, exponents))

            # TODO: bases is the same for every element: store it just once.
            # TODO: for some reason, pickling the deque exponents doesn't store
            # the fact that it has been rotated.
            res.append([mod, width, tmpnums, list(exponents), list(expected)])
            exponents.rotate(1)
            print('.', end='', flush=True)

    do_pickle('multimodexp_input', res)

def uniq(lst):
    return map(operator.itemgetter(0), itertools.groupby(lst))

def gen_encryptdecrypt(nums):
    pool = multiprocessing.Pool(processes = NCPUS)
    width = 32

    maxbitlen = 512
    res = []
    print('Generating primes... ', end='', flush=True)
    candidates = [n for n in nums if n.bit_length() <= maxbitlen]
    primes = [int(p) for p in uniq(map(gmpy2.next_prime, candidates)) if p.bit_length() <= maxbitlen]
    print('done.')
    print_msg('encryption', len(primes) * (len(primes) - 1) // 2)

    for i in range(len(primes)):
        p = primes[i]
        for q in primes[i+1:]:
            if p == q:
                continue
            key = p * q
            key_sqr = key**2

            # Using nums as both the plaintext and the random mask.
            masks = [n % key for n in nums]
            plaintxts = [n % key for n in nums]
            expected = [((1 + key * ptxt) * pow(r, key, key_sqr)) % key_sqr
                            for ptxt, r in zip(plaintxts, masks)]

            res.append([(p, q), width, plaintxts, masks, expected])
            print('.', end='', flush=True)

    do_pickle('encrypt_input', res)

if __name__ == "__main__":
    nums = generate_interesting_numbers()
    gen_modexp(nums)
    gen_multimodexp(nums)
    gen_encryptdecrypt(nums)
