from itertools import chain, product
from collections import deque
from timeit import default_timer as timer

def write_int(dest, sz, n):
    dest.write(n.to_bytes(sz, byteorder = 'little'))

def write_vector(dest, elt_sz, v):
    for n in v:
        write_int(dest, elt_sz, n)

def mktests(op, xs, bits):
    ys = deque(xs)
    res = []
    for i in range(len(xs)):
        yield zip(*[op(x, y, bits) for x, y in zip(xs, ys)])
        ys.rotate(1)

def write_tests(fname, arg):
    op, xs, bits = arg
    t = timer()
    print('Writing {} tests into "{}"... '.format(len(xs)**2, fname), end='', flush=True)
    with open(fname, 'wb') as f:
        fixnum_bytes = bits >> 3
        vec_len = len(xs)
        # FIXME: Need to get this from op somehow
        nvecs = 2 # number of output values
        write_int(f, 4, fixnum_bytes)
        write_int(f, 4, vec_len)
        write_int(f, 4, nvecs)
        write_vector(f, fixnum_bytes, xs)
        for v in mktests(op, xs, bits):
            v = list(v)
            assert len(v) == nvecs, 'bad result length'
            for res in v:
                write_vector(f, fixnum_bytes, res)
    t = timer() - t
    print('done ({:.2f}s).'.format(t))
    return fname

def add_cy(x, y, bits):
    return [(x + y) & ((1<<bits) - 1), (x + y) >> bits]

def sub_br(x, y, bits):
    return [(x - y) & ((1<<bits) - 1), int(x < y)]

def mul_wide(x, y, bits):
    return [(x * y) & ((1<<bits) - 1), (x * y) >> bits]

def test_inputs(nbytes):
    assert nbytes >= 4 and (nbytes & (nbytes - 1)) == 0, "nbytes must be a binary power at least 4"
    q = nbytes // 4
    res = [0]

    nums = [1, 2, 3];
    nums.extend([2**32 - n for n in nums])

    for i in range(q):
        res.extend(n << 32*i for n in nums)

    lognbits = (32*q).bit_length()
    for i in range(2, lognbits - 1):
        # b = 0xF, 0xFF, 0xFFFF, 0xFFFFFFFF, ...
        e = 1 << i
        b = (1 << e) - 1
        c = sum(b << 2*e*j for j in range(32*q // (2*e)))
        res.extend([c, (1 << 32*q) - c - 1])
    return res

def generate_everything(nbytes):
    print('Generating input arguments... ', end='', flush=True)
    bits = nbytes * 8

    t = timer()
    xs = test_inputs(nbytes)
    t = timer() - t
    print('done ({:.2f}s). Created {} arguments.'.format(t, len(xs)))
    if nbytes == 4:
        print('xs = {}'.format(xs))

    ops = {
        'add_cy': (add_cy, xs, bits),
        'sub_br': (sub_br, xs, bits),
        'mul_wide': (mul_wide, xs, bits)
    }
    fnames = map(lambda fn: fn + '_' + str(nbytes), ops.keys())
    return list(map(write_tests, fnames, ops.values()))

if __name__ == '__main__':
    for i in range(2, 9):
        generate_everything(1 << i)
