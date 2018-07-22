from itertools import chain, product
from collections import Iterable, deque
from timeit import default_timer as timer
import operator

OPEN_PAREN = b'('
CLOSE_PAREN = b')'
LEN_DELIMIT = b':'
LEN_BASE = 'd'

def iceildiv(n, d):
    """Return ceil(n / d) as an integer."""
    return (n + d - 1) // d

def write_atom(dest, atom):
    if isinstance(atom, int):
        atom = atom.to_bytes(iceildiv(atom.bit_length(), 8), 'little')
    dest.write(bytes(format(len(atom), LEN_BASE), encoding='ascii'))
    dest.write(LEN_DELIMIT)
    dest.write(atom)

def write_list(dest, lst):
    dest.write(OPEN_PAREN)
    for el in lst:
        if isinstance(el, Iterable):
            write_list(dest, el)
        else:
            write_atom(dest, el)
    dest.write(CLOSE_PAREN)

def modexp_tests(xs, ns, es):
    return [[n, e]
            + [x % n for x in xs]
            + [pow(x, e, n) for x in xs]
            for n, e in product(ns, es) if n > 2 and n % 2 == 1]

def mkmodexptests(fname):
    xs = generate_interesting_numbers()
    t = timer()
    print('Writing {} tests into "{}"... '.format(len(xs)**3, fname), end='', flush=True)
    with open(fname, 'wb') as f:
        write_list(f, modexp_tests(xs, xs, xs))
    t = timer() - t
    print('done ({:.2f}s).'.format(t))
    return fname


def mktests(fname, arg):
    op, xs, bits = arg
    t = timer()
    print('Writing {} tests into "{}"... '.format(len(xs)**2, fname), end='', flush=True)
    with open(fname, 'wb') as f:
        f.write(bits >> 3)
        f.write(len(xs))
        f.write(2) # How many output values?
        f.write(xs)
        f.write(op(xs, bits))
    t = timer() - t
    print('done ({:.2f}s).'.format(t))
    return fname

def add_cy(xs, bits):
    ys = deque(xs)
    res = []
    for i in range(len(xs)):
        ys.rotate(1)
        res.append(zip(*[[(x + y) & ((1<<bits) - 1), (x + y) >> bits] for x, y in zip(xs, ys)]))
    return res

def sub_br(xs, bits):
    ys = deque(xs)
    res = []
    for i in range(len(xs)):
        ys.rotate(1)
        res.append(zip(*[[(x - y) & ((1<<bits) - 1), int(x < y)] for x, y in zip(xs, ys)]))
    return res

def mul_wide(xs, bits):
    ys = deque(xs)
    res = []
    for i in range(len(xs)):
        ys.rotate(1)
        res.append(zip(*[[(x * y) & ((1<<bits) - 1), (x * y) >> bits] for x, y in zip(xs, ys)]))
    return res

def test_inputs_four_bytes():
    nums = [1, 2, 3];
    nums.extend([2^32 - n for n in nums])
    nums.extend([0xFF << i for i in range(4)])
    nums.extend([0, 0xFFFF, 0xFFFF0000, 0xFF00FF00, 0xFF00FF, 0xF0F0F0F0, 0x0F0F0F0F])
    #nums.extend([1 << i for i in range(32)])
    nums.append(0)
    return nums

def test_inputs(nbytes):
    assert nbytes >= 4 && (nbytes & (nbytes - 1)), "nbytes must be a binary power at least 4"
    nums = test_inputs_four_bytes()
    q = nbytes / 4;
    return itertools.product(nums, repeat = q)

def generate_everything():
    print('Generating input arguments... ', end='', flush=True)
    t = timer()
    xs = test_inputs()
    ys = xs
    t = timer() - t
    print('done ({:.2f}s). Created {} arguments.'.format(t, len(xs)))

    ops = {
        'add_cy': (operator.add, xs, ys),
        'sub_br': (sub_br, xs, ys),
        'mul_wide': (mul_wide, xs, ys)
    }
    return list(map(mktests, ops.keys(), ops.values()))

if __name__ == '__main__':
    generate_everything()
#    mkmodexptests('modexp')
