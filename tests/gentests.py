from itertools import chain, product
from collections import Iterable
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


def gentests(op, xs, ys):
    return ([x, y, op(x, y)] for x, y in product(xs, ys))

def mktests(fname, arg):
    op, xs, ys = arg
    t = timer()
    print('Writing {} tests into "{}"... '.format(len(xs) * len(ys), fname), end='', flush=True)
    with open(fname, 'wb') as f:
        write_list(f, gentests(op, xs, ys))
    t = timer() - t
    print('done ({:.2f}s).'.format(t))
    return fname

def sub_br(x, y):
    if x >= y:
        return x - y
    return (x - y) % 2**2048

def generate_interesting_numbers(max_bits=2048, digit_bits=32):
    nums = list(range(12))
    for i in range(-2, 5):
        b = (i * digit_bits - 2) % max_bits
        nums.extend((1 << b) + k for k in range(-4, 4))
        nums.extend(1 << (b + k) for k in range(6))
#        nums.extend(1 << k for k in range(0, max_bits, 29))
#        nums.extend((1 << k) - 1 for k in range(0, max_bits, 29))

    return nums

def generate_everything():
    print('Generating input arguments... ', end='', flush=True)
    t = timer()
    xs = generate_interesting_numbers()
    ys = xs
    t = timer() - t
    print('done ({:.2f}s). Created {} arguments.'.format(t, len(xs)))

    ops = {
        'add_cy': (operator.add, xs, ys),
        'sub_br': (sub_br, xs, ys),
        'mul_wide': (operator.mul, xs, ys)
    }

    return list(map(mktests, ops.keys(), ops.values()))

if __name__ == '__main__':
    generate_everything()
    mkmodexptests('modexp')
