#include <gtest/gtest.h>
#include <tuple>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <fstream>
#include "sexp.h"

// FIXME FIXME!
#include "sexp.cc"

#include "array/fixnum_array.h"
#include "fixnum/default.cu"
#include "functions/monty_mul.cu"
#include "functions/modexp.cu"

using namespace std;

void die_if(bool p, const char *msg) {
    if (p) {
        cerr << "Error: " << msg << endl;
        abort();
    }
}

// should be a tuple
//typedef tuple<byte_array, byte_array, byte_array> binop_args;
typedef vector<byte_array> binop_args;

class extract_all_args : public visitor {
    vector<binop_args> args;
    bool in_list = false;

    vector<byte_array> current_arg;
    bool expecting_atom = false;

public:
    virtual ~extract_all_args() { }

    virtual void visit(const list &lst) override {
        if (in_list) {
            die_if(expecting_atom, "Got a list when expecting an atom.");
            expecting_atom = true;
        } else {
            in_list = true;
        }
    }

    virtual void visit(const atom &el) override {
        die_if( ! expecting_atom, "Got an atom when expecting a list.");
        current_arg.push_back(el.data);
        if (current_arg.size() == 3) {
            //binop_args t = make_tuple(current_arg[0], current_arg[1], current_arg[2]);
            args.push_back(current_arg);
            current_arg.clear();
            expecting_atom = false;
        }
    }

    vector<binop_args> result() {
        return args;
    }
};

vector<binop_args>
fname_to_args(const char *fname) {
    ifstream file(fname);
    die_if ( ! file.good(), "Couldn't open file.");
    auto s = unique_ptr<const sexp>(sexp::create(file));
    extract_all_args v;
    s->accept(v);
    return v.result();
}

::testing::AssertionResult
arrays_are_equal(
    const uint8_t *expected, size_t expected_len,
    const uint8_t *actual, size_t actual_len)
{
    if (expected_len > actual_len) {
        return ::testing::AssertionFailure()
            << "arrays don't have the same length";
    }
    size_t i;
    for (i = 0; i < expected_len; ++i) {
        if (expected[i] != actual[i]) {
            return ::testing::AssertionFailure()
                << "arrays differ: expected[" << i << "] = "
                << static_cast<int>(expected[i])
                << " but actual[" << i << "] = "
                << static_cast<int>(actual[i]);
        }
    }
    for (; i < actual_len; ++i) {
        if (actual[i] != 0) {
            return ::testing::AssertionFailure()
                << "arrays differ: expected[" << i << "] = 0"
                << " but actual[" << i << "] = "
                << static_cast<int>(actual[i]);
        }
    }
    return ::testing::AssertionSuccess();
}


template< typename fixnum_impl_ >
struct TypedPrimitives : public ::testing::Test {
    typedef fixnum_impl_ fixnum_impl;

    TypedPrimitives() {}
};

typedef ::testing::Types<
    default_fixnum_impl<4, uint32_t>,
    default_fixnum_impl<8, uint32_t>,
    default_fixnum_impl<16, uint32_t>,
    default_fixnum_impl<32, uint32_t>,
    default_fixnum_impl<64, uint32_t>,
    default_fixnum_impl<128, uint32_t>,

    default_fixnum_impl<8, uint64_t>,
    default_fixnum_impl<16, uint64_t>,
    default_fixnum_impl<32, uint64_t>,
    default_fixnum_impl<64, uint64_t>,
    default_fixnum_impl<128, uint64_t>,
    default_fixnum_impl<256, uint64_t>
> FixnumImplTypes;

TYPED_TEST_CASE(TypedPrimitives, FixnumImplTypes);

template< typename fixnum_impl >
void set_args_from_tcases(
        const char *fname,
        vector<binop_args> &tcases,
        fixnum_array<fixnum_impl> *&res,
        fixnum_array<fixnum_impl> *&xs,
        fixnum_array<fixnum_impl> *&ys,
        bool truncate = true)
{
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    typedef fixnum_array<fixnum_impl> fixnum_array;
    tcases = fname_to_args(fname);

    // Filter out test cases whose arguments would be truncated
    if ( ! truncate) {
        auto args_too_big = [] (binop_args &args) {
            return args[0].size() > FIXNUM_BYTES
                || args[1].size() > FIXNUM_BYTES;
        };
        auto it = remove_if(tcases.begin(), tcases.end(), args_too_big);
        int nskipped = static_cast<int>(tcases.end() - it);
        if (nskipped > 0) {
            int ntests = tcases.size();
            cerr << "Skipping " << nskipped << " / " << ntests
                 << " (" << setprecision(3) << nskipped * 100.0 / ntests << "%) "
                 << "tests to avoid truncation." << endl;
            tcases.erase(it, tcases.end());
        }
    }

    int n = (int) tcases.size();

    xs = fixnum_array::create(n);
    ys = fixnum_array::create(n);
    res = fixnum_array::create(n);

    for (int i = 0; i < n; ++i) {
        const binop_args &arg = tcases[i];
        const byte_array &x = arg[0];
        const byte_array &y = arg[1];
        int r;

        r = xs->set(i, x.data(), x.size());
        ASSERT_EQ(r, std::min(x.size(), FIXNUM_BYTES));
        r = ys->set(i, y.data(), y.size());
        ASSERT_EQ(r, std::min(y.size(), FIXNUM_BYTES));
    }
}

template< typename fixnum_impl, typename Iter >
void check_result(
    const fixnum_array<fixnum_impl> *res,
    Iter begin, Iter end)
{
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    int n = res->length();
    int cnt = 0;
    for (Iter i = begin; i < end; ++i, ++cnt) {
        die_if(cnt == n, "more results than expected");
        const byte_array &expected = *i;
        uint8_t arr[FIXNUM_BYTES];
        memset(arr, 0, FIXNUM_BYTES);

        size_t b = res->retrieve_into(arr, FIXNUM_BYTES, cnt);
        ASSERT_EQ(b, FIXNUM_BYTES);
        EXPECT_TRUE(arrays_are_equal(expected.data(), expected.size(), arr, FIXNUM_BYTES)
                    << " at index " << cnt);
    }
}

template< typename fixnum_impl >
void check_result(
    const vector<binop_args> &tcases,
    initializer_list<const fixnum_array<fixnum_impl> *> args)
{
    typedef const fixnum_array<fixnum_impl> *arg_ptr;
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;

    // All arguments must be the same length
    int n = (int) tcases.size();
    for_each(args.begin(), args.end(),
        [n] (arg_ptr arg) { EXPECT_EQ(arg->length(), n); });

    // We will concatenate corresponding elements from args into arr for
    // comparison with their corresponding result in tcases.
    int nargs = args.size();
    size_t arrlen = FIXNUM_BYTES * nargs;
    uint8_t *arr = new uint8_t[arrlen];

    for (int i = 0; i < n; ++i) {
        const byte_array &expected = tcases[i][2];

        memset(arr, 0, arrlen);
        uint8_t *dest = arr;
        for (auto arg : args) {
            size_t b = arg->retrieve_into(dest, FIXNUM_BYTES, i);
            ASSERT_EQ(b, FIXNUM_BYTES);
            dest += FIXNUM_BYTES;
        }

        size_t expected_len = std::min(expected.size(), arrlen);
        EXPECT_TRUE(arrays_are_equal(expected.data(), expected_len, arr, arrlen)
                    << " at index i = " << i);
    }
    delete[] arr;
}

template< typename fixnum_impl >
void check_result(
    const vector<binop_args> &tcases,
    const fixnum_array<fixnum_impl> *arg)
{
    check_result(tcases, {arg});
}

template< typename fixnum_impl >
struct add_cy : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::add_cy(r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, add_cy) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs, *ys;
    vector<binop_args> tcases;
    set_args_from_tcases("tests/add_cy", tcases, res, xs, ys);

    auto fn = new add_cy<fixnum_impl>();
    fixnum_array::map(fn, res, xs, ys);
    delete fn;

    // FIXME: check for carries
    check_result(tcases, res);
    delete res;
    delete xs;
    delete ys;
}


template< typename fixnum_impl >
struct sub_br : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::sub_br(r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, sub_br) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs, *ys;
    vector<binop_args> tcases;
    set_args_from_tcases("tests/sub_br", tcases, res, xs, ys);

    auto fn = new sub_br<fixnum_impl>();
    fixnum_array::map(fn, res, xs, ys);
    delete fn;

    // FIXME: check for borrows
    check_result(tcases, res);
    delete res;
    delete xs;
    delete ys;
}


template< typename fixnum_impl >
struct mul_lo : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_lo(r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, mul_lo) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs, *ys;
    vector<binop_args> tcases;
    set_args_from_tcases("tests/mul_wide", tcases, res, xs, ys);

    auto fn = new mul_lo<fixnum_impl>();
    fixnum_array::map(fn, res, xs, ys);
    delete fn;

    check_result(tcases, res);
    delete res;
    delete xs;
    delete ys;
}

template< typename fixnum_impl >
struct mul_wide : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &s, fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_wide(s, r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, mul_wide) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;
    static constexpr bool TRUNCATE = false;

    fixnum_array *his, *los, *xs, *ys;
    vector<binop_args> tcases;
    set_args_from_tcases("tests/mul_wide", tcases, los, xs, ys, TRUNCATE);
    // FIXME: This should be set in set_args_from_tcases somehow.
    his = fixnum_array::create(tcases.size());

    auto fn = new mul_wide<fixnum_impl>();
    fixnum_array::map(fn, his, los, xs, ys);
    delete fn;

    check_result(tcases, {los, his});
    delete his;
    delete los;
    delete xs;
    delete ys;
}


template< typename fixnum_impl >
struct to_monty : public managed {
    typedef typename fixnum_impl::fixnum fixnum;
    const monty_mul<fixnum_impl> *mul;

    to_monty(const monty_mul<fixnum_impl> *mul_)
        : mul(mul_) { }

    __device__ void operator()(fixnum &z, fixnum x) {
        mul->to_monty(z, x);
    }
};

template< typename fixnum_impl >
struct from_monty : public managed {
    typedef typename fixnum_impl::fixnum fixnum;
    const monty_mul<fixnum_impl> *mul;

    from_monty(const monty_mul<fixnum_impl> *mul_)
        : mul(mul_) { }

    __device__ void operator()(fixnum &z, fixnum x) {
        mul->from_monty(z, x);
    }
};

TYPED_TEST(TypedPrimitives, monty_conversion)
{
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs, *ys;

    int n = 13;
    res = fixnum_array::create(n);
    xs = fixnum_array::create(n, 7);
    ys = fixnum_array::create(n);

    // 23 + 39*256 = 10007 = nextprime(1e4)
    static constexpr int modbytes = 8;
    uint8_t modulus[modbytes] = { 23, 0, 0, 0, 0, 0, 0, 0 };

    auto mul = new monty_mul<fixnum_impl>(modulus, modbytes);

    // FIXME: The use of to_monty is awkward; should really consider making a
    // first-class Monty representation.
    auto to = new to_monty<fixnum_impl>(mul);
    fixnum_array::map(to, res, xs);
    delete to;

    // Square
    fixnum_array::map(mul, res, res);

    auto from = new from_monty<fixnum_impl>(mul);
    fixnum_array::map(from, ys, res);
    delete from;

    // check result.
    static constexpr int FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    static constexpr size_t arrlen = FIXNUM_BYTES;
    uint8_t arr1[arrlen];
    uint8_t arr2[arrlen];

    for (int i = 0; i < n; ++i) {
        size_t b;
        memset(arr1, 0, arrlen);
        memset(arr2, 0, arrlen);
        b = xs->retrieve_into(arr1, FIXNUM_BYTES, i);
        ASSERT_EQ(b, FIXNUM_BYTES);
        b = ys->retrieve_into(arr2, FIXNUM_BYTES, i);
        ASSERT_EQ(b, FIXNUM_BYTES);

        for (unsigned j = 0; j < arrlen; ++j) {
            unsigned int t = arr1[j];
            t = (t * t) % 23;
            arr1[j] = t;
        }

        EXPECT_TRUE(arrays_are_equal(arr1, arrlen, arr2, arrlen)
                    << " at index i = " << i);
    }

    delete mul;
    delete res;
    delete xs;
    delete ys;
}

typedef vector<byte_array> tcase_args;

// FIXME: This all sucks.

template< typename T >
const T *cast_to(const sexp *s) {
    const T *x = dynamic_cast<const T *>(s);
    die_if( ! x, "Failed to cast sexp");
    return x;
};

vector<tcase_args>
set_args_modexp(const char *fname) {
    ifstream file(fname);
    die_if ( ! file.good(), "Couldn't open file.");
    const sexp *s = sexp::create(file);
    const list *xs;

    int n = -1;
    vector<tcase_args> result;
    xs = cast_to<list>(s);
    for (const sexp *tc : xs->sexps) {
        tcase_args args;
        xs = cast_to<list>(tc);
        for (const sexp *el : xs->sexps) {
            const atom *a = cast_to<atom>(el);
            args.push_back(a->data);
        }
        if (n < 0)
            n = args.size();
        die_if((int)args.size() != n, "Inconsistent argument lengths");
        result.push_back(args);
    }
    delete s;
    return result;
}

template< typename Iter, typename fixnum_impl >
void
set_fixnum_array(fixnum_array<fixnum_impl> *&xs, Iter begin, Iter end)
{
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    int n = xs->length();
    int cnt = 0;
    for (Iter i = begin; i < end; ++i, ++cnt) {
        die_if(cnt == n, "got too many elements from file");
        int r = xs->set(cnt, i->data(), i->size());
        die_if((unsigned)r != std::min(i->size(), FIXNUM_BYTES), "fixnum overflow");
    }
}

TYPED_TEST(TypedPrimitives, modexp) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;

    fixnum_array *res, *xs;
    vector<tcase_args> tcases = set_args_modexp("tests/modexp");

    // Each tcase is [n, e, *xs, *res], so len(xs) = len(res) = (size - 2)/2
    int arglen = tcases[0].size();
    die_if(arglen & 1, "args length should be even");
    int n = (arglen - 2) / 2;
    res = fixnum_array::create(n);
    xs = fixnum_array::create(n);

    int nskipped = 0;
    int cnt = 0;
    for (const tcase_args &args : tcases) {
        const byte_array &mod = args[0], &exp = args[1];

        ++cnt;
        if (cnt < 151) continue;

        if (mod.size() > FIXNUM_BYTES) {
            ++nskipped;
            continue;
        }
        auto first_x = args.begin() + 2;
        set_fixnum_array(xs, first_x, first_x + n);

        auto fn = new modexp<fixnum_impl>(
            mod.data(), mod.size(), exp.data(), exp.size());
        fixnum_array::map(fn, res, xs);
        delete fn;
        auto first_res = first_x + n;
        check_result(res, first_res, first_res + n);
    }
    int ntests = tcases.size();
    cerr << "Skipped " << nskipped << " / " << ntests
         << " (" << setprecision(3) << nskipped * 100.0 / ntests << "%) "
         << "tests to avoid truncation." << endl;

    delete res;
    delete xs;
}

int main(int argc, char *argv[])
{
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
