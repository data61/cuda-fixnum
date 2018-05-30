#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <memory>
#include <fstream>
#include "sexp.h"

// FIXME FIXME!
#include "sexp.cc"

#include "array/fixnum_array.h"
#include "fixnum/default.cu"
#include "functions/square.cu"

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

vector<binop_args> fname_to_args(const char *fname) {
    ifstream file(fname);
    die_if ( ! file.good(), "Couldn't open file.");
    auto s = unique_ptr<const sexp>(sexp::create(file));
    extract_all_args v;
    s->accept(v);
    return v.result();
}

::testing::AssertionResult arrays_are_equal(
        const uint8_t *expected, size_t expected_len,
        const uint8_t *actual, size_t actual_len) {
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

template< typename fixnum_impl >
struct add_cy : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::add_cy(r, a, b);
    }
};

template< typename fixnum_impl >
struct mul_lo : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_lo(r, a, b);
    }
};

class Primitives : public ::testing::TestWithParam<int> { };

TEST_P(Primitives, precomputed) {
    static constexpr size_t FIXNUM_BYTES = 256;
    typedef default_fixnum_impl<FIXNUM_BYTES, uint64_t> fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    int algo = GetParam();
    const char *fname = algo == 0 ? "tests/add_cy" : "tests/mul_wide";
    vector<binop_args> tcases = fname_to_args(fname);
    int n = (int) tcases.size();
    fixnum_array *xs, *ys, *res;

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

    if (algo == 0) {
        auto fn_add = new add_cy<fixnum_impl>();
        fixnum_array::map(fn_add, res, xs, ys);
        delete fn_add;
    } else {
        auto fn_mul = new mul_lo<fixnum_impl>();
        fixnum_array::map(fn_mul, res, xs, ys);
        delete fn_mul;
    }

    EXPECT_EQ(res->length(), n);
    for (int i = 0; i < n; ++i) {
        const byte_array &expected = tcases[i][2];
        uint8_t arr[FIXNUM_BYTES];
        memset(arr, 0, FIXNUM_BYTES);
        size_t r = res->retrieve_into(arr, FIXNUM_BYTES, i);
        ASSERT_EQ(r, FIXNUM_BYTES);

        size_t expected_len = std::min(expected.size(), FIXNUM_BYTES);
        EXPECT_TRUE(arrays_are_equal(expected.data(), expected_len, arr, FIXNUM_BYTES));
    }

    delete res;
    delete xs;
    delete ys;
}

INSTANTIATE_TEST_CASE_P(All_Primitives, Primitives, ::testing::Values(0, 1));


TEST(basic_arithmetic, square) {
    typedef default_fixnum_impl<16> fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    int n = 1;

    auto res = fixnum_array::create(n);
    auto arr1 = fixnum_array::create(n, 0UL);
    auto arr2 = fixnum_array::create(n, ~0UL);
    auto arr3 = fixnum_array::create(n, 245);
    auto sqr = new square<fixnum_impl>();

    fixnum_array::map(sqr, res, arr1);
    // check that res is all zeros
    constexpr int fn_bytes = fixnum_impl::FIXNUM_BYTES;
    constexpr int bufsz = 4096;
    uint8_t arr[bufsz];
    int nelts;
    for (auto &a : arr) a = 0;
    res->retrieve_all(arr, bufsz, &nelts);
    EXPECT_EQ(nelts - res->length(), 0);
    // NB: Only checking up to fn_bytes because n = 1;
    for (int i = 0; i < fn_bytes; ++i)
        EXPECT_EQ(arr[i], 0);

    fixnum_array::map(sqr, res, arr2);
    res->retrieve_all(arr, bufsz, &nelts);
    EXPECT_EQ(nelts - res->length(), 0);
    uint8_t res2[] = { 1, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255, 255, 255, 255, 255 };
    // NB: Only checking up to fn_bytes because n = 1;
    for (int i = 0; i < fn_bytes; ++i)
        EXPECT_EQ(arr[i], res2[i]);

    fixnum_array::map(sqr, res, arr3);
    res->retrieve_all(arr, bufsz, &nelts);
    EXPECT_EQ(nelts - res->length(), 0);
    EXPECT_EQ(arr[0], 121);
    EXPECT_EQ(arr[1], 234);
    // NB: Only checking up to fn_bytes because n = 1;
    for (int i = 2; i < fn_bytes; ++i)
        EXPECT_EQ(arr[i], 0);

    delete sqr;
    delete res;
    delete arr1;
    delete arr2;
    delete arr3;
}

int main(int argc, char *argv[])
{
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
