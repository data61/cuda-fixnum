#include <gtest/gtest.h>
#include <tuple>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
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

template< typename fixnum_impl >
struct add_cy : public managed {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::add_cy(r, a, b);
    }
};


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

template< typename fixnum_impl >
void check_result(
    const vector<binop_args> &tcases,
    const fixnum_array<fixnum_impl> *res)
{
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    int n = (int) tcases.size();
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
}

template< typename fixnum_impl >
void check_result2(
    const vector<binop_args> &tcases,
    const fixnum_array<fixnum_impl> *rs,
    const fixnum_array<fixnum_impl> *ss)
{
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;
    int n = (int) tcases.size();
    EXPECT_EQ(rs->length(), n);
    EXPECT_EQ(ss->length(), n);
    for (int i = 0; i < n; ++i) {
        static constexpr size_t arrlen = FIXNUM_BYTES * 2;
        uint8_t arr[arrlen];
        size_t b;

        const byte_array &expected = tcases[i][2];

        memset(arr, 0, arrlen);
        b = rs->retrieve_into(arr, FIXNUM_BYTES, i);
        ASSERT_EQ(b, FIXNUM_BYTES);
        b = ss->retrieve_into(arr + FIXNUM_BYTES, FIXNUM_BYTES, i);
        ASSERT_EQ(b, FIXNUM_BYTES);

        size_t expected_len = std::min(expected.size(), arrlen);
        EXPECT_TRUE(arrays_are_equal(expected.data(), expected_len, arr, arrlen)
                    << " at index i = " << i);
    }
}

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
    // FIXME:
    his = fixnum_array::create(tcases.size());

    auto fn = new mul_wide<fixnum_impl>();
    fixnum_array::map(fn, his, los, xs, ys);
    delete fn;

    // FIXME:
    check_result2(tcases, los, his);
    delete his;
    delete los;
    delete xs;
    delete ys;
}


int main(int argc, char *argv[])
{
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
