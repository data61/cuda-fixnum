#include <gtest/gtest.h>
#include <tuple>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <fstream>
#include <string>
#include <sstream>
#include "sexp.h"

// FIXME FIXME!
#include "sexp.cc"

#include "array/fixnum_array.h"
#include "fixnum/default.cu"
#include "functions/monty_mul.cu"
#include "functions/modexp.cu"

using namespace std;

void die_if(bool p, const string &msg) {
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

void read_into(ifstream &file, uint8_t *buf, size_t nbytes) {
    file.read(reinterpret_cast<char *>(buf), nbytes);
    die_if( ! file.good(), "Read error.");
    die_if(static_cast<size_t>(file.gcount()) != nbytes, "Expected more data.");
}

uint32_t read_int(ifstream &file) {
    uint32_t res;
    file.read(reinterpret_cast<char*>(&res), sizeof(res));
    return res;
}

template<typename fixnum_impl>
void read_tcases(
        vector<byte_array> &res,
        fixnum_array<fixnum_impl> *&xs,
        const string &fname) {
    static constexpr int fixnum_bytes = fixnum_impl::FIXNUM_BYTES;
    ifstream file(fname + "_" + std::to_string(fixnum_bytes));
    die_if( ! file.good(), "Couldn't open file.");

    uint32_t fn_bytes, vec_len, noutvecs;
    fn_bytes = read_int(file);
    vec_len = read_int(file);
    noutvecs = read_int(file);

    stringstream ss;
    ss << "Inconsistent reporting of fixnum bytes. "
       << "Expected " << fixnum_bytes << " got " << fn_bytes << ".";
    die_if(fixnum_bytes != fn_bytes, ss.str());

    size_t nbytes = fixnum_bytes * vec_len;
    uint8_t *buf = new uint8_t[nbytes];

    read_into(file, buf, nbytes);
    xs = fixnum_array<fixnum_impl>::create(buf, nbytes, fixnum_bytes);

    res.reserve(noutvecs * vec_len);
    for (uint32_t i = 0; i < vec_len; ++i) {
        for (uint32_t j = 0; j < noutvecs; ++j) {
            read_into(file, buf, nbytes);
            res.emplace_back(buf, buf + nbytes);
        }
    }

    delete[] buf;
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

template< typename fixnum_impl, typename tcase_iter >
void check_result_new(
    tcase_iter &tcase, uint32_t vec_len,
    initializer_list<const fixnum_array<fixnum_impl> *> args,
    int skip = 1)
{
    static constexpr int fixnum_bytes = fixnum_impl::FIXNUM_BYTES;
    size_t nbytes = fixnum_bytes * vec_len;
    uint8_t *buf = new uint8_t[nbytes];

    for (auto arg : args) {
        int n;
        const uint8_t *expected = tcase->data();
        arg->retrieve_all(buf, nbytes, &n);
        EXPECT_EQ(n, vec_len);
        EXPECT_TRUE(arrays_are_equal(expected, nbytes, buf, nbytes));
        tcase += skip;
    }
    delete[] buf;
}

template< typename fixnum_impl >
struct add_cy {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum &cy, fixnum a, fixnum b) {
        int c = fixnum_impl::add_cy(r, a, b);
        cy = (fixnum_impl::slot_layout::laneIdx() == 0) ? c : 0;
    }
};

TYPED_TEST(TypedPrimitives, add_cy) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *cys, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/add_cy");
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);
    cys = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<add_cy>(res, cys, xs, ys);
        check_result_new(tcase, vec_len, {res, cys});
        delete ys;
    }
    delete res;
    delete cys;
    delete xs;
}


template< typename fixnum_impl >
struct sub_br {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum &br, fixnum a, fixnum b) {
        int bb = fixnum_impl::sub_br(r, a, b);
        br = (fixnum_impl::slot_layout::laneIdx() == 0) ? bb : 0;
    }
};

TYPED_TEST(TypedPrimitives, sub_br) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *brs, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/sub_br");
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);
    brs = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<sub_br>(res, brs, xs, ys);
        check_result_new(tcase, vec_len, {res, brs});
        delete ys;
    }
    delete res;
    delete brs;
    delete xs;
}

template< typename fixnum_impl >
struct mul_lo {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_lo(r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, mul_lo) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide");
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_lo>(res, xs, ys);
        check_result_new(tcase, vec_len, {res}, 2);
        delete ys;
    }
    delete res;
    delete xs;
}

template< typename fixnum_impl >
struct mul_hi {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_hi(r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, mul_hi) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide");
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);

    auto tcase = tcases.begin() + 1;
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_hi>(res, xs, ys);
        check_result_new(tcase, vec_len, {res}, 2);
        delete ys;
    }
    delete res;
    delete xs;
}

template< typename fixnum_impl >
struct mul_wide {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &s, fixnum &r, fixnum a, fixnum b) {
        fixnum_impl::mul_wide(s, r, a, b);
    }
};

TYPED_TEST(TypedPrimitives, mul_wide) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *his, *los, *xs;
    vector<byte_array> tcases;

    read_tcases(tcases, xs, "tests/mul_wide");
    int vec_len = xs->length();
    his = fixnum_array::create(vec_len);
    los = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_wide>(his, los, xs, ys);
        check_result_new(tcase, vec_len, {los, his});
        delete ys;
    }
    delete his;
    delete los;
    delete xs;
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

template< typename fixnum_impl >
struct my_modexp {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &z, fixnum x, fixnum m, fixnum e) {
        modexp<fixnum_impl> me(m, e);
        me(z, x);
    };
};

TYPED_TEST(TypedPrimitives, modexp) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;
    static constexpr size_t FIXNUM_BYTES = fixnum_impl::FIXNUM_BYTES;

    fixnum_array *res, *xs, *exps, *mods;
    vector<tcase_args> tcases = set_args_modexp("tests/modexp");

    // Each tcase is [n, e, *xs, *res], so len(xs) = len(res) = (size - 2)/2
    int arglen = tcases[0].size();
    die_if(arglen & 1, "args length should be even");
    int nxs = (arglen - 2) / 2, n = nxs;
    n = nxs;  // Set to < nxs to limit number of tests run.
    res = fixnum_array::create(n);
    xs = fixnum_array::create(n);
    exps = fixnum_array::create(n);
    mods = fixnum_array::create(n);

    int nskipped = 0;
    for (const tcase_args &args : tcases) {
        const byte_array &mod = args[0], &exp = args[1];

        if (mod.size() > FIXNUM_BYTES || !(mod[0] & 1)
            || exp.size() > FIXNUM_BYTES) {
            ++nskipped;
            continue;
        }

        auto first_x = args.begin() + 2;
        set_fixnum_array(xs, first_x, first_x + n);

        // TODO: exps (resp. mods) should be a fixnum_array of
        // "length" n, where every element is exp (resp. mod); i.e. a
        // "constant" fixnum array.
        for (int i = 0; i < n; ++i) {
            exps->set(i, exp.data(), exp.size());
            mods->set(i, mod.data(), mod.size());
        }

        fixnum_array::template map<my_modexp>(res, xs, mods, exps);

        auto first_res = first_x + nxs;
        check_result(res, first_res, first_res + n);
    }
    int ntests = tcases.size();
    cerr << "Skipped " << nskipped << " / " << ntests
         << " (" << setprecision(3) << nskipped * 100.0 / ntests << "%) "
         << "tests to avoid truncation." << endl;

    delete res;
    delete xs;
    delete exps;
    delete mods;
}

int main(int argc, char *argv[])
{
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
