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

#include "array/fixnum_array.h"
#include "fixnum/fixnum.cu"
#include "functions/monty_mul.cu"
#include "functions/modexp.cu"
#include "functions/paillier_encrypt.cu"
#include "functions/paillier_decrypt.cu"

using namespace std;

typedef vector<uint8_t> byte_array;

void die_if(bool p, const string &msg) {
    if (p) {
        cerr << "Error: " << msg << endl;
        abort();
    }
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
        const string &fname,
        int nargs) {
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

    // ninvecs = number of input combinations
    uint32_t ninvecs = 1;
    for (int i = 1; i < nargs; ++i)
        ninvecs *= vec_len;
    res.reserve(noutvecs * ninvecs);
    for (uint32_t i = 0; i < ninvecs; ++i) {
        for (uint32_t j = 0; j < noutvecs; ++j) {
            read_into(file, buf, nbytes);
            res.emplace_back(buf, buf + nbytes);
        }
    }

    delete[] buf;
}

template< typename fixnum_impl, typename tcase_iter >
void check_result(
    tcase_iter &tcase, uint32_t vec_len,
    initializer_list<const fixnum_array<fixnum_impl> *> args,
    int skip = 1,
    uint32_t nvecs = 1)
{
    static constexpr int fixnum_bytes = fixnum_impl::FIXNUM_BYTES;
    size_t total_vec_len = vec_len * nvecs;
    size_t nbytes = fixnum_bytes * total_vec_len;
    // TODO: The fixnum_arrays are in managed memory; there isn't really any
    // point to copying them into buf.
    byte_array buf(nbytes);

    for (auto arg : args) {
        auto buf_iter = buf.begin();
        for (uint32_t i = 0; i < nvecs; ++i) {
            std::copy(tcase->begin(), tcase->end(), buf_iter);
            buf_iter += fixnum_bytes*vec_len;
            tcase += skip;
        }
        EXPECT_TRUE(arrays_are_equal(buf.data(), nbytes, arg->get_ptr(), nbytes));
    }
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

    read_tcases(tcases, xs, "tests/add_cy", 2);
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);
    cys = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<add_cy>(res, cys, xs, ys);
        check_result(tcase, vec_len, {res, cys});
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

    read_tcases(tcases, xs, "tests/sub_br", 2);
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);
    brs = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<sub_br>(res, brs, xs, ys);
        check_result(tcase, vec_len, {res, brs});
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

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_lo>(res, xs, ys);
        check_result(tcase, vec_len, {res}, 2);
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

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    res = fixnum_array::create(vec_len);

    auto tcase = tcases.begin() + 1;
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_hi>(res, xs, ys);
        check_result(tcase, vec_len, {res}, 2);
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

    read_tcases(tcases, xs, "tests/mul_wide", 2);
    int vec_len = xs->length();
    his = fixnum_array::create(vec_len);
    los = fixnum_array::create(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *ys = xs->rotate(i);
        fixnum_array::template map<mul_wide>(his, los, xs, ys);
        check_result(tcase, vec_len, {los, his});
        delete ys;
    }
    delete his;
    delete los;
    delete xs;
}


template< typename fixnum_impl >
struct my_modexp {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &z, fixnum x, fixnum e, fixnum m) {
        modexp<fixnum_impl> me(m, e);
        me(z, x);
    };
};

TYPED_TEST(TypedPrimitives, modexp) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;
    typedef fixnum_array<fixnum_impl> fixnum_array;

    fixnum_array *res, *input, *xs, *zs;
    vector<byte_array> tcases;

    read_tcases(tcases, input, "tests/modexp", 3);
    int vec_len = input->length();
    int vec_len_sqr = vec_len * vec_len;

    res = fixnum_array::create(vec_len_sqr);
    xs = input->repeat(vec_len);
    zs = input->rotations(vec_len);

    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        fixnum_array *tmp = input->rotate(i);
        fixnum_array *ys = tmp->repeat(vec_len);
        fixnum_array::template map<my_modexp>(res, xs, ys, zs);
        check_result(tcase, vec_len, {res}, 1, vec_len);
        delete ys;
        delete tmp;
    }
    delete res;
    delete input;
    delete xs;
    delete zs;
}

template< typename fixnum_impl >
struct pencrypt {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &z, fixnum p, fixnum q, fixnum r, fixnum m) {
        fixnum n;
        fixnum_impl::mul_lo(n, p, q);
        paillier_encrypt<fixnum_impl> enc(n);
        enc(z, m, r);
    };
};

template< typename fixnum_impl >
struct pdecrypt {
    typedef typename fixnum_impl::fixnum fixnum;

    __device__ void operator()(fixnum &z, fixnum ct, fixnum p, fixnum q, fixnum r, fixnum m) {
        if (fixnum_impl::cmp(p, q) == 0
              || fixnum_impl::cmp(r, p) == 0
              || fixnum_impl::cmp(r, q) == 0) {
            z = 0;
            return;
        }
        paillier_decrypt<fixnum_impl> dec(p, q);
        dec(z, 0, ct);
        fixnum n;
        fixnum_impl::mul_lo(n, p, q);
        quorem_preinv<fixnum_impl> qr(n);
        qr(m, 0, m);

        z = (z != m);
    };
};

TYPED_TEST(TypedPrimitives, paillier) {
    typedef typename TestFixture::fixnum_impl fixnum_impl;

    typedef fixnum_impl ctxt;
    // TODO: FIXNUM_BYTES/2 only works when FIXNUM_BYTES > 4
    //typedef default_fixnum_impl<ctxt::FIXNUM_BYTES/2, typename ctxt::word_tp> ptxt;
    typedef fixnum_impl ptxt;

    typedef fixnum_array<ctxt> ctxt_array;
    typedef fixnum_array<ptxt> ptxt_array;

    ctxt_array *ct, *pt, *p;
    vector<byte_array> tcases;
    read_tcases(tcases, p, "tests/paillier_encrypt", 4);

    int vec_len = p->length();
    ct = ctxt_array::create(vec_len);
    pt = ctxt_array::create(vec_len);

    // TODO: Parallelise these tests similar to modexp above.
    ctxt_array *zeros = ctxt_array::create(vec_len, 0);
    auto tcase = tcases.begin();
    for (int i = 0; i < vec_len; ++i) {
        ctxt_array *q = p->rotate(i);
        for (int j = 0; j < vec_len; ++j) {
            ctxt_array *r = p->rotate(j);
            for (int k = 0; k < vec_len; ++k) {
                ctxt_array *m = p->rotate(k);

                ctxt_array::template map<pencrypt>(ct, p, q, r, m);
                check_result(tcase, vec_len, {ct});

                ptxt_array::template map<pdecrypt>(pt, ct, p, q, r, m);

                size_t nbytes = vec_len * ctxt::FIXNUM_BYTES;
                const uint8_t *zptr = reinterpret_cast<const uint8_t *>(zeros->get_ptr());
                const uint8_t *ptptr = reinterpret_cast<const uint8_t *>(pt->get_ptr());
                EXPECT_TRUE(arrays_are_equal(zptr, nbytes, ptptr, nbytes));

                delete m;
            }
            delete r;
        }
        delete q;
    }

    delete p;
    delete ct;
    delete zeros;
}

int main(int argc, char *argv[])
{
    int r;

    testing::InitGoogleTest(&argc, argv);
    r = RUN_ALL_TESTS();
    return r;
}
