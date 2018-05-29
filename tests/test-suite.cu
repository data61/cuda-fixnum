#include <gtest/gtest.h>
#include "array/fixnum_array.h"
#include "fixnum/default.cu"
#include "functions/square.cu"

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
