#include "powers.hpp"

#include <gtest/gtest.h>

TEST(Powers, TestIterativeIntPowsTwo) {
    EXPECT_EQ(1, ct::iterative::power(2, 0));
    EXPECT_EQ(2, ct::iterative::power(2, 1));
    EXPECT_EQ(4, ct::iterative::power(2, 2));
    EXPECT_EQ(8, ct::iterative::power(2, 3));
    EXPECT_EQ(16, ct::iterative::power(2, 4));
    EXPECT_EQ(32, ct::iterative::power(2, 5));
}

TEST(Powers, TestRecursiveIntPowsTwo) {
    EXPECT_EQ(1, ct::recursive::power(2, 0));
    EXPECT_EQ(2, ct::recursive::power(2, 1));
    EXPECT_EQ(4, ct::recursive::power(2, 2));
    EXPECT_EQ(8, ct::recursive::power(2, 3));
    EXPECT_EQ(16, ct::recursive::power(2, 4));
    EXPECT_EQ(32, ct::recursive::power(2, 5));
}

TEST(Powers, TestIterativeFloatPowsTwo) {
    EXPECT_EQ(1.0f, ct::iterative::power(2.0f, 0));
    EXPECT_EQ(2.0f, ct::iterative::power(2.0f, 1));
    EXPECT_EQ(4.0f, ct::iterative::power(2.0f, 2));
    EXPECT_EQ(8.0f, ct::iterative::power(2.0f, 3));
    EXPECT_EQ(16.0f, ct::iterative::power(2.0f, 4));
    EXPECT_EQ(32.0f, ct::iterative::power(2.0f, 5));
}

TEST(Powers, TestRecursiveFloatPowsTwo) {
    EXPECT_EQ(1.0f, ct::recursive::power(2.0f, 0));
    EXPECT_EQ(2.0f, ct::recursive::power(2.0f, 1));
    EXPECT_EQ(4.0f, ct::recursive::power(2.0f, 2));
    EXPECT_EQ(8.0f, ct::recursive::power(2.0f, 3));
    EXPECT_EQ(16.0f, ct::recursive::power(2.0f, 4));
    EXPECT_EQ(32.0f, ct::recursive::power(2.0f, 5));
}