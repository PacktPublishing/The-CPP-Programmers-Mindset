#include "find_root.hpp"


#include <cmath>

#include <gtest/gtest.h>


TEST(FindRoot, LinearFunction) {
    auto fn = [](double x) { return x; };

    auto root = ct::find_root_bisect(fn, 2.0, -3.0, 1e-6);

    EXPECT_NEAR(root, 0.0, 1e-6);
    EXPECT_NEAR(fn(root), 0.0, 1e-6);
}


TEST(FindRoot, Sin) {
    auto fn = [](double x) { return std::sin(x); };
    auto root = ct::find_root_bisect(fn, .7, -0.5, 1e-6);

    EXPECT_NEAR(root, 0.0, 1e-6);
    EXPECT_NEAR(fn(root), 0.0, 1e-6);
}