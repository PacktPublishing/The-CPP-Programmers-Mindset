#include "function_with_noise.hpp"


#include <gtest/gtest.h>


TEST(FunctionWithNoise, ValueIsCloseToDeterministic) {
    ct::FunctionWithNoise fn(12345, 0.0, 1.0);

    auto fn0 = fn(0.0);
    EXPECT_NE(fn0, 1.0);

    // About 99.7% of noise values should be between -3.0 and 3.0
    EXPECT_NEAR(fn0, 1.0, 3.0);

    auto fn1 = fn(1.0);
    EXPECT_NE(fn1, 3.0);

    EXPECT_NEAR(fn1, 3.0, 3.0);
}