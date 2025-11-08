#include <gtest/gtest.h>
#include <vector>
#include "sycl_saxy.hpp"

class SyclSaxpyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(SyclSaxpyTest, BasicComputation) {
    const int N = 5;
    float a = 2.0f;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<float> expected = {12.0f, 24.0f, 36.0f, 48.0f, 60.0f};

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SyclSaxpyTest, ZeroScalar) {
    const int N = 4;
    float a = 0.0f;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> expected = {5.0f, 6.0f, 7.0f, 8.0f};

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SyclSaxpyTest, NegativeScalar) {
    const int N = 3;
    float a = -1.5f;
    std::vector<float> x = {2.0f, 4.0f, 6.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f};
    std::vector<float> expected = {7.0f, 14.0f, 21.0f};

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SyclSaxpyTest, SingleElement) {
    const int N = 1;
    float a = 3.0f;
    std::vector<float> x = {7.0f};
    std::vector<float> y = {5.0f};
    std::vector<float> expected = {26.0f};

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    EXPECT_FLOAT_EQ(y[0], expected[0]);
}

TEST_F(SyclSaxpyTest, LargerArray) {
    const int N = 100;
    float a = 0.5f;
    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<float> expected(N);

    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i * 2);
        expected[i] = a * x[i] + y[i];
    }

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(SyclSaxpyTest, ZeroVectors) {
    const int N = 5;
    float a = 2.5f;
    std::vector<float> x(N, 0.0f);
    std::vector<float> y(N, 0.0f);
    std::vector<float> expected(N, 0.0f);

    ct::sycl_saxpy(a, x.data(), y.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(y[i], expected[i]) << "Mismatch at index " << i;
    }
}