//
// Created by sammorley on 31/12/24.
//


#include <gtest/gtest.h>

#include "gemm.h"
#include "matrix.h"

using namespace ct;



class GemmTest : public ::testing::Test {

protected:
    static constexpr ptrdiff_t n_rows = 250;
    static constexpr ptrdiff_t n_cols = 250;
    Matrix<double> a;
    Matrix<double> b;
    Matrix<double> c;

    GemmTest() : a(n_rows, n_cols), b(n_cols, n_rows), c(n_rows, n_cols) {
    }

    void SetUp() override {
        std::fill_n(a.data(), a.size(), 1.0);
        std::fill_n(b.data(), b.size(), 1.0);
    }


};




TEST_F(GemmTest, TestBlockedAgainstBasic) {
    Matrix<double> expected(n_rows, n_cols);
    dgemm_basic(a, b, expected, 2.0, 0.0);

    dgemm_blocked(a, b, c, 2.0, 0.0, 64);

    EXPECT_LE(matrix_max_abs_difference(c, expected), 1e-12);
}