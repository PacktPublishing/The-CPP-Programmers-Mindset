#ifndef CT_GEMM_H
#define CT_GEMM_H


#include <cstddef>


#include "matrix_view.h"


namespace ct {
/**
 * @brief Performs a basic double-precision general matrix multiplication (DGEMM).
 *
 * This function computes the matrix product of two matrices `a` and `b`, scales
 * the result by a scalar `alpha`, scales the existing values in matrix `c` by
 * a scalar `beta`, and accumulates the results in matrix `c`.
 *
 * The operation is expressed mathematically as:
 *   c = beta * c + alpha * (a * b)
 *
 * Ths implementation uses absolutely no optimizations and simply writes out the product
 * as one would mathematically. This is the slowest, but guaranteed correct, implementation
 * of the DGEMM function.
 *
 * @param a The left-hand side matrix in the multiplication operation. Represents matrix `a`.
 * @param b The right-hand side matrix in the multiplication operation. Represents matrix `b`.
 * @param c The result matrix where the computed values are stored. Represents matrix `c`.
 * @param alpha A scalar value that scales the product of `a` and `b`.
 * @param beta A scalar value that scales the original values of matrix `c` before accumulation.
 */
void dgemm_basic(MatrixView<const double> a, MatrixView<const double> b, MatrixView<double> c, double alpha,
                 double beta);


void dgemm_blocked(MatrixView<const double> a, MatrixView<const double> b, MatrixView<double> c, double alpha,
                   double beta, ptrdiff_t block_size);
}


#endif //CT_GEMM_H
