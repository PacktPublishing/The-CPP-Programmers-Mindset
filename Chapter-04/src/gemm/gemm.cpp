#include "gemm.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "matrix_view.h"
#include "matrix.h"


using namespace ct;


static void check_dimensions(const MatrixView<const double> &a, const MatrixView<const double> &b,
                             const MatrixView<double> &c) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Invalid dimensions: inner dimensions must match");
    }

    if (c.rows() != a.rows() || c.cols() != b.cols()) {
        throw std::invalid_argument("Invalid dimensions: output dimensions must match");
    }
}


void ct::dgemm_basic(MatrixView<const double> A, MatrixView<const double> B, MatrixView<double> C, double alpha,
                     double beta) {
    check_dimensions(A, B, C);

    assert(A.is_row_major());
    assert(B.is_row_major());
    assert(C.is_row_major());

    for (ptrdiff_t i = 0; i < A.rows(); ++i) {
        for (ptrdiff_t j = 0; j < A.cols(); ++j) {
            double tmp = 0;

            for (ptrdiff_t k = 0; k < B.cols(); ++k) {
                auto &a_val = A.data()[i * A.cols() + k];
                auto &b_val = B.data()[k * B.cols() + j];
                tmp += a_val * b_val;
            }

            auto &c_elt = C.data()[i * C.cols() + j];
            c_elt = beta * c_elt + alpha * tmp;
        }
    }
}

void ct::dgemm_blocked(MatrixView<const double> A, MatrixView<const double> B, MatrixView<double> C, double alpha,
                       double beta,
                       ptrdiff_t block_size) {
    check_dimensions(A, B, C);

    // For simplicity, only consider block sizes that are powers of 2
    assert((block_size & (block_size - 1)) == 0);

    assert(A.is_row_major());
    assert(B.is_row_major());
    assert(C.is_row_major());

    Matrix<double> tile(block_size, block_size);

    for (ptrdiff_t i_block = 0; i_block < C.rows(); i_block += block_size) {
        auto i_bound = std::min(C.rows() - i_block, block_size);

        for (ptrdiff_t j_block = 0; j_block < C.cols(); j_block += block_size) {
            auto j_bound = std::min(C.cols() - j_block, block_size);
            std::fill_n(tile.data(), tile.size(), 0.0);

            for (ptrdiff_t k_block = 0; k_block < B.cols(); k_block += block_size) {
                auto k_bound = std::min(B.cols() - k_block, block_size);

                for (ptrdiff_t i = 0; i < i_bound; ++i) {
                    auto i_index = i_block + i;
                    for (ptrdiff_t j = 0; j < j_bound; ++j) {
                        auto j_index = j_block + j;
                        for (ptrdiff_t k = 0; k < k_bound; ++k) {
                            auto k_index = k_block + k;
                            auto& a_val = A.data()[i_index * A.cols() + k_index];
                            auto& b_val = B.data()[k_index * B.cols() + j_index];
                            auto& t_val = tile.data()[i * block_size + j];
                            t_val += a_val*b_val;
                        }
                    }
                }
            }

            for (ptrdiff_t i = 0; i < i_bound; ++i) {
                auto i_index = i_block + i;
                for (ptrdiff_t j = 0; j < j_bound; ++j) {
                    auto j_index = j_block + j;
                    auto &c_elt = C.data()[i_index*C.cols() + j_index];
                    auto& t_val = tile.data()[i * block_size + j];
                    c_elt = beta * c_elt + alpha * t_val;
                }
            }
        }
    }
}
