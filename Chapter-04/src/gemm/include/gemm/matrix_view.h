//
// Created by sammorley on 03/01/25.
//

#ifndef CT_MATRIX_VIEW_H
#define CT_MATRIX_VIEW_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>

namespace ct {

using std::size_t;
using std::ptrdiff_t;


/**
 * @enum MatrixOrder
 * @brief Enumeration representing the memory storage order of a matrix.
 *
 * The `MatrixOrder` enum defines constants for specifying the layout of
 * matrix data in memory. Two options are provided:
 * - `RowMajor`: Data is stored row by row.
 * - `ColMajor`: Data is stored column by column.
 */
enum class MatrixOrder {
    RowMajor,
    ColMajor
};


/**
 * @class MatrixView
 * @brief A lightweight non-owning view for representing and interacting with a matrix in memory.
 *
 * The `MatrixView` class provides functionality to access and modify data stored in a
 * matrix-like structure without taking ownership of the underlying memory.
 * It supports both row-major and column-major order.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template<typename T>
class MatrixView {
    T *data_ = nullptr;
    ptrdiff_t rows_ = 0;
    ptrdiff_t cols_ = 0;
    MatrixOrder order_ = MatrixOrder::RowMajor;

public:
    static constexpr MatrixOrder RowMajor = MatrixOrder::RowMajor;
    static constexpr MatrixOrder ColMajor = MatrixOrder::ColMajor;

    MatrixView(T *data, ptrdiff_t rows, ptrdiff_t cols, MatrixOrder order = RowMajor)
        : data_(data), rows_(rows), cols_(cols), order_(order) {
    }


    [[nodiscard]] constexpr bool is_row_major() const noexcept { return order_ == RowMajor; }
    [[nodiscard]] constexpr bool is_col_major() const noexcept { return order_ == ColMajor; }

    [[nodiscard]] constexpr ptrdiff_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr ptrdiff_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr MatrixOrder order() const noexcept { return order_; }
    [[nodiscard]] constexpr ptrdiff_t stride() const noexcept { return is_row_major() ? cols_ : rows_; }
    [[nodiscard]] constexpr T *data() const noexcept { return data_; }
    [[nodiscard]] constexpr size_t size() const noexcept { return static_cast<size_t>(rows_ * cols_); }

    void reset(T *new_ptr = nullptr) noexcept {
        data_ = new_ptr;
        if (new_ptr == nullptr) {
            rows_ = 0;
            cols_ = 0;
            order_ = MatrixOrder::RowMajor;
        }
    }

    std::span<T> to_span() noexcept {
        return {data(), size()};
    }

    std::span<const T> to_span() const noexcept {
        return {data(), size()};
    }

    void swap(MatrixView &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(order_, other.order_);
    }

    T &at(ptrdiff_t i, ptrdiff_t j) noexcept {
        return is_row_major() ? data_[i * stride() + j] : data_[j * stride() + i];
    }

    const T &at(ptrdiff_t i, ptrdiff_t j) const noexcept {
        return is_row_major() ? data_[i * stride() + j] : data_[j * stride() + i];
    }
};

template<typename T>
bool operator==(const MatrixView<T> &lhs, const MatrixView<T> &rhs) noexcept {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return false;
    }

    if (lhs.is_row_major() == rhs.is_row_major()) {
        return std::ranges::equal(lhs.to_span(), rhs.to_span());
    }

    for (ptrdiff_t i = 0; i < lhs.rows(); ++i) {
        for (ptrdiff_t j = 0; j < lhs.cols(); ++j) {
            if (lhs.at(i, j) != rhs.at(i, j)) {
                return false;
            }
        }
    }

    return true;
}

template<typename T>
bool operator!=(const MatrixView<T> &lhs, const MatrixView<T> &rhs) noexcept {
    return !(lhs == rhs);
}

namespace detail {
/*
 * We want the abs functions from the standard library to participate in the
 * argument dependent lookup, but we don't want to explicitly import only these
 * functions, since this might preclude ADL from finding more appropriate implementations.
 * So we commit the cardinal sin of `using namespace std` in library
 * code, wrapped suitably in an internal namespace so-as not to pollute the main
 * or global namespaces. I saw this trick used in one of the Boost libraries, but I forget
 * which one.
 */
using namespace std;

struct AbsDifferenceHelper {
    template<typename T>
    auto operator()(const T &a, const T &b) const noexcept -> decltype(abs(a - b)) {
        return abs(a - b);
    }
};
}


/**
 * @brief Calculates the maximum absolute difference between two matrices.
 *
 * This function computes the maximum absolute difference element-wise
 * between two matrices passed as `MatrixView` objects. If the dimensions
 * of the matrices do not match, the function returns positive infinity.
 * Otherwise, it iterates over each element of the matrices, calculates
 * the absolute difference, and tracks the maximum difference encountered.
 *
 * @tparam T The type of elements stored in the matrices.
 * @param lhs The first matrix view for comparison.
 * @param rhs The second matrix view for comparison.
 * @return The maximum absolute difference (of type T) between the matrices.
 * Returns positive infinity if the dimensions of the matrices are not the same.
 *
 * @note This function does not alter the input matrices and assumes they
 * are accessible and properly initialized for indexing.
 * @note The function is marked `noexcept` and guarantees exception safety.
 */
template<typename T>
T matrix_max_abs_difference(const MatrixView<T> &lhs, const MatrixView<T> &rhs) noexcept {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return std::numeric_limits<T>::infinity();
    }

    detail::AbsDifferenceHelper abs_diff;

    auto max_diff = -std::numeric_limits<T>::infinity();
    for (ptrdiff_t i = 0; i < lhs.rows(); ++i) {
        for (ptrdiff_t j = 0; j < lhs.cols(); ++j) {
            auto diff = abs_diff(lhs.at(i, j), rhs.at(i, j));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    return max_diff;
}

}//namespace ct

#endif //CT_MATRIX_VIEW_H
