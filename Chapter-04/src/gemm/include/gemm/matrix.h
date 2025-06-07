//
// Created by sammorley on 03/01/25.
//

#ifndef CT_MATRIX_H
#define CT_MATRIX_H

#include <algorithm>
#include <type_traits>
#include <memory>

#include "matrix_view.h"

namespace ct {

/**
 * @class Matrix
 * @brief A class that represents a matrix with automatic memory management and specified alignment.
 *
 * This class provides a representation of a matrix with customizable row and column counts.
 * Memory allocation and deallocation are handled internally with specified alignment.
 * It inherits functionality from the `MatrixView` class to perform matrix operations.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template<typename T>
class Matrix : public MatrixView<T> {
    static constexpr std::align_val_t alignment{64};

    [[nodiscard]]
    static T *allocate(size_t n_rows, size_t n_cols) {
        return static_cast<T *>(::operator new(n_rows * n_cols * sizeof(T), alignment));
    }

    static void deallocate(T *data, size_t n_rows, size_t n_cols) {
        ::operator delete(data, n_rows * n_cols * sizeof(T), alignment);
    }

public:
    explicit Matrix(ptrdiff_t rows, ptrdiff_t cols, MatrixOrder order = MatrixOrder::RowMajor)
        : MatrixView<T>(allocate(rows, cols), rows, cols, order) {
        std::uninitialized_default_construct_n(this->data(), this->size());
    }

    ~Matrix() {
        deallocate(this->data(), this->rows(), this->cols());
        this->reset();
    }

    constexpr operator MatrixView<std::add_const_t<T> >() const noexcept {
        return MatrixView<std::add_const_t<T> >(this->data(), this->rows(), this->cols(), this->order());
    }
};

} // namespace ct

#endif //CT_MATRIX_H
