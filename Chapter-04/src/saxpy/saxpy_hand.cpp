//
// Created by sammorley on 02/01/25.
//




#include "saxpy.h"

#include <span>

#include <immintrin.h>

void ct::saxpy_hand(float a, std::span<const float> x, std::span<float> y) {
    constexpr auto vec_size = sizeof(__m256) / sizeof(float);
    const auto ymm0 = _mm256_set1_ps(a);

    size_t size = y.size();
    size_t pos = 0;
    const auto* x_data = x.data();
    auto* y_data = y.data();

    for (; pos + vec_size <= size; pos += vec_size) {
        auto ymm1 = _mm256_loadu_ps(x_data);
        auto ymm3 = _mm256_loadu_ps(y_data);
        ymm1 = _mm256_mul_ps(ymm0, ymm1);
        ymm3 = _mm256_add_ps(ymm3, ymm1);
        _mm256_storeu_ps(y_data, ymm3);
        x_data += vec_size;
        y_data += vec_size;
    }

    // Clear up the < vec_size terms that remain
    const size_t remainder = size - pos;
    for (size_t i=0; i < remainder; ++i) {
        y_data[i] += a * x_data[i];
    }
}
