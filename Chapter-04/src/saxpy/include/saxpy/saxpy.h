//
// Created by sammorley on 02/01/25.
//

#ifndef CT_SAXPY_H
#define CT_SAXPY_H

#include <span>

namespace ct {


// private function clones
void saxpy_default(float a, std::span<const float> x, std::span<float> y);
void saxpy_sse42(float a, std::span<const float> x, std::span<float> y);
void saxpy_avx2(float a, std::span<const float> x, std::span<float> y);
void saxpy_avx512f(float a, std::span<const float> x, std::span<float> y);

// dispatcher
void saxpy(float a, std::span<const float> x, std::span<float> y);


// Implemented by hand using compiler intrinsics
void saxpy_hand(float a, std::span<const float> x, std::span<float> y);

}


#endif //CT_SAXPY_H
