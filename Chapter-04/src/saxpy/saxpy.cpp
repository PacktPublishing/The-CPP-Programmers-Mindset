//
// Created by sammorley on 02/01/25.
//

#include "saxpy.h"

#include <cassert>
#include <span>

#include "get_best_instructions.h"


void ct::saxpy(float a, std::span<const float> x, std::span<float> y) {
    // the dispatcher is also the perfect place for error checking
    assert(x.size() == y.size());

    switch (get_best_instructions()) {
        case SIMDInstructions::AVX512F:
            saxpy_avx512f(a, x, y); break;
        case SIMDInstructions::AVX2:
            saxpy_avx2(a, x, y); break;
        case SIMDInstructions::SSE42:
            saxpy_sse42(a, x, y); break;
        default:
            saxpy_default(a, x, y);
    }

}
