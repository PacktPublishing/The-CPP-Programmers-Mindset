//
// Created by sammorley on 02/01/25.
//

#ifndef CT_GET_BEST_INSTRUCTIONS_H
#define CT_GET_BEST_INSTRUCTIONS_H

#include "my_cpuid.h"

namespace ct {

enum class SIMDInstructions {
    Default,
    SSE42,
    AVX2,
    AVX512F
};

inline
SIMDInstructions get_best_instructions() {
    unsigned cpuid_registers[4];
    my_cpuid(0, 0, cpuid_registers);  // Query highest CPUID function supported
    if (cpuid_registers[0] >= 7) {  // Check for extended features
        my_cpuid(7, 0, cpuid_registers);
        if (cpuid_registers[1] & (1 << 16)) {  // AVX512F support (bit 16 of EBX)
            return SIMDInstructions::AVX512F;
        }
        if (cpuid_registers[1] & (1 << 5)) {  // AVX2 support (bit 5 of EBX)
            return SIMDInstructions::AVX2;
        }
    }
    my_cpuid(1, 0, cpuid_registers);  // Basic features (EAX=1)
    if (cpuid_registers[2] & (1 << 20)) {  // SSE4.2 support (bit 20 of ECX)
        return SIMDInstructions::SSE42;
    }
    return SIMDInstructions::Default;
}
} // namespace ct


#endif //CT_GET_BEST_INSTRUCTIONS_H
