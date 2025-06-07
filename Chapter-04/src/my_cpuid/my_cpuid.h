//
// Created by sammorley on 03/01/25.
//

#ifndef CT_MY_CPUID_H
#define CT_MY_CPUID_H

#if defined(__X86__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

#if defined(__MSVC__)
#  include <intrin.h>
#  define my_cpuid(i, j, regs) \
        __cpuidx(reinterpret_cast<int*>(regs), i, j)
#elif defined(__GNUC__) || defined(__clang__)
#  include <cpuid.h>
#  define my_cpuid(i, j, regs) __cpuid_count(i, j, \
        regs[0], regs[1], regs[2], regs[3])
#endif

#else

#  define my_cpuid(i, j, regs) static_assert(false, "Not implemented")

#endif

#endif //CT_MY_CPUID_H
