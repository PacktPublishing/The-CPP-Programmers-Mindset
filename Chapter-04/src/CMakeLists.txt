if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(X86 TRUE)
else ()
    set(X86 FALSE)
endif ()

###############################################################################
# Multithreading
###############################################################################

# We've separated out the implementations so we can write some benchmarks
add_subdirectory(open_mp_demo)
add_subdirectory(bench_nearest_neighbor)
add_subdirectory(nearest_neighbor_functions)

###############################################################################
# Cache and memory
###############################################################################

if (X86)
    add_subdirectory(cache_information)
endif ()

add_subdirectory(bench_gemm)
add_subdirectory(test_gemm)
add_subdirectory(my_cpuid)
add_subdirectory(gemm)

###############################################################################
# Single instruction multiple data
###############################################################################

if (X86)
    add_subdirectory(bench_saxpy)
    add_subdirectory(saxpy)
endif ()

###############################################################################
# Branch prediction and speculative execution
###############################################################################
