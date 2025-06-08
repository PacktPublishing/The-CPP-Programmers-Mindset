cmake_minimum_required(VERSION 3.28)
project(ct4 CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(X86 TRUE)
else ()
    set(X86 FALSE)
endif ()

find_package(benchmark CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

set(BLA_SIZEOF_INTEGER 8)
set(BLA_STATIC ON)
find_package(BLAS)

find_package(Threads REQUIRED)

# Set the CMAKE_PREFIX_PATH if necessary
if (APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" CACHE PATH "Path to OpenMP")
    message("Path to Apple OpenMP: ${CMAKE_PREFIX_PATH}")
endif ()

set(OpenMP_RUNTIME_MSVC experimental) # needed to use OpenMP > 2.0
find_package(OpenMP REQUIRED COMPONENTS CXX)


###############################################################################
# Cache and memory
###############################################################################

add_library(my_cpuid INTERFACE my_cpuid.h)
target_include_directories(my_cpuid INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>
)

###############################################################################
# Multithreading
###############################################################################

# We've separated out the implementations so we can write some benchmarks
add_library(nearest_neighbor_functions STATIC
        nearest_neighbor.cpp
        nearest_neighbor.h
)
target_link_Libraries(nearest_neighbor_functions PRIVATE
        Threads::Threads
        OpenMP::OpenMP_CXX)


add_executable(open_mp_demo
        open_mp_demo.cpp
)
target_link_Libraries(open_mp_demo PRIVATE nearest_neighbor_functions)


add_executable(bench_nearest_neighbor
        bench_nearest_neighbor.cpp)
target_Link_libraries(bench_nearest_neighbor PRIVATE
        nearest_neighbor_functions benchmark::benchmark_main)


###############################################################################
# Cache and memory
###############################################################################

if (X86)

    add_executable(cache_information cache_information.cpp)
    target_link_libraries(cache_information PRIVATE my_cpuid)

endif () # X86


add_library(gemm STATIC
        gemm.h
        gemm.cpp
        matrix_view.h
        matrix.h
)

add_executable(bench_gemm bench_gemm.cpp)

if (BLAS_FOUND)

    find_file(BLAS_INCLUDE_DIR cblas.h
            PATHS /usr/include/openblas)
    cmake_path(GET BLAS_INCLUDE_DIR PARENT_PATH BLAS_INCLUDE_DIR)
    message(STATUS "BLAS INCLUDE DIR: ${BLAS_INCLUDE_DIR}")
    target_include_directories(BLAS::BLAS INTERFACE
            ${BLAS_INCLUDE_DIR})


    target_compile_definitions(bench_gemm PRIVATE CT_BLAS=1)
    target_link_Libraries(bench_gemm PRIVATE BLAS::BLAS)
endif ()

target_link_libraries(bench_gemm PRIVATE
        gemm
        benchmark::benchmark_main)

target_compile_options(bench_gemm PRIVATE -g -pg -fno-omit-frame-pointer)


add_executable(test_gemm test_gemm.cpp)

target_link_libraries(test_gemm PRIVATE gemm GTest::gtest_main)


###############################################################################
# Single instruction multiple data
###############################################################################

if (X86)

    add_library(saxpy STATIC saxpy.cpp saxpy.h
            get_best_instructions.h
            saxpy_hand.cpp)
    target_include_directories(saxpy PRIVATE
            ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    target_link_Libraries(saxpy PRIVATE my_cpuid)

    if (MSVC)
        set(ARCH_FLAG "/arch:")
    else ()
        set(ARCH_FLAG "-m")
    endif ()

    set_property(SOURCE saxpy_hand.cpp APPEND PROPERTY
            COMPILE_OPTIONS ${ARCH_FLAG}avx2)


    set(INST default)
    configure_file(saxpy.cpp.in saxpy_${INST}.cpp @ONLY)
    target_sources(saxpy PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp)
    foreach (ARCH IN ITEMS sse4.2 avx2 avx512f)
        string(REPLACE "." "" INST ${ARCH})
        configure_file(saxpy.cpp.in saxpy_${INST}.cpp @ONLY)
        set_property(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp"
                APPEND PROPERTY COMPILE_OPTIONS "${ARCH_FLAG}${ARCH}")
        target_sources(saxpy PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp)
    endforeach ()


    add_executable(bench_saxpy bench_saxpy.cpp)
    target_link_libraries(bench_saxpy PRIVATE saxpy benchmark::benchmark)

    if (BLAS_FOUND)
        target_link_libraries(bench_saxpy PRIVATE BLAS::BLAS)
        target_compile_definitions(bench_saxpy PRIVATE CT_BLAS=1)
    endif ()

endif () # X86


###############################################################################
# Branch prediction and speculative execution
###############################################################################

add_library(clamp_loop STATIC clamp_loop.cpp
        clamp_loop.h)


add_executable(bench_clamp_loop bench_clamp_loop.cpp)
target_link_Libraries(bench_clamp_loop PRIVATE clamp_loop benchmark::benchmark_main)

###############################################################################
# Single instruction multiple data
###############################################################################

add_executable(page_size_report page_size_report.cpp)
