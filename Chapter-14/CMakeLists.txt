cmake_minimum_required(VERSION 3.30)
project(ct14 CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Thrust CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(GTest CONFIG REQUIRED)
enable_testing()
include(GoogleTest)


add_library(cuda_saxpy STATIC
        saxpy_mt.cpp
        saxpy_mt.hpp
        saxpy.cu
        saxpy.cuh)

target_link_libraries(cuda_saxpy PUBLIC
    OpenMP::OpenMP_CXX
    Thrust::Thrust
)

target_compile_options(cuda_saxpy PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)


add_executable(bench_saxpy
        bench_saxpy.cu
)

target_link_libraries(bench_saxpy PRIVATE benchmark::benchmark_main cuda_saxpy)



add_executable(ct14 main.cu)

set_target_properties(ct14 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)


add_library(matmul STATIC matmul.cu
        matmul.cuh)


add_executable(test_matmul test_matmul.cu)

target_link_libraries(test_matmul PRIVATE
        matmul
        Thrust::Thrust
        CUDA::cublas
        GTest::gtest_main
)

gtest_discover_tests(test_matmul)


add_executable(bench_matmul bench_matmul.cu)

target_link_libraries(bench_matmul PRIVATE
        matmul
        benchmark::benchmark
        CUDA::cublas
        Thrust::Thrust
)
