add_executable(bench_gemm bench_gemm.cpp)

find_package(benchmark CONFIG REQUIRED)

find_package(BLAS)

if (BLAS_FOUND)
    find_file(BLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas)
    cmake_path(GET BLAS_INCLUDE_DIR PARENT_PATH BLAS_INCLUDE_DIR)
    message(STATUS "BLAS INCLUDE DIR: ${BLAS_INCLUDE_DIR}")
    target_include_directories(BLAS::BLAS INTERFACE ${BLAS_INCLUDE_DIR})
    target_compile_definitions(bench_gemm PRIVATE CT_BLAS=1)
    target_link_Libraries(bench_gemm PRIVATE BLAS::BLAS)
endif ()

target_link_libraries(bench_gemm PRIVATE
        gemm
        benchmark::benchmark_main
)

target_compile_options(bench_gemm PRIVATE -g -pg -fno-omit-frame-pointer)
