add_executable(bench_saxpy bench_saxpy.cpp)

find_package(benchmark CONFIG REQUIRED)

target_link_libraries(bench_saxpy PRIVATE saxpy benchmark::benchmark)

find_package(BLAS)

if (BLAS_FOUND)
    find_file(BLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas)
    cmake_path(GET BLAS_INCLUDE_DIR PARENT_PATH BLAS_INCLUDE_DIR)
    message(STATUS "BLAS INCLUDE DIR: ${BLAS_INCLUDE_DIR}")
    target_include_directories(BLAS::BLAS INTERFACE ${BLAS_INCLUDE_DIR})
    target_compile_definitions(bench_saxpy PRIVATE CT_BLAS=1)
    target_link_Libraries(bench_saxpy PRIVATE BLAS::BLAS)
endif ()
