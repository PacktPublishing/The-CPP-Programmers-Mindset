add_library(gemm gemm.cpp)

target_include_directories(gemm PRIVATE include)
target_include_directories(gemm PUBLIC include/gemm)
