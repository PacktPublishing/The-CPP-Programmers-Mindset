cmake_minimum_required(VERSION 3.30)
project(ct15)

set(CMAKE_CXX_STANDARD 20)


find_package(benchmark CONFIG REQUIRED)


add_executable(bench_weighted_average
        bench_weighted_average.cpp
)

target_link_libraries(bench_weighted_average
        PRIVATE benchmark::benchmark_main)

target_compile_options(bench_weighted_average
        PRIVATE -fno-omit-frame-pointer -g -fno-exceptions -fno-rtti)

