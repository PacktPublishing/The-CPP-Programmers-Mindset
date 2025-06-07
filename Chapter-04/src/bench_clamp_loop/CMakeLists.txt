add_executable(bench_clamp_loop bench_clamp_loop.cpp)

find_package(benchmark CONFIG REQUIRED)

target_link_Libraries(bench_clamp_loop PRIVATE clamp_loop benchmark::benchmark_main)
