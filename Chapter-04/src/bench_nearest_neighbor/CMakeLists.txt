add_executable(bench_nearest_neighbor bench_nearest_neighbor.cpp)

find_package(benchmark CONFIG REQUIRED)

target_Link_libraries(bench_nearest_neighbor PRIVATE
        nearest_neighbor_functions benchmark::benchmark_main
)
