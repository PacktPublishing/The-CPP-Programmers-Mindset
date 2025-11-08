cmake_minimum_required(VERSION 3.30)
project(c5)

set(CMAKE_CXX_STANDARD 20)


find_package(absl CONFIG REQUIRED)
find_package(boost_container CONFIG REQUIRED)
find_package(boost_container_hash CONFIG REQUIRED)
find_package(boost_unordered CONFIG REQUIRED)


find_package(benchmark CONFIG REQUIRED benchmark_main)
find_package(GTest CONFIG REQUIRED gtest_main)


add_library(bench_container_base INTERFACE
        bench_containers.h
)

target_link_libraries(bench_container_base INTERFACE
        absl::base
        absl::core_headers
        absl::container_common
        absl::hash
        absl::flat_hash_map
        absl::flat_hash_set
        absl::hash_container_defaults
        absl::hash_policy_traits
        absl::inlined_vector
        absl::memory
        absl::node_hash_map
        absl::node_hash_set
        Boost::container
        Boost::unordered
)

add_executable(bench_container_insert
        bench_container_insert.cpp
)
target_link_libraries(bench_container_insert PRIVATE
        benchmark::benchmark_main
        bench_container_base)

add_executable(bench_container_traversal
        bench_container_traversal.cpp)
target_link_libraries(bench_container_traversal PRIVATE
        benchmark::benchmark_main
        bench_container_base)

add_executable(bench_map_random_access
        bench_map_random_access.cpp)
target_link_libraries(bench_map_random_access PRIVATE
        benchmark::benchmark_main
        bench_container_base)


add_executable(bench_hash benchmark_hash.cpp)
target_link_libraries(bench_hash PRIVATE
        benchmark::benchmark_main
        Boost::container_hash
        absl::hash)
