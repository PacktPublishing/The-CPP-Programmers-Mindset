cmake_minimum_required(VERSION 3.30)

project(ct3)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)


enable_testing()
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)


################################################################################
# Designing your own algorithms
################################################################################


## Recursive algorithms

add_library(powers INTERFACE
        powers.hpp)


add_executable(test_powers
        test_powers.cpp
)

target_link_libraries(test_powers PRIVATE
        GTest::gtest_main
        powers
)

gtest_discover_tests(test_powers)


## Divide and conquer algorithms


add_library(quick_sort INTERFACE
        quick_sort.hpp)


add_executable(test_quick_sort
        test_quick_sort.cpp)

target_link_libraries(test_quick_sort PRIVATE
        GTest::gtest_main
        quick_sort
)
gtest_discover_tests(test_quick_sort)



