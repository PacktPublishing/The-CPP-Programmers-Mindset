cmake_minimum_required(VERSION 3.30)

project(ct1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)


####################################################################
# Understanding algorithms
####################################################################

add_library(max_element INTERFACE
        max_element.hpp
)

add_executable(test_max_element
        test_max_element.cpp
)
target_link_libraries(test_max_element PRIVATE
        GTest::gtest_main
        max_element
)
gtest_discover_tests(test_max_element)


# Recursive algorithms

add_library(recursive_words STATIC
        recursive_word.cpp
        recursive_word.hpp)

add_executable(test_recursive_words
        test_recursive_word.cpp
)
target_link_libraries(test_recursive_words PRIVATE
        GTest::gtest_main
        recursive_words
)
gtest_discover_tests(test_recursive_words)
