
cmake_minimum_required(VERSION 3.30)


project(ct2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


enable_testing()
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

##############################################################################
## Using standard algorithms
##############################################################################

add_library(signal_intensity STATIC
        signal_intensity.cpp
        signal_intensity.hpp
        intensity_grid_search.cpp
        intensity_grid_search.hpp
        expanding_search_range.hpp
        expanding_search_range.cpp
        find_root.hpp
        function_with_noise.hpp
)


add_executable(test_signal_intensity
        test_signal_intensity.cpp)

target_link_libraries(test_signal_intensity PRIVATE
        GTest::gtest_main
        signal_intensity
)
gtest_discover_tests(test_signal_intensity)

##############################################################################
## When to use functions
##############################################################################

## Creating interfaces based on functions

add_library(summary_statistics STATIC
        summary_statistics.cpp
        summary_statistics.hpp
)

add_executable(test_summary_stats
        test_summary_statistics.cpp)

target_link_libraries(test_summary_stats PRIVATE
        GTest::gtest_main
        summary_statistics
)
gtest_discover_tests(test_summary_stats)


## Functions as building blocks

add_library(find_root INTERFACE find_root.hpp)


add_executable(test_find_root
        test_find_root.cpp)

target_link_libraries(test_find_root PRIVATE
        GTest::gtest_main
        find_root
)
gtest_discover_tests(test_find_root)

## Function-like objects


add_library(function_with_noise INTERFACE
        function_with_noise.hpp)


add_executable(test_function_with_noise
        test_function_with_noise.cpp)

target_link_libraries(test_function_with_noise PRIVATE
        GTest::gtest_main
        function_with_noise
)
gtest_discover_tests(test_function_with_noise)


##############################################################################
## When to use classes
##############################################################################


add_library(address_book_record STATIC
        address_book_record.cpp
        address_book_record.hpp
        conversion_traits.hpp
)


add_executable(test_address_book_record
        test_address_book_record.cpp
)

target_link_libraries(test_address_book_record PRIVATE
        GTest::gtest_main
        address_book_record
)
gtest_discover_tests(test_address_book_record)

## Classes that represent physical objects


add_library(temperature_sensors STATIC
        temperature_sensors.cpp
        temperature_sensors.hpp
        concrete_temperature_sensor.cpp
        concrete_temperature_sensor.hpp)

add_executable(test_temperature_sensors
        test_temperature_sensors.cpp
)

target_link_libraries(test_temperature_sensors PRIVATE
        GTest::gtest_main
        temperature_sensors
)
gtest_discover_tests(test_temperature_sensors)


