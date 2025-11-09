cmake_minimum_required(VERSION 3.30)
project(ct13)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)


add_library(sync_methods INTERFACE
        caching_function.hpp
        task.hpp
)

add_executable(ct13 main.cpp)

target_link_libraries(ct13 PRIVATE MPI::MPI_CXX)
add_executable(byte_order byte_order.cpp)


