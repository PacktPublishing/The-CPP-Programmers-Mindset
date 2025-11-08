cmake_minimum_required(VERSION 3.30)
project(ct14 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ct14 main.cu)

set_target_properties(ct14 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
