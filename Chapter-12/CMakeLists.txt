cmake_minimum_required(VERSION 3.30)


project(curious_rubber_duckies)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(spdlog CONFIG REQUIRED)


add_executable(duckies main.cpp)


target_link_libraries(duckies PRIVATE
        Boost::program_options
        OpenMP::OpenMP_CXX
        spdlog::spdlog
)


find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

enable_testing()
add_subdirectory(common)
add_subdirectory(readers)
add_subdirectory(clustering)


target_link_libraries(duckies PRIVATE
        duckies_common
        ducky_readers
        ducky_clustering
)



