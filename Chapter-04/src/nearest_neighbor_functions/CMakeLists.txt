add_library(nearest_neighbor_functions nearest_neighbor.cpp)

target_include_directories(nearest_neighbor_functions PRIVATE include)
target_include_directories(nearest_neighbor_functions PUBLIC include/nearest_neighbor_functions)

find_package(Threads REQUIRED)

# Set the CMAKE_PREFIX_PATH if necessary
if (APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" CACHE PATH "Path to OpenMP")
    message("Path to Apple OpenMP: ${CMAKE_PREFIX_PATH}")
endif ()

find_package(OpenMP REQUIRED COMPONENTS CXX)

target_link_Libraries(nearest_neighbor_functions PRIVATE
        Threads::Threads
        OpenMP::OpenMP_CXX
)
