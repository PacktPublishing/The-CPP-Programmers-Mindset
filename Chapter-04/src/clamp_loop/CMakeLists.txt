add_library(clamp_loop clamp_loop.cpp)

target_include_directories(clamp_loop PRIVATE include)
target_include_directories(clamp_loop PUBLIC include/clamp_loop)
