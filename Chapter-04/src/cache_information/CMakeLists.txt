add_executable(cache_information cache_information.cpp)

target_link_libraries(cache_information PRIVATE my_cpuid)
