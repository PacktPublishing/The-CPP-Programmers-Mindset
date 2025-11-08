if (ENABLE_TESTS)
    find_package(GTest CONFIG REQUIRED)
endif ()
include(GenerateExportHeader)

# Import the two solver's that we've implemented
add_subdirectory(euler)
add_subdirectory(rk4)

add_library(ctode SHARED ctode.h ctode.cpp)
target_link_libraries(ctode PRIVATE ctode_euler ctode_rk4)

generate_export_header(ctode)
target_include_directories(ctode PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

if (ENABLE_TESTS)
    add_executable(test_ctode test_ctode.cpp)
    target_link_libraries(test_ctode PRIVATE
            ctode GTest::gtest_main)
endif ()