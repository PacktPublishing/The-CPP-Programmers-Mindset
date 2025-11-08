add_library(ctode_euler STATIC euler.h euler.cpp)

set_target_properties(ctode_euler PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if (ENABLE_TESTS)
    add_executable(test_ctode_euler test_euler.cpp)
    target_link_libraries(test_ctode_euler PRIVATE
            ctode_euler
            GTest::gtest_main
    )
endif ()