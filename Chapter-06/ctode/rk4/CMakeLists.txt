add_library(ctode_rk4 STATIC rk4.h rk4.cpp)

set_target_properties(ctode_rk4 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if (ENABLE_TESTS)
    add_executable(test_ctode_rk4 test_rk4.cpp)
    target_link_libraries(test_ctode_rk4 PRIVATE
            ctode_rk4
            GTest::gtest_main
    )
endif ()
