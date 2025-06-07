add_executable(test_gemm test_gemm.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(test_gemm PRIVATE gemm GTest::gtest_main)
