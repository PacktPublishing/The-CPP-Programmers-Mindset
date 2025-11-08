find_package(RapidJSON CONFIG REQUIRED)
find_package(csv2 CONFIG REQUIRED)
find_package(ctre CONFIG REQUIRED)

add_library(ducky_readers STATIC
        file_reader.cpp
        file_reader.h
        json_reader.cpp
        json_reader.h
        csv_reader.cpp
        csv_reader.h
        free_reader.cpp
        free_reader.h
)

target_link_libraries(ducky_readers
        PRIVATE
        csv2::csv2
        spdlog::spdlog
        ctre::ctre
        PUBLIC
        duckies_common
)

# Clang defaults to a expression fold limit of 256 terms, which is
# too small for our compile-time regexes. We expand this here and
# below.
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    target_compile_options(ducky_readers PRIVATE -fbracket-depth=1024)
endif ()

target_include_directories(ducky_readers PRIVATE
        ${RAPIDJSON_INCLUDE_DIRS}
)


add_executable(test_ducky_readers
        test_csv_reader.cpp
        test_json_reader.cpp
        test_free_reader.cpp
)

target_link_libraries(test_ducky_readers PRIVATE
        ducky_readers
        ctre::ctre
        spdlog::spdlog
        GTest::gtest_main
)

gtest_discover_tests(test_ducky_readers)

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    target_compile_options(test_ducky_readers PRIVATE -fbracket-depth=1024)
endif ()
