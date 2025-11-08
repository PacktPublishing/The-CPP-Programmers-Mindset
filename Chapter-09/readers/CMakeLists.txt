find_package(RapidJSON CONFIG REQUIRED)
find_package(csv2 CONFIG REQUIRED)

add_library(ducky_readers STATIC
        file_reader.cpp
        file_reader.h
        json_reader.cpp
        json_reader.h
        csv_reader.cpp
        csv_reader.h
)

target_link_libraries(ducky_readers
        PRIVATE
        csv2::csv2
        spdlog::spdlog
        PUBLIC
        duckies_common
)

target_include_directories(ducky_readers PRIVATE
        ${RAPIDJSON_INCLUDE_DIRS}
)


add_executable(test_ducky_readers
        test_csv_reader.cpp
        test_json_reader.cpp
)

target_link_libraries(test_ducky_readers PRIVATE
        ducky_readers
        spdlog::spdlog
        GTest::gtest_main
)

gtest_discover_tests(test_ducky_readers)
