add_library(ducky_clustering
        clustering.h
        clustering.cpp
        coord3.cpp
        coord3.h
        kmeans.cpp
        kmeans.h
)

target_link_libraries(ducky_clustering
        PUBLIC
        duckies_common
        spdlog::spdlog
)


add_executable(test_clustering
        test_kmeans.cpp
        test_clustering.cpp
        test_coord3.cpp
)

target_link_libraries(test_clustering
        PRIVATE
        GTest::gtest_main ducky_clustering)


gtest_discover_tests(test_clustering)
