#include <gtest/gtest.h>

#include "clustering.h"

using namespace duckies;

TEST(Clustering, TestClusteringPolarData) {
    RubberDuckData data;
    data.reserve_additional(10);
    constexpr float km100 = 0.89833485f;

    data.insert(90.f, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - km100, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - km100, 90.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - km100, 180.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - km100, -90.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + km100, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + km100, 90.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + km100, 180.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + km100, -90.0f, std::chrono::year_month_day{}, "");

    auto clusters = compute_clusters(data, 1, 10, 5, 100);
    ASSERT_EQ(clusters.size(), 2);

    for (const auto &cluster: clusters) {
        EXPECT_EQ(cluster.num_points, 5);
    }
}