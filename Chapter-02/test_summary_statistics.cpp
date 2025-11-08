#include "summary_statistics.hpp"

#include <gtest/gtest.h>


TEST(SummaryStatistics, DefaultConfig) {
    auto sources = std::vector<std::string>{"dummy"};

    auto stats = ct::compute_summary_statistics(ct::Configuration{}, sources);

    // Our dummy implementation just sets things to non-zero if they are requested
    EXPECT_NE(stats[0].mean, 0.0);
    EXPECT_NE(stats[0].st_dev, 0.0);
    EXPECT_NE(stats[0].min, 0.0);
    EXPECT_NE(stats[0].max, 0.0);
}


TEST(SummaryStatistics, WithoutMinMax) {
    auto sources = std::vector<std::string>{"dummy"};

    ct::Configuration config;
    config.include_min(false);
    config.include_max(false);

    auto stats = ct::compute_summary_statistics(config, sources);

    EXPECT_NE(stats[0].mean, 0.0);
    EXPECT_NE(stats[0].st_dev, 0.0);
    EXPECT_EQ(stats[0].min, 0.0);
    EXPECT_EQ(stats[0].max, 0.0);
}