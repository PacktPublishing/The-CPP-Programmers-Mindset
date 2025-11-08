#include <sstream>

#include <gtest/gtest.h>

#include "json_reader.h"

using namespace duckies;
using namespace std::chrono_literals;

TEST(JSON, TestReadSample) {
    JSONReader reader;
    std::string json_sample(
        "[{\"latitude\": 51.5074,\"longitude\": -0.1278,"
        "\"date\": \"2024-09-10\","
        "\"description\": \"A curious rubber ducky sighted in London.\"}]");
    std::stringstream ss(json_sample);
    RubberDuckData data;
    reader.read_stream(data, ss);

    auto &coordinates = data.coordinates();
    ASSERT_EQ(coordinates.size(), 1);
    auto &coord = coordinates[0];
    EXPECT_NEAR(coord.latitude, 51.5074, 1e-6);
    EXPECT_NEAR(coord.longitude, -0.1278, 1e-6);

    auto &dates = data.dates();
    ASSERT_EQ(dates.size(), 1);
    std::chrono::year_month_day expected_date{
        2024y, std::chrono::September,
        10d
    };
    EXPECT_EQ(dates[0], expected_date);
}