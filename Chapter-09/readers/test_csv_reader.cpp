#include <string_view>

#include <gtest/gtest.h>

#include "csv_reader.h"

using namespace duckies;
using namespace std::chrono_literals;

TEST(CSV, TestReadSample) {
    CSVReader reader;
    std::string_view csv_sample("date, latitude, longitude, description\n"
        "2024-09-10, 51.5074, -0.1278, A mysterious rubber "
        "ducky was seen floating in London.\n");
    RubberDuckData data;
    reader.read_view(data, csv_sample);

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

TEST(CSV, TestReadSamplePermutedCols) {
    CSVReader reader;
    std::string_view csv_sample("latitude, longitude, date, description\n"
        "51.5074, -0.1278, 2024-09-10, A mysterious rubber "
        "ducky was seen floating in London.\n");
    RubberDuckData data;
    reader.read_view(data, csv_sample);

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