#include <gtest/gtest.h>

#include <ctre-unicode.hpp>

#include "free_reader.h"

using namespace duckies;

static constexpr ctll::fixed_string date{RE_DATE};
static constexpr ctll::fixed_string dd_coord{RE_DD};
static constexpr ctll::fixed_string dms_coord{RE_DMS};

TEST(Free, TestDateREISO) {
    std::string s = "2025-04-21";

    auto match = ctre::match < date > (s);
    EXPECT_TRUE(match);
    EXPECT_TRUE(match.get<iso_date_day_group>());
    EXPECT_TRUE(match.get<iso_date_month_group>());
    EXPECT_TRUE(match.get<iso_date_year_group>());
}

TEST(Free, TestDateRELongEU) {
    std::string s = "21 April 2025";

    auto match = ctre::match < date > (s);
    EXPECT_TRUE(match);
}

TEST(Free, TestDateREShortEU) {
    std::string s = "21 Apr 2025";

    auto match = ctre::match < date > (s);
    EXPECT_TRUE(match);
}

TEST(Free, TestDateRELongNA) {
    std::string s = "April 21 2025";

    auto match = ctre::match < date > (s);
    EXPECT_TRUE(match);
}

TEST(Free, TestDateREShortNA) {
    std::string s = "Apr 21 2025";

    auto match = ctre::match < date > (s);
    EXPECT_TRUE(match);
}

TEST(Free, TestCoordREDD) {
    std::u8string s = u8"51.5074, -0.1278";

    auto match = ctre::match < dd_coord > (s);

    EXPECT_TRUE(match);
    EXPECT_TRUE(match.get<ctll::fixed_string("dd_lat")>());
    EXPECT_TRUE(match.get<ctll::fixed_string("dd_lon")>());
}

TEST(Free, TestCoordREFullDMS) {
    std::u8string s = u8"51\u00B0 30' 26\" N, 00\u00B0 7' 39\" W";

    auto match = ctre::match < dms_coord > (s);

    EXPECT_TRUE(match);
}

TEST(Free, TestCoordREDMSNoDegree) {
    std::u8string s = u8"51\u00B0 30' 26\" N, 7' 39\" W";

    auto match = ctre::match < dms_coord > (s);

    EXPECT_TRUE(match);
}

TEST(Free, TestReadOneEntry) {
    FreeTextReader reader;

    std::string text("June 10 2024: This morning I saw a bright yellow rubber "
        "duck was spotted\n"
        "cruising leisurely on the Thames River in London (51 30' "
        "26''N, 7' 39''W).\n"
        "It had a tiny blue scarf around its neck.\n");

    std::istringstream ss(text);

    RubberDuckData data;
    reader.read_stream(data, ss);

    auto coords = data.coordinates();
    ASSERT_EQ(coords.size(), 1);

    auto &first_coord = coords[0];
    EXPECT_NEAR(first_coord.latitude, 51.0f + 30.f / 60 + 26.f / 3600, 1e-7);
    EXPECT_NEAR(
        first_coord.longitude, -(0.0f + 7.0f / 60 + 39.0f / 3600), 1e-7);
}

TEST(Free, TestReadManyEntries) {
    FreeTextReader reader;

    std::string text("June 10 2024: This morning I saw a bright yellow rubber "
        "duck was spotted\n"
        "cruising leisurely on the Thames River in London (51 30' "
        "26''N, 7' 39''W).\n"
        "It had a tiny blue scarf around its neck.\n"
        "\n"
        "June 10 2024: This morning I saw a bright yellow rubber "
        "duck was spotted\n"
        "cruising leisurely on the Thames River in London (51 30' "
        "26''N, 7' 39''W).\n"
        "It had a tiny blue scarf around its neck.\n");

    std::istringstream ss(text);

    RubberDuckData data;
    reader.read_stream(data, ss);

    auto coords = data.coordinates();
    ASSERT_EQ(coords.size(), 2);
}