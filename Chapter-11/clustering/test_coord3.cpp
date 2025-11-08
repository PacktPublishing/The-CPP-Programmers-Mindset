#include <gtest/gtest.h>

#include <cmath>

#include "coord3.h"

using namespace duckies;

TEST(Coord3, TestCoord3Addition) {
    Coord3 origin{0.0f, 0.0f, 0.0f};

    Coord3 x{1.0f, 0.0f, 0.0f};
    Coord3 y{0.0f, 1.0f, 0.0f};
    Coord3 z{0.0f, 0.0f, 1.0f};

    EXPECT_EQ(origin += x, x);

    Coord3 xy{1.0f, 1.0f, 0.0f};
    EXPECT_EQ(origin += y, xy);

    Coord3 xyz{1.0f, 1.0f, 1.0f};
    EXPECT_EQ(origin += z, xyz);
}

TEST(Coord3, TestScalarMultiplication) {
    Coord3 xyz{1.0f, 1.0f, 1.0f};

    xyz *= 0.5f;

    Coord3 expected{0.5f, 0.5f, 0.5f};
    EXPECT_EQ(xyz, expected);
}

TEST(Coord3, TestEuclideanDistance) {
    Coord3 x{1.0f, 0.74f, -0.25f};
    Coord3 y{1.0f, 0.24f, 0.25f};

    auto expected = std::sqrtf(0.5);

    auto norm = dist2(x, y);

    EXPECT_NEAR(norm, expected, 1e-7);

    EXPECT_NEAR(dist2_squared(x, y), expected * expected, 2e-7f);
}

TEST(Coord3, TestEmbeddingCardinalPositions) {
    std::vector<Coordinate> cardinals{
        {0.0f, 0.0f}, {90.0f, 0.0f},
        {-90.0f, 0.0f}, {0.0f, 90.0f}, {0.0f, -90.0f}, {0.0f, 180.0f}
    };

    std::vector<Coord3> expected{
        {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f}
    };

    auto transformed = coord3_embedding(cardinals);

    ASSERT_EQ(transformed.size(), cardinals.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(dist2(transformed[i], expected[i]), 0.0f, 1e-7f);

        auto rt = to_coordinates(transformed[i]);
        EXPECT_NEAR(rt.latitude, cardinals[i].latitude, 1e-7) << i;
        EXPECT_NEAR(rt.longitude, cardinals[i].longitude, 1e-7) << i;
    }
}