#ifndef DUCKIES_CLUSTERING_POINT3_H
#define DUCKIES_CLUSTERING_POINT3_H

#include <cmath>
#include <iosfwd>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include "duckies_types.h"

namespace duckies {
struct Coord3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

constexpr Coord3 &operator+=(Coord3 &lhs, const Coord3 &rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

constexpr Coord3 &operator*=(Coord3 &lhs, const float &scalar) noexcept {
    lhs.x *= scalar;
    lhs.y *= scalar;
    lhs.z *= scalar;
    return lhs;
}

constexpr float dist2_squared(const Coord3 &lhs, const Coord3 &rhs) noexcept {
    auto x = lhs.x - rhs.x;
    auto y = lhs.y - rhs.y;
    auto z = lhs.z - rhs.z;

    return x * x + y * y + z * z;
}

inline float dist2(const Coord3 &lhs, const Coord3 &rhs) noexcept {
    return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

std::vector<Coord3> coord3_embedding(
    std::span<const Coordinate> coords) noexcept;

inline Coordinate to_coordinates(const Coord3 &coord) noexcept {
    auto r = std::hypot(coord.x, coord.y, coord.z);
    constexpr auto pi = std::numbers::pi_v<float>;

    // Clamp to zero to avoid incorrect angle computation
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    float x = (std::fabs(coord.x) < epsilon) ? 0.0f : coord.x;
    float y = (std::fabs(coord.y) < epsilon) ? 0.0f : coord.y;

    auto lat = 180.0f * std::asinf(coord.z / r) / pi;
    auto lon = 180.0f * std::atan2f(y, x) / pi;
    return {lat, lon};
}

constexpr bool operator==(const Coord3 &lhs, const Coord3 &rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

constexpr bool operator!=(const Coord3 &lhs, const Coord3 &rhs) noexcept {
    return !operator==(lhs, rhs);
}

std::ostream &operator<<(std::ostream &os, const Coord3 &arg);
} // namespace duckies

#endif