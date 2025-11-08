#include "coord3.h"

#include <cmath>
#include <format>
#include <ostream>

using namespace duckies;

std::vector<Coord3> duckies::coord3_embedding(
    std::span<const Coordinate> coords) noexcept {
    std::vector<Coord3> result(coords.size());
    const auto N = coords.size();

#pragma omp parallel for default(none) shared(N, result, coords)
    for (size_t i = 0; i < N; ++i) {
        auto &coords3 = result[i];
        const auto &angles = coords[i];

        auto psi = std::numbers::pi_v<float> * angles.latitude / 180.;
        auto lambda = std::numbers::pi_v<float> * angles.longitude / 180.;

        auto cospsi = std::cosf(psi);

        coords3.x = cospsi * std::cosf(lambda);
        coords3.y = cospsi * std::sinf(lambda);
        coords3.z = std::sinf(psi);
    }

    return result;
}

std::ostream &duckies::operator<<(std::ostream &os, const Coord3 &arg) {
    return os << std::format("{{{}, {}, {}}}", arg.x, arg.y, arg.z);
}