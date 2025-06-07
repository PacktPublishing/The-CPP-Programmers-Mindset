//
// Created by sammorley on 04/01/25.
//

#include "nearest_neighbor.h"

#include <algorithm>
#include <execution>
#include <vector>
#include <ranges>


using namespace ct;

float ct::nearest_neighbor_distance_seq(std::span<const Point2D> points) {
    const auto size = points.size();
    auto min_distance = std::numeric_limits<float>::infinity();

    for (size_t i=0; i<size; ++i) {
        for (size_t j=i+1; j<size; ++j) {
            auto dist = distance(points[i], points[j]);
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
    }

    return min_distance;
}

float ct::nearest_neighbor_distance_omp(std::span<const Point2D> points) {
    const auto size = points.size();
    auto min_distance = std::numeric_limits<float>::infinity();

#pragma omp parallel for reduction(min:min_distance)
    for (size_t i=0; i<size; ++i) {
        for (size_t j=i+1; j<size; ++j) {
            auto dist = distance(points[i], points[j]);
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
    }

    return min_distance;
}
