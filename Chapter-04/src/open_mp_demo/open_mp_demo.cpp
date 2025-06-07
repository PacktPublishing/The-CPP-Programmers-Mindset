//
// Created by sammorley on 04/01/25.
//



#include <random>
#include <vector>
#include <iostream>

// We've moved the code that was inline in the book into
// a separate translation unit so we can benchmark them
// against one another.
#include "nearest_neighbor.h"

using namespace ct;

int main() {

    // Some random data to process
    std::vector<Point2D> points;

    points.reserve(100'000);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist{-1.f, 1.f};
    for (int i=0; i<points.capacity(); ++i) {
        points.emplace_back(dist(rng), dist(rng));
    }

    auto min = nearest_neighbor_distance_omp(points);

    std::cout << "Minimum distance " << min << std::endl;

    return 0;
}