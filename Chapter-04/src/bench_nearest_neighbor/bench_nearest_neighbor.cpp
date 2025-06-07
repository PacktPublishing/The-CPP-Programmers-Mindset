//
// Created by sammorley on 04/01/25.
//


#include <random>

#include <benchmark/benchmark.h>


#include "nearest_neighbor.h"

using namespace ct;




static void BM_sequential(benchmark::State& state) {
    // Some random data to process
    std::vector<Point2D> points;

    points.reserve(state.range(0));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist{-1.f, 1.f};
    for (int i=0; i<points.capacity(); ++i) {
        points.emplace_back(dist(rng), dist(rng));
    }

    for (auto _: state) {
        benchmark::DoNotOptimize(ct::nearest_neighbor_distance_seq(points));
    }
}

static void BM_openmp(benchmark::State& state) {
    // Some random data to process
    std::vector<Point2D> points;

    points.reserve(state.range(0));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist{-1.f, 1.f};
    for (int i=0; i<points.capacity(); ++i) {
        points.emplace_back(dist(rng), dist(rng));
    }

    for (auto _: state) {
        benchmark::DoNotOptimize(ct::nearest_neighbor_distance_omp(points));
    }
}


BENCHMARK(BM_sequential) -> Arg(65536) -> Unit(benchmark::kMillisecond);
BENCHMARK(BM_openmp) -> Arg(65536) -> Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();