//
// Created by sammorley on 04/01/25.
//


#include <vector>
#include <random>

#include <benchmark/benchmark.h>

#include "clamp_loop.h"



static void BM_clamp_loop_min(benchmark::State& state) {
    std::vector<uint16_t> x(1'000);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint16_t> dist(0, std::numeric_limits<uint16_t>::max());

    for (auto _: state) {
        state.PauseTiming();
        for (auto& v : x) {
            v = dist(rng);
        }
        state.ResumeTiming();

        ct::clamp_min(x);
        auto *data = x.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_clamp_loop_min);


static void BM_clamp_loop_conditional(benchmark::State& state) {
    std::vector<uint16_t> x(1'000);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint16_t> dist(0, std::numeric_limits<uint16_t>::max());
    for (auto _: state) {
        state.PauseTiming();
        for (auto& v : x) {
            v = dist(rng);
        }
        state.ResumeTiming();
        ct::clamp_conditional(x);
        auto *data = x.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_clamp_loop_conditional);


BENCHMARK_MAIN();