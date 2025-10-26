
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

inline void outer_product(float* z,
                          const float* x,
                          const float* y,
                          size_t left_size,
                          size_t right_size) noexcept {
    for (size_t i=0; i<left_size; ++i) {
        for (size_t j=0; j<right_size; ++j) {
            z[i*right_size + j] += x[i] * y[j];
        }
    }
}

static void bench_outer_product(benchmark::State& state) {
    auto left_size = static_cast<size_t>(state.range(0));
    auto right_size = static_cast<size_t>(state.range(1));

    std::vector<float> x(left_size, 1.0f);
    std::vector<float> y(right_size, 1.0f);
    std::vector<float> z(left_size * right_size, 0.0f);

    for (auto _ : state) {
        outer_product(z.data(), x.data(), y.data(), left_size, right_size);
        benchmark::DoNotOptimize(z.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(bench_outer_product)->Args({512, 4096})->Args({128, 16384});