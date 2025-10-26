#include <benchmark/benchmark.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "saxpy.cuh"
#include "saxpy_mt.hpp"


// inline constexpr int N = 4096;


#define SAXPY_BENCHMARK(name) BENCHMARK(name)->RangeMultiplier(2)->Range(1<<12, 1<<30)


static void bench_cpu_saxpy(benchmark::State& state) {
    auto N = static_cast<int>(state.range(0));
    thrust::host_vector<float> x(N, 1.0);
    thrust::host_vector<float> y(N, -1.0);
    float a = 2.0;

    for (auto _ : state) {
        ct::cpu_saxpy(a, x.data(), y.data(), N);
        auto* data = y.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}

SAXPY_BENCHMARK(bench_cpu_saxpy);

static void bench_cpu_mt_saxpy(benchmark::State& state) {
    auto N = static_cast<int>(state.range(0));
    thrust::host_vector<float> x(N, 1.0);
    thrust::host_vector<float> y(N, -1.0);
    float a = 2.0;

    for (auto _ : state) {
        ct::cpu_mt_saxpy(a, x.data(), y.data(), N);
        auto* data = y.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}

SAXPY_BENCHMARK(bench_cpu_mt_saxpy);


// OpenMP device offload is best supported on Clang, and requires the
// -fopenmp-targets=<arch> flags to be set on the command line. To
// avoid confusion, we've disabled the benchmark for this implementation.
#ifdef __clang__

static void bench_openmp_saxpy(benchmark::State& state) {
    auto N = static_cast<int>(state.range(0));
    thrust::host_vector<float> x(N, 1.0);
    thrust::host_vector<float> y(N, -1.0);
    float a = 2.0;

    for (auto _ : state) {
        ct::openmp_saxpy(a, x.data(), y.data(), N);
        auto* data = y.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}
SAXPY_BENCHMARK(bench_openmp_saxpy);
#endif


static void bench_thrust_saxpy(benchmark::State& state) {
    auto N = static_cast<int>(state.range(0));
    thrust::device_vector<float> x(N, 1.0);
    thrust::device_vector<float> y(N, -1.0);
    thrust::device_vector<float> out(N, 0.0);
    float a = 2.0;

    for (auto _ : state) {
        ct::thrust_saxpy(a, x, y, out);
        cudaDeviceSynchronize();
    }
}

SAXPY_BENCHMARK(bench_thrust_saxpy);


static void bench_cuda_saxpy(benchmark::State &state) {
    auto N = static_cast<int>(state.range(0));
    thrust::device_vector<float> x(N, 1.0);
    thrust::device_vector<float> y(N, -1.0);

    float a = 2.0;


    for (auto _: state) {
        auto rc = ct::basic_saxpy(
            a,
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            N
        );
        benchmark::DoNotOptimize(rc);
        cudaDeviceSynchronize();
    }
}


SAXPY_BENCHMARK(bench_cuda_saxpy);



