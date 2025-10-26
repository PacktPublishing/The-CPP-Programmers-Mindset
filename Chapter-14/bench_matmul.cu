#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "matmul.cuh"



// Benchmark custom matrix multiplication
static void bench_ct_matmul(benchmark::State& state) {
    int m = static_cast<int>(state.range(0));
    int n = static_cast<int>(state.range(1));
    int k = static_cast<int>(state.range(2));

    // Initialize matrices
    thrust::device_vector<float> d_A(m * k, 1.0f);
    thrust::device_vector<float> d_B(k * n, 1.0f);
    thrust::device_vector<float> d_C(m * n, 0.0f);

    constexpr int tile_size = 16;
    dim3 block(tile_size, tile_size);
    dim3 grid((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size);


    for (auto _ : state) {
        ct::matrix_mul<<<grid, block>>>(
            thrust::raw_pointer_cast(d_C.data()),
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()),
            m, n, k
        );
        cudaDeviceSynchronize();
    }
}

// Benchmark cuBLAS matrix multiplication
static void bench_cublas_matmul(benchmark::State& state) {
    int m = static_cast<int>(state.range(0));
    int n = static_cast<int>(state.range(1));
    int k = static_cast<int>(state.range(2));

    // Initialize matrices
    thrust::device_vector<float> d_A(m * k, 1.0f);
    thrust::device_vector<float> d_B(k * n, 1.0f);
    thrust::device_vector<float> d_C(m * n, 0.0f);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        state.SkipWithError("failed to create cublas handle");
    };

    float alpha = 1.0f;
    float beta = 0.0f;



    for (auto _ : state) {
        auto code = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n, m, k,
            &alpha,
            thrust::raw_pointer_cast(d_B.data()), n,
            thrust::raw_pointer_cast(d_A.data()), k,
            &beta,
            thrust::raw_pointer_cast(d_C.data()), n
        );
        benchmark::DoNotOptimize(code);
        cudaDeviceSynchronize();
    }

    cublasDestroy(handle);
}

// Square matrices (m=n=k)
BENCHMARK(bench_ct_matmul)
    ->Args({64, 64, 64})
    ->Args({256, 256, 256})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(bench_cublas_matmul)
    ->Args({64, 64, 64})
    ->Args({256, 256, 256})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Unit(benchmark::kMicrosecond);


BENCHMARK_MAIN();