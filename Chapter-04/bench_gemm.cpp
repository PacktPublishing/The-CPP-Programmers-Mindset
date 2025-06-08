//
// Created by sammorley on 31/12/24.
//


#include <benchmark/benchmark.h>

#if CT_BLAS
#include <cblas.h>
#endif

#include "gemm.h"
#include "matrix.h"

static void BM_gemm_basic(benchmark::State &state) {
    ct::Matrix<double> a(1000, 1000);
    ct::Matrix<double> b(1000, 1000);
    std::uninitialized_fill_n(b.data(), b.size(), 1.0);
    ct::Matrix<double> c(1000, 1000);
    std::uninitialized_fill_n(c.data(), c.size(), 1.0);

    for (auto _: state) {
        ct::dgemm_basic(a, b, c, 1.0, 0.0);
        auto *c_data = c.data();
        benchmark::DoNotOptimize(c_data);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gemm_basic)->Unit(benchmark::kMillisecond);

static void BM_gemm_blocked(benchmark::State &state) {
    ct::Matrix<double> a(1000, 1000);
    ct::Matrix<double> b(1000, 1000);
    std::uninitialized_fill_n(b.data(), b.size(), 1.0);
    ct::Matrix<double> c(1000, 1000);
    std::uninitialized_fill_n(c.data(), c.size(), 1.0);

    for (auto _: state) {
        ct::dgemm_blocked(a, b, c, 1.0, 0.0, 32);
        auto *c_data = c.data();
        benchmark::DoNotOptimize(c_data);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gemm_blocked)->Unit(benchmark::kMillisecond);


#ifdef CT_BLAS

static void BM_gemm_blas(benchmark::State &state) {
    ct::Matrix<double> a(1000, 1000);
    ct::Matrix<double> b(1000, 1000);
    std::uninitialized_fill_n(b.data(), b.size(), 1.0);
    ct::Matrix<double> c(1000, 1000);
    std::uninitialized_fill_n(c.data(), c.size(), 1.0);

    for (auto _: state) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1000, 1000, 1000, 1.0, a.data(), 1000, b.data(), 1000,
                    0.0, c.data(), 1000);
        auto *c_data = c.data();
        benchmark::DoNotOptimize(c_data);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_gemm_blas)->Unit(benchmark::kMillisecond);

#endif


BENCHMARK_MAIN();
