//
// Created by sammorley on 02/01/25.
//
#include <benchmark/benchmark.h>

#if CT_BLAS
#include <cblas.h>
#endif

#include "saxpy.h"
#include "get_best_instructions.h"

inline constexpr size_t N = 5*1024;


#if CT_BLAS

static void BM_saxpy_blas(benchmark::State &state) {
    std::vector<float> x(N, 1.0);
    std::vector<float> y(N, -1.0);
    float a = 2.0;

    for (auto _: state) {
        cblas_saxpy(N, a, x.data(), 1, y.data(), 1);
        auto *data = y.data();
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}


#endif

int main(int argc, char **argv) {
    switch (ct::get_best_instructions()) {
        case ct::SIMDInstructions::AVX512F:
            benchmark::RegisterBenchmark("saxpy_avx512f",
                                         [](benchmark::State &state) {
                                             std::vector<float> x(N, 1.0);
                                             std::vector<float> y(N, -1.0);
                                             float a = 2.0;

                                             for (auto _: state) {
                                                 ct::saxpy_avx512f(a, x, y);
                                                 auto *data = y.data();
                                                 benchmark::DoNotOptimize(data);
                                                 benchmark::ClobberMemory();
                                             }
                                         }
            );
        case ct::SIMDInstructions::AVX2:
            benchmark::RegisterBenchmark("saxpy_avx2",
                                         [](benchmark::State &state) {
                                             std::vector<float> x(N, 1.0);
                                             std::vector<float> y(N, -1.0);
                                             float a = 2.0;

                                             for (auto _: state) {
                                                 ct::saxpy_avx2(a, x, y);
                                                 auto *data = y.data();
                                                 benchmark::DoNotOptimize(data);
                                                 benchmark::ClobberMemory();
                                             }
                                         }
            );
            benchmark::RegisterBenchmark("saxpy_hand",
                                         [](benchmark::State &state) {
                                             std::vector<float> x(N, 1.0);
                                             std::vector<float> y(N, -1.0);
                                             float a = 2.0;

                                             for (auto _: state) {
                                                 ct::saxpy_hand(a, x, y);
                                                 auto *data = y.data();
                                                 benchmark::DoNotOptimize(data);
                                                 benchmark::ClobberMemory();
                                             }
                                         }
                );
        case ct::SIMDInstructions::SSE42:
            benchmark::RegisterBenchmark("saxpy_sse42",
                                         [](benchmark::State &state) {
                                             std::vector<float> x(N, 1.0);
                                             std::vector<float> y(N, -1.0);
                                             float a = 2.0;

                                             for (auto _: state) {
                                                 ct::saxpy_sse42(a, x, y);
                                                 auto *data = y.data();
                                                 benchmark::DoNotOptimize(data);
                                                 benchmark::ClobberMemory();
                                             }
                                         }
            );
        case ct::SIMDInstructions::Default:
            benchmark::RegisterBenchmark("saxpy_default",
                                         [](benchmark::State &state) {
                                             std::vector<float> x(N, 1.0);
                                             std::vector<float> y(N, -1.0);
                                             float a = 2.0;

                                             for (auto _: state) {
                                                 ct::saxpy_default(a, x, y);
                                                 auto *data = y.data();
                                                 benchmark::DoNotOptimize(data);
                                                 benchmark::ClobberMemory();
                                             }
                                         }
            );
    }


#ifdef CT_BLAS
    benchmark::RegisterBenchmark("saxpy_blas", BM_saxpy_blas);
#endif
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
