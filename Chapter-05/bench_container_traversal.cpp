#include <benchmark/benchmark.h>


#include "bench_containers.h"


template<typename Container>
void bench_container_traversal(benchmark::State &state) {
    Container c;
    for (int64_t i = 0; i < state.range(0); ++i) {
        push(c, i);
    }

    for (auto _: state) {
        int64_t sum = 0;
        for (auto &i: c) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK_TEMPLATE(bench_container_traversal, StdVec) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, StdFList) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, StdList) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, StdSet) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, StdUSet) -> Arg(256);

BENCHMARK_TEMPLATE(bench_container_traversal, AbslBTreeSet) -> Arg(245);
BENCHMARK_TEMPLATE(bench_container_traversal, AbslFlatHashSet) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, AbslNodeHashSet) -> Arg(256);

BENCHMARK_TEMPLATE(bench_container_traversal, BoostVec) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, BoostFlatSet) -> Arg(256);
BENCHMARK_TEMPLATE(bench_container_traversal, BoostStableVec) -> Arg(256);



BENCHMARK_MAIN();