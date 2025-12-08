#include <cstdint>
#include <random>
#include <functional>
#include <string>

#include <benchmark/benchmark.h>

#include <absl/hash/hash.h>
#include <boost/container_hash/hash.hpp>


static void BM_baseline_Int64(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    auto val = dist(rng);
    for (auto _: state) {
        benchmark::DoNotOptimize(val);
    }
}

static void BM_baseline_Double(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::lowest(),
                                                std::numeric_limits<double>::max());
    auto val = dist(rng);
    for (auto _: state) {
        benchmark::DoNotOptimize(val);
    }
}

static void BM_StdHash_Int64(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    std::hash<int64_t> hasher;

    auto val = dist(rng);
    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_AbslHash_Int64(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    absl::Hash<int64_t> hasher;
    auto val = dist(rng);

    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_BoostHash_Int64(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<int64_t>::min(),
                                                std::numeric_limits<int64_t>::max());
    boost::hash<int64_t> hasher;
    auto val = dist(rng);

    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_StdHash_Double(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(),
                                                std::numeric_limits<double>::max());
    std::hash<double> hasher;
    auto val = dist(rng);

    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_AbslHash_Double(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(),
                                                std::numeric_limits<double>::max());
    absl::Hash<double> hasher;

    auto val = dist(rng);
    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_BoostHash_Double(benchmark::State &state) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(),
                                                std::numeric_limits<double>::max());
    boost::hash<double> hasher;

    auto val = dist(rng);
    for (auto _: state) {
        benchmark::DoNotOptimize(hasher(val));
    }
}

static void BM_StdHash_String(benchmark::State &state) {
    std::string test_str("this is a string that is a far too "
        "long to fit in the internal memory");

    for (auto _: state) {
        std::hash<std::string> hasher;
        auto hv = hasher(test_str);
        benchmark::DoNotOptimize(hv);
    }
}


static void BM_AbslHash_String(benchmark::State &state) {
    std::string test_str("this is a string that is a far too "
        "long to fit in the internal memory");

    for (auto _: state) {
        absl::Hash<std::string> hasher;
        auto hv = hasher(test_str);
        benchmark::DoNotOptimize(hv);
    }
}

static void BM_BoostHash_String(benchmark::State &state) {
    std::string test_str("this is a string that is a far too "
        "long to fit in the internal memory");

    for (auto _: state) {
        boost::hash<std::string> hasher;
        auto hv = hasher(test_str);
        benchmark::DoNotOptimize(hv);
    }
}

BENCHMARK(BM_baseline_Int64);
BENCHMARK(BM_StdHash_Int64);
BENCHMARK(BM_AbslHash_Int64);
BENCHMARK(BM_BoostHash_Int64);
BENCHMARK(BM_baseline_Double);
BENCHMARK(BM_StdHash_Double);
BENCHMARK(BM_AbslHash_Double);
BENCHMARK(BM_BoostHash_Double);
BENCHMARK(BM_StdHash_String);
BENCHMARK(BM_AbslHash_String);
BENCHMARK(BM_BoostHash_String);


BENCHMARK_MAIN();