#include "quick_sort.hpp"

#include <algorithm>
#include <random>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

TEST(QuickSort, TestSortTwoElementSetAlreadySorted) {
    std::vector<int> data{0, 1};
    std::vector<int> expected = data;

    // Without the explicit cast template argument deduction fails here
    // on GCC.
    ct::quicksort(std::span<int>(data));

    EXPECT_EQ(data, expected);
}

TEST(QuickSort, TestSortTwoElementSetUnsorted) {
    std::vector<int> data{1, 0};
    std::vector<int> expected{0, 1};

    // Without the explicit cast template argument deduction fails here
    // on GCC.
    ct::quicksort(std::span<int>(data));

    EXPECT_EQ(data, expected);
}

TEST(QuickSort, TestSortRandomSet) {
    std::vector<int> data;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist{0, 100};

    data.reserve(100);
    for (size_t i = 0; i < 100; ++i) {
        data.push_back(dist(rng));
    }

    ct::quicksort(std::span<int>(data));

    EXPECT_TRUE(std::ranges::is_sorted(data));
}