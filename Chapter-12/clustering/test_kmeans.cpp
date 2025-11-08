#include <cmath>
#include <set>

#include <gtest/gtest.h>

#include "kmeans.h"

using namespace duckies;

static constexpr float xy_factor = 0.099833417f;

static constexpr Coord3 polar_data[] = {
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, -1.0f},
    {0.0f, 0.099833417f, 0.995004165},
    {0.099833417f, 0.0f, 0.995004165},
    {-0.099833417f, 0.0f, 0.995004165},
    {0.0f, -0.099833417f, 0.995004165},
    {0.0f, 0.099833417f, -0.995004165},
    {0.099833417f, 0.0f, -0.995004165},
    {-0.099833417f, 0.0f, -0.995004165},
    {0.0f, -0.099833417f, -0.995004165},
};

TEST(KMeans, TestComputeMeans) {
    KMeans k_means(std::span<const Coord3>(polar_data, 10), size_t(2));
    k_means.set_labels({0, 1, 0, 0, 0, 0, 1, 1, 1, 1});

    const auto &sizes = k_means.cluster_sizes();
    EXPECT_EQ(sizes[0], 5);
    EXPECT_EQ(sizes[1], 5);

    k_means.recompute_means();

    auto means = k_means.cluster_means();

    Coord3 north_mean{0.0f, 0.0f, (4 * 0.995004165 + 1.0) / 5.0};
    Coord3 south_mean{0.0f, 0.0f, (4 * -0.995004165 + -1.0) / 5.0};

    EXPECT_NEAR(dist2(north_mean, means[0]), 0.0f, 1e-7);
    EXPECT_NEAR(dist2(south_mean, means[1]), 0.0f, 1e-7);
}

TEST(KMeans, TestMeansEmptyCluster) {
    KMeans k_means(std::span<const Coord3>(polar_data, 10), size_t(3));
    k_means.set_labels({2, 1, 2, 2, 2, 2, 1, 1, 1, 1});
    const auto &sizes = k_means.cluster_sizes();
    ASSERT_EQ(sizes.size(), 3);
    EXPECT_EQ(sizes[0], 0);

    k_means.recompute_means();
    const auto &means = k_means.cluster_means();
    ASSERT_EQ(means.size(), 3);

    EXPECT_EQ(means[0], Coord3(0.0f, 0.0f, 0.0f));
}

TEST(KMeans, TestRandomAssignment) {
    const size_t k = 2;
    KMeans k_means(std::span<const Coord3>(polar_data, 10), k);
    k_means.assign_random_labels(12345);

    std::set<int> seen;
    for (const auto &label: k_means.labels()) {
        EXPECT_GE(label, 0);
        EXPECT_LT(label, k);
        seen.insert(label);
    }

    EXPECT_EQ(seen.size(), k);
}

TEST(KMeans, TestConvergeIterations) {
    KMeans k_means(std::span<const Coord3>(polar_data, 10), size_t(2));
    k_means.assign_random_labels(12345);
    int no_iters = 0;
    bool changed = true;
    for (; changed && no_iters < 10; ++no_iters) {
        k_means.recompute_means();
        changed = k_means.update_labels();
    }

    ASSERT_LT(no_iters, 10);
    const auto &labels = k_means.labels();

    EXPECT_EQ(labels[2], labels[0]);
    EXPECT_EQ(labels[3], labels[0]);
    EXPECT_EQ(labels[4], labels[0]);
    EXPECT_EQ(labels[5], labels[0]);
    EXPECT_EQ(labels[6], labels[1]);
    EXPECT_EQ(labels[7], labels[1]);
    EXPECT_EQ(labels[8], labels[1]);
    EXPECT_EQ(labels[9], labels[1]);
}

TEST(KMeans, TestAlgorithmFunction) {
    KMeans k_means(std::span<const Coord3>(polar_data, 10), size_t(2));
    kmeans_cluster(k_means, 12345, 10);
    const auto &labels = k_means.labels();

    EXPECT_EQ(labels[2], labels[0]);
    EXPECT_EQ(labels[3], labels[0]);
    EXPECT_EQ(labels[4], labels[0]);
    EXPECT_EQ(labels[5], labels[0]);
    EXPECT_EQ(labels[6], labels[1]);
    EXPECT_EQ(labels[7], labels[1]);
    EXPECT_EQ(labels[8], labels[1]);
    EXPECT_EQ(labels[9], labels[1]);
}

TEST(KMeans, TestScoreFunctionSinglePoint) {
    std::vector<Coord3> data{{0.0f, 0.0f, 0.0f}};
    KMeans kmeans(data, 1);
    kmeans.set_labels({0});
    auto score = compute_score(kmeans);

    EXPECT_NEAR(score, 0.5f, 1.0e-7f);
}

TEST(KMeans, TestScoreFunctionTwoClusters) {
    KMeans k_means(std::span<const Coord3>(polar_data, 10), size_t(2));
    k_means.set_labels({0, 1, 0, 0, 0, 0, 1, 1, 1, 1});
    k_means.recompute_means();
    ASSERT_EQ(k_means.n_clusters(), 2);
    auto score = compute_score(k_means);

    EXPECT_GT(score, 0.9f);
}