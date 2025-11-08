#ifndef DUCKIES_CLUSTERING_KMEANS_H
#define DUCKIES_CLUSTERING_KMEANS_H

#include <span>
#include <vector>

#include "clustering.h"
#include "coord3.h"

namespace duckies {
class KMeans {
    std::vector<int> labels_{};
    std::vector<Coord3> cluster_means_{};
    std::vector<size_t> cluster_size_{};

    std::span<const Coord3> data_{};

    size_t n_clusters_ = 0;

public:
    KMeans(std::span<const Coord3> data, size_t n_clusters)
        : data_(data)
          , n_clusters_(n_clusters) {
        cluster_means_.resize(n_clusters);
        cluster_size_.resize(n_clusters);
        labels_.resize(data_.size());
    }

    size_t n_clusters() const noexcept { return n_clusters_; }
    std::span<const Coord3> data() const noexcept { return data_; }
    const std::vector<int> &labels() const noexcept { return labels_; }

    const std::vector<Coord3> &cluster_means() const noexcept {
        return cluster_means_;
    }

    const std::vector<size_t> &cluster_sizes() const noexcept {
        return cluster_size_;
    }

    void set_labels(std::vector<int> labels);

    void assign_random_labels(size_t seed);

    void recompute_means();

    bool update_labels();
};

bool kmeans_cluster(KMeans &driver, size_t seed, size_t max_iterations = 100);

float compute_score(const KMeans &driver);

std::pair<KMeans, float> best_kmeans_cluster(std::span<const Coord3> data,
                                             size_t k, size_t n_repetitions, size_t max_iterations = 100);
} // namespace duckies

#endif