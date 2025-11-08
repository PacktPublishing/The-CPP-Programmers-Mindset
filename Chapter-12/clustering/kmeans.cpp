#include "kmeans.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

#include <spdlog/spdlog.h>

using namespace duckies;

void KMeans::assign_random_labels(size_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_clusters_ - 1);

    spdlog::debug("assigning random labels with seed {}", seed);

    // clear any lingering cluster sizes
    cluster_size_.assign(n_clusters_, 0);

    for (auto &label: labels_) {
        label = dist(rng);
        cluster_size_[label] += 1;
    }

    for (size_t i = 0; i < n_clusters_; ++i) {
        spdlog::debug("cluster {} has {} points", i, cluster_size_[i]);
    }
}

void KMeans::recompute_means() {
    const auto N = data_.size();
    cluster_means_.assign(n_clusters_, Coord3{0.0f, 0.0f, 0.0f});

    for (size_t i = 0; i < N; ++i) {
        auto label = labels_[i];
        cluster_means_[label] += data_[i];
    }

    for (size_t i = 0; i < n_clusters_; ++i) {
        if (cluster_size_[i] > 0) {
            cluster_means_[i] *= 1.0f / static_cast<float>(cluster_size_[i]);
        }
    }
}

template<typename It>
static auto min_distance(
    const Coord3 &pt, const It &begin, const It &end) noexcept {
    It min_value = begin;
    float current_best = std::numeric_limits<float>::infinity();
    for (auto it = begin; it != end; ++it) {
        auto this_distance = dist2_squared(pt, *it);
        if (this_distance < current_best) {
            min_value = it;
            current_best = this_distance;
        }
    }
    return min_value;
}

bool KMeans::update_labels() {
    bool has_changed = false;
    const auto N = data_.size();

    const auto means_begin = cluster_means_.begin();
    const auto means_end = cluster_means_.end();

    for (size_t i = 0; i < N; ++i) {
        const auto &pt = data_[i];
        auto min_pos = min_distance(pt, means_begin, means_end);
        auto new_label = static_cast<int>(min_pos - means_begin);
        auto old_label = labels_[i];

        if (new_label != old_label) {
            spdlog::debug("assigning label {} to point {}", new_label, i);
            labels_[i] = new_label;
            --cluster_size_[old_label];
            ++cluster_size_[new_label];
            has_changed = true;
        }
    }

    return has_changed;
}

void KMeans::set_labels(std::vector<int> labels) {
    assert(labels.size() == data_.size());
    labels_ = std::move(labels);

    cluster_size_.clear();
    spdlog::debug("setting labels to given");

    for (const auto &label: labels_) {
        if (label >= static_cast<int>(cluster_size_.size())) {
            cluster_size_.resize(label + 1);
        }

        cluster_size_[label] += 1;
    }

    n_clusters_ = cluster_size_.size();
    cluster_means_.assign(n_clusters_, Coord3{0.0f, 0.0f, 0.0f});
    spdlog::info("new labels have {} clusters", n_clusters_);
}

bool duckies::kmeans_cluster(KMeans &driver, size_t seed, size_t max_iterations) {
    driver.assign_random_labels(seed);

    size_t i = 0;
    bool has_changed = true;
    for (; i < max_iterations && has_changed; ++i) {
        driver.recompute_means();
        has_changed = driver.update_labels();
    }

    if (i < max_iterations) {
        spdlog::info("kmeans converged in {} iterations", i);
    } else {
        spdlog::warn(
            "kmeans failed to converge in {} iterations", max_iterations);
    }
    return i < max_iterations;
}

float duckies::compute_score(const KMeans &driver) {
    const auto &means = driver.cluster_means();
    const auto &labels = driver.labels();
    const auto data = driver.data();
    const auto num_clusters = driver.n_clusters();

    float silhouette = 0.0f;

    for (size_t i = 0; i < data.size(); ++i) {
        float a_i = dist2(data[i], means[labels[i]]);
        float b_i = 1.0f;

        for (size_t cluster_i = 0; cluster_i < num_clusters; ++cluster_i) {
            if (cluster_i != labels[i]) {
                auto dist = dist2(means[cluster_i], data[i]);
                if (dist < b_i) {
                    b_i = dist;
                }
            }
        }

        silhouette += (b_i - a_i) / std::max(a_i, b_i);
    }

    silhouette /= static_cast<float>(data.size());

    const auto &sizes = driver.cluster_sizes();
    // auto num_zeros = std::count(sizes.begin(), sizes.end(), 0);
    // auto penalty = static_cast<float>(num_zeros);
    float penalty = 0.0f;
    for (size_t i = 0; i < num_clusters; ++i) {
        if (sizes[i] == 0) {
            penalty += 1.0f;
        } else if (sizes[i] == 1) {
            penalty += 0.5f;
        }
    }
    penalty /= num_clusters;


    return silhouette - penalty;
}

std::pair<KMeans, float> duckies::best_kmeans_cluster(
    std::span<const Coord3> data, size_t k, size_t n_repetitions,
    size_t max_iterations) {
    std::vector<KMeans> drivers;
    std::vector<float> scores(n_repetitions, -1.0f);
    drivers.reserve(n_repetitions);
    for (size_t i = 0; i < n_repetitions; ++i) {
        drivers.emplace_back(data, k);
    }

#pragma omp parallel for default(none) shared(drivers, scores)
    for (size_t i = 0; i < n_repetitions; ++i) {
        size_t seed = std::random_device{}();
        if (kmeans_cluster(drivers[i], seed, max_iterations)) {
            scores[i] = compute_score(drivers[i]);
        }
    }

    auto best = std::max_element(scores.begin(), scores.end());
    auto best_idx = static_cast<size_t>(best - scores.begin());

    return {std::move(drivers[best_idx]), *best};
}