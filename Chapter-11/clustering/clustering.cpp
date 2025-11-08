#include "clustering.h"

#include <algorithm>
#include <utility>

#include <spdlog/spdlog.h>

#include "kmeans.h"

using namespace duckies;

std::vector<Cluster> duckies::compute_clusters(const RubberDuckData &data,
                                               size_t min_clusters, size_t max_clusters, size_t num_repetitions,
                                               size_t max_iterations) {
    using ClusterScore = std::pair<KMeans, float>;

    auto embedded_data = coord3_embedding(data.coordinates());

    std::vector<std::pair<Coord3, size_t> > best;
    float best_score = -1.0f;
    auto update_best = [&](ClusterScore &other) {
        if (other.second > best_score) {
            best_score = other.second;
            auto &kmeans = other.first;
            best.clear();
            best.reserve(kmeans.n_clusters());
            for (size_t i = 0; i < kmeans.n_clusters(); ++i) {
                best.emplace_back(
                    kmeans.cluster_means()[i], kmeans.cluster_sizes()[i]);
            }
        }
    };

    constexpr size_t working_set_size = 3;
    std::vector<ClusterScore> working_set;
    working_set.reserve(working_set_size);
    for (size_t i = 0; i < working_set_size; ++i) {
        working_set.emplace_back(KMeans(embedded_data, 0), -1.0f);
    }

    auto best_working = [&] {
        return std::max_element(working_set.begin(), working_set.end(),
                                [](const ClusterScore &l, const ClusterScore &r) {
                                    return l.second < r.second;
                                });
    };

    auto score = [&](size_t i) { return working_set[i].second; };

    auto update = [&](size_t i, size_t new_k) {
        if (working_set[i].first.n_clusters() != new_k) {
            spdlog::info("running kmeans with {} clusters", new_k);
            working_set[i] = best_kmeans_cluster(
                embedded_data, new_k, num_repetitions, max_iterations);
        }
    };

    size_t low_k = min_clusters;
    size_t high_k = max_clusters;

    while (high_k > low_k) {
        spdlog::info("checking clustering in range [{}, {}]", low_k, high_k);
        auto mid_k = (low_k + high_k) / 2;
        update(0, low_k);
        update(1, mid_k);
        update(2, high_k);

        auto iter_best = best_working();
        spdlog::info("the best configuration has "
                     "{} clusters with a score of {}",
                     iter_best->first.n_clusters(), iter_best->second);
        update_best(*iter_best);

        if (score(2) >= score(1) && score(2) > score(0)) {
            low_k = mid_k + 1;
        } else {
            high_k = mid_k - 1;
        }
    }

    std::vector<Cluster> result;
    result.reserve(best.size());

    for (const auto &cluster: best) {
        result.emplace_back(to_coordinates(cluster.first), cluster.second);
    }

    return result;
}