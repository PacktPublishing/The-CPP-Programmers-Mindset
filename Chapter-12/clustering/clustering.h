#ifndef DUCKIES_CLUSTERING_CLUSTERING_H_
#define DUCKIES_CLUSTERING_CLUSTERING_H_

#include <vector>

#include "duckies_types.h"

namespace duckies {
struct Cluster {
    Coordinate position;
    size_t num_points;
};

[[nodiscard]]
std::vector<Cluster> compute_clusters(const RubberDuckData &data,
                                      size_t min_clusters, size_t max_clusters, size_t num_repetitions,
                                      size_t max_iterations);
} // namespace duckies

#endif // DUCKIES_CLUSTERING_CLUSTERING_H_