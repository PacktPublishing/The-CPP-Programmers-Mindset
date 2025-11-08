#include "summary_statistics.hpp"


std::vector<ct::SummaryStatistics> ct::compute_summary_statistics(const Configuration &config,
                                                                  std::span<std::string> sources) {
    std::vector<SummaryStatistics> stats;
    stats.reserve(sources.size());

    for (const auto &src: sources) {
        stats.emplace_back(0.0, 0.0, 0.0, 0.0);
        auto &these_stats = stats.back();

        // simulate computing the stats
        if (config.include_mean()) {
            these_stats.mean = 1.0;
        }

        if (config.include_std()) {
            these_stats.st_dev = 2.718;
        }

        if (config.include_min()) {
            these_stats.min = -50.0;
        }

        if (config.include_max()) {
            these_stats.max = 50.0;
        }
    }

    return stats;
}